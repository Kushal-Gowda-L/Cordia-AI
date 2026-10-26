import type { Column, Dataset } from "./csv";

/**
 * A risk model trained in the browser on whichever dataset is loaded, so the
 * twin works with any team's data rather than one fixed schema. It is an
 * L2-regularised logistic regression: every numeric / yes-no column except
 * the outcome is standardised (missing values filled with the column median),
 * every category column is one-hot encoded, and the outcome is read as
 * "has the condition" vs. not. 80% of rows train it and the other 20% (a
 * fixed shuffle, so results repeat) measure it, giving the AUC and accuracy
 * shown in the panel. Simple on purpose: it trains in about a second on
 * 60,000 rows, its per-column effects can be read off directly, and its
 * probabilities are reasonably calibrated -- which matters because the
 * estimate drives the heart.
 */

/** One row's values, one entry per dataset column (NaN = missing; category index for categories). */
export type RowValues = number[];

interface Feature {
  column: number;
  category?: number; // one-hot level, for category columns
  center: number;
  scale: number;
  label: string;
}

export interface RiskModel {
  targetColumn: number;
  targetLabel: string;
  positiveLabel: string;
  features: Feature[];
  weights: Float64Array;
  bias: number;
  auc: number;
  accuracy: number;
  positiveRate: number;
  nTrain: number;
  nTest: number;
}

/** 1 = has the outcome, 0 = not, NaN = unknown. */
export function outcomeOf(col: Column, v: number): number {
  if (!Number.isFinite(v)) return NaN;
  if (col.kind === "category") {
    return /^(no|none|normal|absent|negative|healthy|false|0)\b/i.test(col.categories[v]) ? 0 : 1;
  }
  if (col.stats.distinct === 2) return v === col.stats.max ? 1 : 0;
  return v > Math.min(0, col.stats.min) ? 1 : 0; // e.g. UCI "num" 0-4: any level of disease
}

export function positiveLabelOf(col: Column): string {
  if (col.kind === "binary" && col.positiveLabel) return col.positiveLabel;
  if (col.kind === "category") return col.categories.filter((c, i) => outcomeOf(col, i) === 1).join(" / ") || "positive";
  if (col.stats.distinct === 2) return String(col.stats.max);
  return `above ${Math.min(0, col.stats.min)}`;
}

/** Columns a model can learn from: everything but free text, the outcome, and mostly-empty columns. */
export function usableFeatureColumns(ds: Dataset, target: number): number[] {
  return ds.columns
    .map((c, i) => ({ c, i }))
    .filter(({ c, i }) => i !== target && c.kind !== "text" && c.stats.missing < ds.rows * 0.5 && c.stats.distinct > 1)
    .map(({ i }) => i);
}

function buildFeatures(ds: Dataset, columns: number[]): Feature[] {
  const out: Feature[] = [];
  for (const j of columns) {
    const c = ds.columns[j];
    if (c.kind === "category") {
      c.categories.forEach((cat, k) => out.push({ column: j, category: k, center: 0, scale: 1, label: `${c.name} = ${cat}` }));
    } else {
      let sum = 0;
      let sq = 0;
      let n = 0;
      for (const v of c.values) {
        if (Number.isFinite(v)) {
          sum += v;
          sq += v * v;
          n++;
        }
      }
      const mean = sum / n;
      const sd = Math.sqrt(Math.max(1e-12, sq / n - mean * mean));
      out.push({ column: j, center: mean, scale: sd, label: c.name });
    }
  }
  return out;
}

function encode(ds: Dataset, features: Feature[], row: RowValues, into: Float64Array) {
  features.forEach((f, k) => {
    const v = row[f.column];
    if (f.category !== undefined) into[k] = v === f.category ? 1 : 0;
    else into[k] = ((Number.isFinite(v) ? v : ds.columns[f.column].stats.median) - f.center) / f.scale;
  });
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, z))));

/** Area under the ROC curve via the rank-sum formula. */
function auc(scores: number[], labels: number[]): number {
  const idx = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b]);
  let rankSum = 0;
  let pos = 0;
  for (let r = 0; r < idx.length; ) {
    let e = r;
    while (e + 1 < idx.length && scores[idx[e + 1]] === scores[idx[r]]) e++;
    const avgRank = (r + e) / 2 + 1;
    for (let k = r; k <= e; k++) if (labels[idx[k]]) {
      rankSum += avgRank;
      pos++;
    }
    r = e + 1;
  }
  const neg = labels.length - pos;
  return pos && neg ? (rankSum - (pos * (pos + 1)) / 2) / (pos * neg) : NaN;
}

export function rowValues(ds: Dataset, row: number): RowValues {
  return ds.columns.map((c) => c.values[row]);
}

export function trainRiskModel(ds: Dataset, target: number): RiskModel {
  const tcol = ds.columns[target];
  const features = buildFeatures(ds, usableFeatureColumns(ds, target));
  const d = features.length;

  const rows: number[] = [];
  for (let i = 0; i < ds.rows; i++) if (Number.isFinite(outcomeOf(tcol, tcol.values[i]))) rows.push(i);
  if (rows.length < 20) throw new Error("Too few rows have a value in the outcome column to train on.");
  // Fixed-seed shuffle, so the reported accuracy repeats on every load.
  let seed = 12345;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  const nTest = Math.max(1, Math.round(rows.length * 0.2));
  const test = rows.slice(0, nTest);
  // Capped so training stays around a second on big files; more rows add little to a linear model.
  const train = rows.slice(nTest, nTest + 25000);

  const X = new Float64Array(train.length * d);
  const y = new Float64Array(train.length);
  const buf = new Float64Array(d);
  train.forEach((r, i) => {
    encode(ds, features, rowValues(ds, r), buf);
    X.set(buf, i * d);
    y[i] = outcomeOf(tcol, tcol.values[r]);
  });
  const positiveRate = y.reduce((a, b) => a + b, 0) / y.length;

  // Full-batch gradient descent with momentum; the inputs are standardised so one step size suits all.
  const w = new Float64Array(d);
  const vel = new Float64Array(d + 1);
  let b = Math.log(Math.max(1e-6, positiveRate) / Math.max(1e-6, 1 - positiveRate));
  const lambda = 1e-3;
  const lr = 0.5;
  const grad = new Float64Array(d);
  for (let it = 0; it < 300; it++) {
    grad.fill(0);
    let gb = 0;
    for (let i = 0; i < train.length; i++) {
      let z = b;
      const o = i * d;
      for (let k = 0; k < d; k++) z += w[k] * X[o + k];
      const err = sigmoid(z) - y[i];
      gb += err;
      for (let k = 0; k < d; k++) grad[k] += err * X[o + k];
    }
    for (let k = 0; k < d; k++) {
      vel[k] = 0.9 * vel[k] + lr * (grad[k] / train.length + lambda * w[k]);
      w[k] -= vel[k];
    }
    vel[d] = 0.9 * vel[d] + lr * (gb / train.length);
    b -= vel[d];
  }

  const model: RiskModel = {
    targetColumn: target,
    targetLabel: tcol.name,
    positiveLabel: positiveLabelOf(tcol),
    features,
    weights: w,
    bias: b,
    auc: NaN,
    accuracy: NaN,
    positiveRate,
    nTrain: train.length,
    nTest,
  };
  const scores = test.map((r) => predictRisk(ds, model, rowValues(ds, r)));
  const labels = test.map((r) => outcomeOf(tcol, tcol.values[r]));
  model.auc = auc(scores, labels);
  model.accuracy = scores.filter((s, i) => (s >= 0.5 ? 1 : 0) === labels[i]).length / test.length;
  return model;
}

export function predictRisk(ds: Dataset, m: RiskModel, row: RowValues): number {
  const x = new Float64Array(m.features.length);
  encode(ds, m.features, row, x);
  let z = m.bias;
  for (let k = 0; k < x.length; k++) z += m.weights[k] * x[k];
  return sigmoid(z);
}

/** The inputs pushing this row's estimate up or down the most, relative to an average row. */
export function topFactors(ds: Dataset, m: RiskModel, row: RowValues, n = 4): { label: string; effect: number }[] {
  const x = new Float64Array(m.features.length);
  encode(ds, m.features, row, x);
  const byColumn = new Map<number, { label: string; effect: number }>();
  m.features.forEach((f, k) => {
    const effect = m.weights[k] * (f.category !== undefined ? x[k] - 1 / ds.columns[f.column].categories.length : x[k]);
    const prev = byColumn.get(f.column);
    byColumn.set(f.column, { label: ds.columns[f.column].name, effect: (prev?.effect ?? 0) + effect });
  });
  return [...byColumn.values()]
    .filter((f) => Math.abs(f.effect) > 0.05)
    .sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect))
    .slice(0, n);
}
