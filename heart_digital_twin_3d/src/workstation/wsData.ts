import type { Column, Dataset } from "../dataset/csv";
import { F, type State } from "./models";

/**
 * Fills the workstation's patient fields from any team dataset (CSV). Each
 * field is matched to a column by name (several spellings per field, from the
 * UCI, Framingham, cardio-70k, BRFSS and Kaggle heart files), and the column's
 * coding is read into the field's units: days to years, mmol/L to mg/dL,
 * 0-based Kaggle codes to the UCI ones, yes/no text to 0/1. Fields the
 * dataset doesn't have keep the patient's current value.
 */

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

type Read = (c: Column, v: number) => number | null;
const asIs: Read = (_c, v) => v;
const flag: Read = (c, v) => (c.kind === "category" ? (/^(no|never|none|false|0)/i.test(c.categories[v]) ? 0 : 1) : v > 0 ? 1 : 0);
const label = (c: Column, v: number) => (c.kind === "category" ? (c.categories[v] ?? "") : "");

interface Alias {
  names: string[];
  read: Read;
}

const ALIASES: Record<string, Alias> = {
  age: {
    names: ["age", "ageyears", "patientage", "agecategory", "agegroup"],
    read: (c, v) => (c.stats.median > 150 ? v / 365.25 : c.stats.max <= 14 && c.stats.min >= 1 ? 20 + (v - 1) * 5 : v),
  },
  sex: {
    names: ["sex", "male", "gender", "ismale"],
    read: (c, v) => {
      if (c.kind === "category") return /^m/i.test(c.categories[v]) ? 1 : 0;
      if (c.kind === "binary" && c.positiveLabel) return /^m/i.test(c.positiveLabel) ? v : 1 - v;
      if (c.stats.min === 1 && c.stats.max === 2) return v === 2 ? 1 : 0;
      return v ? 1 : 0;
    },
  },
  height_cm: {
    names: ["height", "heightcm"],
    read: (c, v) => (c.stats.median < 3 ? v * 100 : v),
  },
  weight_kg: { names: ["weight", "weightkg"], read: asIs },
  sys_bp: {
    names: ["sysbp", "trestbps", "trtbps", "aphi", "sbp", "systolic", "systolicbp", "restingbp", "restingbloodpressure"],
    read: asIs,
  },
  dia_bp: {
    names: ["diabp", "aplo", "dbp", "diastolic", "diastolicbp"],
    read: asIs,
  },
  heart_rate: {
    names: ["heartrate", "restinghr", "restingheartrate", "hr", "pulse", "pulserate"],
    read: asIs,
  },
  total_chol: {
    names: ["totchol", "chol", "cholesterol", "serumcholesterol", "totalcholesterol"],
    read: (c, v) =>
      c.stats.max <= 3 && c.stats.min >= 1 ? ([190, 220, 260][Math.round(v) - 1] ?? 190) : c.stats.median < 20 ? v * 38.67 : v,
  },
  glucose: {
    names: ["glucose", "gluc", "fastingglucose", "bloodsugar", "fbs", "fastingbs", "fastingbloodsugar"],
    read: (c, v) =>
      c.kind === "binary" || c.stats.max <= 1
        ? v
          ? 140
          : 95
        : c.stats.max <= 3 && c.stats.min >= 1
          ? ([90, 112, 140][Math.round(v) - 1] ?? 95)
          : c.stats.median < 20
            ? v * 18
            : v,
  },
  smoker: {
    names: ["currentsmoker", "smoker", "smoke", "smoking", "smokingstatus"],
    read: flag,
  },
  cigs_per_day: {
    names: ["cigsperday", "cigarettesperday", "cigs"],
    read: asIs,
  },
  active: { names: ["active", "physactivity", "physicalactivity"], read: flag },
  alcohol_heavy: {
    names: ["alco", "hvyalcoholconsump", "alcoholheavy", "heavyalcohol", "alcoholdrinking"],
    read: flag,
  },
  fruits_veg: { names: ["fruitsveg", "fruits", "veggies"], read: flag },
  diabetes: {
    names: ["diabetes", "diabetic", "diabetesdx"],
    read: (c, v) =>
      c.kind === "category" ? (/^yes/i.test(c.categories[v]) ? 1 : 0) : c.stats.max === 2 ? (v === 2 ? 1 : 0) : v > 0 ? 1 : 0,
  },
  hypertension_dx: {
    names: ["prevalenthyp", "hypertension", "highbp", "hypertensiondx"],
    read: flag,
  },
  bp_meds: {
    names: ["bpmeds", "bpmedication", "bloodpressuremeds"],
    read: flag,
  },
  prior_stroke: {
    names: ["prevalentstroke", "stroke", "priorstroke"],
    read: flag,
  },
  gen_health: {
    names: ["genhlth", "genhealth", "generalhealth"],
    read: (c, v) => {
      const t = label(c, v).toLowerCase();
      if (t) return t.startsWith("exc") ? 1 : t.startsWith("very") ? 2 : t.startsWith("good") ? 3 : t.startsWith("fair") ? 4 : 5;
      return v >= 1 && v <= 5 ? Math.round(v) : null;
    },
  },
  diff_walking: {
    names: ["diffwalk", "diffwalking", "difficultywalking"],
    read: flag,
  },
  cp: {
    names: ["cp", "chestpaintype", "chestpain"],
    read: (c, v) => {
      const t = label(c, v).toLowerCase();
      if (t) return /^(ta|typ)/.test(t) ? 1 : /^(ata|aty)/.test(t) ? 2 : /^(nap|non)/.test(t) ? 3 : 4;
      return c.stats.min === 0 ? Math.round(v) + 1 : Math.round(v);
    },
  },
  restecg: {
    names: ["restecg", "restingecg"],
    read: (c, v) => {
      const t = label(c, v).toLowerCase();
      if (t) return t.startsWith("lvh") ? 2 : t.startsWith("st") ? 1 : 0;
      return Math.min(2, Math.max(0, Math.round(v)));
    },
  },
  thalach: {
    names: ["thalach", "thalachh", "maxhr", "maxheartrate", "maximumheartrate", "peakhr"],
    read: asIs,
  },
  exang: {
    names: ["exang", "exng", "exerciseangina", "exerciseinducedangina"],
    read: flag,
  },
  oldpeak: {
    names: ["oldpeak", "stdepression"],
    read: (_c, v) => Math.max(0, v),
  },
  slope: {
    names: ["slope", "stslope", "slp"],
    read: (c, v) => {
      const t = label(c, v).toLowerCase();
      if (t) return t.startsWith("up") ? 1 : t.startsWith("flat") ? 2 : 3;
      return c.stats.min === 0 ? Math.round(v) + 1 : Math.round(v);
    },
  },
  ca: {
    names: ["ca", "caa", "majorvessels"],
    read: (_c, v) => (v >= 0 && v <= 3 ? Math.round(v) : -1),
  },
  thal: {
    names: ["thal", "thall"],
    read: (c, v) => {
      const t = label(c, v).toLowerCase();
      if (t) return t.startsWith("norm") ? 3 : t.startsWith("fix") ? 6 : t.startsWith("rev") ? 7 : -1;
      if (c.stats.max <= 3) return [-1, 6, 3, 7][Math.round(v)] ?? -1; // Kaggle 0-3
      return [3, 6, 7].includes(Math.round(v)) ? Math.round(v) : -1;
    },
  },
  ef: {
    names: ["ejectionfraction", "ef", "lvef"],
    read: (c, v) => (c.stats.max <= 1 ? v * 100 : v),
  },
};

/** Which column (by index) fills each patient field; BMI fills weight when there's no weight column. */
export interface Matches {
  fields: Record<string, number>;
  bmi: number | null;
}

export function matchColumns(ds: Dataset): Matches {
  const names = ds.columns.map((c) => norm(c.name));
  const used = new Set<number>();
  const fields: Record<string, number> = {};
  // Earlier spellings win; every column fills at most one field.
  for (const key in ALIASES)
    for (const n of ALIASES[key].names) {
      const j = names.findIndex((x, i) => x === n && !used.has(i) && ds.columns[i].kind !== "text");
      if (j >= 0) {
        fields[key] = j;
        used.add(j);
        break;
      }
    }
  const b = names.findIndex((x, i) => x === "bmi" && !used.has(i));
  return { fields, bmi: fields.weight_kg === undefined && b >= 0 ? b : null };
}

const clampTo = (k: string, v: number) => {
  const f = F[k];
  if (f.v) return Object.keys(f.v).map(Number).includes(v) ? v : null;
  return Math.min(f.max!, Math.max(f.min!, f.s! < 1 ? Math.round(v * 10) / 10 : Math.round(v)));
};

/** One dataset row as patient fields, on top of `from` (missing cells keep `from`'s values). */
export function rowState(ds: Dataset, m: Matches, row: number, from: State): State {
  const s = { ...from };
  for (const k in m.fields) {
    const c = ds.columns[m.fields[k]];
    const raw = c.values[row];
    if (!Number.isFinite(raw)) continue;
    const v = ALIASES[k].read(c, raw);
    if (v === null || !Number.isFinite(v)) continue;
    const fitted = clampTo(k, v);
    if (fitted !== null) s[k] = fitted;
  }
  if (m.bmi !== null) {
    const b = ds.columns[m.bmi].values[row];
    if (Number.isFinite(b)) s.weight_kg = clampTo("weight_kg", b * Math.pow(s.height_cm / 100, 2))!;
  }
  // A dataset without a smoking amount: a smoker smokes a typical 15/day, a non-smoker none.
  if (m.fields.smoker !== undefined && m.fields.cigs_per_day === undefined) s.cigs_per_day = s.smoker ? 15 : 0;
  if (s.smoker === 0) s.cigs_per_day = 0;
  return s;
}
