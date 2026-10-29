/**
 * Reads any heart dataset in CSV form into typed columns, entirely in the
 * browser. Nothing is assumed about the schema: the delimiter (comma,
 * semicolon, tab), header row, and each column's kind are detected, and
 * common encodings are turned into numbers so they can drive the twin and
 * train a model:
 *
 *  - Yes/No, True/False, Present/Absent, Positive/Negative, Male/Female -> 1/0
 *    (values starting "Yes"/"No", e.g. "No, borderline diabetes", count too)
 *  - age bands such as "55-59" -> 57, "80 or older" / "80+" -> 82
 *  - a combined blood pressure such as "158/88" -> two columns, systolic and
 *    diastolic
 *  - anything else with few distinct values -> a category (one-hot in the model)
 */

export type ColumnKind = "numeric" | "binary" | "category" | "text";

export interface Column {
  name: string;
  kind: ColumnKind;
  /** Numbers for numeric/binary columns (NaN when missing); category index for categories. */
  values: Float64Array;
  /** Category labels, for kind "category". */
  categories: string[];
  /** For binary text columns: what 1 stands for, e.g. "Yes" or "Male". */
  positiveLabel?: string;
  /** For binary text columns: what 0 stands for. */
  negativeLabel?: string;
  /** Text column values (kind "text" only, e.g. IDs). */
  text?: string[];
  stats: { min: number; max: number; median: number; p01: number; p99: number; distinct: number; missing: number };
  /** One-line note on how the raw values were read, shown in the mapping panel. */
  note?: string;
}

export interface Dataset {
  name: string;
  rows: number;
  columns: Column[];
  /** Rows dropped by the size cap (very large files are sampled). */
  sampledFrom?: number;
}

const MAX_ROWS = 60000;
const MAX_CATEGORIES = 15;

function detectDelimiter(head: string): string {
  const firstLine = head.split(/\r?\n/, 1)[0];
  let best = ",";
  let bestCount = 0;
  for (const d of [",", ";", "\t", "|"]) {
    const count = firstLine.split(d).length - 1;
    if (count > bestCount) {
      best = d;
      bestCount = count;
    }
  }
  return best;
}

/** RFC-4180-style split, handling quoted fields with embedded delimiters, quotes and newlines. */
function parseRecords(text: string, delim: string): string[][] {
  const records: string[][] = [];
  let field = "";
  let record: string[] = [];
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else quoted = false;
      } else field += ch;
    } else if (ch === '"' && field === "") quoted = true;
    else if (ch === delim) {
      record.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      record.push(field);
      field = "";
      if (record.length > 1 || record[0] !== "") records.push(record);
      record = [];
    } else field += ch;
  }
  record.push(field);
  if (record.length > 1 || record[0] !== "") records.push(record);
  return records;
}

const isNumber = (s: string) => s !== "" && Number.isFinite(Number(s));
const MISSING = new Set(["", "na", "n/a", "nan", "null", "none", "?", "-", "missing"]);

const BINARY_PAIRS: [RegExp, RegExp, string, string][] = [
  [/^(yes|y)\b/i, /^(no|n)\b/i, "Yes", "No"],
  [/^true$/i, /^false$/i, "True", "False"],
  [/^(male|m)$/i, /^(female|f)$/i, "Male", "Female"],
  [/^present$/i, /^absent$/i, "Present", "Absent"],
  [/^(positive|pos)$/i, /^(negative|neg)$/i, "Positive", "Negative"],
  [/^abnormal$/i, /^normal$/i, "Abnormal", "Normal"],
];

function ageBand(s: string): number | null {
  let m = s.match(/^(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)$/);
  if (m) return (Number(m[1]) + Number(m[2])) / 2;
  m = s.match(/^(\d+(?:\.\d+)?)\s*(?:\+|or older|and over|or more|and above)$/i);
  if (m) return Number(m[1]) + 2;
  m = s.match(/^(?:under|below|<)\s*(\d+(?:\.\d+)?)$/i);
  if (m) return Number(m[1]) - 2;
  return null;
}

function percentile(sorted: number[], q: number): number {
  if (!sorted.length) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

function withStats(c: Omit<Column, "stats">): Column {
  const finite: number[] = [];
  let missing = 0;
  for (const v of c.values) {
    if (Number.isFinite(v)) finite.push(v);
    else missing++;
  }
  finite.sort((a, b) => a - b);
  let distinct = 0;
  for (let i = 0; i < finite.length; i++) if (i === 0 || finite[i] !== finite[i - 1]) distinct++;
  return {
    ...c,
    stats: {
      min: finite.length ? finite[0] : NaN,
      max: finite.length ? finite[finite.length - 1] : NaN,
      median: percentile(finite, 0.5),
      p01: percentile(finite, 0.01),
      p99: percentile(finite, 0.99),
      distinct,
      missing,
    },
  };
}

function buildColumns(name: string, raw: string[]): Column[] {
  const present = raw.map((s) => s.trim()).map((s) => (MISSING.has(s.toLowerCase()) ? "" : s));
  const nonEmpty = present.filter((s) => s !== "");
  const n = present.length;
  const values = new Float64Array(n);

  // Plain numbers
  if (nonEmpty.length && nonEmpty.filter(isNumber).length / nonEmpty.length >= 0.97) {
    present.forEach((s, i) => (values[i] = isNumber(s) ? Number(s) : NaN));
    const col = withStats({ name, kind: "numeric", values, categories: [] });
    if (col.stats.distinct === 2 && col.stats.min === 0 && col.stats.max === 1) col.kind = "binary";
    return [col];
  }

  // "120/80" -> systolic + diastolic
  if (nonEmpty.length && nonEmpty.filter((s) => /^\d+(\.\d+)?\s*\/\s*\d+(\.\d+)?$/.test(s)).length / nonEmpty.length >= 0.97) {
    const sys = new Float64Array(n);
    const dia = new Float64Array(n);
    present.forEach((s, i) => {
      const [a, b] = s.split("/").map((x) => Number(x.trim()));
      sys[i] = s ? a : NaN;
      dia[i] = s ? b : NaN;
    });
    const note = `Split from "${name}" (written as systolic/diastolic)`;
    return [
      withStats({ name: `${name} (systolic)`, kind: "numeric", values: sys, categories: [], note }),
      withStats({ name: `${name} (diastolic)`, kind: "numeric", values: dia, categories: [], note }),
    ];
  }

  // Age bands
  if (nonEmpty.length && nonEmpty.every((s) => ageBand(s) !== null || isNumber(s))) {
    present.forEach((s, i) => (values[i] = s === "" ? NaN : isNumber(s) ? Number(s) : (ageBand(s) as number)));
    return [withStats({ name, kind: "numeric", values, categories: [], note: "Ranges such as 55-59 read as their midpoint" })];
  }

  const distinct = [...new Set(nonEmpty)];

  // Two-valued text (Yes/No, Male/Female ...)
  for (const [pos, neg, posLabel, negLabel] of BINARY_PAIRS) {
    if (nonEmpty.length && nonEmpty.every((s) => pos.test(s) || neg.test(s))) {
      present.forEach((s, i) => (values[i] = s === "" ? NaN : pos.test(s) ? 1 : 0));
      const note = distinct.length > 2 ? `${distinct.length} answers grouped as ${posLabel} / ${negLabel}` : undefined;
      return [withStats({ name, kind: "binary", values, categories: [], positiveLabel: posLabel, negativeLabel: negLabel, note })];
    }
  }

  if (distinct.length <= MAX_CATEGORIES) {
    const categories = distinct.sort();
    const index = new Map(categories.map((c, i) => [c, i]));
    present.forEach((s, i) => (values[i] = s === "" ? NaN : (index.get(s) as number)));
    return [withStats({ name, kind: "category", values, categories })];
  }

  values.fill(NaN);
  return [withStats({ name, kind: "text", values, categories: [], text: present })];
}

export function parseCsv(text: string, name: string): Dataset {
  text = text.replace(/^﻿/, "");
  const delim = detectDelimiter(text.slice(0, 5000));
  let records = parseRecords(text, delim);
  if (records.length < 2) throw new Error("The file needs a header row and at least one data row.");

  // No header row if every field of the first record is a number (e.g. ECG beat files).
  let header = records[0].map((h) => h.trim());
  if (header.every(isNumber)) header = header.map((_, i) => `Column ${i + 1}`);
  else records = records.slice(1);
  header = header.map((h, i) => h || `Column ${i + 1}`);

  let sampledFrom: number | undefined;
  if (records.length > MAX_ROWS) {
    sampledFrom = records.length;
    // Even spacing keeps the sample representative and repeatable.
    const step = records.length / MAX_ROWS;
    records = Array.from({ length: MAX_ROWS }, (_, i) => records[Math.floor(i * step)]);
  }

  const columns: Column[] = [];
  header.forEach((h, j) => {
    columns.push(...buildColumns(h, records.map((r) => r[j] ?? "")));
  });
  return { name, rows: records.length, columns, sampledFrom };
}

/** Display text for one cell. */
export function cellText(c: Column, row: number): string {
  if (c.kind === "text") return c.text?.[row] || "—";
  const v = c.values[row];
  if (!Number.isFinite(v)) return "—";
  if (c.kind === "category") return c.categories[v];
  if (c.kind === "binary" && c.positiveLabel) return v ? c.positiveLabel : (c.negativeLabel ?? "No");
  return Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/\.?0+$/, "");
}
