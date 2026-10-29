import type { HeartParams } from "../simulation/heartModel";
import type { ParamSources, TwinScenario, ScenarioKey } from "../patient/patientTwin";
import type { Column, Dataset } from "./csv";

/**
 * Connects any dataset's columns to the heart. Each "role" is one thing the
 * heart simulation can use (age, blood pressure, heart rate ...). Columns are
 * matched to roles by name -- across the naming styles of the common public
 * heart datasets (UCI, Framingham, Kaggle cardio, BRFSS, the 2020 CDC survey,
 * heart-failure records) -- and the person can change any match in the panel.
 * Each role also reads its column's units: age in days or as a survey age-group
 * code, a yes/no "high blood pressure" flag instead of mm Hg, cholesterol as
 * 1-3 levels or mmol/L, and so on, and says how it read them.
 */

export type RoleKey =
  | "age"
  | "sex"
  | "restingHR"
  | "maxHR"
  | "systolic"
  | "diastolic"
  | "cholesterol"
  | "glucose"
  | "ejectionFraction"
  | "oxygen"
  | "exerciseAngina"
  | "stDepression";

export interface Reading {
  value: number;
  note?: string;
}

export interface RoleDef {
  key: RoleKey;
  label: string;
  unit: string;
  exact: string[];
  contains: string[];
  /** What it changes in the simulated heart, shown as a hint. */
  effect: string;
  read: (c: Column, v: number) => Reading;
}

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
const asIs = (_c: Column, v: number): Reading => ({ value: v });
const flagOrValue = (yes: number, no: number, what: string) => (c: Column, v: number): Reading =>
  c.kind === "binary"
    ? { value: v ? yes : no, note: `Yes/no column: "yes" read as ${yes}, "no" as ${no} ${what}` }
    : { value: v };
const fraction = (c: Column, v: number): Reading =>
  c.stats.max <= 1 ? { value: v * 100, note: "Written as a fraction, read as a percentage" } : { value: v };

export const ROLES: RoleDef[] = [
  {
    key: "age",
    label: "Age",
    unit: "years",
    exact: ["age", "ageyears", "agecategory", "agegroup", "patientage"],
    contains: ["age"],
    effect: "Stiffer arteries with age",
    read: (c, v) => {
      if (c.stats.median > 150) return { value: v / 365.25, note: "Age in days, converted to years" };
      if (c.stats.max <= 14 && c.stats.min >= 1 && Number.isInteger(c.stats.median)) {
        return { value: 18 + (v - 1) * 5 + 2, note: "Age-group code (1 = 18-24 ... 13 = 80+), read as the group's middle age" };
      }
      return { value: v };
    },
  },
  {
    key: "sex",
    label: "Sex",
    unit: "",
    exact: ["sex", "gender", "male", "ismale"],
    contains: ["gender"],
    effect: "Typical stroke volume",
    read: (c, v) => {
      if (c.kind === "binary" && c.positiveLabel === "Male") return { value: v };
      if (c.kind === "category") return { value: /^m/i.test(c.categories[v]) ? 1 : 0 };
      if (c.stats.min === 1 && c.stats.max === 2) return { value: v === 2 ? 1 : 0, note: "Coded 1/2: read as 1 = female, 2 = male" };
      return { value: v ? 1 : 0, note: "Coded 0/1: read as 1 = male" };
    },
  },
  {
    key: "restingHR",
    label: "Resting heart rate",
    unit: "beats/min",
    exact: ["heartrate", "restinghr", "restingheartrate", "hr", "pulse", "pulserate", "restingpulse"],
    contains: ["heartrate", "pulse"],
    effect: "How fast the heart beats at rest",
    read: asIs,
  },
  {
    key: "maxHR",
    label: "Peak exercise heart rate",
    unit: "beats/min",
    exact: ["thalach", "thalachh", "maxhr", "maxheartrate", "maximumheartrate", "peakhr", "maxheartrateachieved"],
    contains: ["maxhr", "maxheart", "maximumheart"],
    effect: "Heart rate in the exercise scenario",
    read: asIs,
  },
  {
    key: "systolic",
    label: "Systolic blood pressure",
    unit: "mm Hg",
    exact: ["trestbps", "trtbps", "aphi", "sysbp", "sbp", "systolic", "systolicbp", "restingbp", "restingbloodpressure", "bloodpressure", "highbp", "hypertension", "prevalenthyp"],
    contains: ["systolic"],
    effect: "Pressure the left pump works against",
    read: flagOrValue(145, 120, "mm Hg"),
  },
  {
    key: "diastolic",
    label: "Diastolic blood pressure",
    unit: "mm Hg",
    exact: ["aplo", "diabp", "dbp", "diastolic", "diastolicbp"],
    contains: ["diastolic"],
    effect: "Pressure between beats; feeds the heart's own arteries",
    read: flagOrValue(92, 78, "mm Hg"),
  },
  {
    key: "cholesterol",
    label: "Cholesterol",
    unit: "mg/dL",
    exact: ["chol", "cholesterol", "totchol", "serumcholesterol", "highchol"],
    contains: ["chol"],
    effect: "Risk estimate only",
    read: (c, v) => {
      if (c.kind === "binary") return { value: v ? 250 : 190, note: 'Yes/no "high cholesterol": read as 250 / 190 mg/dL' };
      if (c.stats.max <= 3 && c.stats.min >= 1) return { value: [190, 240, 290][Math.round(v) - 1] ?? 190, note: "Levels 1-3 (normal / above / well above), read as 190 / 240 / 290 mg/dL" };
      if (c.stats.median < 20) return { value: v * 38.67, note: "mmol/L, converted to mg/dL" };
      return { value: v };
    },
  },
  {
    key: "glucose",
    label: "High blood sugar / diabetes",
    unit: "",
    exact: ["fbs", "fastingbs", "fastingbloodsugar", "gluc", "glucose", "diabetes", "diabetic", "bloodsugar"],
    contains: ["diabet", "glucose", "sugar"],
    effect: "Stiffer blood vessels",
    read: (c, v) => {
      if (c.kind === "binary") return { value: v };
      if (c.stats.max <= 3 && c.stats.min >= 0 && Number.isInteger(c.stats.median)) {
        return { value: v >= 2 ? 1 : 0, note: "Levels: the higher levels (2+) read as high blood sugar" };
      }
      if (c.stats.median < 20) return { value: v >= 7 ? 1 : 0, note: "mmol/L: 7 or more read as high" };
      return { value: v >= 126 ? 1 : 0, note: "mg/dL: 126 or more read as high" };
    },
  },
  {
    key: "ejectionFraction",
    label: "Ejection fraction",
    unit: "%",
    exact: ["ejectionfraction", "ef", "lvef"],
    contains: ["ejection"],
    effect: "Sets how strongly the heart squeezes",
    read: fraction,
  },
  {
    key: "oxygen",
    label: "Oxygen saturation",
    unit: "%",
    exact: ["spo2", "oxygensaturation", "o2sat", "saturation", "o2saturation"],
    contains: ["spo2", "saturation"],
    effect: "Blood oxygen level",
    read: fraction,
  },
  {
    key: "exerciseAngina",
    label: "Chest pain on exercise",
    unit: "",
    exact: ["exang", "exng", "exerciseangina", "exerciseinducedangina"],
    contains: ["angina"],
    effect: "Weaker squeeze during exercise",
    read: (c, v) => ({ value: c.kind === "binary" || c.stats.max <= 1 ? v : v > 0 ? 1 : 0 }),
  },
  {
    key: "stDepression",
    label: "ST depression on exercise",
    unit: "mm",
    exact: ["oldpeak", "stdepression"],
    contains: ["oldpeak"],
    effect: "Weaker squeeze during exercise",
    read: asIs,
  },
];

export const ROLE_BY_KEY = Object.fromEntries(ROLES.map((r) => [r.key, r])) as Record<RoleKey, RoleDef>;

export type Mapping = Partial<Record<RoleKey, number>>; // role -> column index

const TARGET_NAMES = [
  "target", "num", "output", "cardio", "heartdisease", "heartdiseaseorattack", "heartattackrisk", "heartattack",
  "label", "outcome", "diagnosis", "class", "deathevent", "condition", "disease", "hd", "chd", "tenyearchd", "y",
];

export function guessTarget(ds: Dataset): number | null {
  const names = ds.columns.map((c) => norm(c.name));
  for (const t of TARGET_NAMES) {
    const i = names.indexOf(t);
    if (i >= 0 && ds.columns[i].kind !== "text") return i;
  }
  const last = ds.columns.length - 1;
  const c = ds.columns[last];
  if (c && (c.kind === "binary" || (c.kind === "numeric" && c.stats.distinct <= 5 && c.stats.min >= 0))) return last;
  return null;
}

export function guessMapping(ds: Dataset, target: number | null): Mapping {
  const mapping: Mapping = {};
  const used = new Set<number>(target === null ? [] : [target]);
  const names = ds.columns.map((c) => norm(c.name));
  const pick = (role: RoleDef, test: (n: string) => boolean) => {
    if (mapping[role.key] !== undefined) return;
    const i = names.findIndex((n, j) => !used.has(j) && ds.columns[j].kind !== "text" && test(n));
    if (i >= 0) {
      mapping[role.key] = i;
      used.add(i);
    }
  };
  // Earlier names in a role's list win (e.g. "sysBP" in mm Hg over a yes/no "prevalentHyp").
  for (const role of ROLES) for (const name of role.exact) pick(role, (n) => n === name);
  for (const role of ROLES) pick(role, (n) => role.contains.some((k) => n.includes(k)));
  return mapping;
}

/** The roles' values for one row, in the heart's units, with how each was read. */
export function readRoles(ds: Dataset, mapping: Mapping, row: number[]): Partial<Record<RoleKey, Reading>> {
  const out: Partial<Record<RoleKey, Reading>> = {};
  for (const role of ROLES) {
    const j = mapping[role.key];
    if (j === undefined) continue;
    const v = row[j];
    if (!Number.isFinite(v)) continue;
    out[role.key] = role.read(ds.columns[j], v);
  }
  return out;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const r1 = (v: number) => Math.round(v * 10) / 10;

/**
 * The same physiology rules as src/patient_twin.py, generalised to whatever
 * the dataset has: each heart setting comes from a mapped column when there
 * is one, otherwise from a stated estimate or a typical value. `risk` is the
 * dataset model's estimate (0-1), which lowers pumping strength the way the
 * server's predicted severity does, unless an ejection fraction is given.
 */
export function deriveScenarios(
  roles: Partial<Record<RoleKey, Reading>>,
  risk: number | null,
  columnName: (role: RoleKey) => string
): Record<ScenarioKey, TwinScenario> {
  const params: Partial<HeartParams> = {};
  const sources: ParamSources = {};
  const put = (key: keyof HeartParams, value: number, source: string) => {
    params[key] = r1(value);
    sources[key] = source;
  };
  const has = (k: RoleKey) => roles[k] !== undefined;
  const val = (k: RoleKey) => roles[k]!.value;
  const from = (k: RoleKey, text: string) => `From data ("${columnName(k)}"): ${text}${roles[k]?.note ? ` -- ${roles[k]!.note}` : ""}`;

  const age = has("age") ? val("age") : 50;
  const sbp = has("systolic") ? clamp(val("systolic"), 80, 220) : 120;
  const restHR = has("restingHR") ? clamp(val("restingHR"), 40, 180) : 72;
  const sugar = has("glucose") && val("glucose") >= 1;

  put("heartRate", restHR, has("restingHR") ? from("restingHR", `${r1(val("restingHR"))} beats/min`) : "Not in data: typical resting value");
  put("systolicBP", sbp, has("systolic") ? from("systolic", `${r1(val("systolic"))} mm Hg`) : "Not in data: typical value");
  const dbp = has("diastolic") ? clamp(val("diastolic"), 40, Math.min(140, sbp - 10)) : sbp >= 110 ? 80 : sbp - 30;
  put("diastolicBP", dbp, has("diastolic") ? from("diastolic", `${r1(val("diastolic"))} mm Hg`) : "Not in data: typical value");

  if (has("sex")) {
    const male = val("sex") === 1;
    put("baseStrokeVolume", male ? 72 : 62, `Estimated: typical resting stroke volume for an adult ${male ? "male" : "female"}`);
  } else put("baseStrokeVolume", 70, "Not in data: typical value");

  let vrNote = has("systolic") || has("age") ? `Estimated from blood pressure ${r1(sbp)} mm Hg and age ${Math.round(age)} (arterial stiffening)` : "Not in data: typical value";
  if (sugar) vrNote += ", raised for high blood sugar";
  put("vascularResistance", clamp(50 + 0.6 * (sbp - 120) + 0.3 * Math.max(0, age - 50) + (sugar ? 6 : 0), 15, 95), vrNote);
  put("afterload", clamp(50 + 0.5 * (sbp - 120), 15, 95), has("systolic") ? `Estimated from blood pressure ${r1(sbp)} mm Hg` : "Not in data: typical value");

  if (has("ejectionFraction")) {
    const ef = clamp(val("ejectionFraction"), 10, 80);
    // Calibrated on this simulator: contractility 50 settles at an EF of about 55%, and each point of
    // contractility moves EF by roughly 0.77 points (see README, "Validated behaviour").
    put("contractility", clamp(50 + (ef - 55) * 1.3, 8, 95), from("ejectionFraction", `ejection fraction ${r1(ef)}%`));
  } else if (risk !== null) {
    put("contractility", clamp(55 - 20 * risk, 10, 90), `Estimated: typical baseline, lowered in step with the dataset model's risk estimate (${Math.round(risk * 100)}%)`);
  } else put("contractility", 55, "Not in data: typical value");

  put("preload", 50, "Not in data: typical value");
  put("oxygenSaturation", has("oxygen") ? clamp(val("oxygen"), 70, 100) : 98, has("oxygen") ? from("oxygen", `${r1(val("oxygen"))}%`) : "Not in data: typical value");
  put("aorticStenosis", 0, "Set with the valve controls");
  put("mitralRegurgitation", 0, "Set with the valve controls");

  // Exercise: the same heart at a peak heart rate, from the data when it has one.
  const stress: Partial<HeartParams> = { ...params };
  const stressSources: ParamSources = { ...sources };
  const peak = has("maxHR") ? clamp(val("maxHR"), 60, 180) : clamp(0.85 * (220 - age), 100, 180);
  stress.heartRate = r1(peak);
  stressSources.heartRate = has("maxHR") ? from("maxHR", `${r1(val("maxHR"))} beats/min`) : `Estimated: 85% of the age-predicted maximum (220 - age ${Math.round(age)})`;
  stress.systolicBP = r1(Math.min(220, sbp + 0.35 * (peak - restHR)));
  stressSources.systolicBP = `Estimated: resting ${r1(sbp)} mm Hg plus a typical exercise rise with heart rate`;
  const ischemia = (has("exerciseAngina") && val("exerciseAngina") >= 1 ? 15 : 0) + (has("stDepression") ? Math.min(25, 6 * Math.max(0, val("stDepression"))) : 0);
  stress.contractility = r1(clamp((params.contractility ?? 55) + 20 - ischemia, 10, 95));
  stressSources.contractility = `Estimated: resting value +20 for exercise${ischemia > 0 ? ", reduced for exercise ischemia in the data" : ""}`;
  stress.vascularResistance = r1(clamp((params.vascularResistance ?? 50) - 15, 10, 95));
  stressSources.vascularResistance = "Estimated: resting value, lowered for exercise vasodilation";
  stress.preload = 65;
  stressSources.preload = "Estimated: raised for increased venous return during exercise";

  return {
    rest: { label: "At rest", params, sources },
    stress: { label: "During exercise", params: stress, sources: stressSources },
  };
}
