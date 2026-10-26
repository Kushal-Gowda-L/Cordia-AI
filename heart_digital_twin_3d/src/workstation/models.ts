import DATA_JSON from "./cordia-data.json";

/**
 * The Cordia Heart Twin workstation's patient state and browser risk models,
 * ported as-is from the workstation page: one patient state of ~28 fields,
 * mapped into four logistic models trained on four different datasets (UCI
 * angiography, Framingham 10-year CHD, the 70k cardiovascular exams, CDC
 * BRFSS), plus damped yearly trends for the time machine.
 */

export type State = Record<string, number>;

export interface Field {
  g: string; // group
  l: string; // label
  u?: string; // unit
  min?: number;
  max?: number;
  s?: number; // step
  v?: Record<string, string>; // choices
  d: number; // default
  h?: number; // healthy target
}

export const F: Record<string, Field> = {
  age: { g: "Patient", l: "Age", u: "y", min: 18, max: 95, s: 1, d: 58 },
  sex: { g: "Patient", l: "Sex", v: { 0: "female", 1: "male" }, d: 1 },
  height_cm: {
    g: "Patient",
    l: "Height",
    u: "cm",
    min: 130,
    max: 210,
    s: 1,
    d: 172,
  },
  weight_kg: {
    g: "Patient",
    l: "Weight",
    u: "kg",
    min: 35,
    max: 200,
    s: 1,
    d: 84,
  },
  sys_bp: {
    g: "Vitals & labs",
    l: "Systolic BP",
    u: "mmHg",
    min: 85,
    max: 240,
    s: 1,
    d: 148,
    h: 120,
  },
  dia_bp: {
    g: "Vitals & labs",
    l: "Diastolic BP",
    u: "mmHg",
    min: 45,
    max: 150,
    s: 1,
    d: 92,
    h: 80,
  },
  heart_rate: {
    g: "Vitals & labs",
    l: "Resting HR",
    u: "bpm",
    min: 35,
    max: 180,
    s: 1,
    d: 82,
    h: 65,
  },
  total_chol: {
    g: "Vitals & labs",
    l: "Total cholesterol",
    u: "mg/dl",
    min: 100,
    max: 450,
    s: 1,
    d: 248,
    h: 180,
  },
  glucose: {
    g: "Vitals & labs",
    l: "Fasting glucose",
    u: "mg/dl",
    min: 50,
    max: 400,
    s: 1,
    d: 112,
    h: 95,
  },
  smoker: {
    g: "Lifestyle",
    l: "Current smoker",
    v: { 0: "no", 1: "yes" },
    d: 1,
    h: 0,
  },
  cigs_per_day: {
    g: "Lifestyle",
    l: "Cigarettes/day",
    u: "",
    min: 0,
    max: 60,
    s: 1,
    d: 12,
    h: 0,
  },
  active: {
    g: "Lifestyle",
    l: "Physically active",
    v: { 0: "no", 1: "yes" },
    d: 0,
    h: 1,
  },
  alcohol_heavy: {
    g: "Lifestyle",
    l: "Heavy alcohol",
    v: { 0: "no", 1: "yes" },
    d: 0,
    h: 0,
  },
  fruits_veg: {
    g: "Lifestyle",
    l: "Fruit & veg daily",
    v: { 0: "no", 1: "yes" },
    d: 1,
    h: 1,
  },
  diabetes: { g: "History", l: "Diabetes", v: { 0: "no", 1: "yes" }, d: 0 },
  hypertension_dx: {
    g: "History",
    l: "Hypertension diagnosed",
    v: { 0: "no", 1: "yes" },
    d: 1,
  },
  bp_meds: { g: "History", l: "BP medication", v: { 0: "no", 1: "yes" }, d: 0 },
  prior_stroke: {
    g: "History",
    l: "Previous stroke",
    v: { 0: "no", 1: "yes" },
    d: 0,
  },
  gen_health: {
    g: "History",
    l: "Self-rated health",
    v: { 1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor" },
    d: 3,
  },
  diff_walking: {
    g: "History",
    l: "Difficulty walking",
    v: { 0: "no", 1: "yes" },
    d: 0,
  },
  cp: {
    g: "Cardiology work-up",
    l: "Chest pain",
    v: {
      1: "typical angina",
      2: "atypical angina",
      3: "non-anginal",
      4: "asymptomatic",
    },
    d: 4,
  },
  restecg: {
    g: "Cardiology work-up",
    l: "Resting ECG",
    v: { 0: "normal", 1: "ST-T abnormal", 2: "LV hypertrophy" },
    d: 0,
  },
  thalach: {
    g: "Cardiology work-up",
    l: "Max HR (stress test)",
    u: "bpm",
    min: 60,
    max: 210,
    s: 1,
    d: 138,
  },
  exang: {
    g: "Cardiology work-up",
    l: "Exercise angina",
    v: { 0: "no", 1: "yes" },
    d: 1,
  },
  oldpeak: {
    g: "Cardiology work-up",
    l: "ST depression",
    u: "mm",
    min: 0,
    max: 6.5,
    s: 0.1,
    d: 1.6,
  },
  slope: {
    g: "Cardiology work-up",
    l: "ST slope",
    v: { "-1": "not done", 1: "upsloping", 2: "flat", 3: "downsloping" },
    d: 2,
  },
  ca: {
    g: "Cardiology work-up",
    l: "Vessels (fluoroscopy)",
    v: { "-1": "not done", 0: "0", 1: "1", 2: "2", 3: "3" },
    d: -1,
  },
  thal: {
    g: "Cardiology work-up",
    l: "Thallium scan",
    v: {
      "-1": "not done",
      3: "normal",
      6: "fixed defect",
      7: "reversible defect",
    },
    d: -1,
  },
  ef: {
    g: "Heart function",
    l: "Ejection fraction",
    u: "%",
    min: 10,
    max: 80,
    s: 1,
    d: 58,
  },
  ectopy: {
    g: "Heart function",
    l: "Abnormal (ectopic) beats",
    u: "%",
    min: 0,
    max: 40,
    s: 1,
    d: 0,
  },
};

const DRIFT: Record<string, number> = {
  sys_bp: 0.6,
  dia_bp: 0.1,
  total_chol: 0.8,
  glucose: 0.6,
  weight_kg: 0.25,
};

interface LrModel {
  name: string;
  title: string;
  outcome: string;
  features: string[];
  median: number[];
  mean: number[];
  scale: number[];
  coef: number[];
  intercept: number;
  calibration: [number, number];
}
interface UciTerm {
  type: "num" | "onehot";
  key: string;
  mean?: number;
  std?: number;
  median?: number;
  value?: number;
  coef: number;
}
interface UciModel {
  terms: UciTerm[];
  intercept: number;
}

const DATA = DATA_JSON as unknown as {
  models: {
    framingham: LrModel;
    cardio: LrModel;
    brfss: LrModel;
    uci: UciModel;
  };
  state: State;
  example: boolean;
};
const M = DATA.models;

export function defaultState(): State {
  const base: State = {};
  for (const k in F) base[k] = F[k].d;
  return { ...base, ...DATA.state };
}

const sig = (z: number) => 1 / (1 + Math.exp(-z));
const logit = (p: number) => {
  p = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(p / (1 - p));
};
export const bmi = (s: State) => s.weight_kg / Math.pow(s.height_cm / 100, 2);

type Mapper = (s: State) => Record<string, number>;
const MAP: Record<"framingham" | "cardio" | "brfss" | "uci", Mapper> = {
  framingham: (s) => ({
    male: s.sex,
    age: s.age,
    currentSmoker: s.smoker,
    cigsPerDay: s.smoker ? s.cigs_per_day : 0,
    BPMeds: s.bp_meds,
    prevalentStroke: s.prior_stroke,
    prevalentHyp: +(s.hypertension_dx == 1 || s.sys_bp >= 140 || s.dia_bp >= 90),
    diabetes: +(s.diabetes == 1 || s.glucose >= 126),
    totChol: s.total_chol,
    sysBP: s.sys_bp,
    diaBP: s.dia_bp,
    BMI: bmi(s),
    heartRate: s.heart_rate,
    glucose: s.glucose,
  }),
  cardio: (s) => ({
    age: s.age,
    male: s.sex,
    height: s.height_cm,
    weight: s.weight_kg,
    bmi: bmi(s),
    ap_hi: s.sys_bp,
    ap_lo: s.dia_bp,
    cholesterol: s.total_chol < 200 ? 1 : s.total_chol < 240 ? 2 : 3,
    gluc: s.glucose < 100 ? 1 : s.glucose < 126 ? 2 : 3,
    smoke: s.smoker,
    alco: s.alcohol_heavy,
    active: s.active,
  }),
  brfss: (s) => ({
    HighBP: +(s.hypertension_dx == 1 || s.bp_meds == 1 || s.sys_bp >= 140 || s.dia_bp >= 90),
    HighChol: +(s.total_chol >= 240),
    CholCheck: 1,
    BMI: Math.round(bmi(s)),
    Smoker: s.smoker,
    Stroke: s.prior_stroke,
    Diabetes: s.diabetes == 1 || s.glucose >= 126 ? 2 : s.glucose >= 100 ? 1 : 0,
    PhysActivity: s.active,
    Fruits: s.fruits_veg,
    Veggies: s.fruits_veg,
    HvyAlcoholConsump: s.alcohol_heavy,
    GenHlth: s.gen_health,
    MentHlth: 0,
    PhysHlth: 0,
    DiffWalk: s.diff_walking,
    Sex: s.sex,
    Age: s.age < 25 ? 1 : Math.min(13, 2 + Math.floor((s.age - 25) / 5)),
  }),
  uci: (s) => ({
    age: s.age,
    sex: s.sex,
    cp: s.cp,
    trestbps: s.sys_bp,
    chol: s.total_chol,
    fbs: +(s.glucose > 120 || s.diabetes == 1),
    restecg: s.restecg,
    thalach: s.thalach,
    exang: s.exang,
    oldpeak: s.oldpeak,
    slope: s.slope,
    ca: s.ca,
    thal: s.thal,
  }),
};

function lrPredict(m: LrModel, x: Record<string, number>) {
  let z = m.intercept;
  m.features.forEach((f, i) => {
    const v = x[f] == null || isNaN(x[f]) ? m.median[i] : x[f];
    z += (m.coef[i] * (v - m.mean[i])) / m.scale[i];
  });
  const [a, b] = m.calibration;
  return sig(a * logit(sig(z)) + b);
}

function lrContrib(m: LrModel, x: Record<string, number>) {
  return m.features
    .map((f, i) => ({
      f,
      c: (m.coef[i] * ((x[f] == null || isNaN(x[f]) ? m.median[i] : x[f]) - m.mean[i])) / m.scale[i],
    }))
    .sort((p, q) => Math.abs(q.c) - Math.abs(p.c));
}

function uciPredict(m: UciModel, x: Record<string, number>) {
  let z = m.intercept;
  for (const t of m.terms) {
    if (t.type === "num") {
      const v = x[t.key] == null || isNaN(x[t.key]) ? t.median! : x[t.key];
      z += (t.coef * (v - t.mean!)) / t.std!;
    } else if (Number(x[t.key]) === t.value) z += t.coef;
  }
  return sig(z);
}

export const MODELS = [
  {
    key: "cad",
    title: "Coronary artery disease",
    sub: "diagnostic · UCI",
    src: "uci",
  },
  {
    key: "chd10",
    title: "10-year CHD risk",
    sub: "prognostic · Framingham",
    src: "framingham",
  },
  {
    key: "cvd",
    title: "Cardiovascular disease",
    sub: "vitals/lifestyle · 70k exams",
    src: "cardio",
  },
  {
    key: "population",
    title: "Heart disease / MI",
    sub: "population screen · BRFSS",
    src: "brfss",
  },
] as const;

export type Outputs = Record<(typeof MODELS)[number]["key"], number>;

export function evaluate(s: State): Outputs {
  const out = {} as Outputs;
  for (const d of MODELS) out[d.key] = d.src === "uci" ? uciPredict(M.uci, MAP.uci(s)) : lrPredict(M[d.src], MAP[d.src](s));
  return out;
}

export function framinghamRisk(s: State) {
  return lrPredict(M.framingham, MAP.framingham(s));
}

const DRIVER_NAMES: Record<string, string> = {
  sysBP: "systolic BP",
  age: "age",
  cigsPerDay: "smoking",
  male: "male sex",
  totChol: "cholesterol",
  glucose: "glucose",
  diaBP: "diastolic BP",
  BMI: "BMI",
  prevalentHyp: "hypertension",
  diabetes: "diabetes",
  heartRate: "heart rate",
  BPMeds: "BP medication",
  currentSmoker: "smoker",
  prevalentStroke: "prior stroke",
};

/** The four inputs moving the 10-year risk most, for this state. */
export function mainDrivers(s: State) {
  return lrContrib(M.framingham, MAP.framingham(s))
    .slice(0, 4)
    .map((d) => ({ name: DRIVER_NAMES[d.f] ?? d.f, raises: d.c > 0 }));
}

/** Damped trends + ageing, `y` years ahead (mirrors twin/simulator.py). */
export function project(s: State, y: number): State {
  const o = { ...s };
  const n = Math.ceil(y);
  let t = 0;
  for (let k = 1; k <= n; k++) t += Math.pow(0.8, k);
  const damp = n ? (t * y) / n : 0;
  for (const k in DRIFT) o[k] = Math.min(F[k].max!, o[k] + DRIFT[k] * damp);
  o.age = s.age + y;
  return o;
}

export function healthyTargets(s: State): State {
  const o = { ...s };
  for (const k in F) if (F[k].h !== undefined) o[k] = F[k].h!;
  const hw = 24.9 * Math.pow(o.height_cm / 100, 2);
  if (o.weight_kg > hw) o.weight_kg = Math.round(hw);
  return o;
}

export const IS_EXAMPLE = DATA.example;

const clamp = (v: number, a = 0, b = 1) => Math.min(Math.max(v, a), b);

/** Illustrative coronary narrowing from the models and risk factors (as the workstation's heart shows it). */
export function stenosis(s: State, out: Outputs) {
  const cad = out.cad ?? 0.3;
  const chd = out.chd10 ?? 0.1;
  const burden = clamp(
    0.55 * cad + 0.25 * clamp(chd / 0.3) + 0.12 * clamp((s.total_chol - 170) / 140) + 0.08 * s.smoker + 0.05 * +(s.glucose >= 126),
  );
  const ca = Number(s.ca);
  const w = ca >= 0 ? [ca >= 1 ? 1 : 0.25, ca >= 3 ? 1 : 0.3, ca >= 2 ? 1 : 0.3] : [1, 0.7, 0.8];
  return {
    LAD: clamp(0.95 * burden * w[0]),
    LCx: clamp(0.8 * burden * w[1]),
    RCA: clamp(0.88 * burden * w[2]),
  };
}
