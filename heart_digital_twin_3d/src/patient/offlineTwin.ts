import type { HeartParams } from "../simulation/heartModel";
import type { ParamSources, PatientTwin, PredictionInputs, RiskReading } from "./patientTwin";

/**
 * Offline stand-in for the CordiaAI backend, so the twin runs on its own
 * with `npm run dev` (no FastAPI server, no login). It is used when the
 * twin is opened without a patient and the backend can't be reached, or
 * when the URL has ?offline=1.
 *
 * Two parts, both mirroring the server:
 *  - riskModel: a multinomial logistic regression on the same ten inputs and
 *    the same five classes as the server's FT-Transformer, fitted on
 *    datasets/uci/heart_disease_raw.csv (all 920 rows; zero cholesterol /
 *    BP treated as missing and median-filled, chest-pain type one-hot).
 *    5-fold cross-validated: disease vs. no disease AUC 0.88, accuracy 79%;
 *    5-class accuracy 56%. It is a lighter, less accurate stand-in, not
 *    the production model, and the UI says so.
 *  - buildTwin: a line-for-line port of src/patient_twin.py and
 *    src/risk_reading.py, so the heart is set up exactly as the server would.
 */

const CLASSES = ["No disease", "Mild", "Moderate", "Severe", "Critical"];

// Feature order: age, sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, cp1, cp2, cp3
const MEAN = [53.51087, 0.78913, 132.13696, 245.22283, 0.15, 0.60326, 137.69239, 0.3663, 0.85326, 0.05, 0.18913, 0.22174];
const STD = [9.42469, 0.40815, 17.92976, 51.78533, 0.35727, 0.80544, 25.14524, 0.48206, 1.05805, 0.21806, 0.39183, 0.41564];
const COEF = [
  [-0.39265, -0.57368, -0.02486, -0.14181, -0.13522, -0.10433, 0.39148, -0.2961, -0.59743, 0.25564, 0.83831, 0.46077],
  [-0.29441, -0.01105, 0.01317, 0.11838, -0.02615, -0.22075, 0.0402, 0.21979, -0.23395, 0.06847, 0.07439, -0.11261],
  [0.16863, 0.08398, -0.05893, 0.01662, 0.05564, -0.08406, 0.02193, 0.04913, 0.06314, -0.08155, -0.37333, -0.20703],
  [0.3079, 0.14059, 0.02766, -0.004, 0.12422, 0.015, -0.29481, 0.1376, 0.15953, -0.09461, -0.12896, -0.01049],
  [0.21052, 0.36017, 0.04296, 0.01081, -0.01849, 0.39414, -0.1588, -0.11042, 0.6087, -0.14795, -0.41041, -0.13065],
];
const INTERCEPT = [1.38551, 1.16032, -0.06628, -0.31178, -2.16777];

/** Class probabilities, in CLASSES order. */
export function riskModel(r: PredictionInputs): number[] {
  const x = [r.age, r.sex, r.trestbps, r.chol, r.fbs, r.restecg, r.thalach, r.exang, r.oldpeak, +(r.cp === 1), +(r.cp === 2), +(r.cp === 3)];
  const z = x.map((v, i) => (v - MEAN[i]) / STD[i]);
  const logits = COEF.map((w, k) => INTERCEPT[k] + w.reduce((s, wi, i) => s + wi * z[i], 0));
  const top = Math.max(...logits);
  const e = logits.map((l) => Math.exp(l - top));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

const round1 = (v: number) => Math.round(v * 10) / 10;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// --- src/risk_reading.py
function readRisk(probs: number[]): RiskReading {
  const disease = 1 - probs[0];
  const pct = disease * 100;
  const predicts = pct >= 50;
  const [band, verdict]: [RiskReading["band"], string] =
    pct < 35
      ? ["unlikely", "Heart disease unlikely"]
      : pct < 65
        ? ["borderline", "Borderline — leans towards " + (predicts ? "heart disease" : "no heart disease")]
        : ["likely", "Heart disease likely"];
  const conditional: Record<string, number> = {};
  for (let i = 1; i < CLASSES.length; i++) conditional[CLASSES[i]] = disease > 1e-6 ? probs[i] / disease : 0;
  const top = disease > 1e-6 ? CLASSES.slice(1).reduce((a, b) => (conditional[b] > conditional[a] ? b : a)) : null;
  return {
    disease_percent: round1(pct),
    no_disease_percent: round1(probs[0] * 100),
    decision_point: 50,
    zones: { unlikely_below: 35, likely_from: 65 },
    band,
    verdict,
    predicts_disease: predicts,
    severity_if_disease: Object.fromEntries(Object.entries(conditional).map(([k, v]) => [k, round1(v * 100)])),
    likely_severity_if_disease: top,
    likely_severity_percent: top ? round1(conditional[top] * 100) : null,
  };
}

// --- src/patient_twin.py
const SEVERITY_CONTRACTILITY_PENALTY = [0, 4, 8, 13, 18];
const SIM_HR_MAX = 180;

type Params = Partial<HeartParams>;

function restScenario(r: PredictionInputs, severityIdx: number): [Params, ParamSources] {
  const params: Record<string, number> = {};
  const sources: Record<string, string> = {};
  const put = (key: keyof HeartParams, value: number, source: string) => {
    params[key] = round1(value);
    sources[key] = source;
  };
  const sbp = r.trestbps;
  const male = r.sex === 1;
  put("heartRate", 72, "Not in record: the record's heart rate (thalach) is the stress-test peak, so a typical resting value is used");
  put("systolicBP", sbp, `From record: resting blood pressure ${sbp} mm Hg`);
  put("diastolicBP", sbp >= 110 ? 80 : sbp - 30, "Not in record: the record holds systolic pressure only, so a typical diastolic value is used");
  put("baseStrokeVolume", male ? 72 : 62, `Estimated: typical resting stroke volume for an adult ${male ? "male" : "female"}`);

  const vr = 50 + 0.6 * (sbp - 120) + 0.3 * Math.max(0, r.age - 50) + (r.fbs ? 6 : 0);
  let vrNote = `Estimated from resting BP ${sbp} mm Hg and age ${r.age} (arterial stiffening)`;
  if (r.fbs) vrNote += ", raised for elevated fasting blood sugar";
  put("vascularResistance", clamp(vr, 15, 95), vrNote);

  const afterload = 50 + 0.5 * (sbp - 120) + (r.restecg === 2 ? 10 : 0);
  let alNote = `Estimated from resting BP ${sbp} mm Hg`;
  if (r.restecg === 2) alNote += "; raised for left ventricular hypertrophy on the resting ECG (chronic pressure overload)";
  put("afterload", clamp(afterload, 15, 95), alNote);

  const contractility = 55 - SEVERITY_CONTRACTILITY_PENALTY[severityIdx] - (r.restecg === 1 ? 5 : 0);
  let cNote = "Estimated: typical baseline, reduced in step with the risk model's predicted severity";
  if (r.restecg === 1) cNote += " and for ST-T wave abnormality on the resting ECG";
  put("contractility", clamp(contractility, 10, 90), cNote);

  put("preload", 50, "Not in record: typical value");
  put("oxygenSaturation", 98, "Not in record: typical value");
  put("aorticStenosis", 0, "Not in record: no valve disease assumed");
  put("mitralRegurgitation", 0, "Not in record: no valve disease assumed");
  return [params as Params, sources as ParamSources];
}

function stressScenario(r: PredictionInputs, rest: Params, restSources: ParamSources): [Params, ParamSources] {
  const params: Record<string, number> = { ...(rest as Record<string, number>) };
  const sources: Record<string, string> = { ...(restSources as Record<string, string>) };
  const hr = r.thalach;
  params.heartRate = Math.min(hr, SIM_HR_MAX);
  sources.heartRate =
    `From record: peak exercise heart rate (thalach) ${hr} BPM` + (hr > SIM_HR_MAX ? `, capped at the simulator's ${SIM_HR_MAX} BPM` : "");

  params.systolicBP = round1(Math.min(220, r.trestbps + 0.35 * (hr - 72)));
  sources.systolicBP = `Estimated: resting ${r.trestbps} mm Hg plus a typical exercise rise with heart rate`;

  const ischemia = (r.exang ? 15 : 0) + Math.min(25, 6 * Math.max(0, r.oldpeak));
  params.contractility = round1(clamp(params.contractility + 20 - ischemia, 10, 95));
  let note = "Estimated: resting value +20 for exercise inotropy";
  if (ischemia > 0) {
    const causes: string[] = [];
    if (r.exang) causes.push("exercise-induced angina");
    if (r.oldpeak > 0) causes.push(`ST depression ${r.oldpeak}`);
    note += `, reduced for exercise ischemia (${causes.join(", ")})`;
  }
  sources.contractility = note;

  params.vascularResistance = round1(clamp(params.vascularResistance - 15, 10, 95));
  sources.vascularResistance = "Estimated: resting value, lowered for exercise vasodilation";
  params.preload = 65;
  sources.preload = "Estimated: raised for increased venous return during exercise";
  return [params as Params, sources as ParamSources];
}

export function buildOfflineTwin(record: PredictionInputs, name = "What-if"): PatientTwin {
  const probs = riskModel(record);
  const severityIdx = probs.indexOf(Math.max(...probs));
  const [rest, restSources] = restScenario(record, severityIdx);
  const [stress, stressSources] = stressScenario(record, rest, restSources);
  return {
    patient: { name, age: record.age, sex: record.sex === 1 ? "Male" : "Female" },
    prediction: {
      risk_percent: round1((1 - probs[0]) * 100),
      severity: CLASSES[severityIdx],
      severity_percent: round1(probs[severityIdx] * 100),
      class_probabilities: Object.fromEntries(CLASSES.map((c, i) => [c, round1(probs[i] * 100)])),
      reading: readRisk(probs),
    },
    record,
    scenarios: {
      rest: { label: "At rest", params: rest, sources: restSources },
      stress: { label: "Peak exercise (stress test)", params: stress, sources: stressSources },
    },
  };
}
