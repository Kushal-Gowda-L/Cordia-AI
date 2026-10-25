import type { PredictionInputs } from "./patientTwin";

export type InputKey = keyof PredictionInputs;

interface BaseField {
  key: InputKey;
  label: string; // everyday wording
  clinical: string; // the clinical term, shown alongside for clinicians
  /** True if this input directly sets part of the simulated heart (every input changes the risk). */
  drivesHeart: boolean;
  /** Recorded during the exercise stress test, so it shows up in the "during exercise" heart. */
  exercise?: boolean;
  /** Shown under the control when the model's response to this input is known to be unreliable. */
  caution?: string;
  /** Offered to patients in their own account; everything else stays as their clinic recorded it. */
  patient?: { label: string; hint: string };
}

export interface SliderField extends BaseField {
  kind: "slider";
  unit: string;
  min: number;
  max: number;
  step: number;
}

export interface ChoiceField extends BaseField {
  kind: "choice";
  options: { value: number; label: string }[];
}

export type InputField = SliderField | ChoiceField;

// Ranges match the data-entry form and the backend's validation.
export const INPUT_FIELDS: InputField[] = [
  { key: "age", kind: "slider", label: "Age", clinical: "age", unit: "years", min: 28, max: 77, step: 1, drivesHeart: true },
  {
    key: "sex",
    kind: "choice",
    label: "Sex",
    clinical: "sex",
    options: [
      { value: 1, label: "Male" },
      { value: 0, label: "Female" },
    ],
    drivesHeart: true,
  },
  {
    key: "cp",
    kind: "choice",
    label: "Type of chest pain",
    clinical: "chest pain type",
    options: [
      { value: 1, label: "Typical heart pain (typical angina)" },
      { value: 2, label: "Unusual chest pain (atypical angina)" },
      { value: 3, label: "Pain not from the heart (non-anginal)" },
      { value: 4, label: "No chest pain (asymptomatic)" },
    ],
    drivesHeart: false,
  },
  {
    key: "trestbps",
    kind: "slider",
    label: "Blood pressure at rest",
    clinical: "resting systolic blood pressure",
    unit: "mm Hg",
    min: 90,
    max: 200,
    step: 1,
    drivesHeart: true,
    patient: { label: "Blood pressure", hint: "The top number of a blood pressure reading" },
  },
  {
    key: "chol",
    kind: "slider",
    label: "Cholesterol",
    clinical: "serum cholesterol",
    unit: "mg/dl",
    min: 100,
    max: 600,
    step: 1,
    drivesHeart: false,
    // Training-data defect: zeros used as a missing-value code (see README,
    // "Known model issue: cholesterol").
    caution:
      "Known model issue: this model's estimate falls as cholesterol rises — the opposite of the medical evidence — because of a flaw in its training data. Don't rely on changes to this value.",
  },
  {
    key: "fbs",
    kind: "choice",
    label: "High blood sugar when fasting",
    clinical: "fasting blood sugar > 120 mg/dl",
    options: [
      { value: 0, label: "No" },
      { value: 1, label: "Yes" },
    ],
    drivesHeart: true,
    patient: { label: "High blood sugar or diabetes", hint: "Whether your blood sugar runs high" },
  },
  {
    key: "restecg",
    kind: "choice",
    label: "Heart tracing at rest",
    clinical: "resting ECG",
    options: [
      { value: 0, label: "Normal" },
      { value: 1, label: "Minor changes (ST-T abnormality)" },
      { value: 2, label: "Thickened heart muscle (LV hypertrophy)" },
    ],
    drivesHeart: true,
  },
  {
    key: "thalach",
    kind: "slider",
    label: "Highest heart rate during exercise",
    clinical: "maximum heart rate achieved",
    unit: "beats/min",
    min: 60,
    max: 202,
    step: 1,
    drivesHeart: true,
    exercise: true,
  },
  {
    key: "exang",
    kind: "choice",
    label: "Chest pain during exercise",
    clinical: "exercise-induced angina",
    options: [
      { value: 0, label: "No" },
      { value: 1, label: "Yes" },
    ],
    drivesHeart: true,
    exercise: true,
  },
  {
    key: "oldpeak",
    kind: "slider",
    label: "Heart strain on the exercise tracing",
    clinical: "ST depression (oldpeak)",
    unit: "mm",
    min: -2,
    max: 6,
    step: 0.1,
    drivesHeart: true,
    exercise: true,
  },
];

/** Everyday band for a top (systolic) blood-pressure number, following the common adult categories. */
export function bloodPressureBand(systolic: number): string {
  if (systolic < 120) return "Normal";
  if (systolic < 130) return "Slightly raised";
  if (systolic < 140) return "High";
  return "Very high";
}

export function fieldFor(key: InputKey): InputField {
  return INPUT_FIELDS.find((f) => f.key === key)!;
}

export function formatInput(field: InputField, value: number): string {
  if (field.kind === "choice") return field.options.find((o) => o.value === value)?.label ?? String(value);
  const shown = field.step < 1 ? value.toFixed(1) : String(Math.round(value));
  return `${shown} ${field.unit}`;
}

/** Human-readable summary of what changed between two sets of inputs. */
export function describeChanges(from: PredictionInputs, to: PredictionInputs): string {
  const parts = INPUT_FIELDS.filter((f) => from[f.key] !== to[f.key]).map(
    (f) => `${f.label}: ${formatInput(f, from[f.key])} → ${formatInput(f, to[f.key])}`
  );
  return parts.length ? parts.join("; ") : "Recalculated";
}
