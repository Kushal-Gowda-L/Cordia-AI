import type { HeartParams } from "../simulation/heartModel";

/**
 * Talks to the CordiaAI backend. The twin is served from the same origin as
 * the main app (at /twin/), so it reuses that app's stored login token. The
 * mapping from inputs to heart physiology lives server-side in
 * src/patient_twin.py; this side only fetches and applies it.
 */

export type ScenarioKey = "rest" | "stress";
export type ParamSources = Partial<Record<keyof HeartParams, string>>;

/** Opened from a patient's own account (?self=1): simpler wording and only patient-friendly controls. */
export const PATIENT_VIEW = new URLSearchParams(window.location.search).get("self") === "1";

/** The ten fields the risk model is trained on -- also the twin's controls. */
export interface PredictionInputs {
  age: number;
  sex: number;
  cp: number;
  trestbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalach: number;
  exang: number;
  oldpeak: number;
}

export interface TwinScenario {
  label: string;
  params: Partial<HeartParams>;
  sources: ParamSources;
}

/**
 * The two-step reading of the model's output (src/risk_reading.py):
 * step 1 -- total chance of disease against the 50% decision point, with
 * 35%/65% display zones; step 2 -- the disease levels rescaled to add up to
 * 100% among disease cases.
 */
export interface RiskReading {
  disease_percent: number;
  no_disease_percent: number;
  decision_point: number;
  zones: { unlikely_below: number; likely_from: number };
  band: "unlikely" | "borderline" | "likely";
  verdict: string;
  predicts_disease: boolean;
  severity_if_disease: Record<string, number>;
  likely_severity_if_disease: string | null;
  likely_severity_percent: number | null;
}

export interface Prediction {
  risk_percent: number;
  severity: string;
  severity_percent: number;
  class_probabilities: Record<string, number>;
  reading: RiskReading;
}

export interface PatientTwin {
  patient: { name: string; age: number; sex: string };
  prediction: Prediction;
  record: PredictionInputs;
  scenarios: Record<ScenarioKey, TwinScenario>;
}

/** A typical adult, used when the twin is opened without a patient record. */
export const DEFAULT_INPUTS: PredictionInputs = {
  age: 54,
  sex: 1,
  cp: 4,
  trestbps: 132,
  chol: 246,
  fbs: 0,
  restecg: 0,
  thalach: 150,
  exang: 0,
  oldpeak: 1,
};

/** Which record the URL asks for: ?patient=<id> (clinician) or ?self=1 (patient). */
export function twinRequestPath(): string | null {
  const q = new URLSearchParams(window.location.search);
  if (q.get("self") === "1") return "/api/me/twin";
  const id = q.get("patient");
  return id && /^\d+$/.test(id) ? `/api/patients/${id}/twin` : null;
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("cordia_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function readJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    if (res.status === 401) {
      throw new Error("Please sign in to CordiaAI, then open the digital twin from an assessment.");
    }
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  }
  return res.json();
}

export async function fetchPatientTwin(path: string): Promise<PatientTwin> {
  return readJson(await fetch(path, { headers: authHeaders() }));
}

/** Re-runs the risk model and re-derives the heart for a set of what-if inputs. */
export async function fetchWhatIf(inputs: PredictionInputs): Promise<PatientTwin> {
  return readJson(
    await fetch("/api/twin/what-if", {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify(inputs),
    })
  );
}
