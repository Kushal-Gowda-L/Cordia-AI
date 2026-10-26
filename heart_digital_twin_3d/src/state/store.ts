import { create } from "zustand";
import { DEFAULT_PARAMS, HeartOutputs, HeartParams } from "../simulation/heartModel";
import type { PatientTwin, Prediction, PredictionInputs, ScenarioKey, TwinScenario } from "../patient/patientTwin";
import { steadyStateVitals, type Vitals } from "../simulation/steadyState";
import { describeChanges, fieldFor, type InputKey } from "../patient/inputFields";

export type ViewMode = "anatomy" | "chambers" | "bloodFlow" | "pressure" | "valves" | "electrical" | "risk";

/** How to look inside the closed heart: a see-through wall and a slicing plane. */
export interface XRaySettings {
  wallOpacity: number; // 0.1 (glass) .. 1 (solid) -- the outer heart muscle
  cutDepth: number; // 0 (no cut) .. 1 (sliced to the back); front-to-back plane
}

export interface StructureToggles {
  chambers: boolean;
  vessels: boolean;
  valves: boolean;
  bloodFlow: boolean;
  conduction: boolean;
  labels: boolean;
}

export type TwinStatus = "loading" | "ready" | "error";

export interface RiskPoint {
  step: number;
  risk: number;
  label: string;
}

const MAX_HISTORY = 40;

interface HeartStore {
  params: HeartParams;
  resetToken: number;

  paused: boolean;
  togglePaused: () => void;
  speed: number; // simulation speed multiplier
  setSpeed: (s: number) => void;

  viewMode: ViewMode;
  setViewMode: (m: ViewMode) => void;

  toggles: StructureToggles;
  toggleStructure: (key: keyof StructureToggles) => void;

  xray: XRaySettings;
  setXRay: (patch: Partial<XRaySettings>) => void;

  // True when the backend couldn't be reached and the in-browser estimate
  // (src/patient/offlineTwin.ts) stands in for the server's model.
  offline: boolean;
  setOffline: (offline: boolean) => void;

  // Live simulation output, updated ~20Hz for the 2D dashboard/graphs.
  // The 3D scene reads simulation output every frame directly from the
  // driver instead of through here, to avoid 60Hz React re-renders.
  liveOutputs: HeartOutputs | null;
  setLiveOutputs: (o: HeartOutputs) => void;

  // The ten prediction inputs drive everything else: each change re-runs the
  // risk model and re-derives the heart's physiology (useWhatIfSync in App).
  twinStatus: TwinStatus;
  twinError: string;
  patient: PatientTwin["patient"] | null;
  recordTwin: PatientTwin | null;
  recordInputs: PredictionInputs | null;
  inputs: PredictionInputs | null;
  appliedInputs: PredictionInputs | null; // the inputs the current prediction/heart describe
  prediction: Prediction | null;
  scenarios: Record<ScenarioKey, TwinScenario> | null;
  scenario: ScenarioKey;
  riskHistory: RiskPoint[];
  whatIfRequestId: number;
  whatIfPending: boolean;
  whatIfError: string;
  // Settled heart numbers for the record and for the current inputs, so the
  // dashboard can show how far a what-if has moved the heart from the record.
  recordVitals: Record<ScenarioKey, Vitals> | null;
  currentVitals: Vitals | null;

  setTwinError: (message: string) => void;
  loadTwin: (twin: PatientTwin, hasPatient: boolean) => void;
  setInput: (key: InputKey, value: number) => void;
  beginWhatIf: () => number;
  applyWhatIf: (twin: PatientTwin, requested: PredictionInputs) => void;
  setWhatIfError: (message: string) => void;
  setScenario: (scenario: ScenarioKey) => void;
  resetToRecord: () => void;
}

const DEFAULT_TOGGLES: StructureToggles = {
  chambers: true,
  vessels: true,
  valves: true,
  bloodFlow: true,
  conduction: false,
  labels: true,
};

/** Whether the chambers inside are on show: always in the opened views, and in the closed ones once see-through or cut. */
export function showsInterior(s: { viewMode: ViewMode; xray: XRaySettings }): boolean {
  if (s.viewMode !== "anatomy" && s.viewMode !== "risk") return true;
  return s.xray.wallOpacity < 0.98 || s.xray.cutDepth > 0;
}

function scenarioParams(scenarios: Record<ScenarioKey, TwinScenario>, scenario: ScenarioKey): HeartParams {
  return { ...DEFAULT_PARAMS, ...scenarios[scenario].params };
}

function vitalsFor(scenarios: Record<ScenarioKey, TwinScenario>, scenario: ScenarioKey): Vitals {
  return steadyStateVitals(scenarioParams(scenarios, scenario));
}

function withPoint(history: RiskPoint[], risk: number, label: string): RiskPoint[] {
  const step = history.length ? history[history.length - 1].step + 1 : 0;
  return [...history, { step, risk, label }].slice(-MAX_HISTORY);
}

export const useHeartStore = create<HeartStore>((set, get) => ({
  params: { ...DEFAULT_PARAMS },
  resetToken: 0,

  paused: false,
  togglePaused: () => set((s) => ({ paused: !s.paused })),
  speed: 1,
  setSpeed: (s) => set({ speed: s }),

  viewMode: "chambers",
  setViewMode: (m) => set({ viewMode: m }),

  toggles: { ...DEFAULT_TOGGLES },
  toggleStructure: (key) => set((s) => ({ toggles: { ...s.toggles, [key]: !s.toggles[key] } })),

  xray: { wallOpacity: 1, cutDepth: 0 },
  setXRay: (patch) => set((s) => ({ xray: { ...s.xray, ...patch } })),

  offline: false,
  setOffline: (offline) => set({ offline }),

  liveOutputs: null,
  setLiveOutputs: (o) => set({ liveOutputs: o }),

  twinStatus: "loading",
  twinError: "",
  patient: null,
  recordTwin: null,
  recordInputs: null,
  inputs: null,
  appliedInputs: null,
  prediction: null,
  scenarios: null,
  scenario: "rest",
  riskHistory: [],
  whatIfRequestId: 0,
  whatIfPending: false,
  whatIfError: "",
  recordVitals: null,
  currentVitals: null,

  setTwinError: (message) => set({ twinStatus: "error", twinError: message }),

  loadTwin: (twin, hasPatient) => {
    const inputs = { ...twin.record };
    const recordVitals = { rest: vitalsFor(twin.scenarios, "rest"), stress: vitalsFor(twin.scenarios, "stress") };
    set((s) => ({
      recordVitals,
      currentVitals: recordVitals.rest,
      twinStatus: "ready",
      twinError: "",
      patient: hasPatient ? twin.patient : null,
      recordTwin: twin,
      recordInputs: inputs,
      inputs,
      appliedInputs: inputs,
      prediction: twin.prediction,
      scenarios: twin.scenarios,
      scenario: "rest",
      params: scenarioParams(twin.scenarios, "rest"),
      riskHistory: withPoint([], twin.prediction.risk_percent, hasPatient ? "Recorded values" : "Starting values"),
      resetToken: s.resetToken + 1,
    }));
  },

  setInput: (key, value) => {
    const { inputs, scenario, scenarios } = get();
    if (!inputs || inputs[key] === value) return;
    // Exercise-test inputs only show in the exercising heart, so switch to it.
    const toExercise = Boolean(fieldFor(key).exercise) && scenario === "rest" && scenarios !== null;
    set({
      inputs: { ...inputs, [key]: value },
      ...(toExercise && scenarios ? { scenario: "stress" as const, params: scenarioParams(scenarios, "stress"), currentVitals: vitalsFor(scenarios, "stress") } : {}),
    });
  },

  beginWhatIf: () => {
    const id = get().whatIfRequestId + 1;
    set({ whatIfRequestId: id, whatIfPending: true, whatIfError: "" });
    return id;
  },

  applyWhatIf: (twin, requested) => {
    const { appliedInputs, riskHistory, scenario } = get();
    const label = appliedInputs ? describeChanges(appliedInputs, requested) : "Updated";
    set({
      prediction: twin.prediction,
      scenarios: twin.scenarios,
      params: scenarioParams(twin.scenarios, scenario),
      currentVitals: vitalsFor(twin.scenarios, scenario),
      appliedInputs: requested,
      whatIfPending: false,
      whatIfError: "",
      riskHistory: withPoint(riskHistory, twin.prediction.risk_percent, label),
    });
  },

  setWhatIfError: (message) => set({ whatIfPending: false, whatIfError: message }),

  setScenario: (scenario) => {
    const { scenarios } = get();
    if (!scenarios) return;
    set({ scenario, params: scenarioParams(scenarios, scenario), currentVitals: vitalsFor(scenarios, scenario) });
  },

  resetToRecord: () => {
    const { recordTwin, recordInputs, recordVitals, appliedInputs, riskHistory, whatIfRequestId, patient } = get();
    if (!recordTwin || !recordInputs) return;
    const changed = JSON.stringify(appliedInputs) !== JSON.stringify(recordInputs);
    set((s) => ({
      inputs: recordInputs,
      appliedInputs: recordInputs,
      prediction: recordTwin.prediction,
      scenarios: recordTwin.scenarios,
      scenario: "rest",
      params: scenarioParams(recordTwin.scenarios, "rest"),
      currentVitals: recordVitals ? recordVitals.rest : null,
      riskHistory: changed
        ? withPoint(riskHistory, recordTwin.prediction.risk_percent, patient ? "Reset to recorded values" : "Reset to starting values")
        : riskHistory,
      // Bumping the request id discards any what-if still in flight.
      whatIfRequestId: whatIfRequestId + 1,
      whatIfPending: false,
      whatIfError: "",
      resetToken: s.resetToken + 1,
      paused: false,
      speed: 1,
    }));
  },
}));
