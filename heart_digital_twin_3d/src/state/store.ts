import { create } from "zustand";
import { DIGITAL } from "../variant";
import { DEFAULT_PARAMS, HeartOutputs, HeartParams } from "../simulation/heartModel";
import type { PatientTwin, Prediction, PredictionInputs, ScenarioKey, TwinScenario } from "../patient/patientTwin";
import { steadyStateVitals, type Vitals } from "../simulation/steadyState";
import { describeChanges, fieldFor, type InputKey } from "../patient/inputFields";

export type ViewMode =
  | "anatomy"
  | "chambers"
  | "bloodFlow"
  | "pressure"
  | "valves"
  | "electrical"
  | "risk"
  | "realistic"
  // Holographic twin only (digital.html):
  | "hologram"
  | "lowpoly"
  | "dual";

/** Views showing a whole, closed model heart (no cutaway), which start turning when chosen. */
export function isModelView(m: ViewMode): boolean {
  return m === "realistic" || m === "hologram" || m === "lowpoly" || m === "dual";
}

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

/** Valve disease set directly with the valve controls, on top of whatever the data gives. */
export interface ValveDisease {
  aorticStenosis: number; // 0-1
  mitralRegurgitation: number; // 0-1
}
const NO_VALVE_DISEASE: ValveDisease = { aorticStenosis: 0, mitralRegurgitation: 0 };

/** Where the twin's inputs come from: the CordiaAI risk model's ten fields, or a loaded dataset. */
export type TwinSource = "example" | "dataset";

/** One dataset row turned into a twin (see src/dataset/). No prediction when no outcome column is chosen. */
export interface DatasetTwin {
  name: string;
  prediction: Prediction | null;
  scenarios: Record<ScenarioKey, TwinScenario>;
}

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

  /** Turntable: the camera circles the heart continuously (360 degrees). */
  autoRotate: boolean;
  setAutoRotate: (on: boolean) => void;

  toggles: StructureToggles;
  toggleStructure: (key: keyof StructureToggles) => void;

  xray: XRaySettings;
  setXRay: (patch: Partial<XRaySettings>) => void;

  // True when the backend couldn't be reached and the in-browser estimate
  // (src/patient/offlineTwin.ts) stands in for the server's model.
  offline: boolean;
  setOffline: (offline: boolean) => void;

  valves: ValveDisease;
  setValves: (patch: Partial<ValveDisease>) => void;

  source: TwinSource;
  // The example / patient twin as last loaded, so leaving dataset mode can return to it.
  exampleSnapshot: { twin: PatientTwin; hasPatient: boolean } | null;
  useExample: () => void;
  loadDatasetTwin: (twin: DatasetTwin, label: string) => void;
  applyDatasetTwin: (twin: DatasetTwin, label: string) => void;

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
  // The holographic page starts clean; labels are one click away.
  labels: !DIGITAL,
};

/** Whether the chambers inside are on show: always in the opened views, and in the closed ones once see-through or cut. */
export function showsInterior(s: { viewMode: ViewMode; xray: XRaySettings }): boolean {
  if (s.viewMode !== "anatomy" && s.viewMode !== "risk") return true;
  return s.xray.wallOpacity < 0.98 || s.xray.cutDepth > 0;
}

function scenarioParams(scenarios: Record<ScenarioKey, TwinScenario>, scenario: ScenarioKey, valves = NO_VALVE_DISEASE): HeartParams {
  return { ...DEFAULT_PARAMS, ...scenarios[scenario].params, ...valves };
}

function vitalsFor(scenarios: Record<ScenarioKey, TwinScenario>, scenario: ScenarioKey, valves = NO_VALVE_DISEASE): Vitals {
  return steadyStateVitals(scenarioParams(scenarios, scenario, valves));
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

  viewMode: DIGITAL ? "hologram" : "chambers",
  // The model hearts are made to be looked at all round, so they start turning.
  setViewMode: (m) => set(isModelView(m) ? { viewMode: m, autoRotate: true } : { viewMode: m }),

  autoRotate: DIGITAL,
  setAutoRotate: (on) => set({ autoRotate: on }),

  toggles: { ...DEFAULT_TOGGLES },
  toggleStructure: (key) => set((s) => ({ toggles: { ...s.toggles, [key]: !s.toggles[key] } })),

  xray: { wallOpacity: 1, cutDepth: 0 },
  setXRay: (patch) => set((s) => ({ xray: { ...s.xray, ...patch } })),

  offline: false,
  setOffline: (offline) => set({ offline }),

  valves: { ...NO_VALVE_DISEASE },
  setValves: (patch) => {
    const { scenarios, scenario } = get();
    const valves = { ...get().valves, ...patch };
    set(scenarios ? { valves, params: scenarioParams(scenarios, scenario, valves), currentVitals: vitalsFor(scenarios, scenario, valves) } : { valves });
  },

  source: "example",
  exampleSnapshot: null,
  useExample: () => {
    const snap = get().exampleSnapshot;
    set({ source: "example" });
    if (snap) get().loadTwin(snap.twin, snap.hasPatient);
  },

  loadDatasetTwin: (twin, label) => {
    const { valves, scenario } = get();
    // Vitals of the row as it is in the data (no valve disease), so the cards show the effect of edits and valve changes.
    const recordVitals = { rest: vitalsFor(twin.scenarios, "rest"), stress: vitalsFor(twin.scenarios, "stress") };
    const risk = twin.prediction?.risk_percent;
    set((s) => ({
      source: "dataset",
      twinStatus: "ready",
      twinError: "",
      patient: { name: twin.name, age: 0, sex: "" },
      recordTwin: { patient: { name: twin.name, age: 0, sex: "" }, prediction: twin.prediction, scenarios: twin.scenarios } as unknown as PatientTwin,
      recordInputs: null,
      prediction: twin.prediction,
      scenarios: twin.scenarios,
      params: scenarioParams(twin.scenarios, scenario, valves),
      recordVitals,
      currentVitals: vitalsFor(twin.scenarios, scenario, valves),
      riskHistory: risk === undefined ? [] : withPoint([], risk, label),
      // Discard any what-if for the example still in flight.
      whatIfRequestId: s.whatIfRequestId + 1,
      whatIfPending: false,
      whatIfError: "",
      resetToken: s.resetToken + 1,
    }));
  },

  applyDatasetTwin: (twin, label) => {
    const { valves, scenario, riskHistory } = get();
    const risk = twin.prediction?.risk_percent;
    set({
      prediction: twin.prediction,
      scenarios: twin.scenarios,
      params: scenarioParams(twin.scenarios, scenario, valves),
      currentVitals: vitalsFor(twin.scenarios, scenario, valves),
      riskHistory: risk === undefined ? riskHistory : withPoint(riskHistory, risk, label),
    });
  },

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
      exampleSnapshot: { twin, hasPatient },
      valves: { ...NO_VALVE_DISEASE },
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
      ...(toExercise && scenarios ? { scenario: "stress" as const, params: scenarioParams(scenarios, "stress", get().valves), currentVitals: vitalsFor(scenarios, "stress", get().valves) } : {}),
    });
  },

  beginWhatIf: () => {
    const id = get().whatIfRequestId + 1;
    set({ whatIfRequestId: id, whatIfPending: true, whatIfError: "" });
    return id;
  },

  applyWhatIf: (twin, requested) => {
    const { appliedInputs, riskHistory, scenario, valves, source } = get();
    if (source !== "example") return;
    const label = appliedInputs ? describeChanges(appliedInputs, requested) : "Updated";
    set({
      prediction: twin.prediction,
      scenarios: twin.scenarios,
      params: scenarioParams(twin.scenarios, scenario, valves),
      currentVitals: vitalsFor(twin.scenarios, scenario, valves),
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
    const { valves } = get();
    set({ scenario, params: scenarioParams(scenarios, scenario, valves), currentVitals: vitalsFor(scenarios, scenario, valves) });
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
      valves: { ...NO_VALVE_DISEASE },
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
