import { create } from "zustand";
import type { Prediction, RiskReading } from "../patient/patientTwin";
import { useHeartStore } from "../state/store";
import { cellText, parseCsv, type Dataset } from "./csv";
import { deriveScenarios, guessMapping, guessTarget, readRoles, ROLE_BY_KEY, type Mapping, type RoleKey } from "./mapping";
import { outcomeOf, predictRisk, rowValues, trainRiskModel, type RiskModel, type RowValues } from "./riskModel";

/**
 * Dataset mode: one loaded CSV, the column matches, the model trained on it,
 * and the row currently shown as the heart. Every change here is pushed into
 * the heart store (loadDatasetTwin for a new row, applyDatasetTwin for an
 * edit), which re-runs the simulation exactly as the CordiaAI inputs do.
 */

export const SAMPLES: { id: string; label: string; detail: string; load: () => Promise<string> }[] = [
  {
    id: "uci",
    label: "UCI heart disease",
    detail: "920 patients, 4 hospitals, angiography outcome",
    load: () => import("./samples/uci_heart_disease.csv?raw").then((m) => m.default),
  },
  {
    id: "framingham",
    label: "Framingham",
    detail: "4,240 adults, 10-year coronary heart disease",
    load: () => import("./samples/framingham.csv?raw").then((m) => m.default),
  },
];

interface DatasetState {
  ds: Dataset | null;
  status: "empty" | "loading" | "ready" | "error";
  error: string;
  target: number | null;
  mapping: Mapping;
  model: RiskModel | null;
  modelError: string;
  row: number;
  original: RowValues;
  values: RowValues;

  loadText: (text: string, name: string) => void;
  loadFile: (file: File) => Promise<void>;
  loadSample: (id: string) => Promise<void>;
  setTarget: (column: number | null) => void;
  setRole: (role: RoleKey, column: number | null) => void;
  selectRow: (row: number) => void;
  randomRow: (outcome?: 0 | 1) => void;
  setValue: (column: number, value: number) => void;
  resetRow: () => void;
}

function makeReading(risk: number, outcome: string): RiskReading {
  const pct = Math.round(risk * 1000) / 10;
  const band = pct < 35 ? "unlikely" : pct < 65 ? "borderline" : "likely";
  const verdict = band === "unlikely" ? `${outcome}: unlikely` : band === "likely" ? `${outcome}: likely` : `${outcome}: borderline`;
  return {
    disease_percent: pct,
    no_disease_percent: Math.round((100 - pct) * 10) / 10,
    decision_point: 50,
    zones: { unlikely_below: 35, likely_from: 65 },
    band,
    verdict,
    predicts_disease: pct >= 50,
    severity_if_disease: {},
    likely_severity_if_disease: null,
    likely_severity_percent: null,
  };
}

function makePrediction(risk: number, model: RiskModel): Prediction {
  const pct = Math.round(risk * 1000) / 10;
  return {
    risk_percent: pct,
    severity: "",
    severity_percent: pct,
    class_probabilities: {},
    reading: makeReading(risk, outcomeName(model)),
  };
}

export function outcomeName(model: RiskModel): string {
  return `${model.targetLabel} = ${model.positiveLabel}`;
}

export const useDatasetStore = create<DatasetState>((set, get) => {
  /** Recomputes risk and heart settings for the current values and hands them to the heart. */
  function push(fresh: boolean, label: string) {
    const { ds, model, mapping, values, row } = get();
    if (!ds) return;
    const risk = model ? predictRisk(ds, model, values) : null;
    const roles = readRoles(ds, mapping, values);
    const scenarios = deriveScenarios(roles, risk, (k) => ds.columns[mapping[k] as number].name);
    const twin = {
      name: `${ds.name} · row ${row + 1}`,
      prediction: model && risk !== null ? makePrediction(risk, model) : null,
      scenarios,
    };
    const heart = useHeartStore.getState();
    if (fresh) heart.loadDatasetTwin(twin, label);
    else heart.applyDatasetTwin(twin, label);
  }

  function train(target: number | null): Pick<DatasetState, "model" | "modelError"> {
    const { ds } = get();
    if (!ds || target === null) return { model: null, modelError: "" };
    try {
      return { model: trainRiskModel(ds, target), modelError: "" };
    } catch (e) {
      return { model: null, modelError: (e as Error).message };
    }
  }

  return {
    ds: null,
    status: "empty",
    error: "",
    target: null,
    mapping: {},
    model: null,
    modelError: "",
    row: 0,
    original: [],
    values: [],

    loadText: (text, name) => {
      try {
        const ds = parseCsv(text, name);
        const target = guessTarget(ds);
        set({ ds, target, mapping: guessMapping(ds, target), status: "ready", error: "" });
        set(train(target));
        get().selectRow(0);
      } catch (e) {
        set({ status: "error", error: (e as Error).message });
      }
    },

    loadFile: async (file) => {
      set({ status: "loading", error: "" });
      const text = await file.text();
      // Let the "Reading…" state paint before the (synchronous) parse and training.
      setTimeout(() => get().loadText(text, file.name.replace(/\.(csv|tsv|txt)$/i, "")), 30);
    },

    loadSample: async (id) => {
      const sample = SAMPLES.find((s) => s.id === id);
      if (!sample) return;
      set({ status: "loading", error: "" });
      const text = await sample.load();
      setTimeout(() => get().loadText(text, sample.label), 30);
    },

    setTarget: (column) => {
      const { ds, mapping } = get();
      if (!ds) return;
      // The outcome can't also be a heart input.
      const cleaned = Object.fromEntries(Object.entries(mapping).filter(([, j]) => j !== column)) as Mapping;
      set({ target: column, mapping: cleaned, ...train(column) });
      push(true, "Outcome changed");
    },

    setRole: (role, column) => {
      const mapping = { ...get().mapping };
      if (column === null) delete mapping[role];
      else mapping[role] = column;
      set({ mapping });
      push(true, `${ROLE_BY_KEY[role].label} column changed`);
    },

    selectRow: (row) => {
      const { ds } = get();
      if (!ds) return;
      const r = Math.max(0, Math.min(ds.rows - 1, Math.round(row)));
      const values = rowValues(ds, r);
      set({ row: r, original: values, values: [...values] });
      push(true, `Row ${r + 1} as recorded`);
    },

    randomRow: (outcome) => {
      const { ds, target } = get();
      if (!ds) return;
      const candidates: number[] = [];
      for (let i = 0; i < ds.rows; i++) {
        if (outcome === undefined || target === null || outcomeOf(ds.columns[target], ds.columns[target].values[i]) === outcome) candidates.push(i);
      }
      if (candidates.length) get().selectRow(candidates[Math.floor(Math.random() * candidates.length)]);
    },

    setValue: (column, value) => {
      const { ds, values } = get();
      if (!ds || values[column] === value) return;
      const next = [...values];
      next[column] = value;
      set({ values: next });
      push(false, `${ds.columns[column].name} → ${valueText(ds, column, value)}`);
    },

    resetRow: () => {
      const { original, ds } = get();
      if (!ds) return;
      set({ values: [...original] });
      push(false, "Reset to the row as recorded");
    },
  };
});

/** Display text for a value of a column (not necessarily one stored in the data). */
export function valueText(ds: Dataset, column: number, value: number): string {
  const c = ds.columns[column];
  return cellText({ ...c, values: Float64Array.of(value), text: undefined }, 0);
}
