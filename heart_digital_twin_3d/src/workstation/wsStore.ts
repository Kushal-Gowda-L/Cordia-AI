import { create } from "zustand";
import type { Dataset } from "../dataset/csv";
import { defaultState, healthyTargets, project, type State } from "./models";
import { matchColumns, rowState, type Matches } from "./wsData";

/**
 * The workstation's patient: the loaded patient (`base`, what "Reset to
 * patient" returns to and the trajectory's dashed line), the edited copy `S`,
 * and how many years ahead the time machine is looking. A dataset, when one
 * is loaded, supplies `base` row by row.
 */
interface WsState {
  base: State;
  S: State;
  years: number;
  playing: boolean;
  ds: Dataset | null;
  matches: Matches | null;
  row: number;
  setField: (key: string, value: number) => void;
  applyHealthy: () => void;
  reset: () => void;
  setYears: (y: number) => void;
  setPlaying: (on: boolean) => void;
  loadDataset: (ds: Dataset) => void;
  showRow: (row: number) => void;
  closeDataset: () => void;
}

const START = defaultState();

export const useWs = create<WsState>((set, get) => ({
  base: START,
  S: START,
  years: 0,
  playing: false,
  ds: null,
  matches: null,
  row: 0,
  setField: (key, value) => set((s) => ({ S: { ...s.S, [key]: value } })),
  applyHealthy: () => set((s) => ({ S: healthyTargets(s.S) })),
  reset: () => set((s) => ({ S: { ...s.base }, years: 0, playing: false })),
  setYears: (years) => set({ years }),
  setPlaying: (playing) => set({ playing }),
  loadDataset: (ds) => {
    set({ ds, matches: matchColumns(ds), row: 0 });
    get().showRow(0);
  },
  showRow: (row) => {
    const { ds, matches } = get();
    if (!ds || !matches) return;
    const r = ((row % ds.rows) + ds.rows) % ds.rows;
    // Unmatched fields fall back to the example patient, so each row stands on its own.
    const base = rowState(ds, matches, r, START);
    set({ row: r, base, S: { ...base }, years: 0, playing: false });
  },
  closeDataset: () =>
    set({
      ds: null,
      matches: null,
      row: 0,
      base: START,
      S: START,
      years: 0,
      playing: false,
    }),
}));

export const stateAt = (S: State, y: number) => (y > 0 ? project(S, y) : S);
