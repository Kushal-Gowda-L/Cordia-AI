import { create } from "zustand";
import type { HeartParams } from "../simulation/heartModel";
import type { State } from "../workstation/models";

/**
 * The dashboard's simulation controls. The baseline heart rate and blood
 * pressure live in the patient record (the workstation store's `S`, so the
 * risk pages follow them); activity adds its effect on top for the beating
 * heart, and SpO2 is set here.
 */

export type Activity = "resting" | "walking" | "jogging" | "running" | "sleeping";
export const ACTIVITIES: { key: Activity; label: string; intensity: number }[] = [
  { key: "resting", label: "Resting", intensity: 0 },
  { key: "walking", label: "Walking", intensity: 30 },
  { key: "jogging", label: "Jogging", intensity: 55 },
  { key: "running", label: "Running", intensity: 85 },
  { key: "sleeping", label: "Sleeping", intensity: 0 },
];

interface CordiaState {
  activity: Activity;
  intensity: number; // 0-100
  spo2: number;
  setActivity: (a: Activity) => void;
  setIntensity: (v: number) => void;
  setSpo2: (v: number) => void;
}

export const useCordia = create<CordiaState>((set) => ({
  activity: "resting",
  intensity: 0,
  spo2: 98,
  setActivity: (activity) => set({ activity, intensity: ACTIVITIES.find((a) => a.key === activity)!.intensity }),
  setIntensity: (intensity) =>
    set({ intensity, activity: intensity < 10 ? "resting" : intensity < 42 ? "walking" : intensity < 70 ? "jogging" : "running" }),
  setSpo2: (spo2) => set({ spo2 }),
}));

const clamp = (v: number, a: number, b: number) => Math.min(Math.max(v, a), b);

/** What the heart is doing right now: the patient's baseline plus the activity's effect. */
export function liveVitals(s: State, activity: Activity, intensity: number, spo2: number) {
  const sleeping = activity === "sleeping";
  const hr = clamp(s.heart_rate + intensity * 0.95 - (sleeping ? 12 : 0), 35, 190);
  const sys = clamp(s.sys_bp + intensity * 0.55 - (sleeping ? 10 : 0), 70, 240);
  const dia = clamp(s.dia_bp + intensity * 0.05 - (sleeping ? 6 : 0), 40, 150);
  const resp = Math.round(clamp((sleeping ? 12 : 16) + intensity * 0.22 + (spo2 < 94 ? (94 - spo2) * 0.6 : 0), 8, 40));
  return { hr, sys, dia, spo2, resp };
}

/** Those vitals as the simulated heart's settings. */
export function heartParams(s: State, v: ReturnType<typeof liveVitals>, intensity: number): HeartParams {
  return {
    heartRate: clamp(v.hr, 40, 180),
    systolicBP: clamp(v.sys, 80, 220),
    diastolicBP: clamp(v.dia, 40, 140),
    baseStrokeVolume: s.sex ? 72 : 62,
    contractility: clamp(50 + (s.ef - 55) * 1.3 + intensity * 0.35, 5, 100),
    preload: clamp(50 + intensity * 0.3, 0, 100),
    afterload: clamp(50 + 0.5 * (s.sys_bp - 120), 15, 95),
    oxygenSaturation: clamp(v.spo2, 70, 100),
    vascularResistance: clamp(50 + 0.6 * (s.sys_bp - 120) + 0.3 * Math.max(0, s.age - 50) - intensity * 0.3, 10, 95),
    aorticStenosis: 0,
    mitralRegurgitation: 0,
  };
}

/** A plain-words status for the live readings. */
export function simStatus(v: ReturnType<typeof liveVitals>, activity: Activity, intensity: number) {
  if (v.hr > 160 || v.hr < 42 || v.sys >= 180 || v.spo2 < 90) return { tone: "bad", text: "Critical (Check now)" };
  if (v.spo2 < 94 || (intensity < 10 && (v.hr > 100 || v.sys >= 140 || v.dia >= 90)) || v.hr < 50)
    return { tone: "warn", text: intensity < 10 ? "Elevated (Resting)" : "Elevated (Active)" };
  if (activity === "sleeping") return { tone: "ok", text: "Normal (Sleeping)" };
  if (intensity >= 10) return { tone: "ok", text: `Normal (${ACTIVITIES.find((a) => a.key === activity)!.label})` };
  return { tone: "ok", text: "Normal (Resting)" };
}
