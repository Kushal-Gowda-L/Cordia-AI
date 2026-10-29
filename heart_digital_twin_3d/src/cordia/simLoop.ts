import { useEffect } from "react";
import { HeartSimulation, type HeartOutputs } from "../simulation/heartModel";
import { liveOutputsRef } from "../simulation/liveOutputsRef";
import { useHeartStore } from "../state/store";

/** The ECG at every simulation step (8 ms), for traces that keep each QRS sharp at any frame rate. */
export const ECG_TRACE: { t: number; v: number }[] = [];

/** Runs the heart simulation in real time, whichever page is showing. */
export function useSimulation() {
  useEffect(() => {
    const sim = new HeartSimulation();
    sim.reset(useHeartStore.getState().params);
    let last = performance.now();
    let acc = 0;
    let id = 0;
    const frame = (now: number) => {
      const { params, speed, setLiveOutputs } = useHeartStore.getState();
      let remaining = Math.min((now - last) / 1000, 0.25) * speed;
      acc += (now - last) / 1000;
      last = now;
      let out: HeartOutputs | null = null;
      while (remaining > 1e-6) {
        out = sim.step(Math.min(remaining, 0.008), params);
        remaining -= 0.008;
        ECG_TRACE.push({ t: out.t, v: out.ecg });
      }
      while (ECG_TRACE.length > 1200) ECG_TRACE.shift();
      if (out) {
        liveOutputsRef.current = out;
        if (acc > 0.05) {
          acc = 0;
          setLiveOutputs(out);
        }
      }
      id = requestAnimationFrame(frame);
    };
    id = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(id);
  }, []);
}

