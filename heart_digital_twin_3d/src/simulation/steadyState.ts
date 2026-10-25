import { HeartParams, HeartSimulation } from "./heartModel";

/** Settled values for one set of parameters, used to compare the twin with the record. */
export interface Vitals {
  heartRate: number;
  strokeVolume: number;
  cardiacOutput: number;
  ejectionFraction: number;
  workload: number;
  systolicBP: number;
}

// From a cold start the simulation settles within ~10 s of simulated time;
// 20 s is comfortably past that and takes a few milliseconds to compute.
const SETTLE_SECONDS = 20;
const DT = 1 / 60;

export function steadyStateVitals(params: HeartParams): Vitals {
  const sim = new HeartSimulation();
  sim.reset(params);
  let out = sim.step(DT, params);
  for (let i = 1; i < SETTLE_SECONDS / DT; i++) out = sim.step(DT, params);
  return {
    heartRate: params.heartRate,
    strokeVolume: out.effectiveStrokeVolume,
    cardiacOutput: out.cardiacOutput,
    ejectionFraction: out.ejectionFraction,
    workload: out.cardiacWorkload,
    systolicBP: params.systolicBP,
  };
}
