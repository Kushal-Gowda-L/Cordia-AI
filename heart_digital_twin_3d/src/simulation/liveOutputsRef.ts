import { HeartOutputs } from "./heartModel";

/**
 * A plain mutable holder for the latest simulation frame, written every
 * animation frame by SimulationDriver. The 3D mesh components read this
 * directly inside their own useFrame callbacks instead of subscribing to
 * React/zustand state, so chamber contraction and particle motion stay at
 * a smooth 60fps without triggering a React re-render on every tick.
 * The 2D dashboard/graphs use the separate, throttled zustand
 * `liveOutputs` field instead (see state/store.ts) -- DOM updates don't
 * need 60Hz.
 */
export const liveOutputsRef: { current: HeartOutputs | null } = { current: null };
