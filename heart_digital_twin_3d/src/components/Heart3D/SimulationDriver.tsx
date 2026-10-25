import { useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { HeartSimulation } from "../../simulation/heartModel";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { useHeartStore } from "../../state/store";

/**
 * Runs entirely inside the Canvas render loop. Every frame it advances the
 * real physiological simulation by (dt * speed) unless paused, writes the
 * result into the 60fps-safe liveOutputsRef for the 3D meshes, and pushes
 * a throttled copy into the zustand store for the 2D dashboard/ECG/graph
 * components (~20Hz is plenty for readable DOM updates and avoids
 * re-rendering React on every animation frame).
 */
export function SimulationDriver() {
  const simRef = useRef<HeartSimulation>(new HeartSimulation());
  const accumRef = useRef(0);
  const params = useHeartStore((s) => s.params);
  const paused = useHeartStore((s) => s.paused);
  const speed = useHeartStore((s) => s.speed);
  const resetToken = useHeartStore((s) => s.resetToken);
  const setLiveOutputs = useHeartStore((s) => s.setLiveOutputs);
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  const speedRef = useRef(speed);
  speedRef.current = speed;

  useEffect(() => {
    simRef.current.reset(paramsRef.current);
  }, [resetToken]);

  useFrame((_, delta) => {
    if (!pausedRef.current) {
      const dt = Math.min(delta, 0.05) * speedRef.current;
      const outputs = simRef.current.step(dt, paramsRef.current);
      liveOutputsRef.current = outputs;

      accumRef.current += delta;
      if (accumRef.current > 0.05) {
        accumRef.current = 0;
        setLiveOutputs(outputs);
      }
    }
  });

  return null;
}
