import { useEffect, useRef, useState } from "react";
import { useHeartStore } from "../../state/store";

const WIDTH = 460;
const HEIGHT = 130;
const HISTORY_SECONDS = 3.2;

export function ECGMonitor() {
  const liveOutputs = useHeartStore((s) => s.liveOutputs);
  const paused = useHeartStore((s) => s.paused);
  const resetToken = useHeartStore((s) => s.resetToken);
  const pointsRef = useRef<{ t: number; v: number }[]>([]);
  const [, forceTick] = useState(0);

  useEffect(() => {
    pointsRef.current = [];
    forceTick((n) => n + 1);
  }, [resetToken]);

  useEffect(() => {
    if (!liveOutputs || paused) return;
    const pts = pointsRef.current;
    // A reset snaps liveOutputs.t back near 0; drop any stale points from
    // before the reset so the trace doesn't jump backward in time.
    if (pts.length && liveOutputs.t < pts[pts.length - 1].t) pts.length = 0;
    pts.push({ t: liveOutputs.t, v: liveOutputs.ecg });
    const cutoff = liveOutputs.t - HISTORY_SECONDS;
    while (pts.length > 1 && pts[0].t < cutoff) pts.shift();
    forceTick((n) => n + 1);
  }, [liveOutputs, paused]);

  const pts = pointsRef.current;
  const tMax = pts.length ? pts[pts.length - 1].t : 0;
  const tMin = tMax - HISTORY_SECONDS;
  const path = pts
    .map((p, i) => {
      const x = ((p.t - tMin) / HISTORY_SECONDS) * WIDTH;
      const y = HEIGHT / 2 - p.v * 55;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="panel ecg-panel">
      <div className="panel-title">
        <span>Heartbeat signal (ECG)</span>
        <span className="panel-subtitle">Simulated trace, in step with the beating heart</span>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="ecg-svg" preserveAspectRatio="none">
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={0} y1={HEIGHT * f} x2={WIDTH} y2={HEIGHT * f} stroke="#1a2738" strokeWidth={1} />
        ))}
        <path d={path} fill="none" stroke="#39e6c1" strokeWidth={2} />
      </svg>
    </div>
  );
}
