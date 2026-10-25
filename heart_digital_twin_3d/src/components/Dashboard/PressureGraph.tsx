import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { useHeartStore } from "../../state/store";

const HISTORY_SECONDS = 3.2;

interface Row {
  t: number;
  pLV: number;
  pRV: number;
  pLA: number;
  pRA: number;
  pAo: number;
}

export function PressureGraph() {
  const liveOutputs = useHeartStore((s) => s.liveOutputs);
  const paused = useHeartStore((s) => s.paused);
  const resetToken = useHeartStore((s) => s.resetToken);
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    setRows([]);
  }, [resetToken]);

  useEffect(() => {
    if (!liveOutputs || paused) return;
    const nextRow: Row = {
      t: Number(liveOutputs.t.toFixed(2)),
      pLV: Math.round(liveOutputs.pLV),
      pRV: Math.round(liveOutputs.pRV),
      pLA: Math.round(liveOutputs.pLA),
      pRA: Math.round(liveOutputs.pRA),
      pAo: Math.round(liveOutputs.pAo),
    };
    setRows((prev) => {
      // A reset snaps liveOutputs.t back near 0; drop stale post-reset
      // points so the trace doesn't jump backward in time.
      if (prev.length && nextRow.t < prev[prev.length - 1].t) prev = [];
      const cutoff = liveOutputs.t - HISTORY_SECONDS;
      const trimmed = prev[0] && prev[0].t < cutoff ? prev.filter((r) => r.t >= cutoff) : prev;
      return [...trimmed, nextRow];
    });
  }, [liveOutputs, paused]);

  return (
    <div className="panel pressure-panel">
      <div className="panel-title">
        <span>Pressure inside the heart</span>
        <span className="panel-subtitle">Simulated, in mm Hg. The left pump (LV) works hardest, pushing blood round the whole body.</span>
      </div>
      <div className="pressure-chart-wrap">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={rows} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#1a2738" strokeDasharray="2 4" />
            <XAxis dataKey="t" stroke="#7f93a8" fontSize={11} tickFormatter={(v) => `${v}s`} />
            <YAxis stroke="#7f93a8" fontSize={11} domain={[0, 160]} />
            <RTooltip contentStyle={{ background: "#0a1220", border: "1px solid #26374a", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="pLV" name="Left pump (LV)" stroke="#ff5c5c" dot={false} isAnimationActive={false} strokeWidth={1.8} />
            <Line type="monotone" dataKey="pRV" name="Right pump (RV)" stroke="#4da6ff" dot={false} isAnimationActive={false} strokeWidth={1.8} />
            <Line type="monotone" dataKey="pLA" name="Left upper chamber (LA)" stroke="#ffb3b3" dot={false} isAnimationActive={false} strokeWidth={1.4} />
            <Line type="monotone" dataKey="pRA" name="Right upper chamber (RA)" stroke="#b3d9ff" dot={false} isAnimationActive={false} strokeWidth={1.4} />
            <Line type="monotone" dataKey="pAo" name="Main artery (aorta)" stroke="#c77dff" dot={false} isAnimationActive={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
