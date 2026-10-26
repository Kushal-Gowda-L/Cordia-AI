import { useHeartStore, type Tuning } from "../../state/store";

const ROWS: { key: keyof Tuning; label: string; hint: string; min: number; max: number; unit: string }[] = [
  { key: "heartRate", label: "Heart rate", hint: "How fast the heart beats", min: 40, max: 180, unit: "beats/min" },
  { key: "systolicBP", label: "Blood pressure (top number)", hint: "Systolic pressure the left pump works against", min: 80, max: 220, unit: "mm Hg" },
  { key: "contractility", label: "Pumping strength", hint: "How hard the heart muscle squeezes", min: 10, max: 95, unit: "%" },
];

/**
 * Real-time handles on the beating heart. They start at whatever the data
 * gives and move every view at once -- the 3D hearts, valves, chambers,
 * blood flow, ECG and pressure traces -- without waiting for the risk
 * model. Changing the data itself takes over again for any value it sets.
 */
export function LiveControls() {
  const params = useHeartStore((s) => s.params);
  const tuning = useHeartStore((s) => s.tuning);
  const setTuning = useHeartStore((s) => s.setTuning);
  const clearTuning = useHeartStore((s) => s.clearTuning);
  const ready = useHeartStore((s) => s.scenarios !== null);
  const tuned = Object.keys(tuning).length > 0;
  return (
    <div className="panel input-controls">
      <div className="panel-title">
        <span>Live heart</span>
        <span className="panel-subtitle">
          Starts from the data shown. Drag to change the beating heart in real time; every view follows.
        </span>
      </div>
      {ROWS.map((r) => (
        <div key={r.key} className={`input-row ${tuning[r.key] !== undefined ? "changed" : ""}`}>
          <div className="input-head">
            <label htmlFor={`live-${r.key}`}>{r.label}</label>
            <span className="input-value">
              {Math.round(params[r.key])} {r.unit}
            </span>
          </div>
          <div className="input-clinical">{tuning[r.key] !== undefined ? "Adjusted by hand" : `${r.hint} (from the data)`}</div>
          <input
            id={`live-${r.key}`}
            type="range"
            min={r.min}
            max={r.max}
            step={1}
            disabled={!ready}
            value={Math.round(params[r.key])}
            onChange={(e) => setTuning({ [r.key]: Number(e.target.value) })}
          />
        </div>
      ))}
      {tuned && (
        <button type="button" className="btn btn-small" onClick={clearTuning}>
          Back to the data values
        </button>
      )}
    </div>
  );
}
