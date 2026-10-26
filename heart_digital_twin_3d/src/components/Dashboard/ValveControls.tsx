import { useHeartStore } from "../../state/store";

const LEVELS = [
  { max: 0.05, label: "Healthy" },
  { max: 0.35, label: "Mild" },
  { max: 0.65, label: "Moderate" },
  { max: 1.01, label: "Severe" },
];
const level = (v: number) => LEVELS.find((l) => v < l.max)!.label;

/**
 * Valve disease on top of whatever drives the heart. Both feed the
 * simulation itself (src/simulation/heartModel.ts): a narrowed aortic valve
 * opens less and the left pump has to build more pressure to empty; a leaky
 * mitral valve lets blood squirt back into the left atrium every beat,
 * shown as a jet in the 3D view.
 */
export function ValveControls() {
  const valves = useHeartStore((s) => s.valves);
  const setValves = useHeartStore((s) => s.setValves);
  const rows: { key: keyof typeof valves; label: string; hint: string }[] = [
    { key: "aorticStenosis", label: "Narrowed aortic valve", hint: "Aortic stenosis: the exit valve of the left pump opens less" },
    { key: "mitralRegurgitation", label: "Leaky mitral valve", hint: "Mitral regurgitation: blood leaks back into the left atrium each beat" },
  ];
  return (
    <div className="panel input-controls">
      <div className="panel-title">
        <span>Valve conditions</span>
        <span className="panel-subtitle">Add valve disease to see how the valves and pumping change. Switch to the Valves view to watch them.</span>
      </div>
      {rows.map((r) => (
        <div key={r.key} className={`input-row ${valves[r.key] > 0 ? "changed" : ""}`}>
          <div className="input-head">
            <label htmlFor={`valve-${r.key}`}>{r.label}</label>
            <span className="heart-tag">&#9829;</span>
            <span className="input-value">{level(valves[r.key])}</span>
          </div>
          <div className="input-clinical">{r.hint}</div>
          <input
            id={`valve-${r.key}`}
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={valves[r.key]}
            onChange={(e) => setValves({ [r.key]: Number(e.target.value) })}
          />
        </div>
      ))}
    </div>
  );
}
