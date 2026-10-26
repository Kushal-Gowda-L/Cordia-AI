import { useHeartStore } from "../../state/store";

const VALVES = [
  { key: "aortic", label: "AORTIC" },
  { key: "pulmonic", label: "PULMONARY" },
  { key: "mitral", label: "MITRAL" },
  { key: "tricuspid", label: "TRICUSPID" },
] as const;

/**
 * Read-outs over the holographic twin's viewer: live heart rate, pressure,
 * pumping figures and each valve's state as the simulation runs.
 */
export function HoloHud() {
  const out = useHeartStore((s) => s.liveOutputs);
  const hr = useHeartStore((s) => s.params.heartRate);
  const valves = useHeartStore((s) => s.valves);
  if (!out) return null;
  const note = (key: string) =>
    key === "aortic" && valves.aorticStenosis > 0.05
      ? " · NARROWED"
      : key === "mitral" && valves.mitralRegurgitation > 0.05
        ? " · LEAKING"
        : "";
  return (
    <>
      <div className="holo-hud holo-hud-left" aria-label="Live readings">
        <div className="holo-hud-title">LIVE TWIN</div>
        <div>
          HR <b>{Math.round(hr)}</b> bpm
        </div>
        <div>
          BP <b>{Math.round(out.pAo)}</b> mmHg · MAP <b>{Math.round(out.meanArterialPressure)}</b>
        </div>
        <div>
          EF <b>{Math.round(out.ejectionFraction)}</b>% · CO <b>{out.cardiacOutput.toFixed(1)}</b> L/min
        </div>
        <div>
          LV <b>{Math.round(out.vLV)}</b> mL · RV <b>{Math.round(out.vRV)}</b> mL
        </div>
      </div>
      <div className="holo-hud holo-hud-right" aria-label="Valves">
        <div className="holo-hud-title">VALVES</div>
        {VALVES.map((v) => {
          const open = out.valves[v.key] > 0.25;
          return (
            <div key={v.key} className={open ? "holo-open" : ""}>
              <span className="holo-dot" /> {v.label} {open ? "OPEN" : "CLOSED"}
              {note(v.key)}
            </div>
          );
        })}
      </div>
    </>
  );
}
