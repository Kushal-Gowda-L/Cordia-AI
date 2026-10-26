import { useHeartStore } from "../../state/store";

const VALVES = [
  { key: "aortic", label: "AORTIC" },
  { key: "pulmonic", label: "PULMONARY" },
  { key: "mitral", label: "MITRAL" },
  { key: "tricuspid", label: "TRICUSPID" },
] as const;

/**
 * Read-outs over the holographic twin's viewer: live heart rate, pressure,
 * pumping figures, the predicted risk and each valve's state as the
 * simulation runs -- the same live values as the cards below the viewer.
 */
export function HoloHud() {
  const out = useHeartStore((s) => s.liveOutputs);
  const params = useHeartStore((s) => s.params);
  const risk = useHeartStore((s) => s.prediction?.risk_percent ?? null);
  const valves = useHeartStore((s) => s.valves);
  if (!out) return null;
  const riskTone = risk === null ? "" : risk < 35 ? "holo-low" : risk < 65 ? "holo-mid" : "holo-high";
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
          HR <b>{Math.round(params.heartRate)}</b> bpm · BP{" "}
          <b>
            {Math.round(params.systolicBP)}/{Math.round(params.diastolicBP)}
          </b>
        </div>
        <div>
          EF <b>{Math.round(out.ejectionFraction)}</b>% · CO <b>{out.cardiacOutput.toFixed(1)}</b> L/min
        </div>
        <div>
          LV <b>{Math.round(out.vLV)}</b> mL · aorta <b>{Math.round(out.pAo)}</b> mmHg
        </div>
        {risk !== null && (
          <div className={riskTone}>
            RISK <b>{Math.round(risk)}%</b>
          </div>
        )}
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
