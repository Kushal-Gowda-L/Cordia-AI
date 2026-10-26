import { PATIENT_VIEW } from "../../patient/patientTwin";
import { useHeartStore } from "../../state/store";
import { useDatasetStore } from "../../dataset/datasetStore";
import { DIGITAL } from "../../variant";

// Inside the CordiaAI dashboard the page already shows the title, so the
// embedded twin shows only its controls.
const EMBEDDED = window.self !== window.top;

export function Header() {
  const paused = useHeartStore((s) => s.paused);
  const togglePaused = useHeartStore((s) => s.togglePaused);
  const speed = useHeartStore((s) => s.speed);
  const setSpeed = useHeartStore((s) => s.setSpeed);
  const resetToRecord = useHeartStore((s) => s.resetToRecord);
  const canReset = useHeartStore((s) => s.recordTwin !== null);
  const patient = useHeartStore((s) => s.patient);
  const fromDataset = useHeartStore((s) => s.source === "dataset");
  const reset = () => {
    useHeartStore.getState().setValves({ aorticStenosis: 0, mitralRegurgitation: 0 });
    useHeartStore.getState().clearTuning();
    if (fromDataset) useDatasetStore.getState().resetRow();
    else resetToRecord();
  };

  return (
    <header className="app-header">
      {!EMBEDDED ? (
        <div>
          <div className="app-header-title">
            {DIGITAL ? "Holographic digital twin" : "Digital twin"}
            {patient ? ` — ${patient.name}` : ""}
          </div>
          <div className="app-header-sub">
            CordiaAI &middot; {DIGITAL ? "live 3D heart for any team dataset" : "simulated heart based on the details shown"}
          </div>
        </div>
      ) : (
        <div className="app-header-sub">Simulated heart based on the details shown</div>
      )}
      <div className="app-header-controls">
        <button type="button" className="btn" onClick={togglePaused}>
          {paused ? "Resume" : "Pause"}
        </button>
        <button type="button" className="btn" onClick={reset} disabled={!canReset}>
          {PATIENT_VIEW ? "Back to my results" : fromDataset ? "Reset row" : patient ? "Reset to record" : "Reset"}
        </button>
        <label className="speed-control">
          Speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value={0.25}>0.25&times;</option>
            <option value={0.5}>0.5&times;</option>
            <option value={1}>1&times;</option>
            <option value={2}>2&times;</option>
          </select>
        </label>
      </div>
    </header>
  );
}
