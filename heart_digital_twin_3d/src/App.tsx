import { useEffect } from "react";
import { Heart3D } from "./components/Heart3D/Heart3D";
import { Header } from "./components/Dashboard/Header";
import { VitalCards } from "./components/Dashboard/VitalCards";
import { ECGMonitor } from "./components/Dashboard/ECGMonitor";
import { PressureGraph } from "./components/Dashboard/PressureGraph";
import { ViewModeToggle } from "./components/Dashboard/ViewModeToggle";
import { InfoPanel } from "./components/Dashboard/InfoPanel";
import { PatientSummary } from "./components/Dashboard/PatientSummary";
import { InputControls } from "./components/Dashboard/InputControls";
import { RiskCharts } from "./components/Dashboard/RiskCharts";
import { DEFAULT_INPUTS, fetchPatientTwin, fetchWhatIf, twinRequestPath } from "./patient/patientTwin";
import { useHeartStore } from "./state/store";

// Opened from an assessment (?patient=<id> or ?self=1) the twin starts from
// that patient's record; opened on its own it starts from typical values.
function useInitialTwin() {
  useEffect(() => {
    const path = twinRequestPath();
    const { loadTwin, setTwinError } = useHeartStore.getState();
    const request = path ? fetchPatientTwin(path) : fetchWhatIf(DEFAULT_INPUTS);
    request.then((twin) => loadTwin(twin, path !== null)).catch((e: Error) => setTwinError(e.message));
  }, []);
}

// Whenever the inputs differ from the ones the current estimate describes,
// re-run the risk model (debounced, so a slider drag sends one request).
// Responses to superseded requests are dropped.
function useWhatIfSync() {
  const inputs = useHeartStore((s) => s.inputs);
  const applied = useHeartStore((s) => s.appliedInputs);
  useEffect(() => {
    if (!inputs || !applied || JSON.stringify(inputs) === JSON.stringify(applied)) return;
    const requested = inputs;
    const timer = setTimeout(() => {
      const { beginWhatIf } = useHeartStore.getState();
      const id = beginWhatIf();
      const isCurrent = () => useHeartStore.getState().whatIfRequestId === id;
      fetchWhatIf(requested)
        .then((twin) => isCurrent() && useHeartStore.getState().applyWhatIf(twin, requested))
        .catch((e: Error) => isCurrent() && useHeartStore.getState().setWhatIfError(e.message));
    }, 250);
    return () => clearTimeout(timer);
  }, [inputs, applied]);
}

export default function App() {
  useInitialTwin();
  useWhatIfSync();
  return (
    <div className="app-shell">
      <Header />
      <div className="app-banner">
        Decision-support visualisation, simulated from the details shown &mdash; not a diagnostic measurement.
      </div>
      <main className="app-grid">
        <div className="app-grid-left">
          <PatientSummary />
          <InputControls />
        </div>
        <div className="app-grid-center">
          <Heart3D />
          <ViewModeToggle />
          <VitalCards />
        </div>
        <div className="app-grid-right">
          <RiskCharts />
          <ECGMonitor />
          <PressureGraph />
          <InfoPanel />
        </div>
      </main>
    </div>
  );
}
