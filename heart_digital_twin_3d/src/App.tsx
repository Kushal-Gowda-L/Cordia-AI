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
import { SourceToggle } from "./components/Dashboard/SourceToggle";
import { DatasetPanel } from "./components/Dashboard/DatasetPanel";
import { ValveControls } from "./components/Dashboard/ValveControls";
import { PATIENT_VIEW } from "./patient/patientTwin";
import { DEFAULT_INPUTS, fetchPatientTwin, fetchWhatIf, FORCE_OFFLINE, twinRequestPath, type PredictionInputs } from "./patient/patientTwin";
import { buildOfflineTwin } from "./patient/offlineTwin";
import { useHeartStore } from "./state/store";

// Opened from an assessment (?patient=<id> or ?self=1) the twin starts from
// that patient's record; opened on its own it starts from typical values.
// Opened on its own with no reachable backend (e.g. plain `npm run dev`), or
// with ?offline=1, it switches to the in-browser estimate instead of failing.
function useInitialTwin() {
  useEffect(() => {
    const path = twinRequestPath();
    const { loadTwin, setTwinError, setOffline } = useHeartStore.getState();
    const goOffline = () => {
      setOffline(true);
      loadTwin(buildOfflineTwin(DEFAULT_INPUTS, "Example patient"), false);
    };
    if (path) {
      fetchPatientTwin(path).then((twin) => loadTwin(twin, true)).catch((e: Error) => setTwinError(e.message));
    } else if (FORCE_OFFLINE) {
      goOffline();
    } else {
      fetchWhatIf(DEFAULT_INPUTS).then((twin) => loadTwin(twin, false)).catch(goOffline);
    }
  }, []);
}

function whatIf(inputs: PredictionInputs) {
  return useHeartStore.getState().offline ? Promise.resolve(buildOfflineTwin(inputs)) : fetchWhatIf(inputs);
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
      whatIf(requested)
        .then((twin) => isCurrent() && useHeartStore.getState().applyWhatIf(twin, requested))
        .catch((e: Error) => isCurrent() && useHeartStore.getState().setWhatIfError(e.message));
    }, useHeartStore.getState().offline ? 60 : 250);
    return () => clearTimeout(timer);
  }, [inputs, applied]);
}

export default function App() {
  useInitialTwin();
  useWhatIfSync();
  const fromDataset = useHeartStore((s) => s.source === "dataset");
  return (
    <div className="app-shell">
      <Header />
      <div className="app-banner">
        Decision-support visualisation, simulated from the details shown &mdash; not a diagnostic measurement.
      </div>
      <main className="app-grid">
        <div className="app-grid-left">
          {!PATIENT_VIEW && <SourceToggle />}
          {fromDataset ? (
            <DatasetPanel />
          ) : (
            <>
              <PatientSummary />
              <InputControls />
            </>
          )}
          {!PATIENT_VIEW && <ValveControls />}
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
