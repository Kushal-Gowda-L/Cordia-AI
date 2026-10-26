import { useHeartStore } from "../../state/store";
import { PATIENT_VIEW } from "../../patient/patientTwin";

/** Key for the risk view, drawn over the 3D canvas. */
export function RiskLegend() {
  const viewMode = useHeartStore((s) => s.viewMode);
  const reading = useHeartStore((s) => s.prediction?.reading);
  const pending = useHeartStore((s) => s.whatIfPending);
  const fromDataset = useHeartStore((s) => s.source === "dataset");
  if (viewMode !== "risk" || !reading) return null;
  const tone = reading.band === "unlikely" ? "low" : reading.band === "borderline" ? "mid" : "high";
  return (
    <div className="risk-legend" role="note">
      <div className="risk-legend-head">
        <span className={`risk-legend-pct tone-${tone}`}>{reading.disease_percent}%</span>
        <span>
          {PATIENT_VIEW ? "Chance of heart disease" : fromDataset ? "Estimated by the model trained on your data" : "Predicted chance of heart disease"}
          {pending && " · updating…"}
          <br />
          <b>{reading.verdict}</b>
        </span>
      </div>
      <div className="risk-legend-ramp" aria-hidden="true">
        <span style={{ left: `${Math.min(100, Math.max(0, reading.disease_percent))}%` }} />
      </div>
      <div className="risk-legend-scale" aria-hidden="true">
        <span>0%</span>
        <span>35%</span>
        <span>65%</span>
        <span>100%</span>
      </div>
      <p>
        Shading is strongest where the left anterior descending artery supplies the muscle, where coronary disease most often
        shows &mdash; a guide, not a scan of this heart.
      </p>
    </div>
  );
}
