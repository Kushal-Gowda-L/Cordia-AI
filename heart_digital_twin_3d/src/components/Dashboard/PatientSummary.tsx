import type { HeartParams } from "../../simulation/heartModel";
import { PATIENT_VIEW, type ScenarioKey } from "../../patient/patientTwin";
import { useHeartStore } from "../../state/store";

const SCENARIOS: { key: ScenarioKey; label: string }[] = PATIENT_VIEW
  ? [
      { key: "rest", label: "Resting" },
      { key: "stress", label: "Being active" },
    ]
  : [
      { key: "rest", label: "At rest" },
      { key: "stress", label: "During exercise" },
    ];

// The simulation's internal settings, in everyday words.
const MODEL_ROWS: { key: keyof HeartParams; label: string; unit: string }[] = [
  { key: "heartRate", label: "Heart rate", unit: "beats/min" },
  { key: "systolicBP", label: "Blood pressure (top number)", unit: "mm Hg" },
  { key: "diastolicBP", label: "Blood pressure (bottom number)", unit: "mm Hg" },
  { key: "baseStrokeVolume", label: "Blood per heartbeat", unit: "mL" },
  { key: "contractility", label: "Pumping strength", unit: "%" },
  { key: "afterload", label: "Resistance to pumping", unit: "%" },
  { key: "vascularResistance", label: "Blood vessel tightness", unit: "%" },
  { key: "preload", label: "Blood returning to the heart", unit: "%" },
  { key: "oxygenSaturation", label: "Oxygen level", unit: "%" },
];

type SourceKind = "record" | "estimated" | "typical";

function sourceKind(source: string): SourceKind {
  if (source.startsWith("From record")) return "record";
  if (source.startsWith("Estimated")) return "estimated";
  return "typical";
}

const KIND_LABEL: Record<SourceKind, string> = {
  record: "From details",
  estimated: "Estimated",
  typical: "Typical value",
};

// Matches the zones in src/risk_reading.py (display convention, not clinical categories).
export function riskTone(pct: number): "low" | "mid" | "high" {
  return pct < 35 ? "low" : pct < 65 ? "mid" : "high";
}

export function PatientSummary() {
  const status = useHeartStore((s) => s.twinStatus);
  const error = useHeartStore((s) => s.twinError);
  const patient = useHeartStore((s) => s.patient);
  const prediction = useHeartStore((s) => s.prediction);
  const recordTwin = useHeartStore((s) => s.recordTwin);
  const scenarios = useHeartStore((s) => s.scenarios);
  const scenario = useHeartStore((s) => s.scenario);
  const setScenario = useHeartStore((s) => s.setScenario);
  const pending = useHeartStore((s) => s.whatIfPending);
  const whatIfError = useHeartStore((s) => s.whatIfError);

  if (status === "loading") return <div className="panel">Loading patient details&hellip;</div>;
  if (status === "error" || !prediction || !scenarios) {
    return (
      <div className="panel panel-error">
        <div className="panel-title">
          <span>The digital twin could not be loaded</span>
        </div>
        <p>{error}</p>
      </div>
    );
  }

  const baseline = recordTwin?.prediction.risk_percent;
  const delta = baseline === undefined ? 0 : prediction.risk_percent - baseline;
  const baselineName = PATIENT_VIEW ? "your results" : patient ? "the record" : "the starting values";
  const current = scenarios[scenario];

  return (
    <div className="panel patient-summary">
      <div className="panel-title">
        <span>{patient ? patient.name : "Example patient"}</span>
        <span className="panel-subtitle">
          {patient ? `${patient.age} years · ${patient.sex}` : "Not linked to a patient record"}
        </span>
      </div>

      <div className="risk-block">
        <div className={`risk-figure tone-${riskTone(prediction.risk_percent)}`}>{prediction.risk_percent}%</div>
        <div className="risk-caption">
          Chance of heart disease
          {pending && <span className="updating">updating&hellip;</span>}
        </div>
        {Math.abs(delta) >= 0.05 && (
          <div className={`risk-delta ${delta > 0 ? "up" : "down"}`}>
            {delta > 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(1)} points compared with {baselineName} ({baseline}%)
          </div>
        )}
        <div className={`verdict verdict-${riskTone(prediction.risk_percent)}`}>{prediction.reading.verdict}</div>
        {!PATIENT_VIEW && prediction.reading.likely_severity_if_disease && (
          <div className="risk-severity">
            If present, most likely: <b>{prediction.reading.likely_severity_if_disease}</b> (
            {prediction.reading.likely_severity_percent}% of disease cases)
          </div>
        )}
      </div>

      {whatIfError && <p className="inline-error">Couldn't update the estimate: {whatIfError}</p>}

      <div className="panel-label">Show the heart</div>
      <div className="scenario-toggle" role="group" aria-label="Show the heart">
        {SCENARIOS.map((s) => (
          <button
            key={s.key}
            type="button"
            className={`seg-button ${scenario === s.key ? "active" : ""}`}
            aria-pressed={scenario === s.key}
            onClick={() => setScenario(s.key)}
          >
            {s.label}
          </button>
        ))}
      </div>

      <details className="model-setup">
        <summary>How the heart model is set up</summary>
        <table className="derivation">
          <tbody>
            {MODEL_ROWS.map((row) => {
              const source = current.sources[row.key] ?? "";
              const kind = sourceKind(source);
              return (
                <tr key={row.key} title={source}>
                  <td>{row.label}</td>
                  <td className="num">
                    {current.params[row.key]} {row.unit}
                  </td>
                  <td>
                    <span className={`src-badge src-${kind}`}>{KIND_LABEL[kind]}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="patient-note">
          Worked out from {PATIENT_VIEW ? "your results" : "the patient's details"}. The record has no heart-scan data, so values marked
          &ldquo;Estimated&rdquo; follow set rules &mdash; hover over a row to see the rule.
        </p>
      </details>
    </div>
  );
}
