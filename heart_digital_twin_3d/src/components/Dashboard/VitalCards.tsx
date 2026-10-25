import { PATIENT_VIEW } from "../../patient/patientTwin";
import type { Vitals } from "../../simulation/steadyState";
import { useHeartStore } from "../../state/store";

function phaseLabel(tFrac: number): string {
  if (tFrac < 0.09) return "Upper chambers squeezing";
  if (tFrac < 0.17) return "Signal passing to the pumps";
  if (tFrac < 0.45) return "Pumping blood out";
  if (tFrac < 0.55) return "Relaxing";
  return "Filling with blood";
}

/** How one number compares with the record, and whether that direction is good for the heart. */
interface DeltaSpec {
  key: keyof Vitals;
  /** "percent" for flows and workload, otherwise the difference in the card's own unit. */
  as: "percent" | "units" | "points";
  unit?: string;
  /** Which direction is better; "neutral" when neither is (e.g. heart rate). */
  better: "up" | "down" | "neutral";
}

interface Card {
  label: string;
  clinical: string;
  value: string;
  unit?: string;
  text?: boolean;
  delta?: DeltaSpec;
}

function Delta({ spec, now, record, baselineName }: { spec: DeltaSpec; now: Vitals; record: Vitals; baselineName: string }) {
  const a = now[spec.key];
  const b = record[spec.key];
  const diff = spec.as === "percent" ? ((a - b) / b) * 100 : a - b;
  if (Math.abs(diff) < 1) return <div className="vital-delta same">Same as {baselineName}</div>;
  const up = diff > 0;
  const tone = spec.better === "neutral" ? "neutral" : (spec.better === "up") === up ? "better" : "worse";
  const amount =
    spec.as === "percent"
      ? `${Math.abs(diff).toFixed(0)}%`
      : spec.as === "points"
        ? `${Math.abs(diff).toFixed(0)} points`
        : `${Math.abs(diff).toFixed(0)} ${spec.unit ?? ""}`.trim();
  return (
    <div className={`vital-delta ${tone}`} title={`Settled value compared with ${baselineName}`}>
      {up ? "▲" : "▼"} {amount} vs {baselineName}
    </div>
  );
}

export function VitalCards() {
  const params = useHeartStore((s) => s.params);
  const out = useHeartStore((s) => s.liveOutputs);
  const now = useHeartStore((s) => s.currentVitals);
  const recordVitals = useHeartStore((s) => s.recordVitals);
  const scenario = useHeartStore((s) => s.scenario);
  const hasPatient = useHeartStore((s) => s.patient !== null);
  const record = recordVitals ? recordVitals[scenario] : null;
  const baselineName = PATIENT_VIEW ? "your results" : hasPatient ? "record" : "start";
  const dash = "—";

  const cards: Card[] = [
    { label: "Heart rate", clinical: "heart rate", value: params.heartRate.toFixed(0), unit: "beats/min",
      delta: { key: "heartRate", as: "units", unit: "beats/min", better: "neutral" } },
    {
      label: "Blood per heartbeat",
      clinical: "stroke volume",
      value: out ? out.effectiveStrokeVolume.toFixed(0) : dash,
      unit: "mL",
      delta: { key: "strokeVolume", as: "percent", better: "up" },
    },
    {
      label: "Blood pumped per minute",
      clinical: "cardiac output",
      value: out ? out.cardiacOutput.toFixed(1) : dash,
      unit: "L/min",
      delta: { key: "cardiacOutput", as: "percent", better: "up" },
    },
    {
      label: "Blood pressure",
      clinical: "systolic / diastolic",
      value: `${params.systolicBP.toFixed(0)}/${params.diastolicBP.toFixed(0)}`,
      unit: "mm Hg",
      delta: { key: "systolicBP", as: "units", unit: "mm Hg", better: "down" },
    },
    {
      label: "Pumping efficiency",
      clinical: "ejection fraction",
      value: out ? out.ejectionFraction.toFixed(0) : dash,
      unit: "%",
      delta: { key: "ejectionFraction", as: "points", better: "up" },
    },
    { label: "Oxygen level", clinical: "oxygen saturation", value: params.oxygenSaturation.toFixed(0), unit: "%" },
    {
      label: "Heart workload",
      clinical: "heart rate × pressure ÷ 100",
      value: out ? out.cardiacWorkload.toFixed(0) : dash,
      delta: { key: "workload", as: "percent", better: "down" },
    },
    { label: "Heartbeat stage", clinical: "cardiac cycle phase", value: out ? phaseLabel(out.tFrac) : dash, text: true },
  ];

  return (
    <>
      <div className="vital-cards">
        {cards.map((c) => (
          <div key={c.label} className="vital-card">
            <div className="vital-card-label">{c.label}</div>
            <div className="vital-card-clinical">{c.clinical}</div>
            <div className={`vital-card-value ${c.text ? "text" : ""}`}>
              {c.value}
              {c.unit && <span className="vital-card-unit">{c.unit}</span>}
            </div>
            {c.delta && now && record && <Delta spec={c.delta} now={now} record={record} baselineName={baselineName} />}
          </div>
        ))}
      </div>
      <div className="clinical-strip">
        <span className="clinical-strip-title">Clinical measures (simulated)</span>
        <span>
          Mean arterial pressure <b>{out ? out.meanArterialPressure.toFixed(0) : dash}</b> mm Hg
        </span>
        <span>
          End-diastolic / end-systolic volume{" "}
          <b>{out ? `${out.edv.toFixed(0)} / ${out.esv.toFixed(0)}` : dash}</b> mL
        </span>
        <span>
          Coronary perfusion pressure <b>{out ? out.coronaryPerfusionPressure.toFixed(0) : dash}</b> mm Hg
        </span>
      </div>
    </>
  );
}
