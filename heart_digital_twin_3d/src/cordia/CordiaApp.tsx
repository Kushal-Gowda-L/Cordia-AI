import { useEffect, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { parseCsv } from "../dataset/csv";
import { SAMPLES } from "../dataset/datasetStore";
import { HeartSimulation, type HeartOutputs } from "../simulation/heartModel";
import { liveOutputsRef } from "../simulation/liveOutputsRef";
import { useHeartStore } from "../state/store";
import { evaluate, F, framinghamRisk, healthyTargets, mainDrivers, MODELS, project, stenosis, type State } from "../workstation/models";
import { useWs } from "../workstation/wsStore";
import { ACTIVITIES, heartParams, liveVitals, simStatus, useCordia, type Activity } from "./cordiaStore";
import { ShowcaseHeart, type ShowcaseView } from "./ShowcaseHeart";

/**
 * The Cordia AI dashboard: the project's sidebar, the illustrated 3D heart
 * twin with its labels and views in the middle, and the live parameters, ECG
 * and simulation controls on the right. The other sidebar pages show the
 * rest of the project around the same live patient: monitoring, the four
 * risk models, any team dataset, the physiology, the 10-year history and
 * preventive targets.
 */

type Section = "overview" | "twin" | "monitoring" | "risk" | "dataset" | "simulation" | "history" | "preventive" | "about";

const clamp = (v: number, a: number, b: number) => Math.min(Math.max(v, a), b);
const pct = (p: number) => `${Math.round(p * 100)}%`;
const riskTone = (p: number) => (p < 0.1 ? "ok" : p < 0.2 ? "lime" : p < 0.4 ? "warn" : "bad");

/* ------------------------------------------------------------------ simulation */

/** The ECG at every simulation step (8 ms), for traces that keep each QRS sharp at any frame rate. */
const ECG_TRACE: { t: number; v: number }[] = [];

/** Runs the heart simulation in real time, whichever page is showing. */
function useSimulation() {
  useEffect(() => {
    const sim = new HeartSimulation();
    sim.reset(useHeartStore.getState().params);
    let last = performance.now();
    let acc = 0;
    let id = 0;
    const frame = (now: number) => {
      const { params, speed, setLiveOutputs } = useHeartStore.getState();
      let remaining = Math.min((now - last) / 1000, 0.25) * speed;
      acc += (now - last) / 1000;
      last = now;
      let out: HeartOutputs | null = null;
      while (remaining > 1e-6) {
        out = sim.step(Math.min(remaining, 0.008), params);
        remaining -= 0.008;
        ECG_TRACE.push({ t: out.t, v: out.ecg });
      }
      while (ECG_TRACE.length > 1200) ECG_TRACE.shift();
      if (out) {
        liveOutputsRef.current = out;
        if (acc > 0.05) {
          acc = 0;
          setLiveOutputs(out);
        }
      }
      id = requestAnimationFrame(frame);
    };
    id = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(id);
  }, []);
}

function useLive() {
  const S = useWs((w) => w.S);
  const { activity, intensity, spo2 } = useCordia();
  const v = useMemo(() => liveVitals(S, activity, intensity, spo2), [S, activity, intensity, spo2]);
  return { S, activity, intensity, spo2, v };
}

/** Keeps the simulated heart and its risk colour in step with the patient and the controls. */
function useDriveHeart() {
  const { S, intensity, v } = useLive();
  const out = useMemo(() => evaluate(S), [S]);
  useEffect(() => {
    const risk = Math.round(Math.max(out.cad, out.cvd, out.chd10) * 100);
    useHeartStore.setState({ params: heartParams(S, v, intensity), twinStatus: "ready" });
    useHeartStore.setState((st) => ({ prediction: { ...(st.prediction ?? {}), risk_percent: risk } as typeof st.prediction }));
  }, [S, v, intensity, out]);
}

/* ------------------------------------------------------------------ icons */

const P = {
  overview: "M3 11 12 4l9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z",
  twin: "M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.5-7 10-7 10zM8 11h2l1-2 2 4 1-2h2",
  monitoring: "M3 12h4l2-6 4 12 2-6h6",
  risk: "M5 20V12M10 20V8M15 20v-6M20 20V5",
  dataset: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  simulation: "M4 7h10M18 7h2M4 17h4M12 17h8M14 4v6M8 14v6",
  history: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  preventive: "M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6zM8.5 12l2.5 2.5 4.5-5",
  about: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v6M12 7.5v.5",
};
function Icon({ d, size = 22 }: { d: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

const NAV: { key: Section; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "twin", label: "Digital Twin Heart" },
  { key: "monitoring", label: "Patient Monitoring" },
  { key: "risk", label: "Risk Assessment" },
  { key: "dataset", label: "Dataset & Model" },
  { key: "simulation", label: "Heart Simulation" },
  { key: "history", label: "Patient History" },
  { key: "preventive", label: "Preventive Care" },
  { key: "about", label: "About Project" },
];

const TITLES: Record<Section, [string, string]> = {
  overview: ["Overview", "CordiaAI at a glance: the live patient and the system behind it"],
  twin: ["Digital Twin Heart", "Interactive 3D visualization of the heart with real-time parameters"],
  monitoring: ["Patient Monitoring", "Live vital signs, ECG, pressures and chamber volumes from the twin"],
  risk: ["Risk Assessment", "Four risk models trained on four datasets, re-scored live in the browser"],
  dataset: ["Dataset & Model", "Load any team heart dataset; each row becomes the patient"],
  simulation: ["Heart Simulation", "The beat-by-beat physiology driving the twin"],
  history: ["Patient History", "Where this patient's 10-year coronary risk is heading"],
  preventive: ["Preventive Care", "What changing each modifiable factor would do to the risk"],
  about: ["About Project", "AI Powered System for Predictive and Preventive Cardiac Care"],
};

function Logo() {
  return (
    <div className="brand">
      <svg viewBox="0 0 64 56" width="52" height="46" aria-hidden="true">
        <path d="M32 52S4 34 4 17A13 13 0 0 1 32 10a13 13 0 0 1 28 7c0 17-28 35-28 35z" fill="none" stroke="#ff3b4e" strokeWidth="4" strokeLinejoin="round" />
        <path d="M10 27h12l4-8 6 16 5-12 3 4h14" fill="none" stroke="#ff3b4e" strokeWidth="3.4" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
      <div>
        <div className="brand-name">
          CORDIA <span>AI</span>
        </div>
        <div className="brand-sub">AI Powered System for Predictive and Preventive Cardiac Care</div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ right rail */

/** A small moving trace under each vital card: three cycles across the card, scrolling at the live rate. */
function Wave({ kind, period }: { kind: "ecg" | "bp" | "pleth" | "resp"; period: number }) {
  const d = useMemo(() => {
    const pts: string[] = [];
    const C = 100 / 3;
    for (let x = 0; x <= 200; x += 0.25) {
      const t = (x % C) / C;
      let y = 0;
      if (kind === "ecg")
        y =
          -0.12 * Math.exp(-Math.pow((t - 0.25) / 0.04, 2)) +
          0.25 * Math.exp(-Math.pow((t - 0.44) / 0.012, 2)) -
          1.0 * Math.exp(-Math.pow((t - 0.47) / 0.012, 2)) +
          0.4 * Math.exp(-Math.pow((t - 0.5) / 0.012, 2)) -
          0.22 * Math.exp(-Math.pow((t - 0.72) / 0.06, 2));
      if (kind === "bp") y = -0.75 * Math.exp(-Math.pow((t - 0.3) / 0.1, 2)) - 0.3 * Math.exp(-Math.pow((t - 0.58) / 0.06, 2)) + 0.35 * t;
      if (kind === "pleth") y = -0.7 * Math.exp(-Math.pow((t - 0.35) / 0.14, 2)) - 0.22 * Math.exp(-Math.pow((t - 0.65) / 0.08, 2)) + 0.3 * t;
      if (kind === "resp") y = -0.6 * Math.sin(t * Math.PI * 2);
      pts.push(`${x === 0 ? "M" : "L"}${x} ${(15 + y * 12).toFixed(1)}`);
    }
    return pts.join(" ");
  }, [kind]);
  return (
    <div className="wave">
      <svg viewBox="0 0 200 30" preserveAspectRatio="none">
        <path d={d} style={{ animationDuration: `${period * 3}s` }} />
      </svg>
    </div>
  );
}

function Vital({ icon, title, value, unit, wave, period, tone }: { icon: ReactNode; title: ReactNode; value: ReactNode; unit: string; wave: "ecg" | "bp" | "pleth" | "resp"; period: number; tone: string }) {
  return (
    <div className={`vital tone-${tone}`}>
      <div className="vital-icon">{icon}</div>
      <div>
        <div className="vital-title">{title}</div>
        <div className="vital-value">
          {value} <small>{unit}</small>
        </div>
      </div>
      <Wave kind={wave} period={period} />
    </div>
  );
}

const ICON = {
  heart: (
    <svg viewBox="0 0 24 24" width="38" height="38">
      <path d="M12 21s-8-5-8-11a4.5 4.5 0 0 1 8-2.8A4.5 4.5 0 0 1 20 10c0 6-8 11-8 11z" fill="#ff3b4e" />
    </svg>
  ),
  steth: (
    <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="#7fc4ff" strokeWidth="1.6" strokeLinecap="round">
      <path d="M6 3v6a4 4 0 0 0 8 0V3M10 13v2a5 5 0 0 0 10 0v-2" />
      <circle cx="20" cy="11" r="2" />
    </svg>
  ),
  lungs: (
    <svg viewBox="0 0 24 24" width="38" height="38" fill="#3ec5ff">
      <path d="M11 4v7l-2-1.5V7c0-1-1-2-2-1.5C4 7 3 12 3 16c0 2 1 3 3 3 3 0 4-2 4-4v-3l1-1h2l1 1v3c0 2 1 4 4 4 2 0 3-1 3-3 0-4-1-9-4-10.5-1-.5-2 .5-2 1.5v2.5L13 11V4z" />
    </svg>
  ),
};

function LiveParameters() {
  const { v, activity, intensity } = useLive();
  const status = simStatus(v, activity, intensity);
  const period = 60 / v.hr;
  return (
    <section className="card">
      <h3>
        <Icon d={P.twin} size={24} /> Live Parameters <span className="muted">(Simulated)</span>
      </h3>
      <div className="vitals">
        <Vital icon={ICON.heart} title="Heart Rate" value={Math.round(v.hr)} unit="BPM" wave="ecg" period={period} tone="red" />
        <Vital icon={ICON.steth} title="Blood Pressure" value={`${Math.round(v.sys)} / ${Math.round(v.dia)}`} unit="mmHg" wave="bp" period={period} tone="blue" />
        <Vital
          icon={ICON.lungs}
          title={
            <>
              SpO<sub>2</sub>
            </>
          }
          value={Math.round(v.spo2)}
          unit="%"
          wave="pleth"
          period={period * 2}
          tone="cyan"
        />
        <Vital icon={ICON.lungs} title="Respiratory Rate" value={v.resp} unit="breaths/min" wave="resp" period={60 / v.resp} tone="cyan" />
      </div>
      <div className={`status tone-${status.tone}`}>
        <span className="status-dot" />
        <div>
          <div className="muted">Simulation Status</div>
          <b>{status.text}</b>
        </div>
      </div>
    </section>
  );
}

/** The twin's ECG, sampled from the simulation every frame and swept across a monitor grid. */
function EcgStrip({ seconds = 4, height = 150 }: { seconds?: number; height?: number }) {
  const cv = useRef<HTMLCanvasElement>(null);
  const hr = useHeartStore((s) => Math.round(s.params.heartRate));
  useEffect(() => {
    const c = cv.current!;
    const g = c.getContext("2d")!;
    const pts = ECG_TRACE;
    let id = 0;
    const draw = () => {
      const out = liveOutputsRef.current;
      const dpr = window.devicePixelRatio || 1;
      const W = c.clientWidth;
      const H = c.clientHeight;
      if (c.width !== Math.round(W * dpr)) {
        c.width = Math.round(W * dpr);
        c.height = Math.round(H * dpr);
      }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      g.strokeStyle = "rgba(255,70,90,0.10)";
      g.lineWidth = 1;
      for (let x = 0; x < W; x += 16) {
        g.beginPath();
        g.moveTo(x + 0.5, 0);
        g.lineTo(x + 0.5, H);
        g.stroke();
      }
      for (let y = 0; y < H; y += 16) {
        g.beginPath();
        g.moveTo(0, y + 0.5);
        g.lineTo(W, y + 0.5);
        g.stroke();
      }
      if (pts.length > 1) {
        const tMax = pts[pts.length - 1].t;
        g.strokeStyle = "#ff3b4e";
        g.lineWidth = 2;
        g.shadowColor = "#ff3b4e";
        g.shadowBlur = 6;
        g.beginPath();
        let first = true;
        pts.forEach((p) => {
          if (p.t < tMax - seconds) return;
          const x = W - ((tMax - p.t) / seconds) * W;
          const y = H * 0.58 - p.v * H * 0.36;
          if (first) g.moveTo(x, y);
          else g.lineTo(x, y);
          first = false;
        });
        g.stroke();
        g.shadowBlur = 0;
      }
      id = requestAnimationFrame(draw);
    };
    id = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(id);
  }, [seconds]);
  return (
    <div className="ecg" style={{ height }}>
      <canvas ref={cv} />
      <span className="ecg-hr">HR: {hr} BPM</span>
    </div>
  );
}

function HeartActivity() {
  return (
    <section className="card">
      <h3>
        <Icon d={P.monitoring} size={24} /> Heart Activity <span className="muted">(ECG)</span>
        <span className="live-pill">● Live</span>
      </h3>
      <EcgStrip />
    </section>
  );
}

function Slider({ label, value, min, max, step = 1, onChange, tone }: { label: ReactNode; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; tone: string }) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <label className={`slider tone-${tone}`}>
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} style={{ ["--fill" as string]: `${fill}%` }} />
      <output>{Math.round(value)}</output>
    </label>
  );
}

function SimulationControls() {
  const S = useWs((w) => w.S);
  const setField = useWs((w) => w.setField);
  const { activity, intensity, spo2, setActivity, setIntensity, setSpo2 } = useCordia();
  return (
    <section className="card">
      <h3>
        <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
          <path
            fill="#3ec5ff"
            d="M19.4 13a7.8 7.8 0 0 0 0-2l2-1.6-2-3.4-2.4 1a7.5 7.5 0 0 0-1.7-1L15 3.5h-4L10.7 6a7.5 7.5 0 0 0-1.7 1l-2.4-1-2 3.4 2 1.6a7.8 7.8 0 0 0 0 2l-2 1.6 2 3.4 2.4-1a7.5 7.5 0 0 0 1.7 1l.3 2.5h4l.3-2.5a7.5 7.5 0 0 0 1.7-1l2.4 1 2-3.4zM13 15.5a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7z"
          />
        </svg>
        Simulation Controls
      </h3>
      <label className="select">
        <span>Activity State :</span>
        <select value={activity} onChange={(e) => setActivity(e.target.value as Activity)}>
          {ACTIVITIES.map((a) => (
            <option key={a.key} value={a.key}>
              {a.label}
            </option>
          ))}
        </select>
      </label>
      <Slider label="Heart Rate (BPM)" value={S.heart_rate} min={F.heart_rate.min!} max={F.heart_rate.max!} onChange={(v) => setField("heart_rate", v)} tone="red" />
      <Slider label="Systolic BP (mmHg)" value={S.sys_bp} min={F.sys_bp.min!} max={F.sys_bp.max!} onChange={(v) => setField("sys_bp", v)} tone="blue" />
      <Slider label="Diastolic BP (mmHg)" value={S.dia_bp} min={F.dia_bp.min!} max={F.dia_bp.max!} onChange={(v) => setField("dia_bp", v)} tone="blue" />
      <Slider
        label={
          <>
            SpO<sub>2</sub> (%)
          </>
        }
        value={spo2}
        min={70}
        max={100}
        onChange={setSpo2}
        tone="cyan"
      />
      <Slider label="Activity Intensity (%)" value={intensity} min={0} max={100} onChange={setIntensity} tone="grey" />
    </section>
  );
}

/* ------------------------------------------------------------------ twin page */

const THUMBS: { view: ShowcaseView; label: string }[] = [
  { view: "front", label: "Front View" },
  { view: "left", label: "Left View" },
  { view: "right", label: "Right View" },
  { view: "back", label: "Back View" },
];

function TwinPage() {
  const [rotate, setRotate] = useState(true);
  const [labels, setLabels] = useState(true);
  const [zoomed, setZoomed] = useState(false);
  const [view, setView] = useState<{ name: ShowcaseView; n: number }>({ name: "front", n: 0 });
  const go = (name: ShowcaseView) => {
    setRotate(false);
    setView((v) => ({ name, n: v.n + 1 }));
  };
  const tool = (label: string, d: string, on: boolean, act: () => void) => (
    <button className={`tool ${on ? "on" : ""}`} aria-pressed={on} onClick={act}>
      <Icon d={d} size={26} />
      {label}
    </button>
  );
  return (
    <>
      <div className="stage">
        <ShowcaseHeart rotate={rotate} labels={labels} view={view} distance={zoomed ? 3.2 : 4.7} />
      </div>
      <div className="tools">
        {tool("Rotate", "M5 3l14 8-6 1.5L10 19z", rotate, () => setRotate(!rotate))}
        {tool("Zoom", "M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13zM15.5 15.5 21 21", zoomed, () => setZoomed(!zoomed))}
        {tool("Reset View", "M4 12a8 8 0 1 0 2.5-5.8M4 4v4h4", false, () => {
          setZoomed(false);
          setRotate(true);
          setView((v) => ({ name: "front", n: v.n + 1 }));
        })}
        {tool("Toggle Labels", "M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z", labels, () => setLabels(!labels))}
      </div>
      <div className="thumbs">
        {THUMBS.map((t) => (
          <button key={t.view} className={`thumb ${!rotate && view.name === t.view ? "on" : ""}`} onClick={() => go(t.view)}>
            <img src={`cordia/view-${t.view}.jpg`} alt="" />
            {t.label}
          </button>
        ))}
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ other pages */

function Stat({ label, value, sub, tone = "" }: { label: string; value: ReactNode; sub?: ReactNode; tone?: string }) {
  return (
    <div className={`stat tone-${tone}`}>
      <div className="muted">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="muted small">{sub}</div>}
    </div>
  );
}

function RiskCards({ s, reference }: { s: State; reference?: State }) {
  const out = evaluate(s);
  const ref = reference ? evaluate(reference) : null;
  return (
    <div className="grid4">
      {MODELS.map((m) => {
        const p = out[m.key];
        const d = ref ? (p - ref[m.key]) * 100 : 0;
        return (
          <div className={`stat tone-${riskTone(p)}`} key={m.key}>
            <div className="muted">{m.title}</div>
            <div className="stat-value">{pct(p)}</div>
            <div className="bar">
              <i style={{ width: pct(p) }} />
            </div>
            <div className="muted small">
              {m.sub}
              {ref && Math.abs(d) >= 0.5 && (
                <b className={d > 0 ? "up" : "down"}>
                  {" "}
                  {d > 0 ? "+" : ""}
                  {d.toFixed(0)} pts
                </b>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function OverviewPage() {
  const { S, v } = useLive();
  const out = evaluate(S);
  const live = useHeartStore((s) => s.liveOutputs);
  const top = Math.max(out.cad, out.cvd, out.chd10);
  return (
    <div className="page">
      <div className="grid4">
        <Stat label="Heart rate" value={`${Math.round(v.hr)} bpm`} tone="red" />
        <Stat label="Blood pressure" value={`${Math.round(v.sys)}/${Math.round(v.dia)}`} sub="mmHg" tone="blue" />
        <Stat label="Ejection fraction (twin)" value={live ? `${Math.round(live.ejectionFraction)}%` : "…"} sub={live ? `Cardiac output ${live.cardiacOutput.toFixed(1)} L/min` : ""} tone="cyan" />
        <Stat label="Highest model risk" value={pct(top)} sub="of the four risk models" tone={riskTone(top)} />
      </div>
      <div className="grid2">
        <div className="panel">
          <h4>What CordiaAI does</h4>
          <p>
            CordiaAI predicts cardiovascular risk from several kinds of patient data and turns it into a living model of the patient's heart, so
            clinicians and patients can see the risk, its causes and what would lower it.
          </p>
          <ul className="ticks">
            <li>Multimodal prediction: clinical records, ECG, echocardiography, chest X-ray, cardiac MRI and CT</li>
            <li>Explainable results: SHAP, LIME and counterfactuals for the records, Grad-CAM for the ECG</li>
            <li>Privacy: federated training across hospitals with differential privacy</li>
            <li>Digital twin: a beating 3D heart driven by the patient's values, with a 10-year forecast</li>
          </ul>
        </div>
        <div className="panel">
          <h4>System pipeline</h4>
          <ol className="pipeline">
            <li>
              <b>Data</b> UCI, Framingham, cardio-70k, BRFSS, PTB-XL ECGs, EchoNet, NIH X-ray, ACDC MRI, ImageCAS CT
            </li>
            <li>
              <b>Models</b> FT-Transformer for records, ViT for ECG and images, Swin-UNETR for MRI and CT
            </li>
            <li>
              <b>Fusion</b> cross-attention fusion and a 5-model stacking ensemble
            </li>
            <li>
              <b>Explain</b> SHAP / LIME / DiCE, Grad-CAM, fairness audit (AIF360)
            </li>
            <li>
              <b>Twin</b> risk and physiology drive this 3D heart, beat by beat
            </li>
            <li>
              <b>Report</b> clinical web app with an offline AI summary for patients
            </li>
          </ol>
        </div>
      </div>
    </div>
  );
}

function MonitoringPage() {
  const live = useHeartStore((s) => s.liveOutputs);
  const { v } = useLive();
  const rows: [string, string][] = live
    ? [
        ["Aortic pressure", `${Math.round(live.pAo)} mmHg`],
        ["Left ventricle pressure", `${Math.round(live.pLV)} mmHg`],
        ["Mean arterial pressure", `${Math.round(live.meanArterialPressure)} mmHg`],
        ["Stroke volume", `${Math.round(live.edv - live.esv)} mL`],
        ["End-diastolic volume", `${Math.round(live.edv)} mL`],
        ["End-systolic volume", `${Math.round(live.esv)} mL`],
      ]
    : [];
  return (
    <div className="page">
      <div className="grid4">
        <Stat label="Heart rate" value={`${Math.round(v.hr)} bpm`} tone="red" />
        <Stat label="Blood pressure" value={`${Math.round(v.sys)}/${Math.round(v.dia)}`} sub="mmHg" tone="blue" />
        <Stat label="SpO₂" value={`${Math.round(v.spo2)}%`} tone="cyan" />
        <Stat label="Respiratory rate" value={`${v.resp}/min`} tone="cyan" />
      </div>
      <div className="panel">
        <h4>ECG (lead II, simulated)</h4>
        <EcgStrip seconds={6} height={220} />
      </div>
      <div className="grid2">
        <div className="panel">
          <h4>Pressures and volumes this beat</h4>
          <table className="kv">
            <tbody>
              {rows.map(([k, val]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{val}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h4>Chambers right now</h4>
          {live &&
            (
              [
                ["Left ventricle", live.vLV, 160, "red"],
                ["Right ventricle", live.vRV, 170, "blue"],
                ["Left atrium", live.vLA, 90, "red"],
                ["Right atrium", live.vRA, 90, "blue"],
              ] as [string, number, number, string][]
            ).map(([k, vol, max, tone]) => (
              <div className={`meter tone-${tone}`} key={k}>
                <span>{k}</span>
                <div className="bar">
                  <i style={{ width: `${clamp((vol / max) * 100, 0, 100)}%` }} />
                </div>
                <output>{Math.round(vol)} mL</output>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

function RiskPage() {
  const S = useWs((w) => w.S);
  const base = useWs((w) => w.base);
  const st = stenosis(S, evaluate(S));
  return (
    <div className="page">
      <RiskCards s={S} reference={base} />
      <div className="grid2">
        <div className="panel">
          <h4>Main drivers of 10-year risk</h4>
          {mainDrivers(S).map((d) => (
            <div className="driver" key={d.name}>
              <span>{d.name}</span>
              <b className={d.raises ? "up" : "down"}>{d.raises ? "▲ raises" : "▼ lowers"}</b>
            </div>
          ))}
          <p className="muted small">Changes against the loaded patient are shown on each card. Adjust the controls on the right to re-score.</p>
        </div>
        <div className="panel">
          <h4>Coronary arteries (illustrative narrowing)</h4>
          {(
            [
              ["Left anterior descending (LAD)", st.LAD],
              ["Left circumflex (LCx)", st.LCx],
              ["Right coronary (RCA)", st.RCA],
            ] as [string, number][]
          ).map(([k, val]) => (
            <div className={`meter tone-${riskTone(val / 2)}`} key={k}>
              <span>{k}</span>
              <div className="bar">
                <i style={{ width: pct(val) }} />
              </div>
              <output>{pct(val)}</output>
            </div>
          ))}
          <p className="muted small">Estimated from the models and risk factors, not from imaging.</p>
        </div>
      </div>
    </div>
  );
}

function DatasetPage() {
  const { ds, row, matches, loadDataset, showRow, closeDataset } = useWs();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const read = async (name: string, text: Promise<string>) => {
    setBusy(true);
    setError("");
    try {
      const parsed = parseCsv(await text, name);
      if (!parsed.rows) throw new Error("no rows found");
      loadDataset(parsed);
    } catch (e) {
      setError(`Couldn't read ${name}: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };
  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) read(f.name, f.text());
    e.target.value = "";
  };
  const filled = matches ? Object.keys(matches.fields) : [];
  return (
    <div className="page">
      <div className="panel">
        <h4>Team dataset</h4>
        <p className="muted">Any heart CSV works: columns are matched to the patient fields by name, and units and codings are converted.</p>
        <div className="row-btns">
          <button className="btn primary" onClick={() => fileRef.current?.click()} disabled={busy}>
            Load CSV…
          </button>
          {SAMPLES.map((s) => (
            <button key={s.id} className="btn" onClick={() => read(s.label, s.load())} disabled={busy} title={s.detail}>
              {s.label} sample
            </button>
          ))}
          {ds && (
            <button className="btn" onClick={closeDataset}>
              Back to the example patient
            </button>
          )}
          <input ref={fileRef} type="file" accept=".csv,.tsv,.txt,text/csv" hidden onChange={onFile} />
        </div>
        {error && <p className="err">{error}</p>}
        {ds && (
          <>
            <p>
              <b>{ds.name}</b> · {ds.rows.toLocaleString()} rows · {filled.length + (matches?.bmi != null ? 1 : 0)} of {ds.columns.length} columns used
            </p>
            <div className="row-btns">
              <button className="btn" onClick={() => showRow(row - 1)} aria-label="Previous row">
                ◀
              </button>
              <span>
                Patient row <b>{row + 1}</b> of {ds.rows.toLocaleString()}
              </span>
              <button className="btn" onClick={() => showRow(row + 1)} aria-label="Next row">
                ▶
              </button>
              <button className="btn" onClick={() => showRow(Math.floor(Math.random() * ds.rows))}>
                Random patient
              </button>
            </div>
            <div className="chips">
              {filled.map((k) => (
                <span key={k} className="chip">
                  {F[k].l} ← {ds.columns[matches!.fields[k]].name}
                </span>
              ))}
              {matches?.bmi != null && <span className="chip">Weight ← {ds.columns[matches.bmi].name} (BMI)</span>}
            </div>
          </>
        )}
      </div>
      <div className="panel">
        <h4>Models in the browser</h4>
        <table className="kv">
          <tbody>
            <tr>
              <td>Coronary artery disease</td>
              <td>UCI Heart Disease, 920 patients from 4 hospitals (angiography)</td>
            </tr>
            <tr>
              <td>10-year CHD risk</td>
              <td>Framingham Heart Study, 4,240 adults</td>
            </tr>
            <tr>
              <td>Cardiovascular disease</td>
              <td>70,000 medical exams (vitals and lifestyle)</td>
            </tr>
            <tr>
              <td>Heart disease or attack</td>
              <td>CDC BRFSS 2015, 253,000 adults (population screen)</td>
            </tr>
          </tbody>
        </table>
        <p className="muted small">
          The full system adds an FT-Transformer ensemble for records, a ViT for 12-lead ECGs (PTB-XL), and imaging branches for echo, X-ray, MRI and CT.
        </p>
      </div>
    </div>
  );
}

function SimulationPage() {
  const live = useHeartStore((s) => s.liveOutputs);
  const params = useHeartStore((s) => s.params);
  const names: Record<string, string> = { mitral: "Mitral", aortic: "Aortic", tricuspid: "Tricuspid", pulmonic: "Pulmonary" };
  return (
    <div className="page">
      <div className="grid2">
        <div className="panel">
          <h4>How the twin beats</h4>
          <p>
            Each ventricle is a time-varying elastance: its stiffness rises and falls with every beat, pushing blood through four valves into the aorta
            and lungs and drawing it back from the veins. Heart rate, blood pressure, pumping strength and filling are set from the patient, and the
            simulation runs in real time, so every control on the right changes the beating heart within a beat.
          </p>
          <table className="kv">
            <tbody>
              <tr>
                <td>Heart rate</td>
                <td>{Math.round(params.heartRate)} bpm</td>
              </tr>
              <tr>
                <td>Pumping strength</td>
                <td>{Math.round(params.contractility)} / 100</td>
              </tr>
              <tr>
                <td>Vessel resistance</td>
                <td>{Math.round(params.vascularResistance)} / 100</td>
              </tr>
              <tr>
                <td>Filling (preload)</td>
                <td>{Math.round(params.preload)} / 100</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h4>Valves</h4>
          {live &&
            Object.entries(live.valves).map(([k, open]) => (
              <div className={`valve ${open > 0.05 ? "open" : ""}`} key={k}>
                <span className="valve-dot" />
                {names[k] ?? k} valve <b>{open > 0.05 ? "open" : "closed"}</b>
              </div>
            ))}
          <h4 style={{ marginTop: 18 }}>Output</h4>
          {live && (
            <div className="grid2 tight">
              <Stat label="Ejection fraction" value={`${Math.round(live.ejectionFraction)}%`} tone="cyan" />
              <Stat label="Cardiac output" value={`${live.cardiacOutput.toFixed(1)} L/min`} tone="cyan" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function HistoryPage() {
  const S = useWs((w) => w.S);
  const base = useWs((w) => w.base);
  const W = 640;
  const H = 260;
  const xs = (y: number) => 44 + ((W - 60) * y) / 10;
  const ys = (p: number) => H - 30 - ((H - 50) * Math.min(p, 0.8)) / 0.8;
  const path = (st: State) =>
    Array.from({ length: 21 }, (_, i) => i / 2)
      .map((y, i) => `${i ? "L" : "M"}${xs(y).toFixed(1)} ${ys(framinghamRisk(project(st, y))).toFixed(1)}`)
      .join(" ");
  const now = framinghamRisk(S);
  const later = framinghamRisk(project(S, 10));
  return (
    <div className="page">
      <div className="grid4">
        <Stat label="10-year CHD risk today" value={pct(now)} tone={riskTone(now)} />
        <Stat label="In 5 years" value={pct(framinghamRisk(project(S, 5)))} tone={riskTone(framinghamRisk(project(S, 5)))} />
        <Stat label="In 10 years" value={pct(later)} tone={riskTone(later)} />
        <Stat label="Age in 10 years" value={`${Math.round(S.age + 10)}`} sub="with yearly trends in BP, cholesterol, glucose, weight" />
      </div>
      <div className="panel">
        <h4>10-year trajectory</h4>
        <svg viewBox={`0 0 ${W} ${H}`} className="chart">
          {[0, 0.2, 0.4, 0.6, 0.8].map((p) => (
            <g key={p}>
              <line x1={44} x2={W - 16} y1={ys(p)} y2={ys(p)} />
              <text x={4} y={ys(p) + 4}>
                {p * 100}%
              </text>
            </g>
          ))}
          {[0, 2, 4, 6, 8, 10].map((y) => (
            <text key={y} x={xs(y) - 8} y={H - 8}>
              +{y}y
            </text>
          ))}
          <path d={path(base)} className="base" />
          <path d={path(S)} className="cur" />
        </svg>
        <p className="muted small">Solid: the patient as set now · dashed: the patient as loaded. Projected with the Framingham model and population drift rates.</p>
      </div>
    </div>
  );
}

const TARGET_TEXT: Record<string, string> = {
  sys_bp: "Bring systolic BP to 120 mmHg",
  dia_bp: "Bring diastolic BP to 80 mmHg",
  heart_rate: "Resting heart rate around 65 bpm",
  total_chol: "Total cholesterol to 180 mg/dL",
  glucose: "Fasting glucose to 95 mg/dL",
  smoker: "Stop smoking",
  cigs_per_day: "No cigarettes",
  active: "Be physically active",
  alcohol_heavy: "Avoid heavy drinking",
  fruits_veg: "Fruit and vegetables daily",
  weight_kg: "Reach a healthy weight (BMI under 25)",
};

/** Healthy targets, changing only what is worse than its target (a low blood pressure stays as it is). */
function targetsFor(S: State): State {
  const t = healthyTargets(S);
  const o = { ...S };
  for (const k of Object.keys(TARGET_TEXT)) {
    if (F[k].v || k === "weight_kg" || S[k] > t[k]) o[k] = t[k];
  }
  if (o.smoker === 0) o.cigs_per_day = 0;
  return o;
}

function PreventivePage() {
  const S = useWs((w) => w.S);
  const reset = useWs((w) => w.reset);
  const all = targetsFor(S);
  const applyHealthy = () => useWs.setState({ S: all });
  const now = evaluate(S);
  const best = evaluate(all);
  const steps = Object.keys(TARGET_TEXT)
    .filter((k) => all[k] !== S[k])
    .map((k) => {
      const one = { ...S, [k]: all[k] };
      if (k === "smoker") one.cigs_per_day = 0;
      const gain = (now.chd10 - evaluate(one).chd10) * 100;
      return { k, gain };
    })
    .filter((s) => s.gain > 0.05 || ["active", "fruits_veg", "alcohol_heavy"].includes(s.k))
    .sort((a, b) => b.gain - a.gain);
  return (
    <div className="page">
      <div className="grid2">
        <div className="panel">
          <h4>Biggest wins for this patient</h4>
          {steps.length ? (
            steps.map((s) => (
              <div className="driver" key={s.k}>
                <span>{TARGET_TEXT[s.k]}</span>
                <b className={s.gain > 0.5 ? "down" : "muted"}>{s.gain > 0.5 ? `−${s.gain.toFixed(0)} pts` : "small"}</b>
              </div>
            ))
          ) : (
            <p>Every modifiable factor is already at its healthy target.</p>
          )}
          <p className="muted small">Points of 10-year CHD risk (Framingham) saved by each change on its own.</p>
          <div className="row-btns">
            <button className="btn primary" onClick={applyHealthy}>
              Apply all healthy targets
            </button>
            <button className="btn" onClick={reset}>
              Reset to patient
            </button>
          </div>
        </div>
        <div className="panel">
          <h4>With every target met</h4>
          <div className="grid2 tight">
            {MODELS.map((m) => (
              <Stat key={m.key} label={m.title} value={`${pct(now[m.key])} → ${pct(best[m.key])}`} tone={riskTone(best[m.key])} />
            ))}
          </div>
          <p className="muted small">Applying the targets updates the twin, the live parameters and every page.</p>
        </div>
      </div>
    </div>
  );
}

function AboutPage() {
  return (
    <div className="page">
      <div className="grid2">
        <div className="panel">
          <h4>The project</h4>
          <p>
            CordiaAI is a group project building an AI system for predictive and preventive cardiac care: it combines clinical records, ECG signals and
            cardiac imaging to estimate a patient's cardiovascular risk, explains which factors drive it, and shows it on a digital twin of the heart.
          </p>
          <p>
            The digital twin turns any team dataset into a beating heart: load a CSV, pick a patient, and the heart, vital signs, ECG and risk follow
            their values in real time. It is built to work with whichever dataset each team member uses.
          </p>
        </div>
        <div className="panel">
          <h4>Built with</h4>
          <table className="kv">
            <tbody>
              <tr>
                <td>Risk models</td>
                <td>FT-Transformer ensemble, logistic models (Framingham, UCI, cardio-70k, BRFSS), XGBoost baseline</td>
              </tr>
              <tr>
                <td>Signals & imaging</td>
                <td>ViT-B/16 (ECG, X-ray, echo), Swin-UNETR (MRI, CT)</td>
              </tr>
              <tr>
                <td>Trust</td>
                <td>SHAP, LIME, DiCE, Grad-CAM, AIF360 fairness audit, FedAvg + DP-SGD</td>
              </tr>
              <tr>
                <td>Twin</td>
                <td>React, Three.js (react-three-fiber), a time-varying elastance heart model</td>
              </tr>
              <tr>
                <td>App</td>
                <td>FastAPI backend, clinical web app, offline AI summary (Ollama)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <p className="muted small">For research and education. The twin's anatomy and simulated values are illustrative, not a diagnosis.</p>
    </div>
  );
}

/* ------------------------------------------------------------------ app */

export function CordiaApp() {
  const [section, setSection] = useState<Section>("twin");
  const [menu, setMenu] = useState(false);
  useSimulation();
  useDriveHeart();
  const [title, sub] = TITLES[section];
  return (
    <div className="app">
      <nav className={`sidebar ${menu ? "open" : ""}`}>
        <div className="sidebar-head">
          <Logo />
          <button className="menu-btn" aria-label="Menu" aria-expanded={menu} onClick={() => setMenu(!menu)}>
            ☰
          </button>
        </div>
        <ul>
          {NAV.map((n) => (
            <li key={n.key}>
              <button
                className={section === n.key ? "active" : ""}
                aria-current={section === n.key ? "page" : undefined}
                onClick={() => {
                  setSection(n.key);
                  setMenu(false);
                }}
              >
                <Icon d={P[n.key]} />
                {n.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <main className="main">
        <header>
          <h1>{title}</h1>
          <p>{sub}</p>
        </header>
        {section === "twin" ? (
          <TwinPage />
        ) : section === "overview" ? (
          <OverviewPage />
        ) : section === "monitoring" ? (
          <MonitoringPage />
        ) : section === "risk" ? (
          <RiskPage />
        ) : section === "dataset" ? (
          <DatasetPage />
        ) : section === "simulation" ? (
          <SimulationPage />
        ) : section === "history" ? (
          <HistoryPage />
        ) : section === "preventive" ? (
          <PreventivePage />
        ) : (
          <AboutPage />
        )}
      </main>
      <aside className="rail">
        <LiveParameters />
        <HeartActivity />
        <SimulationControls />
      </aside>
    </div>
  );
}
