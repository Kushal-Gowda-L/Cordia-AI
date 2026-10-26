import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { Heart3D } from "../components/Heart3D/Heart3D";
import { parseCsv } from "../dataset/csv";
import { SAMPLES } from "../dataset/datasetStore";
import type { Prediction } from "../patient/patientTwin";
import { useHeartStore, type CameraView, type StructureToggles, type ViewMode } from "../state/store";
import { evaluate, F, framinghamRisk, IS_EXAMPLE, mainDrivers, MODELS, project, stenosis, type Outputs, type State } from "./models";
import { stateAt, useWs } from "./wsStore";

/**
 * The Cordia Heart Twin workstation, with the 3D heart digital twin as its
 * live heart: patient inputs on the left, the twin in the middle, the four
 * risk models, trajectory, time machine and view controls on the right.
 * Every change re-scores the models and re-tunes the twin's simulated heart.
 */

const clamp = (v: number, a: number, b: number) => Math.min(Math.max(v, a), b);
const color = (p: number) => (p < 0.1 ? "var(--ok)" : p < 0.2 ? "var(--lime)" : p < 0.4 ? "var(--warn)" : "var(--bad)");

/**
 * How much the twin's pumping strength is trimmed so its simulated ejection
 * fraction settles on the patient's (see useEfCalibration): a ventricle
 * pushing against a high pressure needs more strength for the same EF.
 */
let efTrim = 0;
const baseContractility = (s: State) => 50 + (s.ef - 55) * 1.3;

/** The patient's state as the twin's heart settings and risk. */
function driveHeart(s: State, out: Outputs) {
  const risk = Math.max(out.cad, out.cvd, out.chd10);
  const pct = Math.round(risk * 100);
  const band = pct < 35 ? "unlikely" : pct < 65 ? "borderline" : "likely";
  const prediction = {
    risk_percent: pct,
    severity: band,
    severity_percent: pct,
    class_probabilities: {},
    reading: {
      disease_percent: pct,
      no_disease_percent: 100 - pct,
      decision_point: 50,
      zones: { unlikely_below: 35, likely_from: 65 },
      band,
      verdict: band === "unlikely" ? "Heart disease unlikely" : band === "borderline" ? "Borderline" : "Heart disease likely",
      predicts_disease: pct >= 50,
      severity_if_disease: {},
      likely_severity_if_disease: null,
      likely_severity_percent: null,
    },
  } as Prediction;
  const params = {
    heartRate: clamp(s.heart_rate, 40, 180),
    systolicBP: clamp(s.sys_bp, 80, 220),
    diastolicBP: clamp(s.dia_bp, 40, 140),
    baseStrokeVolume: s.sex ? 72 : 62,
    // A failing ventricle squeezes less: pumping strength follows the ejection fraction.
    contractility: clamp(baseContractility(s) + efTrim, 5, 100),
    preload: 50,
    afterload: clamp(50 + 0.5 * (s.sys_bp - 120) + (s.restecg == 2 ? 10 : 0), 15, 95),
    vascularResistance: clamp(50 + 0.6 * (s.sys_bp - 120) + 0.3 * Math.max(0, s.age - 50) + (s.diabetes ? 6 : 0), 15, 95),
    oxygenSaturation: 98,
    aorticStenosis: 0,
    mitralRegurgitation: 0,
  };
  useHeartStore.setState({ params, prediction, twinStatus: "ready" });
}

/** Nudges the twin's pumping strength every beat or so until its simulated EF matches the patient's. */
function useEfCalibration(s: State) {
  const ref = useRef(s);
  ref.current = s;
  useEffect(() => {
    const id = setInterval(() => {
      const { liveOutputs, params, paused, speed } = useHeartStore.getState();
      if (!liveOutputs || paused || speed === 0) return;
      const err = ref.current.ef - liveOutputs.ejectionFraction;
      if (Math.abs(err) < 0.5) return;
      efTrim = clamp(efTrim + 0.7 * err, -45, 60);
      const contractility = clamp(baseContractility(ref.current) + efTrim, 5, 100);
      if (contractility !== params.contractility) useHeartStore.setState({ params: { ...params, contractility } });
    }, 600);
    return () => clearInterval(id);
  }, []);
}

/**
 * Abnormal (ectopic) beats: at each beat, with the patient's ectopy %, the
 * next beat comes early (0.62 of a normal beat) and the rhythm then resumes.
 */
function useEctopy(ectopy: number, heartRate: number) {
  const ref = useRef({ ectopy, heartRate });
  ref.current = { ectopy, heartRate };
  useEffect(() => {
    let timer = 0;
    let early = false;
    const setHr = (hr: number) => {
      const { params } = useHeartStore.getState();
      if (params.heartRate !== hr) useHeartStore.setState({ params: { ...params, heartRate: hr } });
    };
    const tick = () => {
      const hr = clamp(ref.current.heartRate, 40, 180);
      if (early) setHr(hr);
      early = Math.random() < ref.current.ectopy / 100;
      if (early) setHr(Math.min(180, hr / 0.62));
      timer = window.setTimeout(tick, (60 / hr) * (early ? 620 : 1000));
    };
    timer = window.setTimeout(tick, 1000);
    return () => clearTimeout(timer);
  }, []);
}

const fmt = (k: string, v: number) => (F[k].s! < 1 ? Number(v).toFixed(1) : Math.round(v)) + (F[k].u ? " " + F[k].u : "");

function Controls() {
  const S = useWs((w) => w.S);
  const setField = useWs((w) => w.setField);
  const matched = useWs((w) => w.matches?.fields ?? null);
  const groups = useMemo(() => {
    const g: Record<string, string[]> = {};
    for (const k in F) (g[F[k].g] ??= []).push(k);
    return g;
  }, []);
  return (
    <div id="controls">
      {Object.entries(groups).map(([g, keys]) => (
        <div key={g}>
          <h2>{g}</h2>
          {keys.map((k) => {
            const f = F[k];
            const tag =
              matched && (matched[k] !== undefined || (k === "weight_kg" && useWs.getState().matches?.bmi != null)) ? (
                <span className="tag" title="Filled from the loaded dataset">
                  data
                </span>
              ) : null;
            return f.v ? (
              <div className="sel" key={k}>
                <label htmlFor={"i_" + k}>
                  {f.l} {tag}
                </label>
                <select id={"i_" + k} value={String(S[k])} onChange={(e) => setField(k, Number(e.target.value))}>
                  {Object.entries(f.v).map(([v, t]) => (
                    <option key={v} value={v}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div key={k}>
                <div className="row">
                  <label htmlFor={"i_" + k}>
                    {f.l} {tag}
                  </label>
                  <output>{fmt(k, S[k])}</output>
                </div>
                <input
                  type="range"
                  id={"i_" + k}
                  min={f.min}
                  max={f.max}
                  step={f.s}
                  value={S[k]}
                  onChange={(e) => setField(k, Number(e.target.value))}
                />
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/** Load any team dataset: one row at a time becomes the patient. */
function DatasetLoader() {
  const { ds, row, matches, loadDataset, showRow, closeDataset } = useWs();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const read = async (name: string, text: Promise<string>) => {
    setBusy(true);
    setError("");
    try {
      const parsed = parseCsv(await text, name);
      if (!parsed.rows) throw new Error("No rows found");
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
  const nMatched = matches ? Object.keys(matches.fields).length + (matches.bmi !== null ? 1 : 0) : 0;
  return (
    <div className="dataset">
      <h2>Team dataset</h2>
      <div className="btns">
        <button onClick={() => fileRef.current?.click()} disabled={busy}>
          Load CSV…
        </button>
        {SAMPLES.map((s) => (
          <button key={s.id} onClick={() => read(s.label, s.load())} disabled={busy} title={s.detail}>
            {s.label}
          </button>
        ))}
        <input ref={fileRef} type="file" accept=".csv,.tsv,.txt,text/csv" hidden onChange={onFile} />
      </div>
      {busy && <div className="note">Reading…</div>}
      {error && <div className="note bad">{error}</div>}
      {ds ? (
        <>
          <div className="note">
            <b>{ds.name}</b> · {ds.rows.toLocaleString()} rows · {nMatched} of {ds.columns.length} columns fill patient fields (tagged{" "}
            <span className="tag">data</span>)
          </div>
          <div className="row-nav">
            <button onClick={() => showRow(row - 1)} aria-label="Previous row">
              ◀
            </button>
            <label>
              Row{" "}
              <input
                type="number"
                min={1}
                max={ds.rows}
                value={row + 1}
                onChange={(e) => Number.isFinite(e.target.valueAsNumber) && showRow(e.target.valueAsNumber - 1)}
              />{" "}
              of {ds.rows.toLocaleString()}
            </label>
            <button onClick={() => showRow(row + 1)} aria-label="Next row">
              ▶
            </button>
            <button onClick={() => showRow(Math.floor(Math.random() * ds.rows))}>Random</button>
            <button onClick={closeDataset} title="Back to the example patient">
              ✕
            </button>
          </div>
        </>
      ) : (
        <div className="note">Any heart CSV works: columns are matched by name and units are converted. Each row becomes the patient.</div>
      )}
    </div>
  );
}

function Cards({ out, reference, s }: { out: Outputs; reference: Outputs; s: State }) {
  const drivers = mainDrivers(s);
  return (
    <div id="cards">
      {MODELS.map((d) => {
        const p = out[d.key];
        const dl = (p - reference[d.key]) * 100;
        return (
          <div className="card" key={d.key}>
            <div className="t">
              {d.title} <span style={{ opacity: 0.7 }}>· {d.sub}</span>
            </div>
            <div>
              <span className="v" style={{ color: color(p) }}>
                {(p * 100).toFixed(0)}%
              </span>
              <span
                className="d"
                style={{
                  color: dl < -0.5 ? "var(--ok)" : dl > 0.5 ? "var(--bad)" : "var(--muted)",
                }}
              >
                {Math.abs(dl) < 0.5 ? "±0" : (dl > 0 ? "+" : "") + dl.toFixed(0)} pts
              </span>
            </div>
            <div className="bar">
              <i
                style={{
                  width: `${(p * 100).toFixed(1)}%`,
                  background: color(p),
                }}
              />
            </div>
          </div>
        );
      })}
      <div className="card">
        <div className="t">Main drivers of 10-year risk</div>
        {drivers.map((d) => (
          <div
            key={d.name}
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 12,
            }}
          >
            <span>{d.name}</span>
            <span style={{ color: d.raises ? "var(--bad)" : "var(--ok)" }}>{d.raises ? "▲ raises" : "▼ lowers"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Trajectory({ S, base, years }: { S: State; base: State; years: number }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = cv.current;
    const g = c?.getContext("2d");
    if (!c || !g) return;
    const W = c.width,
      H = c.height;
    g.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    const pad = 34,
      xs = (y: number) => pad + ((W - pad - 10) * y) / 10,
      ys = (p: number) => H - 24 - ((H - 40) * Math.min(p, 0.8)) / 0.8;
    g.strokeStyle = css.getPropertyValue("--line").trim() || "#2a3443";
    g.fillStyle = css.getPropertyValue("--muted").trim() || "#8b98a9";
    g.font = "18px system-ui";
    g.lineWidth = 1;
    for (const p of [0, 0.2, 0.4, 0.6, 0.8]) {
      g.beginPath();
      g.moveTo(pad, ys(p));
      g.lineTo(W - 10, ys(p));
      g.stroke();
      g.fillText(Math.round(p * 100) + "%", 0, ys(p) + 5);
    }
    for (const y of [0, 5, 10]) g.fillText(y + "y", xs(y) - 8, H - 4);
    const line = (st: State, dash: number[], col: string) => {
      g.setLineDash(dash);
      g.strokeStyle = col;
      g.lineWidth = 3;
      g.beginPath();
      for (let y = 0; y <= 10; y += 0.5) {
        const p = framinghamRisk(project(st, y));
        if (y) g.lineTo(xs(y), ys(p));
        else g.moveTo(xs(y), ys(p));
      }
      g.stroke();
      g.setLineDash([]);
    };
    line(base, [8, 8], "#8b98a9");
    line(S, [], "#ff5d6c");
    g.fillStyle = "#4fb3ff";
    g.fillRect(xs(years) - 1, 14, 2, H - 38);
  }, [S, base, years]);
  return <canvas id="traj" ref={cv} width={560} height={300} />;
}

function TimeMachine() {
  const { years, playing, setYears, setPlaying } = useWs();
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const y = Math.min(10, useWs.getState().years + 0.25);
      setYears(y);
      if (y >= 10) setPlaying(false);
    }, 250);
    return () => clearInterval(id);
  }, [playing, setYears, setPlaying]);
  return (
    <>
      <h2>Time machine</h2>
      <div className="row">
        <label htmlFor="yrs">Years from now</label>
        <output>{years}</output>
      </div>
      <input type="range" id="yrs" min={0} max={10} step={0.5} value={years} onChange={(e) => setYears(Number(e.target.value))} />
      <div className="btns">
        <button
          onClick={() => {
            if (playing) return setPlaying(false);
            setYears(0);
            setPlaying(true);
          }}
        >
          {playing ? "■ Stop" : "▶ Play 10 years"}
        </button>
      </div>
    </>
  );
}

const TWIN_VIEWS: { mode: ViewMode; label: string; title: string }[] = [
  {
    mode: "hologram",
    label: "Hologram",
    title: "Holographic twin with fibres, coronary arteries and flow",
  },
  { mode: "lowpoly", label: "Low-poly", title: "Faceted digital mesh" },
  { mode: "realistic", label: "Realistic", title: "Anatomical red heart" },
  {
    mode: "dual",
    label: "Red + blue",
    title: "The real heart and its digital twin side by side",
  },
  {
    mode: "chambers",
    label: "Inside",
    title: "Cut-away chambers with pressures and volumes",
  },
  {
    mode: "valves",
    label: "Valves",
    title: "The four valves opening and closing",
  },
  {
    mode: "electrical",
    label: "Electrical",
    title: "The conduction system firing each beat",
  },
  {
    mode: "risk",
    label: "Risk map",
    title: "Where coronary disease most often shows, shaded by risk",
  },
];

const TOGGLES: [keyof StructureToggles, string][] = [
  ["labels", "Labels"],
  ["vessels", "Blood vessels"],
  ["valves", "Valves"],
  ["chambers", "Chambers"],
  ["bloodFlow", "Blood flow"],
  ["conduction", "Electrical pathway"],
];

const CAMERA: [CameraView, string][] = [
  ["front", "Anterior"],
  ["back", "Posterior"],
  ["left", "Left lateral"],
  ["four", "4-chamber cut"],
];

function ViewControls() {
  const xray = useHeartStore((s) => s.xray);
  const speed = useHeartStore((s) => s.speed);
  const toggles = useHeartStore((s) => s.toggles);
  const { setXRay, setSpeed, toggleStructure, setCameraView } = useHeartStore.getState();
  const cut = Math.round(xray.cutDepth * 100);
  const op = Math.round(xray.wallOpacity * 100);
  return (
    <>
      <h2>View</h2>
      <div className="row">
        <label htmlFor="cut">Cut-away depth</label>
        <output>{cut ? cut + "%" : "off"}</output>
      </div>
      <input type="range" id="cut" min={0} max={100} value={cut} onChange={(e) => setXRay({ cutDepth: Number(e.target.value) / 100 })} />
      <div className="row">
        <label htmlFor="op">Outer wall opacity</label>
        <output>{op}%</output>
      </div>
      <input type="range" id="op" min={15} max={100} value={op} onChange={(e) => setXRay({ wallOpacity: Number(e.target.value) / 100 })} />
      <div className="row">
        <label htmlFor="sp">Animation speed</label>
        <output>{speed.toFixed(1)}×</output>
      </div>
      <input
        type="range"
        id="sp"
        min={0}
        max={200}
        value={Math.round(speed * 100)}
        onChange={(e) => setSpeed(Number(e.target.value) / 100)}
      />
      <div id="toggles">
        {TOGGLES.map(([k, l]) => (
          <label className="chk" key={k}>
            <input type="checkbox" checked={toggles[k]} onChange={() => toggleStructure(k)} /> {l}
          </label>
        ))}
      </div>
      <div className="btns">
        {CAMERA.map(([v, l]) => (
          <button
            key={v}
            onClick={() => {
              setCameraView(v);
              if (v === "four") setXRay({ cutDepth: 0.55, wallOpacity: 1 });
            }}
          >
            {l}
          </button>
        ))}
      </div>
    </>
  );
}

function Hud({ s, out, years }: { s: State; out: Outputs; years: number }) {
  const live = useHeartStore((st) => st.liveOutputs);
  const st = stenosis(s, out);
  return (
    <div id="hud">
      <b>{Math.round(s.heart_rate)} bpm</b> · BP {Math.round(s.sys_bp)}/{Math.round(s.dia_bp)} · EF {Math.round(s.ef)}%
      <br />
      LAD {Math.round(st.LAD * 100)}% · LCx {Math.round(st.LCx * 100)}% · RCA {Math.round(st.RCA * 100)}%{" "}
      <span style={{ opacity: 0.7 }}>(visual stenosis)</span>
      {live && (
        <>
          <br />
          Twin: EF {Math.round(live.ejectionFraction)}% · CO {live.cardiacOutput.toFixed(1)} L/min · LV {Math.round(live.vLV)} mL
        </>
      )}
      {years > 0 && (
        <>
          <br />
          Simulated: +{years} years (age {Math.round(s.age)})
        </>
      )}
    </div>
  );
}

function ViewChips() {
  const viewMode = useHeartStore((s) => s.viewMode);
  const setViewMode = useHeartStore((s) => s.setViewMode);
  return (
    <div className="chips" role="group" aria-label="3D twin view">
      {TWIN_VIEWS.map((v) => (
        <button
          key={v.mode}
          className={viewMode === v.mode ? "on" : ""}
          aria-pressed={viewMode === v.mode}
          title={v.title}
          onClick={() => setViewMode(v.mode)}
        >
          {v.label}
        </button>
      ))}
    </div>
  );
}

export function Workstation() {
  const { S, base, years, ds, applyHealthy, reset } = useWs();
  const s = useMemo(() => stateAt(S, years), [S, years]);
  const out = useMemo(() => evaluate(s), [s]);
  const reference = useMemo(() => evaluate(base), [base]);
  useEffect(() => driveHeart(s, out), [s, out]);
  useEctopy(s.ectopy, s.heart_rate);
  useEfCalibration(s);

  return (
    <div id="app">
      <aside id="left">
        <h1>🫀 Cordia Heart Twin</h1>
        <div className="sub">
          Adjust the patient — the risk models re-score and the heart responds live.
          {IS_EXAMPLE && !ds ? " Showing an example patient." : ""}
        </div>
        <div className="btns">
          <button className="primary" title="Set every modifiable factor to its healthy target" onClick={applyHealthy}>
            Apply healthy targets
          </button>
          <button onClick={reset}>Reset to patient</button>
        </div>
        <DatasetLoader />
        <Controls />
      </aside>

      <main id="stage">
        <Heart3D />
        <Hud s={s} out={out} years={years} />
        <ViewChips />
      </main>

      <aside className="right">
        <h2 style={{ marginTop: 2 }}>
          Model outputs <span className="pill">browser models</span>
        </h2>
        <Cards out={out} reference={reference} s={s} />
        <h2>Trajectory (10-year CHD risk)</h2>
        <Trajectory S={S} base={base} years={years} />
        <div className="note">Solid: current values with yearly trends · dashed: the patient you loaded.</div>
        <TimeMachine />
        <ViewControls />
        <div className="note" id="foot">
          Models: {MODELS.map((m) => m.src).join(", ")} (logistic components in the browser). The 3D heart is the Cordia digital twin: a
          beat-by-beat simulation set from the patient's heart rate, blood pressure and ejection fraction (its pumping strength is tuned
          until its simulated EF matches), with its coronary colour following the models' risk. Anatomy and lesion visuals are illustrative
          and driven by the model outputs — not patient imaging. Drag to rotate · wheel to zoom · right-drag to pan.
        </div>
      </aside>
    </div>
  );
}
