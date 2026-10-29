import { useState, type ReactNode } from "react";
import { INSIDE_LABELS, REFERENCE_LABELS, ShowcaseHeart, type ShowcaseView } from "../cordia/ShowcaseHeart";
import { useSimulation } from "../cordia/simLoop";
import { useHeartStore } from "../state/store";

/**
 * The heart on its own (heart.html): the reference-look 3D heart, matched to
 * the Cordia dashboard render, turning 360 degrees on its platform with the
 * same labels, tools and views. Look Inside cuts the front away to show the
 * four valves opening and closing and the chambers filling, beat by beat.
 */

const THUMBS: { view: ShowcaseView; label: string }[] = [
  { view: "front", label: "Front View" },
  { view: "left", label: "Left View" },
  { view: "right", label: "Right View" },
  { view: "back", label: "Back View" },
];

function Icon({ d }: { d: string }) {
  return (
    <svg width={26} height={26} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

export function HeartPage() {
  useSimulation();
  const [rotate, setRotate] = useState(true);
  const [labels, setLabels] = useState(true);
  const [zoomed, setZoomed] = useState(false);
  const [inside, setInside] = useState(false);
  const [view, setView] = useState<{ name: ShowcaseView; n: number }>({ name: "front", n: 0 });
  const hr = useHeartStore((s) => s.params.heartRate);
  const setXRay = useHeartStore((s) => s.setXRay);

  const go = (name: ShowcaseView) => {
    setRotate(false);
    setView((v) => ({ name, n: v.n + 1 }));
  };
  const lookInside = (on: boolean) => {
    setInside(on);
    setXRay({ cutDepth: on ? 0.55 : 0, wallOpacity: 1 });
  };
  const tool = (label: string, d: string, on: boolean, act: () => void): ReactNode => (
    <button className={`tool ${on ? "on" : ""}`} aria-pressed={on} onClick={act}>
      <Icon d={d} />
      {label}
    </button>
  );

  return (
    <div className="heart-page">
      <header>
        <div>
          <h1>Digital Twin Heart</h1>
          <p>Interactive 3D visualization of the heart with real-time parameters</p>
        </div>
        <label className="hr-chip">
          <span className="beat" style={{ animationDuration: `${60 / hr}s` }}>
            ♥
          </span>
          <b>{Math.round(hr)}</b> BPM
          <input
            type="range"
            min={40}
            max={160}
            value={Math.round(hr)}
            aria-label="Heart rate"
            onChange={(e) => useHeartStore.setState((s) => ({ params: { ...s.params, heartRate: Number(e.target.value) } }))}
          />
        </label>
      </header>
      <div className="stage">
        <ShowcaseHeart
          rotate={rotate}
          labels={labels}
          view={view}
          distance={zoomed ? 3.3 : 5.1}
          look="reference"
          targetY={0.02}
          labelSet={inside ? INSIDE_LABELS : REFERENCE_LABELS}
        />
      </div>
      <div className="tools">
        {tool("Rotate", "M5 3l14 8-6 1.5L10 19z", rotate, () => setRotate(!rotate))}
        {tool("Zoom", "M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13zM15.5 15.5 21 21", zoomed, () => setZoomed(!zoomed))}
        {tool("Reset View", "M4 12a8 8 0 1 0 2.5-5.8M4 4v4h4", false, () => {
          setZoomed(false);
          setRotate(true);
          lookInside(false);
          setView((v) => ({ name: "front", n: v.n + 1 }));
        })}
        {tool("Toggle Labels", "M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z", labels, () => setLabels(!labels))}
        {tool("Look Inside", "M4 4h16v16H4zM12 4v16M8 9l-2 3 2 3M16 9l2 3-2 3", inside, () => lookInside(!inside))}
      </div>
      <div className="thumbs">
        {THUMBS.map((t) => (
          <button key={t.view} className={`thumb ${!rotate && view.name === t.view ? "on" : ""}`} onClick={() => go(t.view)}>
            <img src={`heart/view-${t.view}.jpg`} alt="" />
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
}
