import { StructureToggles, useHeartStore, ViewMode } from "../../state/store";

const MODES: { id: ViewMode; label: string; hint: string }[] = [
  { id: "realistic", label: "Realistic heart", hint: "A lifelike outer heart, turning so you can see it from every side" },
  { id: "anatomy", label: "Outside", hint: "The heart as seen from outside" },
  { id: "chambers", label: "Inside", hint: "Opened up to show the four chambers" },
  { id: "bloodFlow", label: "Blood flow", hint: "Highlights blood moving through the heart" },
  { id: "pressure", label: "Pressure", hint: "Colours each chamber by how much pressure is in it" },
  { id: "valves", label: "Valves", hint: "Highlights the valves opening and closing" },
  { id: "electrical", label: "Electrical signal", hint: "Shows the signal that starts each heartbeat" },
  { id: "risk", label: "Risk", hint: "Colours the heart by the predicted chance of heart disease" },
];

const TOGGLES: { id: keyof StructureToggles; label: string }[] = [
  { id: "chambers", label: "Chambers" },
  { id: "vessels", label: "Blood vessels" },
  { id: "valves", label: "Valves" },
  { id: "bloodFlow", label: "Blood flow" },
  { id: "conduction", label: "Electrical pathway" },
  { id: "labels", label: "Labels" },
];

export function ViewModeToggle() {
  const viewMode = useHeartStore((s) => s.viewMode);
  const setViewMode = useHeartStore((s) => s.setViewMode);
  const toggles = useHeartStore((s) => s.toggles);
  const toggleStructure = useHeartStore((s) => s.toggleStructure);
  const xray = useHeartStore((s) => s.xray);
  const setXRay = useHeartStore((s) => s.setXRay);
  const closedHeart = viewMode === "anatomy" || viewMode === "risk";

  return (
    <div className="panel viewer-toolbar">
      <div className="viewer-modes" role="group" aria-label="Heart view">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            title={m.hint}
            aria-pressed={viewMode === m.id}
            className={`seg-button ${viewMode === m.id ? "active" : ""}`}
            onClick={() => setViewMode(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>
      <div className="viewer-toggles">
        <span className="viewer-toggles-title">Show:</span>
        {TOGGLES.map((t) => (
          <label key={t.id} className="toggle-chip">
            <input type="checkbox" checked={toggles[t.id]} onChange={() => toggleStructure(t.id)} />
            {t.label}
          </label>
        ))}
      </div>
      <div className="viewer-xray">
        <span className="viewer-toggles-title">Look inside:</span>
        <label className="xray-slider" title={closedHeart ? "" : "Applies to the Outside and Risk views"}>
          See-through wall
          <input
            type="range"
            min={0}
            max={90}
            step={1}
            disabled={!closedHeart}
            value={Math.round((1 - xray.wallOpacity) * 100)}
            onChange={(e) => setXRay({ wallOpacity: 1 - Number(e.target.value) / 100 })}
          />
          <output>{Math.round((1 - xray.wallOpacity) * 100)}%</output>
        </label>
        <label className="xray-slider" title={viewMode === "realistic" ? "Applies to every view except Realistic heart" : "Slices the heart from the front, like a dissection"}>
          Cut away front
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            disabled={viewMode === "realistic"}
            value={Math.round(xray.cutDepth * 100)}
            onChange={(e) => setXRay({ cutDepth: Number(e.target.value) / 100 })}
          />
          <output>{Math.round(xray.cutDepth * 100)}%</output>
        </label>
        {(xray.wallOpacity < 1 || xray.cutDepth > 0) && (
          <button type="button" className="btn btn-small" onClick={() => setXRay({ wallOpacity: 1, cutDepth: 0 })}>
            Close up
          </button>
        )}
      </div>
    </div>
  );
}
