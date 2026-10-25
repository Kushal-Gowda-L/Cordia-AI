import { StructureToggles, useHeartStore, ViewMode } from "../../state/store";

const MODES: { id: ViewMode; label: string; hint: string }[] = [
  { id: "anatomy", label: "Outside", hint: "The heart as seen from outside" },
  { id: "chambers", label: "Inside", hint: "Opened up to show the four chambers" },
  { id: "bloodFlow", label: "Blood flow", hint: "Highlights blood moving through the heart" },
  { id: "pressure", label: "Pressure", hint: "Colours each chamber by how much pressure is in it" },
  { id: "valves", label: "Valves", hint: "Highlights the valves opening and closing" },
  { id: "electrical", label: "Electrical signal", hint: "Shows the signal that starts each heartbeat" },
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
    </div>
  );
}
