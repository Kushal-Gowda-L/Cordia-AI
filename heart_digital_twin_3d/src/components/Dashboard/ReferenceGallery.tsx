import { useHeartStore, type ViewMode } from "../../state/store";

const REFS: { file: string; title: string; text: string; mode: ViewMode }[] = [
  {
    file: "lowpoly-heart.jpg",
    title: "Low-poly digital heart",
    text: "Faceted blue mesh with its wireframe and glowing nodes.",
    mode: "lowpoly",
  },
  {
    file: "fibre-hologram-heart.jpg",
    title: "Fibre hologram",
    text: "See-through glowing heart traced with fine muscle fibres and vessels.",
    mode: "hologram",
  },
  {
    file: "red-and-blue-hearts.jpg",
    title: "Real heart and its digital twin",
    text: "The red anatomical heart beside its blue holographic twin, arteries in red.",
    mode: "dual",
  },
];

/** The reference images the holographic twin's looks are based on, each linked to the view that recreates it. */
export function ReferenceGallery() {
  const viewMode = useHeartStore((s) => s.viewMode);
  const setViewMode = useHeartStore((s) => s.setViewMode);
  return (
    <section className="panel reference-gallery" aria-label="Design references">
      <div className="panel-title">Design references</div>
      <div className="reference-grid">
        {REFS.map((r) => (
          <figure key={r.file} className={`reference-card ${viewMode === r.mode ? "active" : ""}`}>
            <img src={`${import.meta.env.BASE_URL}references/${r.file}`} alt={r.title} loading="lazy" />
            <figcaption>
              <b>{r.title}</b>
              <span>{r.text}</span>
              <button type="button" className="btn btn-small" onClick={() => setViewMode(r.mode)}>
                {viewMode === r.mode ? "Showing in the twin" : "Show in the twin"}
              </button>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
