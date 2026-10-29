import { useDatasetStore } from "../../dataset/datasetStore";
import { useHeartStore } from "../../state/store";

/** Switches the twin between the CordiaAI risk model's ten inputs and a loaded dataset. */
export function SourceToggle() {
  const source = useHeartStore((s) => s.source);
  const useExample = useHeartStore((s) => s.useExample);
  const hasPatient = useHeartStore((s) => s.exampleSnapshot?.hasPatient ?? false);
  const ds = useDatasetStore((s) => s.ds);
  const selectRow = useDatasetStore((s) => s.selectRow);
  const row = useDatasetStore((s) => s.row);

  return (
    <div className="panel source-toggle">
      <div className="panel-label">Drive the heart from</div>
      <div className="scenario-toggle" role="group" aria-label="Drive the heart from">
        <button type="button" className={`seg-button ${source === "example" ? "active" : ""}`} aria-pressed={source === "example"} onClick={() => source !== "example" && useExample()}>
          {hasPatient ? "Patient record" : "CordiaAI model"}
        </button>
        <button
          type="button"
          className={`seg-button ${source === "dataset" ? "active" : ""}`}
          aria-pressed={source === "dataset"}
          onClick={() => {
            if (source === "dataset") return;
            if (ds) selectRow(row);
            else useHeartStore.setState({ source: "dataset" });
          }}
        >
          Your dataset (CSV)
        </button>
      </div>
    </div>
  );
}
