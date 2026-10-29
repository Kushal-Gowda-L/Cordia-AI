import { useState } from "react";
import type { Dataset } from "../../dataset/csv";
import { cellText } from "../../dataset/csv";
import { outcomeName, SAMPLES, useDatasetStore, valueText } from "../../dataset/datasetStore";
import { ROLES, type RoleKey } from "../../dataset/mapping";
import { topFactors } from "../../dataset/riskModel";
import { useHeartStore } from "../../state/store";
import { HeartSetup, riskTone } from "./PatientSummary";

/** Load any CSV, choose its outcome, and pick the row the heart shows. */
export function DatasetPanel() {
  const status = useDatasetStore((s) => s.status);
  const error = useDatasetStore((s) => s.error);
  const ds = useDatasetStore((s) => s.ds);
  return (
    <>
      <LoadCard />
      {status === "loading" && <div className="panel">Reading the file and training a model on it&hellip;</div>}
      {status === "error" && (
        <div className="panel panel-error">
          <div className="panel-title">
            <span>That file couldn&rsquo;t be read</span>
          </div>
          <p>{error}</p>
        </div>
      )}
      {status === "ready" && ds && (
        <>
          <RowCard ds={ds} />
          <EditRow ds={ds} />
          <ColumnMatches ds={ds} />
        </>
      )}
    </>
  );
}

function LoadCard() {
  const ds = useDatasetStore((s) => s.ds);
  const loadFile = useDatasetStore((s) => s.loadFile);
  const loadSample = useDatasetStore((s) => s.loadSample);
  const [dragging, setDragging] = useState(false);

  return (
    <div className="panel">
      <div className="panel-title">
        <span>Your dataset</span>
        <span className="panel-subtitle">
          Any heart dataset as a CSV file. Columns are matched to the heart automatically and a risk model is trained on
          it here in your browser &mdash; nothing is uploaded anywhere.
        </span>
      </div>
      <label
        className={`drop-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files[0];
          if (f) loadFile(f);
        }}
      >
        <input
          type="file"
          accept=".csv,.tsv,.txt,text/csv"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) loadFile(f);
            e.target.value = "";
          }}
        />
        <b>Choose a CSV file</b> or drop it here
      </label>
      <div className="sample-row">
        <span>Or try:</span>
        {SAMPLES.map((s) => (
          <button key={s.id} type="button" className="btn btn-small" title={s.detail} onClick={() => loadSample(s.id)}>
            {s.label}
          </button>
        ))}
      </div>
      {ds && (
        <p className="dataset-meta">
          <b>{ds.name}</b>: {ds.rows.toLocaleString()} rows
          {ds.sampledFrom ? ` (an even sample of ${ds.sampledFrom.toLocaleString()})` : ""}, {ds.columns.length} columns
        </p>
      )}
    </div>
  );
}

function RowCard({ ds }: { ds: Dataset }) {
  const target = useDatasetStore((s) => s.target);
  const model = useDatasetStore((s) => s.model);
  const modelError = useDatasetStore((s) => s.modelError);
  const row = useDatasetStore((s) => s.row);
  const values = useDatasetStore((s) => s.values);
  const setTarget = useDatasetStore((s) => s.setTarget);
  const selectRow = useDatasetStore((s) => s.selectRow);
  const randomRow = useDatasetStore((s) => s.randomRow);
  const prediction = useHeartStore((s) => s.prediction);
  const baseline = useHeartStore((s) => s.recordTwin?.prediction?.risk_percent);

  const recorded = target === null ? null : cellText(ds.columns[target], row);
  const factors = model ? topFactors(ds, model, values) : [];

  return (
    <div className="panel patient-summary">
      <div className="panel-label">Outcome to predict</div>
      <select className="full-select" value={target ?? ""} onChange={(e) => setTarget(e.target.value === "" ? null : Number(e.target.value))}>
        <option value="">None (heart from the values only)</option>
        {ds.columns.map((c, j) =>
          c.kind === "text" ? null : (
            <option key={j} value={j}>
              {c.name}
            </option>
          )
        )}
      </select>
      {modelError && <p className="inline-error">{modelError}</p>}
      {model && (
        <p className="model-stats">
          Trained on {model.nTrain.toLocaleString()} rows, checked on {model.nTest.toLocaleString()} held-out rows:{" "}
          <b>AUC {model.auc.toFixed(2)}</b>, accuracy {(model.accuracy * 100).toFixed(0)}%. {(model.positiveRate * 100).toFixed(0)}% of
          rows have {outcomeName(model)}.
        </p>
      )}

      <div className="panel-label">Row shown as the heart</div>
      <div className="row-picker">
        <button type="button" className="btn btn-small" disabled={row === 0} onClick={() => selectRow(row - 1)} aria-label="Previous row">
          &lsaquo;
        </button>
        <input
          type="number"
          min={1}
          max={ds.rows}
          value={row + 1}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v) && v >= 1) selectRow(v - 1);
          }}
          aria-label="Row number"
        />
        <span className="row-of">of {ds.rows.toLocaleString()}</span>
        <button type="button" className="btn btn-small" disabled={row >= ds.rows - 1} onClick={() => selectRow(row + 1)} aria-label="Next row">
          &rsaquo;
        </button>
        <button type="button" className="btn btn-small" onClick={() => randomRow()}>
          Random
        </button>
      </div>
      {model && (
        <div className="sample-row">
          <span>Random row with outcome:</span>
          <button type="button" className="btn btn-small" onClick={() => randomRow(1)}>
            Yes
          </button>
          <button type="button" className="btn btn-small" onClick={() => randomRow(0)}>
            No
          </button>
        </div>
      )}

      {prediction && model && (
        <div className="risk-block">
          <div className={`risk-figure tone-${riskTone(prediction.risk_percent)}`}>{prediction.risk_percent}%</div>
          <div className="risk-caption">Estimated chance of {outcomeName(model)}</div>
          {baseline !== undefined && Math.abs(prediction.risk_percent - baseline) >= 0.05 && (
            <div className={`risk-delta ${prediction.risk_percent > baseline ? "up" : "down"}`}>
              {prediction.risk_percent > baseline ? "▲" : "▼"} {Math.abs(prediction.risk_percent - baseline).toFixed(1)} points compared
              with the row as recorded ({baseline}%)
            </div>
          )}
          <div className="risk-severity">
            Dataset average {(model.positiveRate * 100).toFixed(0)}% &middot; this row is recorded as <b>{recorded}</b>
          </div>
          {factors.length > 0 && (
            <ul className="factor-list">
              {factors.map((f) => (
                <li key={f.label} className={f.effect > 0 ? "up" : "down"}>
                  {f.effect > 0 ? "▲" : "▼"} {f.label}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      <HeartSetup />
    </div>
  );
}

function EditRow({ ds }: { ds: Dataset }) {
  const mapping = useDatasetStore((s) => s.mapping);
  const target = useDatasetStore((s) => s.target);
  const model = useDatasetStore((s) => s.model);
  const resetRow = useDatasetStore((s) => s.resetRow);
  const changed = useDatasetStore((s) => s.values.some((v, j) => !Object.is(v, s.original[j])));

  const heartColumns = new Map<number, RoleKey>();
  for (const role of ROLES) if (mapping[role.key] !== undefined) heartColumns.set(mapping[role.key] as number, role.key);
  const modelColumns = new Set(model?.features.map((f) => f.column) ?? []);
  const others = ds.columns
    .map((c, j) => j)
    .filter((j) => j !== target && !heartColumns.has(j) && modelColumns.has(j));

  return (
    <div className="panel input-controls">
      <div className="panel-title">
        <span>Change this row&rsquo;s values</span>
        <span className="panel-subtitle">
          A what-if on the loaded data (the file isn&rsquo;t changed). Values marked <span className="heart-tag">&#9829;</span> set the
          heart directly; every value changes the risk estimate{model ? "" : " once an outcome is chosen"}.
        </span>
      </div>
      {changed && (
        <button type="button" className="btn btn-small" onClick={resetRow}>
          Reset to the row as recorded
        </button>
      )}
      {[...heartColumns.keys()].map((j) => (
        <ValueControl key={j} ds={ds} column={j} role={heartColumns.get(j)} />
      ))}
      {heartColumns.size === 0 && <p className="input-group-note">No column matched the heart yet &mdash; see &ldquo;Column matches&rdquo; below.</p>}
      {others.length > 0 && (
        <details className="other-columns">
          <summary>Other columns the model uses ({others.length})</summary>
          {others.map((j) => (
            <ValueControl key={j} ds={ds} column={j} />
          ))}
        </details>
      )}
    </div>
  );
}

function ValueControl({ ds, column, role }: { ds: Dataset; column: number; role?: RoleKey }) {
  const value = useDatasetStore((s) => s.values[column]);
  const original = useDatasetStore((s) => s.original[column]);
  const setValue = useDatasetStore((s) => s.setValue);
  const c = ds.columns[column];
  const roleDef = role ? ROLES.find((r) => r.key === role) : undefined;
  const isChanged = !Object.is(value, original);
  const id = `ds-col-${column}`;
  const missing = !Number.isFinite(value);

  let control;
  if (c.kind === "binary") {
    const labels = [c.negativeLabel ?? "No (0)", c.positiveLabel ?? "Yes (1)"];
    control = (
      <div className="segmented" role="radiogroup" aria-labelledby={`${id}-label`}>
        {[0, 1].map((v) => (
          <button key={v} type="button" role="radio" aria-checked={value === v} className={`seg-button ${value === v ? "active" : ""}`} onClick={() => setValue(column, v)}>
            {labels[v]}
          </button>
        ))}
      </div>
    );
  } else if (c.kind === "category") {
    control = (
      <select id={id} value={missing ? "" : value} onChange={(e) => setValue(column, Number(e.target.value))}>
        {missing && <option value="">— (missing)</option>}
        {c.categories.map((cat, k) => (
          <option key={k} value={k}>
            {cat}
          </option>
        ))}
      </select>
    );
  } else {
    const lo = Math.min(c.stats.p01, Number.isFinite(original) ? original : Infinity);
    const hi = Math.max(c.stats.p99, Number.isFinite(original) ? original : -Infinity);
    const span = hi - lo;
    const integer = [c.stats.min, c.stats.max, c.stats.median].every(Number.isInteger);
    const step = integer ? 1 : span > 50 ? 0.5 : span > 5 ? 0.1 : 0.01;
    control = (
      <input id={id} type="range" min={lo} max={hi} step={step} value={missing ? c.stats.median : value} onChange={(e) => setValue(column, Number(e.target.value))} />
    );
  }

  return (
    <div className={`input-row ${isChanged ? "changed" : ""}`}>
      <div className="input-head">
        <label id={`${id}-label`} htmlFor={id}>
          {c.name}
        </label>
        {roleDef && (
          <span className="heart-tag" title={`${roleDef.label}: ${roleDef.effect}`}>
            &#9829;
          </span>
        )}
        <span className="input-value">{missing ? "missing" : valueText(ds, column, value)}</span>
      </div>
      {roleDef && <div className="input-clinical">{roleDef.label} &middot; {roleDef.effect.toLowerCase()}</div>}
      {control}
      {isChanged && (
        <div className="input-baseline">
          Recorded: {Number.isFinite(original) ? valueText(ds, column, original) : "missing"}
          <button type="button" className="link-button" onClick={() => setValue(column, original)}>
            Undo
          </button>
        </div>
      )}
    </div>
  );
}

function ColumnMatches({ ds }: { ds: Dataset }) {
  const mapping = useDatasetStore((s) => s.mapping);
  const target = useDatasetStore((s) => s.target);
  const setRole = useDatasetStore((s) => s.setRole);
  const found = ROLES.filter((r) => mapping[r.key] !== undefined).length;
  const notes = ds.columns.filter((c) => c.note);

  return (
    <div className="panel">
      <details className="column-matches" open={found === 0}>
        <summary>
          Column matches &mdash; {found} of {ROLES.length} heart inputs found
        </summary>
        <p className="input-group-note">
          Matched by column name; change any that are wrong. Anything not found uses a typical value, shown in &ldquo;How the heart
          model is set up&rdquo;.
        </p>
        <table className="derivation">
          <tbody>
            {ROLES.map((r) => (
              <tr key={r.key}>
                <td title={r.effect}>
                  {r.label}
                  {r.unit && <span className="unit"> ({r.unit})</span>}
                </td>
                <td>
                  <select value={mapping[r.key] ?? ""} onChange={(e) => setRole(r.key, e.target.value === "" ? null : Number(e.target.value))}>
                    <option value="">Not in data</option>
                    {ds.columns.map((c, j) =>
                      c.kind === "text" || j === target ? null : (
                        <option key={j} value={j}>
                          {c.name}
                        </option>
                      )
                    )}
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {notes.length > 0 && (
          <ul className="read-notes">
            {notes.map((c) => (
              <li key={c.name}>
                <b>{c.name}</b>: {c.note}
              </li>
            ))}
          </ul>
        )}
      </details>
    </div>
  );
}
