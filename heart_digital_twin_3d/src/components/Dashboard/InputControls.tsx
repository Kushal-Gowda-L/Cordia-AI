import { bloodPressureBand, formatInput, INPUT_FIELDS, type InputField } from "../../patient/inputFields";
import { PATIENT_VIEW } from "../../patient/patientTwin";
import { useHeartStore } from "../../state/store";

function FieldControl({ field, baselineName }: { field: InputField; baselineName: string }) {
  const value = useHeartStore((s) => s.inputs?.[field.key]);
  const baseline = useHeartStore((s) => s.recordInputs?.[field.key]);
  const setInput = useHeartStore((s) => s.setInput);
  // The caution describes the server's model; the offline estimate doesn't share that flaw.
  const offline = useHeartStore((s) => s.offline);
  if (value === undefined) return null;

  const changed = baseline !== undefined && baseline !== value;
  const id = `input-${field.key}`;
  const label = PATIENT_VIEW && field.patient ? field.patient.label : field.label;
  const shown =
    PATIENT_VIEW && field.key === "trestbps" ? `${formatInput(field, value)} · ${bloodPressureBand(value)}` : formatInput(field, value);

  return (
    <div className={`input-row ${changed ? "changed" : ""}`}>
      <div className="input-head">
        <label id={`${id}-label`} htmlFor={id}>
          {label}
        </label>
        {!PATIENT_VIEW && field.drivesHeart && (
          <span className="heart-tag" title="Also directly changes the simulated heart">
            &#9829;
          </span>
        )}
        <span className="input-value">{shown}</span>
      </div>
      <div className="input-clinical">{PATIENT_VIEW && field.patient ? field.patient.hint : field.clinical}</div>
      {field.caution && !offline && <div className="input-caution">{field.caution}</div>}

      {field.kind === "slider" ? (
        <input
          id={id}
          type="range"
          min={field.min}
          max={field.max}
          step={field.step}
          value={value}
          onChange={(e) => setInput(field.key, Number(e.target.value))}
        />
      ) : field.options.length <= 2 ? (
        <div className="segmented" role="radiogroup" aria-labelledby={`${id}-label`}>
          {field.options.map((o) => (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={o.value === value}
              className={`seg-button ${o.value === value ? "active" : ""}`}
              onClick={() => setInput(field.key, o.value)}
            >
              {o.label}
            </button>
          ))}
        </div>
      ) : (
        <select id={id} value={value} onChange={(e) => setInput(field.key, Number(e.target.value))}>
          {field.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      )}

      {changed && baseline !== undefined && (
        <div className="input-baseline">
          {baselineName}: {formatInput(field, baseline)}
          <button type="button" className="link-button" onClick={() => setInput(field.key, baseline)}>
            Undo
          </button>
        </div>
      )}
    </div>
  );
}

const HEALTH_FIELDS = INPUT_FIELDS.filter((f) => !f.exercise);
const EXERCISE_FIELDS = INPUT_FIELDS.filter((f) => f.exercise);
const PATIENT_FIELDS = INPUT_FIELDS.filter((f) => f.patient);

export function InputControls() {
  const ready = useHeartStore((s) => s.inputs !== null);
  const hasPatient = useHeartStore((s) => s.patient !== null);
  if (!ready) return null;
  const baselineName = hasPatient ? "Record" : "Start";

  if (PATIENT_VIEW) {
    return (
      <div className="panel input-controls">
        <div className="panel-title">
          <span>Try it yourself</span>
          <span className="panel-subtitle">
            See how your heart and your estimate would change if these were different. Your other results are the
            ones your clinic recorded. Nothing you try here is saved.
          </span>
        </div>
        {PATIENT_FIELDS.map((f) => (
          <FieldControl key={f.key} field={f} baselineName="Your result" />
        ))}
      </div>
    );
  }

  return (
    <div className="panel input-controls">
      <div className="panel-title">
        <span>Try changing the details</span>
        <span className="panel-subtitle">
          {hasPatient
            ? "A what-if only — the patient's saved record is not changed."
            : "Example values — adjust them to explore."}{" "}
          Every detail changes the risk estimate; those marked <span className="heart-tag">&#9829;</span> also
          directly change the heart.
        </span>
      </div>

      <div className="input-group-title">Health details</div>
      {HEALTH_FIELDS.map((f) => (
        <FieldControl key={f.key} field={f} baselineName={baselineName} />
      ))}

      <div className="input-group-title">Exercise test results</div>
      <p className="input-group-note">Changing these shows the heart during exercise, where they apply.</p>
      {EXERCISE_FIELDS.map((f) => (
        <FieldControl key={f.key} field={f} baselineName={baselineName} />
      ))}
    </div>
  );
}
