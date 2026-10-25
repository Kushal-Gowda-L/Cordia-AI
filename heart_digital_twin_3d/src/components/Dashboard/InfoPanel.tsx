export function InfoPanel() {
  return (
    <div className="panel info-panel">
      <div className="panel-title">
        <span>Reading the heart view</span>
      </div>
      <ul className="legend-list">
        <li>
          <i className="dot dot-red" /> Oxygen-rich blood (left side of the heart)
        </li>
        <li>
          <i className="dot dot-blue" /> Oxygen-poor blood (right side of the heart)
        </li>
        <li>
          <i className="dot dot-yellow" /> Electrical signal that starts each heartbeat
        </li>
        <li>
          <i className="dot dot-glow" /> Red glow on the muscle: the heart is working harder than in the recorded
          results (brighter means harder)
        </li>
      </ul>
      <dl className="chamber-key">
        <div>
          <dt>RA</dt>
          <dd>Right upper chamber &mdash; receives blood from the body</dd>
        </div>
        <div>
          <dt>RV</dt>
          <dd>Right pump &mdash; sends blood to the lungs</dd>
        </div>
        <div>
          <dt>LA</dt>
          <dd>Left upper chamber &mdash; receives blood from the lungs</dd>
        </div>
        <div>
          <dt>LV</dt>
          <dd>Left pump &mdash; sends blood to the whole body</dd>
        </div>
      </dl>
      <p>
        The heart, the heartbeat signal and the pressures are simulated from the patient's details. They show how
        those details are expected to affect the heart; they are not measurements taken from this patient.
      </p>
    </div>
  );
}
