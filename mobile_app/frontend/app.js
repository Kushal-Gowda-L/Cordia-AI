// CordiaAI web client: sign-in, patient list, assessment report (clinician
// and patient versions), follow-up visits and visit comparison, patient
// access codes, digital twin, plain-language summary and ECG analysis. Talks to the FastAPI backend in
// mobile_app/backend/main.py.
const API = "";

function getToken() { return localStorage.getItem("cordia_token"); }
function setToken(t) { localStorage.setItem("cordia_token", t); }
function clearToken() { localStorage.removeItem("cordia_token"); }
function getUsername() { return localStorage.getItem("cordia_username") || ""; }
function setUsername(u) { localStorage.setItem("cordia_username", u); }
function getRole() { return localStorage.getItem("cordia_role") || "doctor"; }
function setRole(r) { localStorage.setItem("cordia_role", r); }
const isDoctor = () => getRole() === "doctor";

const $ = (id) => document.getElementById(id);

// Patient names and server text are rendered into HTML, so always escape.
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------- Field labels
// Everyday wording for the ten model inputs, with the clinical term alongside.
const CHEST_PAIN = {
  1: "Typical heart pain (typical angina)",
  2: "Unusual chest pain (atypical angina)",
  3: "Pain not from the heart (non-anginal)",
  4: "No chest pain (asymptomatic)",
};
const REST_ECG = { 0: "Normal", 1: "Minor changes (ST-T abnormality)", 2: "Thickened heart muscle (LV hypertrophy)" };
const YES_NO = { 0: "No", 1: "Yes" };
const ST_SLOPE = { 1: "Rising (upsloping)", 2: "Flat", 3: "Falling (downsloping)" };
const THAL = { 3: "Normal", 6: "Permanent defect (fixed)", 7: "Defect during exercise (reversible)" };

const FIELDS = {
  age: { label: "Age", clinical: "age", format: (v) => `${Math.round(v)} years` },
  sex: { label: "Sex", clinical: "sex", format: (v) => (Math.round(v) === 1 ? "Male" : "Female") },
  cp: { label: "Type of chest pain", clinical: "chest pain type", format: (v) => CHEST_PAIN[Math.round(v)] ?? String(v) },
  trestbps: { label: "Blood pressure at rest", clinical: "resting systolic blood pressure", format: (v) => `${Math.round(v)} mm Hg` },
  chol: { label: "Cholesterol", clinical: "serum cholesterol", format: (v) => `${Math.round(v)} mg/dl` },
  fbs: { label: "High blood sugar when fasting", clinical: "fasting blood sugar > 120 mg/dl", format: (v) => YES_NO[Math.round(v)] ?? String(v) },
  restecg: { label: "Heart tracing at rest", clinical: "resting ECG", format: (v) => REST_ECG[Math.round(v)] ?? String(v) },
  thalach: { label: "Highest heart rate during exercise", clinical: "maximum heart rate achieved", format: (v) => `${Math.round(v)} beats/min` },
  exang: { label: "Chest pain during exercise", clinical: "exercise-induced angina", format: (v) => YES_NO[Math.round(v)] ?? String(v) },
  oldpeak: { label: "Heart strain on the exercise tracing", clinical: "ST depression", format: (v) => `${Number(v).toFixed(1)} mm` },
  slope: { label: "Shape of the exercise tracing at peak", clinical: "ST segment slope", format: (v) => ST_SLOPE[Math.round(v)] ?? String(v) },
  ca: { label: "Narrowed main heart arteries seen", clinical: "major vessels on fluoroscopy", format: (v) => String(Math.round(v)) },
  thal: { label: "Blood-flow scan of the heart muscle", clinical: "thallium stress test", format: (v) => THAL[Math.round(v)] ?? String(v) },
};
const FIELD_ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"];
// Optional tests: listed on a report only when they were done.
const OPTIONAL_FIELDS = ["slope", "ca", "thal"];

function fieldLabel(key) { return FIELDS[key]?.label ?? key; }
function fieldValue(key, value) { return FIELDS[key] ? FIELDS[key].format(value) : String(value); }
function optionalInt(id) { const v = $(id).value; return v === "" ? null : parseInt(v, 10); }

// LIME rules name the raw columns ("trestbps > 140.00"); show the plain label.
function plainRule(rule) {
  return rule.replace(/\b(age|sex|cp|trestbps|chol|fbs|restecg|thalach|exang|oldpeak)\b/g, (key) => fieldLabel(key));
}

const ECG_CLASS_LABELS = {
  NORM: "Normal ECG",
  MI: "Signs of a previous heart attack (myocardial infarction)",
  STTC: "ST/T wave changes",
  CD: "Conduction disturbance",
  HYP: "Thickened heart muscle (hypertrophy)",
};

// Zones match src/risk_reading.py: below 35% unlikely, 35-65% borderline,
// 65% and above likely. They are a display convention for flagging results
// near the model's 50% decision point, not clinical categories.
function riskTone(pct) { return pct < 35 ? "low" : pct < 65 ? "mid" : "high"; }
const BAND_TONE = { unlikely: "low", borderline: "mid", likely: "high" };

// Where a result sits on the three zones, with the 50% decision point marked.
function zoneScale(reading) {
  const { unlikely_below: lo, likely_from: hi } = reading.zones;
  const pct = Math.min(100, Math.max(0, reading.disease_percent));
  return `
    <div class="zone-scale" role="img" aria-label="${pct}% on a scale: below ${lo}% unlikely, ${lo} to ${hi}% borderline, ${hi}% and above likely; ${reading.decision_point}% is the model's decision point">
      <div class="zone-track">
        <span class="zone zone-low" style="width:${lo}%"></span><span class="zone zone-mid" style="width:${hi - lo}%"></span><span class="zone zone-high" style="width:${100 - hi}%"></span>
        <span class="zone-cut" style="left:${reading.decision_point}%"></span>
        <span class="zone-marker" style="left:${pct}%"></span>
      </div>
      <div class="zone-labels">
        <span style="width:${lo}%">Unlikely<small>below ${lo}%</small></span>
        <span style="width:${hi - lo}%">Borderline<small>${lo}–${hi}%</small></span>
        <span style="width:${100 - hi}%">Likely<small>${hi}% and above</small></span>
      </div>
      <div class="zone-cut-label">Dashed line: the model's ${reading.decision_point}% decision point</div>
    </div>`;
}

function formatDate(seconds) {
  return seconds ? new Date(seconds * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "";
}
function formatDateTime(seconds) {
  return seconds ? new Date(seconds * 1000).toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
}

// ---------------------------------------------------------------- Views
const AUTH_VIEWS = ["view-login", "view-signup"];

function applyRoleVisibility() {
  document.querySelectorAll(".role-doctor-only").forEach((el) => el.classList.toggle("hidden", !isDoctor()));
  document.querySelectorAll(".role-patient-only").forEach((el) => el.classList.toggle("hidden", isDoctor()));
}

function showView(id) {
  const shell = $("app-shell");
  if (AUTH_VIEWS.includes(id)) {
    shell.classList.add("hidden");
    AUTH_VIEWS.forEach((v) => $(v).classList.toggle("hidden", v !== id));
    return;
  }
  AUTH_VIEWS.forEach((v) => $(v).classList.add("hidden"));
  shell.classList.remove("hidden");
  shell.querySelectorAll(".main-area > .view").forEach((v) => v.classList.toggle("active", v.id === id));
  document.querySelectorAll(".side-link[data-view]").forEach((l) => {
    const current = l.dataset.view === id || (id === "view-link" && l.dataset.view === "view-report");
    l.classList.toggle("active", current);
  });
  applyRoleVisibility();
  window.scrollTo(0, 0);
}

async function api(path, options = {}) {
  const headers = options.headers || {};
  if (getToken()) headers["Authorization"] = `Bearer ${getToken()}`;
  if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const res = await fetch(API + path, { ...options, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : "Request failed");
  }
  if (res.status === 204) return null;
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

// ---------------------------------------------------------------- Navigation
document.querySelectorAll(".side-link[data-view]").forEach((link) => {
  link.onclick = (e) => {
    e.preventDefault();
    const view = link.dataset.view;
    if (view === "view-home") { showView(view); loadPatients(); }
    else if (view === "view-report") routePatient();
    else showView(view);
  };
});

// ---------------------------------------------------------------- Sign in / sign up
$("go-signup").onclick = (e) => { e.preventDefault(); showView("view-signup"); };
$("go-login").onclick = (e) => { e.preventDefault(); showView("view-login"); };

$("form-login").onsubmit = async (e) => {
  e.preventDefault();
  const errEl = $("login-error");
  errEl.textContent = "";
  try {
    const data = await api("/api/login", {
      method: "POST",
      body: JSON.stringify({ username: $("login-username").value, password: $("login-password").value }),
    });
    setToken(data.token); setUsername(data.username); setRole(data.role);
    await enterApp();
  } catch (err) {
    errEl.textContent = err.message;
  }
};

$("form-signup").onsubmit = async (e) => {
  e.preventDefault();
  const errEl = $("signup-error");
  errEl.textContent = "";
  try {
    const role = document.querySelector('input[name="signup-role"]:checked').value;
    const data = await api("/api/signup", {
      method: "POST",
      body: JSON.stringify({ username: $("signup-username").value, password: $("signup-password").value, role }),
    });
    setToken(data.token); setUsername(data.username); setRole(data.role);
    await enterApp();
  } catch (err) {
    errEl.textContent = err.message;
  }
};

$("btn-logout").onclick = () => { clearToken(); showView("view-login"); };

async function enterApp() {
  $("home-username").textContent = getUsername();
  $("home-role").textContent = isDoctor() ? "Clinician" : "Patient";
  $("btn-open-twin").textContent = isDoctor() ? "Open digital twin" : "Open my heart model";
  // Clinicians' main action on a report is recording the next visit.
  $("btn-open-twin").classList.toggle("btn-primary", !isDoctor());
  $("btn-ai-summary").textContent = isDoctor() ? "Plain-language summary" : "Explain in simple words";
  $("ai-summary-title").textContent = isDoctor() ? "Plain-language summary" : "Your report in simple words";
  applyRoleVisibility();
  if (isDoctor()) {
    showView("view-home");
    await loadPatients();
  } else {
    await routePatient();
  }
}

// ---------------------------------------------------------------- Patients: link a record
// Patients don't enter clinical results themselves: their clinic records them
// and gives the patient a one-time code that links the record to this account.
async function routePatient() {
  const linked = await api("/api/me");
  if (linked) { await openMyReport(); return; }
  showView("view-link");
}

$("form-link").onsubmit = async (e) => {
  e.preventDefault();
  const errEl = $("link-error");
  errEl.textContent = "";
  const code = $("link-code").value.trim();
  if (code.replace(/[-\s]/g, "").length !== 8) {
    errEl.textContent = "Please enter the full 8-character code, for example K7PQ-3MXA.";
    return;
  }
  try {
    await api("/api/me/link", { method: "POST", body: JSON.stringify({ code }) });
    $("link-code").value = "";
    await openMyReport();
  } catch (err) {
    errEl.textContent = err.message;
  }
};

// ---------------------------------------------------------------- Patient list (clinicians)
async function loadPatients() {
  const rows = $("patient-rows");
  try {
    const patients = await api("/api/patients");
    rows.innerHTML = patients.map((p) => `
      <tr data-id="${p.id}" tabindex="0">
        <td class="patient-link">${escapeHtml(p.name)}</td>
        <td class="num">${p.age}</td>
        <td>${escapeHtml(p.sex ?? "")}</td>
        <td><span class="risk-badge risk-${riskTone(p.risk_percent)}">${p.risk_percent}%</span></td>
        <td><span class="verdict-text tone-${BAND_TONE[p.band]}">${escapeHtml(p.verdict)}</span></td>
        <td class="num">${p.visit_count ?? 1}</td>
        <td>${formatDate(p.last_visit_at ?? p.created_at)}</td>
        <td>${p.linked ? '<span class="access-badge">Linked</span>' : '<span class="muted">Not linked</span>'}</td>
      </tr>`).join("");
    $("patient-table").classList.toggle("hidden", patients.length === 0);
    $("empty-state").classList.toggle("hidden", patients.length > 0);
    rows.querySelectorAll("tr").forEach((tr) => {
      const open = () => openReport(Number(tr.dataset.id));
      tr.onclick = open;
      tr.onkeydown = (e) => { if (e.key === "Enter") open(); };
    });
  } catch (err) {
    if (/session|authenticated/i.test(err.message)) { clearToken(); showView("view-login"); }
  }
}

// ---------------------------------------------------------------- Add patient / visit (clinicians)
// One form serves three jobs: a new patient, a follow-up visit for an existing
// patient (starts from the latest visit's values) and correcting the values
// recorded at a visit. A follow-up is saved as a new visit, so earlier visits
// stay as they were and can be compared.
let formMode = { mode: "add", visitId: null };

const FORM_TEXT = {
  add: { title: "Add patient", sub: "All of these details are used to estimate the risk.", submit: "Run assessment" },
  visit: { title: "Follow-up visit", sub: "Starts from the last visit's values. Change what is different today; the earlier visits are kept for comparison.", submit: "Save visit and re-assess" },
  edit: { title: "Edit visit details", sub: "Correct the values recorded at this visit. The visit keeps its date and the risk is re-calculated.", submit: "Save changes and re-assess" },
};

function setFormMode(mode, visitId = null, heading = null) {
  formMode = { mode, visitId };
  $("add-title").textContent = heading || FORM_TEXT[mode].title;
  $("add-sub").textContent = FORM_TEXT[mode].sub;
  $("btn-submit-patient").textContent = FORM_TEXT[mode].submit;
  $("add-error").textContent = "";
}

function fillPatientForm(name, inputs, note) {
  $("p-name").value = name;
  ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"].forEach((key) => { $(`p-${key}`).value = inputs[key]; });
  OPTIONAL_FIELDS.forEach((key) => { $(`p-${key}`).value = inputs[key] === null || inputs[key] === undefined ? "" : String(inputs[key]); });
  $("p-note").value = note || "";
}

$("btn-add-patient").onclick = () => {
  $("form-patient").reset();
  setFormMode("add");
  showView("view-add");
};
$("btn-back-add").onclick = () => {
  if (formMode.mode === "add") showView("view-home");
  else showView("view-report");
};

function openVisitForm(mode, visit) {
  $("form-patient").reset();
  const visits = currentReport.visits;
  const heading = mode === "visit"
    ? `Follow-up visit — ${currentReport.name} (visit ${visits.length + 1})`
    : `Edit visit ${visit.number} — ${currentReport.name}`;
  setFormMode(mode, mode === "edit" ? visit.id : null, heading);
  fillPatientForm(currentReport.name, visit.inputs, mode === "edit" ? visit.note : "");
  showView("view-add");
}

function readPatientForm() {
  return {
    age: parseInt($("p-age").value, 10),
    sex: parseInt($("p-sex").value, 10),
    cp: parseInt($("p-cp").value, 10),
    trestbps: parseInt($("p-trestbps").value, 10),
    chol: parseInt($("p-chol").value, 10),
    fbs: parseInt($("p-fbs").value, 10),
    restecg: parseInt($("p-restecg").value, 10),
    thalach: parseInt($("p-thalach").value, 10),
    exang: parseInt($("p-exang").value, 10),
    oldpeak: parseFloat($("p-oldpeak").value),
    slope: optionalInt("p-slope"),
    ca: optionalInt("p-ca"),
    thal: optionalInt("p-thal"),
  };
}

$("form-patient").onsubmit = async (e) => {
  e.preventDefault();
  const errEl = $("add-error");
  errEl.textContent = "";
  try {
    const payload = { name: $("p-name").value, ...readPatientForm(), note: $("p-note").value };
    const body = JSON.stringify(payload);
    let result;
    if (formMode.mode === "visit") {
      result = await api(`/api/patients/${currentPatientId}/visits`, { method: "POST", body });
    } else if (formMode.mode === "edit") {
      result = await api(`/api/patients/${currentPatientId}/visits/${formMode.visitId}`, { method: "PUT", body });
    } else {
      result = await api("/api/patients", { method: "POST", body });
    }
    await openReport(result.id);
  } catch (err) {
    errEl.textContent = err.message;
  }
};

// ---------------------------------------------------------------- Assessment report
let currentPatientId = null;
let currentReport = null;

$("btn-new-visit").onclick = () => {
  const visits = currentReport.visits;
  openVisitForm("visit", visits[visits.length - 1]);
};
$("btn-edit-visit").onclick = () => {
  const visits = currentReport.visits;
  openVisitForm("edit", visits[visits.length - 1]);
};
$("btn-compare-visits").onclick = () => {
  const visits = currentReport.visits;
  openCompare(visits[visits.length - 2].id, visits[visits.length - 1].id);
};

$("btn-delete-patient").onclick = async () => {
  if (!confirm("Delete this patient and their assessment? This cannot be undone.")) return;
  await api(`/api/patients/${currentPatientId}`, { method: "DELETE" });
  showView("view-home");
  await loadPatients();
};

function probabilityBars(probs, topLabel) {
  return Object.entries(probs).map(([label, pct]) => `
    <div class="bar-row ${label === topLabel ? "top" : ""}">
      <div class="bar-label">${escapeHtml(label)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div class="bar-value">${pct}%</div>
    </div>`).join("");
}

// SHAP contributions to the chance of heart disease, drawn as a diverging bar
// chart: red to the right raises the chance, green to the left lowers it.
// Impacts are probability changes, shown as percentage points.
function driverChart(factors) {
  if (!factors.length) return `<p class="panel-text">Factor contributions are unavailable for this assessment.</p>`;
  const max = Math.max(...factors.map((f) => Math.abs(f.impact)), 1e-9);
  const rows = factors.map((f) => {
    const width = (Math.abs(f.impact) / max) * 50;
    const dir = f.impact >= 0 ? "up" : "down";
    const sign = f.impact >= 0 ? "▲" : "▼";
    return `
      <div class="driver-row">
        <div class="driver-label">${escapeHtml(fieldLabel(f.feature))}<small>${escapeHtml(fieldValue(f.feature, f.value))}</small></div>
        <div class="driver-track"><div class="driver-bar ${dir}" style="width:${width}%"></div></div>
        <div class="driver-value" title="SHAP value ${f.impact >= 0 ? "+" : "−"}${Math.abs(f.impact).toFixed(3)}">${sign} ${(Math.abs(f.impact) * 100).toFixed(1)} pts</div>
      </div>`;
  }).join("");
  return `${rows}
    <div class="driver-axis"><span class="axis-ends"><span>&larr; Lowers the risk</span><span>Raises the risk &rarr;</span></span><span></span></div>`;
}

function inputsTable(raw, { clinical = true } = {}) {
  const keys = [...FIELD_ORDER, ...OPTIONAL_FIELDS.filter((key) => raw[key] !== null && raw[key] !== undefined)];
  return `<table class="kv-table">${keys.map((key) => `
    <tr><td>${escapeHtml(fieldLabel(key))}${clinical ? `<small>${escapeHtml(FIELDS[key].clinical)}</small>` : ""}</td><td>${escapeHtml(fieldValue(key, raw[key]))}</td></tr>`).join("")}
  </table>`;
}

// Every visit of the patient, newest first, with the risk at each and how it
// moved since the visit before.
function visitHistoryPanel(visits) {
  const rows = visits.slice().reverse().map((v) => {
    const prev = visits[v.number - 2];
    const delta = prev ? riskDelta(prev.risk_percent, v.risk_percent) : `<span class="muted">First visit</span>`;
    const latest = v.number === visits.length;
    return `
      <tr>
        <td><b>Visit ${v.number}</b>${latest ? ' <span class="access-badge">Current</span>' : ""}<small>${escapeHtml(formatDateTime(v.created_at))}${v.updated_at ? " · edited" : ""}</small>${v.note ? `<small class="visit-note">${escapeHtml(v.note)}</small>` : ""}</td>
        <td><span class="risk-badge risk-${riskTone(v.risk_percent)}">${v.risk_percent}%</span><small>${delta}</small></td>
        <td class="visit-actions">
          ${prev ? `<button class="btn btn-small" data-compare="${v.id}" title="Compare with visit ${prev.number}">Compare with previous</button>` : ""}
          <button class="btn btn-small" data-edit="${v.id}">Edit</button>
          ${visits.length > 1 ? `<button class="btn btn-small btn-danger" data-delete="${v.id}">Delete</button>` : ""}
        </td>
      </tr>`;
  }).join("");
  return `
    <section class="panel">
      <h2>Visits</h2>
      <p class="panel-sub">${visits.length === 1 ? "One visit so far. Use <b>Record follow-up visit</b> when the patient returns; this visit is kept for comparison." : `${visits.length} visits. This report shows the latest one.`}</p>
      <table class="kv-table visit-table">${rows}</table>
    </section>`;
}

function wireVisitHistory() {
  const byId = Object.fromEntries(currentReport.visits.map((v) => [v.id, v]));
  document.querySelectorAll("[data-compare]").forEach((b) => {
    const v = byId[b.dataset.compare];
    b.onclick = () => openCompare(currentReport.visits[v.number - 2].id, v.id);
  });
  document.querySelectorAll("[data-edit]").forEach((b) => { b.onclick = () => openVisitForm("edit", byId[b.dataset.edit]); });
  document.querySelectorAll("[data-delete]").forEach((b) => {
    b.onclick = async () => {
      const v = byId[b.dataset.delete];
      if (!confirm(`Delete visit ${v.number} (${formatDateTime(v.created_at)})? The patient's other visits are kept. This cannot be undone.`)) return;
      await api(`/api/patients/${currentPatientId}/visits/${v.id}`, { method: "DELETE" });
      await openReport(currentPatientId);
    };
  });
}

// Risk change between two visits in percentage points. Up is shown in red,
// down in green -- a higher chance of heart disease is the worse direction.
function riskDelta(from, to) {
  const d = Math.round((to - from) * 10) / 10;
  if (d === 0) return `<span class="delta delta-same">No change</span>`;
  return d > 0
    ? `<span class="delta delta-up">▲ ${d.toFixed(1)} pts</span>`
    : `<span class="delta delta-down">▼ ${Math.abs(d).toFixed(1)} pts</span>`;
}

function accessPanel(access) {
  const linked = access && access.linked;
  const status = linked
    ? `Linked to the patient account <b>${escapeHtml(access.username)}</b>. They can see this report and their heart model.`
    : "Not linked to a patient account yet.";
  return `
    <section class="panel">
      <h2>Patient access</h2>
      <p class="panel-sub">Let the patient see this report and their heart model in simple language, from their own patient account.</p>
      <p class="access-status">${status}</p>
      ${linked ? "" : `<button class="btn" id="btn-access-code">Create access code</button>`}
      <div id="access-code-box" class="access-code hidden" aria-live="polite"></div>
    </section>`;
}

function wireAccessPanel() {
  const button = $("btn-access-code");
  if (!button) return;
  button.onclick = async () => {
    const box = $("access-code-box");
    button.disabled = true;
    try {
      const data = await api(`/api/patients/${currentPatientId}/access-code`, { method: "POST" });
      box.innerHTML = `
        <div class="access-code-value">${escapeHtml(data.code)}</div>
        <p>Give this code to the patient. They sign in with a patient account and enter it under &ldquo;My heart report&rdquo;. It works once and expires on ${formatDate(data.expires_at)}. Creating another code cancels this one.</p>`;
      button.textContent = "Create a new code";
    } catch (err) {
      box.innerHTML = `<p class="form-error">${escapeHtml(err.message)}</p>`;
    } finally {
      box.classList.remove("hidden");
      button.disabled = false;
    }
  };
}

function renderReport(report) {
  if (!isDoctor()) { renderPatientReport(report); return; }

  const sex = report.raw ? fieldValue("sex", report.raw.sex) : "";
  const visits = report.visits || [];
  const latest = visits[visits.length - 1];
  $("report-name").textContent = report.name;
  $("report-sub").textContent = `Age ${report.age}${sex ? " · " + sex : ""}${latest ? ` · Visit ${latest.number} of ${visits.length}, ${formatDate(latest.created_at)}` : ""}`;
  $("btn-compare-visits").disabled = visits.length < 2;
  $("btn-compare-visits").title = visits.length < 2 ? "Record a follow-up visit first" : "";

  const r = report.reading;
  const findings = report.findings.map((f) => `<li>${escapeHtml(f)}</li>`).join("");
  const whatIf = report.what_if.map((f) => `<li>${escapeHtml(f)}</li>`).join("");
  const counterfactuals = (report.dice_counterfactuals || []).map((cf) => `
    <li>To move towards <b>${escapeHtml(cf.target_class)}</b>: ${Object.entries(cf.changes)
      .map(([key, value]) => `${escapeHtml(fieldLabel(key))} &rarr; ${escapeHtml(fieldValue(key, value))}`).join(", ")}</li>`).join("");
  const lime = (report.lime_factors || []).map((f) => `
    <li>${escapeHtml(plainRule(f.rule))} <span class="muted">(weight ${f.weight >= 0 ? "+" : ""}${f.weight})</span></li>`).join("");

  $("report-body").innerHTML = `
    <div class="report-layout">
      <div>
        <section class="panel">
          <div class="summary-risk">
            <div class="step-label">Step 1 · Is heart disease likely?</div>
            <div class="risk-figure tone-${BAND_TONE[r.band]}">${report.risk_percent}%</div>
            <div class="risk-caption">Chance of heart disease</div>
            <div class="verdict verdict-${BAND_TONE[r.band]}">${escapeHtml(r.verdict)}</div>
            ${zoneScale(r)}
          </div>
          <dl class="fact-list">
            <div><dt>Model's decision (${r.decision_point}% cut-off)</dt><dd>${r.predicts_disease ? "Heart disease" : "No heart disease"}</dd></div>
            <div><dt>If present, most likely severity</dt><dd>${r.likely_severity_if_disease ? `${escapeHtml(r.likely_severity_if_disease)} (${r.likely_severity_percent}%)` : "—"}</dd></div>
            <div><dt>Age</dt><dd>${report.age}</dd></div>
            <div><dt>Sex</dt><dd>${escapeHtml(sex)}</dd></div>
          </dl>
          <p class="summary-note">Step 1 adds up every disease level (mild to critical) into one chance and compares it with the model's ${r.decision_point}% decision point. Step 2 then looks only at the disease levels. The ${r.zones.unlikely_below}% and ${r.zones.likely_from}% zone limits are this app's way of flagging results close to the decision point &mdash; not clinical categories. A model estimate, not a diagnosis.</p>
        </section>
        <section class="panel">
          <h2>Patient details used</h2>
          <p class="panel-sub">The ten values the estimate is based on.</p>
          ${report.raw ? inputsTable(report.raw) : ""}
        </section>
        ${accessPanel(report.patient_access)}
      </div>
      <div>
        ${visits.length ? visitHistoryPanel(visits) : ""}
        <section class="panel">
          <div class="step-label">Step 2</div>
          <h2>If heart disease is present, how severe?</h2>
          <p class="panel-sub">The four disease levels rescaled to add up to 100% &mdash; each level's chance divided by the ${report.risk_percent}% total.${r.likely_severity_if_disease ? ` Most likely: <b>${escapeHtml(r.likely_severity_if_disease)}</b>.` : ""}</p>
          ${probabilityBars(r.severity_if_disease, r.likely_severity_if_disease)}
          <details class="raw-output">
            <summary>Model's raw output (all five outcomes)</summary>
            <p class="panel-sub">These add up to 100% across all five outcomes. &ldquo;No disease&rdquo; can be the largest single bar even when the disease levels together are larger &mdash; which is why Step 1 uses their total (${report.risk_percent}%) instead.</p>
            ${probabilityBars(report.class_probabilities, null)}
          </details>
        </section>
        <section class="panel">
          <h2>What is driving the result</h2>
          <p class="panel-sub">The details that moved the chance of heart disease (${report.risk_percent}%) the most, compared with an average patient. Red raises the risk, green lowers it.</p>
          ${driverChart(report.shap_factors || [])}
        </section>
        <section class="panel">
          <h2>Clinical observations</h2>
          <ul class="plain-list">${findings}</ul>
        </section>
        <section class="panel">
          <h2>Changes that could lower the risk</h2>
          <p class="panel-sub">Found by re-running the model with details changed. For discussion only &mdash; not treatment advice.</p>
          ${counterfactuals ? `<ul class="plain-list">${counterfactuals}</ul>` : ""}
          <ul class="plain-list">${whatIf}</ul>
        </section>
        ${lime ? `<details class="panel"><summary>Technical detail</summary><p class="panel-sub">Rules from a second explanation method (LIME) for the most likely severity.</p><ul class="plain-list">${lime}</ul></details>` : ""}
      </div>
    </div>`;
  wireAccessPanel();
  wireVisitHistory();
  showView("view-report");
}

// The patient's own view: plain language, no model internals, and nothing
// that asks them to interpret clinical classifications.
function renderPatientReport(report) {
  const pct = report.risk_percent;
  const r = report.reading;
  const meaning = {
    unlikely: "The model does <b>not</b> suggest heart disease.",
    borderline: `Your result is <b>borderline</b> &mdash; close to the model's dividing line of ${r.decision_point}%. Further tests can give a clearer answer.`,
    likely: "The model suggests heart disease is <b>likely</b>.",
  }[r.band];
  $("report-name").textContent = "Your heart report";
  $("report-sub").textContent = `${report.name} · age ${report.age}`;
  // Suggestions about exercise-test readings aren't something a patient can act on.
  const helpful = report.what_if
    .filter((item) => !/ST depression|single-factor/i.test(item))
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");

  $("report-body").innerHTML = `
    <div class="report-layout">
      <div>
        <section class="panel">
          <div class="summary-risk">
            <div class="risk-figure tone-${BAND_TONE[r.band]}">${pct}%</div>
            <div class="risk-caption">Chance of heart disease</div>
            <div class="verdict verdict-${BAND_TONE[r.band]}">${escapeHtml(r.verdict)}</div>
            ${zoneScale(r)}
          </div>
          <p class="patient-explain">From the results your clinic recorded, the computer model estimates a <b>${pct}%</b> chance that you have some form of heart disease. ${meaning}</p>
          <p class="summary-note">This is an estimate to talk through with your doctor. It is not a diagnosis.</p>
        </section>
        <section class="panel">
          <h2>What your clinic recorded</h2>
          <p class="panel-sub">The results this estimate is based on.</p>
          ${report.raw ? inputsTable(report.raw, { clinical: false }) : ""}
        </section>
      </div>
      <div>
        <section class="panel">
          <h2>See how your heart works</h2>
          <p class="panel-sub">A moving model of your heart. See it resting and being active, and try a different blood pressure or blood sugar to see what changes.</p>
          <button class="btn btn-primary" id="btn-open-twin-inline">Open my heart model</button>
        </section>
        <section class="panel">
          <h2>What could lower your risk</h2>
          <p class="panel-sub">What the model estimates would change. Always follow your doctor's advice.</p>
          ${helpful ? `<ul class="plain-list">${helpful}</ul>` : `<p class="panel-text">Ask your doctor what would help most in your case.</p>`}
        </section>
        <section class="panel">
          <h2>Questions to ask your doctor</h2>
          <ul class="plain-list">
            <li>What does this result mean for me?</li>
            <li>Do I need any more tests?</li>
            <li>What should I change first?</li>
          </ul>
        </section>
      </div>
    </div>`;
  $("btn-open-twin-inline").onclick = () => $("btn-open-twin").click();
  showView("view-report");
}

async function openReport(patientId) {
  currentPatientId = patientId;
  currentReport = await api(`/api/patients/${patientId}/report`);
  renderReport(currentReport);
}

async function openMyReport() {
  renderReport(await api("/api/me/report"));
}

// ---------------------------------------------------------------- Compare visits (clinicians)
// Any two visits side by side: the risk at each and how it moved, then every
// recorded detail with the changed ones highlighted. Defaults to the previous
// visit against the current one.
const NUMERIC_UNITS = { age: "years", trestbps: "mm Hg", chol: "mg/dl", thalach: "beats/min", oldpeak: "mm" };

function compareValue(key, value) {
  return value === null || value === undefined ? "Not done" : fieldValue(key, value);
}

function fieldChange(key, from, to) {
  if (from === to || (from == null && to == null)) return "";
  if (key in NUMERIC_UNITS && from != null && to != null) {
    const d = Math.round((to - from) * 10) / 10;
    const shown = key === "oldpeak" ? Math.abs(d).toFixed(1) : Math.abs(d);
    return `${d > 0 ? "▲ +" : "▼ −"}${shown} ${NUMERIC_UNITS[key]}`;
  }
  return "Changed";
}

function visitOption(v) {
  return `Visit ${v.number} · ${formatDateTime(v.created_at)} · ${v.risk_percent}%`;
}

function openCompare(fromId, toId) {
  const visits = currentReport.visits;
  const options = visits.map((v) => `<option value="${v.id}">${escapeHtml(visitOption(v))}</option>`).join("");
  $("compare-from").innerHTML = options;
  $("compare-to").innerHTML = options;
  $("compare-from").value = String(fromId);
  $("compare-to").value = String(toId);
  $("compare-title").textContent = `Compare visits — ${currentReport.name}`;
  renderCompare();
  showView("view-compare");
}

function renderCompare() {
  const byId = Object.fromEntries(currentReport.visits.map((v) => [String(v.id), v]));
  const a = byId[$("compare-from").value];
  const b = byId[$("compare-to").value];
  const body = $("compare-body");
  if (a.id === b.id) {
    body.innerHTML = `<div class="panel"><p class="panel-text">Choose two different visits to compare.</p></div>`;
    return;
  }
  const keys = [...FIELD_ORDER, ...OPTIONAL_FIELDS];
  let changedCount = 0;
  const rows = keys.map((key) => {
    const change = fieldChange(key, a.inputs[key], b.inputs[key]);
    if (change) changedCount += 1;
    return `
      <tr class="${change ? "changed" : ""}">
        <td>${escapeHtml(fieldLabel(key))}<small>${escapeHtml(FIELDS[key].clinical)}</small></td>
        <td>${escapeHtml(compareValue(key, a.inputs[key]))}</td>
        <td>${escapeHtml(compareValue(key, b.inputs[key]))}</td>
        <td class="change-cell">${escapeHtml(change) || '<span class="muted">Same</span>'}</td>
      </tr>`;
  }).join("");
  const card = (v) => `
    <div class="compare-card">
      <div class="step-label">Visit ${v.number}</div>
      <div class="compare-date">${escapeHtml(formatDateTime(v.created_at))}</div>
      <div class="risk-figure tone-${BAND_TONE[v.band]}">${v.risk_percent}%</div>
      <div class="risk-caption">Chance of heart disease</div>
      <div class="verdict verdict-${BAND_TONE[v.band]}">${escapeHtml(v.verdict)}</div>
      <div class="compare-severity">Most likely outcome: <b>${escapeHtml(v.severity)}</b></div>
      ${v.note ? `<p class="visit-note">${escapeHtml(v.note)}</p>` : ""}
    </div>`;
  const earlier = a.created_at <= b.created_at;
  const bandMoved = a.band !== b.band ? ` The reading moved from <b>${escapeHtml(a.verdict)}</b> to <b>${escapeHtml(b.verdict)}</b>.` : "";
  body.innerHTML = `
    <section class="panel">
      <div class="compare-cards">
        ${card(a)}
        <div class="compare-arrow">
          <div class="risk-caption">Change in risk</div>
          <div class="compare-delta">${riskDelta(a.risk_percent, b.risk_percent)}</div>
          <div class="muted">${a.risk_percent}% &rarr; ${b.risk_percent}%</div>
        </div>
        ${card(b)}
      </div>
      <p class="summary-note">${changedCount === 0 ? "None of the recorded details differ between these visits." : `${changedCount} of the recorded details changed.`}${bandMoved}${earlier ? "" : " The &ldquo;earlier&rdquo; visit chosen is actually the later one, so changes read backwards in time."} Each risk is the model's estimate from that visit's values; points are percentage points.</p>
    </section>
    <section class="panel">
      <h2>Details recorded at each visit</h2>
      <p class="panel-sub">Highlighted rows changed between visit ${a.number} and visit ${b.number}.</p>
      <div class="table-scroll"><table class="kv-table compare-table">
        <thead><tr><th>Detail</th><th>Visit ${a.number}</th><th>Visit ${b.number}</th><th>Change</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
    </section>`;
}

$("compare-from").onchange = renderCompare;
$("compare-to").onchange = renderCompare;
$("btn-back-compare").onclick = () => showView("view-report");

// ---------------------------------------------------------------- Digital twin
// The 3D heart twin (heart_digital_twin_3d/), served from this origin at
// /twin/ so it reuses the stored login token. ?self=1 gives the patient view.
$("btn-open-twin").onclick = () => {
  const src = isDoctor() ? `/twin/?patient=${currentPatientId}` : "/twin/?self=1";
  $("twin-title").textContent = isDoctor() ? `Digital twin — ${$("report-name").textContent}` : "Your heart model";
  $("twin-sub").textContent = isDoctor()
    ? "Change the patient's details to see how the risk estimate and the heart respond. The saved record is not changed."
    : "A moving model of your heart, built from your results. Try a different blood pressure or blood sugar — your real results are not changed.";
  $("twin-frame").setAttribute("src", src);
  showView("view-twin");
};
// Multi-physics twin (static/heart_physics.html): electrical wave, muscle, pressure-volume
// loop and valve flow on the MRI heart mesh, in the same frame as the digital twin.
$("btn-open-physics").onclick = () => {
  $("twin-title").textContent = isDoctor() ? `Multi-physics heart — ${$("report-name").textContent}` : "Your heart: electrical, muscle and blood flow";
  $("twin-sub").textContent = "An illustrative model on an MRI heart shape, calibrated to the blood pressure and heart rate on record. Not a clinical simulation.";
  $("twin-frame").setAttribute("src", isDoctor() ? `/static/heart_physics.html?patient=${currentPatientId}` : "/static/heart_physics.html?self=1");
  showView("view-twin");
};
$("btn-back-twin").onclick = () => {
  // Unload the frame so its render loop stops while the report is shown.
  $("twin-frame").setAttribute("src", "about:blank");
  showView("view-report");
};

// ---------------------------------------------------------------- Plain-language summary
$("btn-ai-summary").onclick = async () => {
  showView("view-ai-summary");
  const body = $("ai-summary-body");
  body.textContent = "Generating…";
  body.classList.remove("ai-summary-error");
  try {
    const path = isDoctor() ? `/api/patients/${currentPatientId}/narrate` : "/api/me/narrate";
    const data = await api(path, { method: "POST" });
    body.textContent = data.text;
    if (!data.available) body.classList.add("ai-summary-error");
  } catch (err) {
    body.textContent = "The summary could not be generated: " + err.message;
    body.classList.add("ai-summary-error");
  }
};
$("btn-back-ai-summary").onclick = () => showView("view-report");

// ---------------------------------------------------------------- ECG analysis
const ecgFiles = { hea: null, dat: null };

function updateEcgFilesDisplay() {
  const el = $("ecg-files");
  const submit = $("ecg-submit");
  if (ecgFiles.hea && ecgFiles.dat) {
    el.textContent = `${ecgFiles.hea.name} + ${ecgFiles.dat.name}`;
    submit.disabled = false;
  } else if (ecgFiles.hea || ecgFiles.dat) {
    el.textContent = `${(ecgFiles.hea || ecgFiles.dat).name} — a matching .hea and .dat pair is needed`;
    submit.disabled = true;
  } else {
    el.textContent = "No files selected";
    submit.disabled = true;
  }
}

function assignEcgFiles(fileList) {
  Array.from(fileList).forEach((f) => {
    if (f.name.toLowerCase().endsWith(".hea")) ecgFiles.hea = f;
    else if (f.name.toLowerCase().endsWith(".dat")) ecgFiles.dat = f;
  });
  updateEcgFilesDisplay();
}

const ecgDropzone = $("ecg-dropzone");
const ecgFileInput = $("ecg-file-input");
ecgDropzone.onclick = () => ecgFileInput.click();
ecgDropzone.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); ecgFileInput.click(); } };
ecgFileInput.onchange = () => assignEcgFiles(ecgFileInput.files);
ecgDropzone.ondragover = (e) => { e.preventDefault(); ecgDropzone.classList.add("drag-over"); };
ecgDropzone.ondragleave = () => ecgDropzone.classList.remove("drag-over");
ecgDropzone.ondrop = (e) => {
  e.preventDefault();
  ecgDropzone.classList.remove("drag-over");
  assignEcgFiles(e.dataTransfer.files);
};

$("ecg-submit").onclick = async () => {
  const resultEl = $("ecg-result");
  resultEl.classList.remove("hidden");
  resultEl.innerHTML = `<div class="loading-row"><span class="spinner"></span> Analysing the recording…</div>`;
  try {
    const form = new FormData();
    form.append("hea", ecgFiles.hea);
    form.append("dat", ecgFiles.dat);
    const data = await api("/api/ecg-predict", { method: "POST", body: form });
    const label = (key) => ECG_CLASS_LABELS[key] ?? key;
    const bars = Object.entries(data.class_probabilities).map(([key, pct]) => `
      <div class="bar-row ${key === data.predicted_class ? "top" : ""}">
        <div class="bar-label">${escapeHtml(key)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <div class="bar-value">${pct}%</div>
      </div>`).join("");
    resultEl.innerHTML = `
      <div class="ecg-predicted">${escapeHtml(label(data.predicted_class))}</div>
      <p class="panel-sub">Model confidence ${data.confidence_percent}%. Review alongside the full ECG.</p>
      ${bars}`;
  } catch (err) {
    resultEl.innerHTML = `<p class="form-error">${escapeHtml(err.message)}</p>`;
  }
};

// ---------------------------------------------------------------- Start
(async function boot() {
  if (getToken()) {
    try { await enterApp(); return; } catch (e) { clearToken(); }
  }
  showView("view-login");
})();
