// CordiaAI -- multi-physics heart twin viewer. Draws the patient's LV myocardium mesh
// coloured by the electrical/muscle maps from /api/physics/simulate, beats it with the
// simulated volume curve, and plots the pressure-volume loop, pressures and valve flows.
// All physics run on the server (src/imaging/heart_physics.py); this file only draws.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/three/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const patientId = params.get("patient") ? Number(params.get("patient")) : null;
const selfRecord = params.get("self") === "1";

async function api(path, opts = {}) {
  const headers = opts.body instanceof FormData ? {} : { "Content-Type": "application/json" };
  const token = localStorage.getItem("cordia_token");
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(path, { ...opts, headers: { ...headers, ...(opts.headers || {}) } });
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json();
}

// ------------------------------------------------------------------ state
const state = {
  meshes: null, metrics: null, thickening: null, source: "",
  baseline: null, result: null, playing: true, t: 0, last: performance.now(), reqId: 0,
};

// ------------------------------------------------------------------ what-if controls
const SLIDERS = [
  ["heart_rate", "Heart rate", 40, 160, 1, null, (v) => `${v}`],
  ["contractility", "Contractility", 0.3, 1.5, 0.05, 1, (v) => `×${(+v).toFixed(2)}`],
  ["resistance", "Afterload (vascular resistance)", 0.5, 2, 0.05, 1, (v) => `×${(+v).toFixed(2)}`],
  ["arterial_stiffness", "Arterial stiffness", 0.5, 2.5, 0.05, 1, (v) => `×${(+v).toFixed(2)}`],
  ["filling_pressure", "Filling pressure (preload)", 0.5, 2, 0.05, 1, (v) => `×${(+v).toFixed(2)}`],
  ["diastolic_stiffness", "Muscle stiffness in diastole", 0.5, 3, 0.05, 1, (v) => `×${(+v).toFixed(2)}`],
  ["aortic_stenosis", "Aortic valve narrowing", 0, 1, 0.05, 0, (v) => `${Math.round(v * 100)}%`],
  ["conduction_velocity", "Conduction speed (fibrosis < 1)", 0.3, 1.2, 0.05, 1, (v) => `×${(+v).toFixed(2)}`],
];
const SCAR = [
  ["scar_angle", "Scar position around LV", -180, 180, 5, 180, (v) => `${v}°`],
  ["scar_level", "Scar height (apex → base)", 0.1, 0.9, 0.05, 0.5, (v) => `${Math.round(v * 100)}%`],
  ["scar_radius", "Scar size", 5, 30, 1, 18, (v) => `${v} mm`],
];

function sliderRow([key, label, min, max, step, def, fmt]) {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `<label for="w-${key}">${label}</label><input type="range" id="w-${key}" min="${min}" max="${max}" step="${step}"><output id="o-${key}"></output>`;
  const input = row.querySelector("input");
  input.value = def ?? min;
  input.dataset.def = def ?? "";
  const out = row.querySelector("output");
  const show = () => { out.textContent = fmt(input.value); };
  input.addEventListener("input", () => { show(); scheduleRun(); });
  show();
  return row;
}
SLIDERS.forEach((s) => $("sliders").appendChild(sliderRow(s)));
SCAR.forEach((s) => $("scar-sliders").appendChild(sliderRow(s)));
$("w-conduction").onchange = scheduleRun;
$("w-scar").onchange = scheduleRun;

function whatif() {
  const w = {};
  for (const [key, , , , , def] of SLIDERS) {
    const v = +$(`w-${key}`).value;
    if (key === "heart_rate") { if (state.baseline && v !== Math.round(state.baseline.hemodynamics.parameters.hr)) w.heart_rate = v; }
    else if (v !== def) w[key] = v;
  }
  if ($("w-conduction").value !== "normal") w.conduction = $("w-conduction").value;
  if ($("w-scar").checked) w.scar = { angle_deg: +$("w-scar_angle").value, level: +$("w-scar_level").value, radius_mm: +$("w-scar_radius").value };
  return w;
}

$("btn-reset").onclick = () => {
  for (const [key, , min, , , def] of [...SLIDERS, ...SCAR]) {
    const el = $(`w-${key}`);
    el.value = key === "heart_rate" && state.baseline ? Math.round(state.baseline.hemodynamics.parameters.hr) : (def ?? min);
    el.dispatchEvent(new Event("input"));
  }
  $("w-conduction").value = "normal";
  $("w-scar").checked = false;
  scheduleRun();
};

let timer = null;
function scheduleRun() { clearTimeout(timer); timer = setTimeout(run, 250); }

async function simulate(w) {
  return api("/api/physics/simulate", {
    method: "POST",
    body: JSON.stringify({ meshes: state.meshes, metrics: state.metrics, patient_id: patientId, self_record: selfRecord, whatif: w }),
  });
}

async function run() {
  const id = ++state.reqId;
  $("controls").classList.add("busy");
  try {
    const r = await simulate(whatif());
    if (id !== state.reqId) return; // a newer change superseded this one
    state.result = r;
    $("err").textContent = "";
    applyResult();
  } catch (e) {
    if (id === state.reqId) $("err").textContent = `Could not run the twin: ${e.message}`;
  } finally {
    if (id === state.reqId) $("controls").classList.remove("busy");
  }
}

// ------------------------------------------------------------------ colour maps
// Sequential = one hue light -> dark; diverging = blue <-> red through a neutral grey.
const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const DIV = ["#1c5cab", "#3987e5", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#a9302f"];
const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const SEQ_RGB = SEQ.map(hexToRgb), DIV_RGB = DIV.map(hexToRgb);
function ramp(stops, x) {
  x = Math.min(1, Math.max(0, x)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  return stops[i].map((c, k) => c + (stops[i + 1][k] - c) * f);
}
// Vertex colours are linear in three.js: convert from sRGB so the ramps look as designed.
const lin = (c) => c.map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
const NO_DATA = lin([0.55, 0.57, 0.6]);
const SCAR_RGB = lin(hexToRgb("#5b4636"));

const MODES = {
  activation: { label: "Activation time", unit: "ms", stops: SEQ_RGB, range: (r) => [0, Math.max(...r.electrical.activation_ms)],
    value: (r, i) => r.electrical.activation_ms[i], note: "Earliest = light, latest = dark. Eikonal-style graph model with rule-based fibres; LV only." },
  repolarisation: { label: "Repolarisation time", unit: "ms", stops: SEQ_RGB, range: (r) => minmax(r.electrical.repolarisation_ms.filter((v) => v >= 0)),
    value: (r, i) => r.electrical.repolarisation_ms[i], note: "Activation time plus an action potential duration that is longer at the endocardium." },
  thickness: { label: "Wall thickness", unit: "mm", stops: SEQ_RGB, range: () => [3, 15],
    value: (r, i) => r.muscle.wall_thickness_mm[i], note: "Measured on the end-diastolic mesh, endocardium to epicardium. 13 mm or more suggests hypertrophy." },
  thickening: { label: "Systolic wall thickening", unit: "%", stops: DIV_RGB, range: () => [-20, 80], mid: 30,
    value: (r, i) => (state.thickening ? state.thickening[i] : null),
    note: "Measured from the patient's ED and ES MRI. Blue = thickens less than ~30% (weak), red = more. Needs an MRI upload." },
  helix: { label: "Fibre helix angle", unit: "°", stops: DIV_RGB, range: () => [-60, 60],
    value: (r, i) => r.muscle.helix_angle_deg[i], note: "Assumed fibre direction (Streeter rule: +60° endocardium to −60° epicardium), not measured." },
  contracts: { label: "Contracting muscle", unit: "", stops: SEQ_RGB, range: () => [0, 1],
    value: (r, i) => r.muscle.contracts[i], note: "Scar (brown) neither conducts quickly nor contracts." },
  live: { label: "Electrical state", unit: "", note: "Red = depolarised (between activation and repolarisation), blue = at rest. Watch the wave spread from the septum." },
};
function minmax(a) { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; }

function updateLegend() {
  const mode = MODES[$("color-mode").value];
  $("mode-note").textContent = mode.note;
  if (!mode.stops) {
    $("legend").innerHTML = `<span style="display:inline-block;width:12px;height:12px;background:#e34948;border-radius:2px"></span> depolarised <span style="display:inline-block;width:12px;height:12px;background:#cde2fb;border-radius:2px;margin-left:8px"></span> resting`;
    return;
  }
  const [lo, hi] = mode.range(state.result);
  const css = (mode.stops === DIV_RGB ? DIV : SEQ).join(",");
  $("legend").innerHTML = mode.unit ? `<span>${lo.toFixed(0)} ${mode.unit}</span><span class="legend-ramp" style="background:linear-gradient(90deg,${css})"></span><span>${hi.toFixed(0)} ${mode.unit}</span><span>${mode.label}</span>`
    : `<span style="display:inline-block;width:12px;height:12px;background:#5b4636;border-radius:2px"></span> scar <span style="display:inline-block;width:12px;height:12px;background:#0d366b;border-radius:2px;margin-left:8px"></span> contracting`;
}

// ------------------------------------------------------------------ 3D scene
const viewer = $("viewer");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
renderer.setClearColor(0xe9eef3);
viewer.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 1, 2000);
camera.position.set(105, 75, 125); // three-quarter view: septum and lateral wall both visible
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.3));
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(80, 120, 160);
scene.add(sun);
const heart = new THREE.Group();
scene.add(heart);

let myo = null, cavity = null, rv = null, fat = null, particles = null;
let geo = null; // per-vertex arrays used by the animation

function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);

function bufferGeometry(mesh) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(mesh.vertices.flat(), 3));
  g.setIndex(mesh.faces.flat());
  g.computeVertexNormals();
  return g;
}

function buildScene() {
  for (const o of [myo, cavity, rv, fat, particles]) if (o) { heart.remove(o); o.geometry.dispose(); }
  const r = state.result;
  const axis = new THREE.Vector3(...r.muscle.axis), centre = new THREE.Vector3(...r.muscle.centre);
  // Put the long axis upright (base at the top) and the heart's centre at the origin.
  heart.quaternion.setFromUnitVectors(axis, new THREE.Vector3(0, 1, 0));
  heart.position.copy(centre.clone().applyQuaternion(heart.quaternion).multiplyScalar(-1));

  const myoGeom = bufferGeometry(state.meshes ? state.meshes["LV myocardium"] : sampleMeshes["LV myocardium"]);
  myoGeom.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(myoGeom.attributes.position.count * 3), 3));
  myo = new THREE.Mesh(myoGeom, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.65, side: THREE.DoubleSide }));
  heart.add(myo);

  const meshes = state.meshes || sampleMeshes;
  cavity = new THREE.Mesh(bufferGeometry(meshes["LV cavity"]), new THREE.MeshStandardMaterial({ color: 0xc0392b, transparent: true, opacity: 0.25, depthWrite: false }));
  heart.add(cavity);
  if (meshes["RV cavity"]) {
    rv = new THREE.Mesh(bufferGeometry(meshes["RV cavity"]), new THREE.MeshStandardMaterial({ color: 0x5b7fa8, transparent: true, opacity: 0.3, depthWrite: false }));
    heart.add(rv);
  }

  // Per-vertex geometry for the beat: radius from the axis, endocardial radius, transmural position.
  const P = myoGeom.attributes.position.array.slice();
  const n = P.length / 3, rad = r.muscle.radial, tm = r.muscle.transmural, th = r.muscle.wall_thickness_mm;
  const R = new Float32Array(n), Rendo = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = new THREE.Vector3(P[3 * i], P[3 * i + 1], P[3 * i + 2]).sub(centre);
    const rr = d.sub(axis.clone().multiplyScalar(d.dot(axis))).length();
    R[i] = rr; Rendo[i] = Math.max(1, rr - tm[i] * th[i]);
  }
  const C = cavity.geometry.attributes.position.array.slice();
  geo = { P, R, Rendo, rad, axis, centre, C, n };

  // Illustrative fat: an outer layer on the epicardial faces, thicker towards the base (AV groove).
  const faces = (state.meshes || sampleMeshes)["LV myocardium"].faces.filter(([a, b, c]) => tm[a] > 0.99 && tm[b] > 0.99 && tm[c] > 0.99);
  const fatGeom = new THREE.BufferGeometry();
  fatGeom.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(P.length), 3));
  fatGeom.setIndex(faces.flat());
  fat = new THREE.Mesh(fatGeom, new THREE.MeshStandardMaterial({ color: 0xeda100, transparent: true, opacity: 0.45, depthWrite: false, side: THREE.DoubleSide }));
  heart.add(fat);

  // Blood particles inside the LV cavity (cylindrical coordinates about the long axis).
  buildParticles();
  updateVisibility();
}

function buildParticles() {
  const C = geo.C, n = C.length / 3, bins = 12;
  let hmin = Infinity, hmax = -Infinity;
  const hs = new Float32Array(n), rs = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = new THREE.Vector3(C[3 * i], C[3 * i + 1], C[3 * i + 2]).sub(geo.centre);
    hs[i] = d.dot(geo.axis); rs[i] = d.sub(geo.axis.clone().multiplyScalar(hs[i])).length();
    hmin = Math.min(hmin, hs[i]); hmax = Math.max(hmax, hs[i]);
  }
  const radius = new Float32Array(bins);
  for (let i = 0; i < n; i++) { const b = Math.min(bins - 1, Math.floor((hs[i] - hmin) / (hmax - hmin) * bins)); radius[b] = Math.max(radius[b], rs[i]); }
  const N = 450;
  const pts = { h: new Float32Array(N), phi: new Float32Array(N), rho: new Float32Array(N) };
  for (let k = 0; k < N; k++) { pts.h[k] = Math.random(); pts.phi[k] = Math.random() * 2 * Math.PI; pts.rho[k] = Math.sqrt(Math.random()) * 0.7; }
  const u = new THREE.Vector3().crossVectors(geo.axis, Math.abs(geo.axis.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)).normalize();
  const v = new THREE.Vector3().crossVectors(geo.axis, u);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(N * 3), 3));
  particles = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xff5a4f, size: 1.8 }));
  particles.userData = { pts, radius, hmin, hmax, u, v, bins };
  heart.add(particles);
}

function updateVisibility() {
  if (particles) particles.visible = $("show-flow").checked;
  if (rv) rv.visible = $("show-rv").checked;
  if (fat) fat.visible = $("show-fat").checked;
}
["show-flow", "show-rv", "show-fat"].forEach((id) => { $(id).onchange = updateVisibility; });
$("fat-mm").oninput = () => { $("fat-mm-out").textContent = `${$("fat-mm").value} mm`; };

function paintStatic() {
  const r = state.result, key = $("color-mode").value, mode = MODES[key];
  if (!myo || !mode.stops) return;
  const col = myo.geometry.attributes.color.array, [lo, hi] = mode.range(r);
  for (let i = 0; i < geo.n; i++) {
    const v = mode.value(r, i);
    let c;
    if (key !== "contracts" && !r.muscle.contracts[i]) c = SCAR_RGB;
    else if (key === "contracts") c = v ? lin(SEQ_RGB[5]) : SCAR_RGB;
    else if (v === null || v === undefined || v < 0 && key !== "thickening" && key !== "helix") c = NO_DATA;
    else c = lin(ramp(mode.stops, (v - lo) / (hi - lo || 1)));
    col.set(c, 3 * i);
  }
  myo.geometry.attributes.color.needsUpdate = true;
}
$("color-mode").onchange = () => { updateLegend(); paintStatic(); };

// ------------------------------------------------------------------ the beat
function twitch(tau, rrMs) {
  const tp = 270 * Math.sqrt(rrMs / 1000), tr = tp / 2;
  if (tau < 0) return 0;
  if (tau < tp) return 0.5 * (1 - Math.cos(Math.PI * tau / tp));
  if (tau < tp + tr) return 0.5 * (1 + Math.cos(Math.PI * (tau - tp) / tr));
  return 0;
}
function sampleCurve(arr, t, dtMs) {
  const x = t / dtMs, i = Math.floor(x) % arr.length, j = (i + 1) % arr.length, f = x - Math.floor(x);
  return arr[i] + (arr[j] - arr[i]) * f;
}

const LIVE_ON = lin(hexToRgb("#e34948")), LIVE_OFF = lin(hexToRgb("#cde2fb"));
function animate(now) {
  requestAnimationFrame(animate);
  controls.update();
  const r = state.result;
  if (r && myo) {
    const h = r.hemodynamics, rrMs = 60000 / h.parameters.hr, dtMs = h.dt_ms;
    if (state.playing) state.t = (state.t + (now - state.last) * +$("speed").value) % rrMs;
    state.last = now;
    const t = state.t, at = r.electrical.activation_ms, rt = r.electrical.repolarisation_ms, ok = r.muscle.contracts;
    const V = sampleCurve(h.curves.v_lv, t, dtMs), EDV = h.summary.edv_ml;
    const s = Math.sqrt(Math.max(0.05, V / EDV)); // in-plane cavity scale
    const eg = sampleCurve(h.activation_curve, t, dtMs);
    const pos = myo.geometry.attributes.position.array, col = myo.geometry.attributes.color.array;
    const live = $("color-mode").value === "live";
    for (let i = 0; i < geo.n; i++) {
      // Local squeeze follows the local twitch; scar is passive and bulges slightly outward.
      let m = 1;
      if (eg > 0.05) m = ok[i] ? Math.min(2.5, twitch(t - at[i], rrMs) / eg) : -0.3;
      const re = geo.Rendo[i] * (1 - (1 - s) * m);
      const rn = Math.sqrt(Math.max(0, re * re + geo.R[i] * geo.R[i] - geo.Rendo[i] * geo.Rendo[i]));
      const d = rn - geo.R[i];
      for (let k = 0; k < 3; k++) pos[3 * i + k] = geo.P[3 * i + k] + geo.rad[i][k] * d;
      if (live) {
        const on = ok[i] && at[i] >= 0 && t >= at[i] && t < rt[i];
        col.set(!ok[i] ? SCAR_RGB : on ? LIVE_ON : LIVE_OFF, 3 * i);
      }
    }
    myo.geometry.attributes.position.needsUpdate = true;
    myo.geometry.computeVertexNormals();
    if (live) myo.geometry.attributes.color.needsUpdate = true;

    // Cavity: scaled in-plane about the long axis by the same factor.
    const cp = cavity.geometry.attributes.position.array, C = geo.C, A = geo.axis, O = geo.centre;
    for (let i = 0; i < C.length; i += 3) {
      const dx = C[i] - O.x, dy = C[i + 1] - O.y, dz = C[i + 2] - O.z, hh = dx * A.x + dy * A.y + dz * A.z;
      cp[i] = O.x + A.x * hh + (dx - A.x * hh) * s; cp[i + 1] = O.y + A.y * hh + (dy - A.y * hh) * s; cp[i + 2] = O.z + A.z * hh + (dz - A.z * hh) * s;
    }
    cavity.geometry.attributes.position.needsUpdate = true;

    if (fat.visible) {
      const fp = fat.geometry.attributes.position.array, mm = +$("fat-mm").value, lvl = r.muscle.level;
      for (let i = 0; i < geo.n; i++) {
        const off = mm * (0.3 + 0.7 * lvl[i] * lvl[i]);
        for (let k = 0; k < 3; k++) fp[3 * i + k] = pos[3 * i + k] + geo.rad[i][k] * off;
      }
      fat.geometry.attributes.position.needsUpdate = true;
      fat.geometry.computeVertexNormals();
    }
    if (particles.visible) moveParticles(h, t, dtMs, s, (now - state.lastP || 16) * +$("speed").value);
    state.lastP = now;
    $("time-readout").textContent = `t = ${t.toFixed(0)} ms of ${rrMs.toFixed(0)} ms · LV ${V.toFixed(0)} ml · ${sampleCurve(h.curves.p_lv, t, dtMs).toFixed(0)} mmHg`;
    drawCharts();
  }
  renderer.render(scene, camera);
}

function moveParticles(h, t, dtMs, s, stepMs) {
  const { pts, radius, hmin, hmax, u, v, bins } = particles.userData;
  const qav = sampleCurve(h.curves.q_av, t, dtMs), qmv = sampleCurve(h.curves.q_mv, t, dtMs);
  const pos = particles.geometry.attributes.position.array, H = hmax - hmin, A = geo.axis, O = geo.centre;
  for (let k = 0; k < pts.h.length; k++) {
    const b = Math.min(bins - 1, Math.floor(pts.h[k] * bins)), R = Math.max(3, radius[b]) * s;
    // Axial speed = net valve flow / cross-section (ml/s -> mm^3/s over mm^2). Filling runs
    // base -> apex, ejection apex -> base; a slow swirl during filling hints at the diastolic vortex.
    const area = Math.PI * R * R, w = (qav - qmv) * 1000 / area; // mm/s, + towards base
    pts.h[k] += (w * stepMs / 1000) / H * (0.6 + 0.8 * pts.rho[k]);
    pts.phi[k] += qmv * 0.00002 * stepMs;
    if (pts.h[k] > 1) pts.h[k] = 0.02 + Math.random() * 0.1;
    if (pts.h[k] < 0) pts.h[k] = 0.9 + Math.random() * 0.1;
    const hh = hmin + pts.h[k] * H, rr = pts.rho[k] * R, c = Math.cos(pts.phi[k]), sn = Math.sin(pts.phi[k]);
    pos[3 * k] = O.x + A.x * hh + (u.x * c + v.x * sn) * rr;
    pos[3 * k + 1] = O.y + A.y * hh + (u.y * c + v.y * sn) * rr;
    pos[3 * k + 2] = O.z + A.z * hh + (u.z * c + v.z * sn) * rr;
  }
  particles.geometry.attributes.position.needsUpdate = true;
}

$("btn-play").onclick = () => { state.playing = !state.playing; $("btn-play").textContent = state.playing ? "Pause" : "Play"; };

// ------------------------------------------------------------------ charts
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function setupCanvas(cv) {
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
function niceTicks(lo, hi, n = 5) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0 || 1));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
const PAD = { l: 44, r: 12, t: 8, b: 30 };
function axes(ctx, w, h, x0, x1, y0, y1, xl, yl) {
  const X = (x) => PAD.l + (x - x0) / (x1 - x0) * (w - PAD.l - PAD.r), Y = (y) => h - PAD.b - (y - y0) / (y1 - y0) * (h - PAD.t - PAD.b);
  ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--text-dim"); ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (const v of niceTicks(y0, y1)) { ctx.beginPath(); ctx.moveTo(PAD.l, Y(v)); ctx.lineTo(w - PAD.r, Y(v)); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(v, PAD.l - 5, Y(v) + 4); }
  ctx.textAlign = "center";
  for (const v of niceTicks(x0, x1)) ctx.fillText(v, X(v), h - PAD.b + 14);
  ctx.fillText(xl, (PAD.l + w - PAD.r) / 2, h - 3);
  ctx.save(); ctx.translate(11, (h - PAD.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yl, 0, 0); ctx.restore();
  return { X, Y };
}
function line(ctx, xs, ys, X, Y, color, dash = []) {
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.lineJoin = "round";
  ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(ys[i])) : ctx.moveTo(X(x), Y(ys[i])))); ctx.stroke(); ctx.setLineDash([]);
}
function dot(ctx, x, y, color) { ctx.fillStyle = color; ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 7); ctx.fill(); ctx.stroke(); }

const hover = { pv: null, p: null, q: null };
function drawCharts() {
  const r = state.result, b = state.baseline;
  if (!r || !b) return;
  const c = r.hemodynamics.curves, cb = b.hemodynamics.curves, dtMs = r.hemodynamics.dt_ms, t = state.t;
  const i = Math.floor(t / dtMs) % c.v_lv.length;
  { // PV loop
    const { ctx, w, h } = setupCanvas($("c-pv"));
    const [v0, v1] = minmax([...c.v_lv, ...cb.v_lv]), [p0, p1] = minmax([...c.p_lv, ...cb.p_lv]);
    const { X, Y } = axes(ctx, w, h, Math.max(0, v0 - 10), v1 + 10, 0, p1 + 10, "LV volume (ml)", "LV pressure (mmHg)");
    line(ctx, cb.v_lv, cb.p_lv, X, Y, css("--baseline"), [5, 4]);
    line(ctx, c.v_lv, c.p_lv, X, Y, css("--series-1"));
    dot(ctx, X(c.v_lv[i]), Y(c.p_lv[i]), css("--series-1"));
    hoverTip("pv", X, Y, (k) => [c.v_lv[k], c.p_lv[k]], c.v_lv.length, (k) => `t ${(k * dtMs).toFixed(0)} ms · ${c.v_lv[k].toFixed(0)} ml · ${c.p_lv[k].toFixed(0)} mmHg`, ctx, true);
  }
  for (const [id, keys, yl] of [["p", ["p_lv", "p_ao", "p_la"], "Pressure (mmHg)"], ["q", ["q_av", "q_mv"], "Flow (ml/s)"]]) {
    const { ctx, w, h } = setupCanvas($(`c-${id}`));
    const all = keys.flatMap((k) => c[k]), [lo, hi] = minmax(all);
    const { X, Y } = axes(ctx, w, h, 0, c.t_ms[c.t_ms.length - 1], Math.min(0, lo), hi * 1.08, "Time in beat (ms)", yl);
    const colors = [css("--series-1"), css("--series-2"), css("--series-3")];
    keys.forEach((k, j) => line(ctx, c.t_ms, c[k], X, Y, colors[j]));
    ctx.strokeStyle = css("--text-dim"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(t), PAD.t); ctx.lineTo(X(t), h - PAD.b); ctx.stroke();
    const names = { p_lv: "LV", p_ao: "Aorta", p_la: "LA", q_av: "Aortic", q_mv: "Mitral" };
    hoverTip(id, X, Y, (k) => [c.t_ms[k], c[keys[0]][k]], c.t_ms.length,
      (k) => `${c.t_ms[k].toFixed(0)} ms · ` + keys.map((kk) => `${names[kk]} ${c[kk][k].toFixed(0)}`).join(" · "), ctx, false);
  }
}
function hoverTip(id, X, Y, at, n, text, ctx, nearest) {
  const tip = $(`tip-${id}`), m = hover[id];
  if (!m) { tip.style.display = "none"; return; }
  let best = 0, bd = Infinity;
  for (let k = 0; k < n; k++) {
    const [x, y] = at(k), d = nearest ? (X(x) - m.x) ** 2 + (Y(y) - m.y) ** 2 : Math.abs(X(x) - m.x);
    if (d < bd) { bd = d; best = k; }
  }
  const [x, y] = at(best);
  if (!nearest) { ctx.strokeStyle = css("--text"); ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(x), PAD.t); ctx.lineTo(X(x), ctx.canvas.clientHeight - PAD.b); ctx.stroke(); ctx.setLineDash([]); }
  else dot(ctx, X(x), Y(y), css("--text"));
  tip.textContent = text(best);
  tip.style.display = "block";
  tip.style.left = `${Math.min(m.x + 12, ctx.canvas.clientWidth - tip.offsetWidth)}px`;
  tip.style.top = `${m.y + 30}px`;
}
for (const id of ["pv", "p", "q"]) {
  const cv = $(`c-${id}`);
  cv.addEventListener("pointermove", (e) => { const b = cv.getBoundingClientRect(); hover[id] = { x: e.clientX - b.left, y: e.clientY - b.top }; });
  cv.addEventListener("pointerleave", () => { hover[id] = null; });
}

// ------------------------------------------------------------------ panels
const CARDS = [
  ["ef_percent", "Ejection fraction", "%", 1], ["sv_ml", "Stroke volume", "ml", 1], ["edv_ml", "End-diastolic vol.", "ml", 0],
  ["esv_ml", "End-systolic vol.", "ml", 0], ["sbp_mmhg", "Systolic BP", "mmHg", 0], ["dbp_mmhg", "Diastolic BP", "mmHg", 0],
  ["cardiac_output_l_min", "Cardiac output", "L/min", 2], ["stroke_work_j", "Stroke work", "J", 2], ["dpdt_max_mmhg_s", "Peak dP/dt", "mmHg/s", 0],
  ["mitral_e_a_ratio", "Mitral E/A", "", 2], ["mean_la_pressure_mmhg", "Mean LA pressure", "mmHg", 1],
];
function renderPanels() {
  const r = state.result, b = state.baseline, s = r.hemodynamics.summary, sb = b.hemodynamics.summary;
  const card = (label, v, unit, d, dp) => `<div class="stat"><div class="k">${label}</div><div class="v">${v ?? "–"} <span style="font-size:12px;font-weight:400">${unit}</span></div><div class="d">${d}</div></div>`;
  const delta = (v, v0, dp) => (v == null || v0 == null) ? "" : Math.abs(v - v0) < 10 ** -dp / 2 ? "same" : `${v > v0 ? "▲" : "▼"} ${Math.abs(v - v0).toFixed(dp)}`;
  let html = CARDS.map(([k, label, unit, dp]) => card(label, s[k] == null ? null : (+s[k]).toFixed(dp), unit, delta(s[k], sb[k], dp))).join("");
  const e = r.electrical, eb = b.electrical;
  html += card("QRS estimate", e.qrs_estimate_ms, "ms", delta(e.qrs_estimate_ms, eb.qrs_estimate_ms, 0));
  html += card("LV activation", e.lv_total_activation_ms.toFixed(0), "ms", delta(e.lv_total_activation_ms, eb.lv_total_activation_ms, 0));
  $("cards").innerHTML = html;

  const tg = r.targets;
  const names = { edv_ml: "End-diastolic volume", esv_ml: "End-systolic volume", sbp_mmhg: "Systolic BP", dbp_mmhg: "Diastolic BP", hr_bpm: "Heart rate" };
  $("sources").innerHTML = "<tr><th>Value</th><th></th><th>Source</th></tr>" + Object.entries(tg).map(([k, v]) => `<tr><td>${names[k]}</td><td>${v.value}</td><td>${v.source}</td></tr>`).join("");
  $("fit").innerHTML = "<tr><th></th><th>Measured</th><th>Model</th></tr>" + Object.entries(r.hemodynamics.calibration_fit).map(([k, v]) => `<tr><td>${names[k]}</td><td>${v.target}</td><td>${v.model}</td></tr>`).join("");
  $("fat-text").textContent = r.fat.measured ? `Epicardial fat ${r.fat.eat_volume_ml} ml (${r.fat.method}).` : r.fat.reason;
  $("scope").innerHTML = r.out_of_scope.map((x) => `<li>${x}</li>`).join("");
  $("hp-note").textContent = r.note;
  $("hp-source").textContent = state.source + (patientId || selfRecord ? " · blood pressure and heart rate from the patient's record" : "");
}

function applyResult() {
  renderPanels();
  updateLegend();
  paintStatic();
}

// ------------------------------------------------------------------ loading
let sampleMeshes = null;
async function loadBaseline() {
  state.baseline = await simulate({});
  state.result = state.baseline;
  const hr = $("w-heart_rate");
  hr.value = Math.round(state.baseline.hemodynamics.parameters.hr);
  hr.dispatchEvent(new Event("input"));
  clearTimeout(timer); // that input event is not a what-if
  buildScene();
  applyResult();
  if (Object.keys(whatif()).length) run();
}

async function start() {
  resize();
  requestAnimationFrame(animate);
  try {
    const s = await api("/api/physics/sample");
    sampleMeshes = s.meshes;
    state.source = `${s.source} (${s.pid})`;
    await loadBaseline();
  } catch (e) {
    $("hp-source").textContent = "";
    $("err").textContent = `Could not load the heart: ${e.message}`;
  }
}

$("btn-upload").onclick = async () => {
  const ed = $("up-ed").files[0], es = $("up-es").files[0];
  if (!ed || !es) { $("up-status").textContent = "Choose both stacks first."; return; }
  const fd = new FormData();
  fd.append("ed", ed); fd.append("es", es);
  if ($("up-h").value) fd.append("height_cm", $("up-h").value);
  if ($("up-w").value) fd.append("weight_kg", $("up-w").value);
  $("up-status").textContent = "Segmenting (can take a minute on a CPU)…";
  try {
    const m = await api("/api/imaging/mri", { method: "POST", body: fd });
    state.meshes = m.meshes; state.metrics = m.metrics;
    state.thickening = m.regional && m.regional.thickening_at_vertices;
    state.source = `Patient MRI: ${ed.name} / ${es.name}`;
    await loadBaseline();
    $("color-mode").value = "thickening";
    updateLegend(); paintStatic();
    $("up-status").textContent = `Done: EF ${m.metrics.lv_ef_percent.toFixed(0)}% measured.`;
  } catch (e) {
    $("up-status").textContent = `Failed: ${e.message}`;
  }
};

start();
