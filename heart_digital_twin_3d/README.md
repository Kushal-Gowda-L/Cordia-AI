# CordiaAI — Digital Twin of a Heart (interactive 3D)

An educational, scientifically-inspired **interactive 3D cardiovascular digital twin**, built with
React, Three.js (via React Three Fiber + drei), TypeScript, and Recharts. Not a diagnostic or
clinical medical device — this is stated inside the app itself, not just here.

## Run it locally

Needs Node.js 18 or newer. From the repository root:

```bash
cd heart_digital_twin_3d
npm install
npm run dev
```

Then open the URL Vite prints (default `http://localhost:5173`). It renders with WebGL, so it runs
on your graphics card in any current browser; no other setup is needed.

**On its own (no backend).** Opened without a patient and with no CordiaAI backend running, the
twin switches to an **offline estimate** instead of failing, and says so in the left panel. Add
`?offline=1` to the URL to force it even when a backend is up. The offline estimate
([`src/patient/offlineTwin.ts`](src/patient/offlineTwin.ts)) is:

- a multinomial logistic regression on the same ten inputs and five severity classes as the
  server's model, fitted on `datasets/uci/heart_disease_raw.csv` (all 920 rows; a cholesterol or BP
  of 0 treated as missing and median-filled). 5-fold cross-validated: disease vs. no disease
  AUC 0.88, accuracy 79%; five-class accuracy 56%. A lighter stand-in, not the production model.
  Unlike the server's model it raises risk with cholesterol, so the cholesterol caution is hidden
  in offline mode.
- a line-for-line port of [`src/patient_twin.py`](../src/patient_twin.py) and
  [`src/risk_reading.py`](../src/risk_reading.py), so the heart is set up exactly as the server
  would set it up.

**With the backend.** Start the FastAPI app on port 8000 (see the main README); `npm run dev`
proxies `/api` to it. The what-if endpoint needs a signed-in user, so open the twin from a
patient's report in the web app, or sign in there first.

## Use it with any dataset

The left panel's **Drive the heart from** switch has two sources: the CordiaAI model's ten inputs
(above), or **Your dataset (CSV)**. Dataset mode is for any team member's data; nothing is tied to
one schema, and nothing leaves the browser.

1. **Load** a CSV (choose or drop a file), or try the bundled UCI or Framingham samples. The reader
   ([`src/dataset/csv.ts`](src/dataset/csv.ts)) detects the delimiter (`,` `;` tab `|`), a missing
   header row, and each column's kind, and turns common encodings into numbers: Yes/No and
   Male/Female columns, age bands such as `55-59`, and a combined blood pressure such as `158/88`
   (split into systolic and diastolic). Files over 60,000 rows are evenly sampled.
2. **Outcome to predict**: guessed from the column name (`target`, `num`, `cardio`,
   `HeartDisease`, `TenYearCHD`, `DEATH_EVENT` ...), changeable. A logistic regression is trained on
   every other usable column right in the browser ([`src/dataset/riskModel.ts`](src/dataset/riskModel.ts)),
   checked on a 20% hold-out, and its AUC and accuracy are shown. Choose "None" to drive the heart
   from the values alone.
3. **Column matches**: columns are matched by name to the twelve things the heart uses (age, sex,
   resting and peak heart rate, systolic and diastolic BP, cholesterol, blood sugar, ejection
   fraction, oxygen saturation, exercise angina, ST depression), and each match can be changed.
   Units are read per column: age in days or as a survey age-group code, a yes/no "high BP" flag,
   cholesterol as 1-3 levels or mmol/L, fractions vs. percentages; each reading is stated.
   ([`src/dataset/mapping.ts`](src/dataset/mapping.ts))
4. **Pick a row** (number, next/previous, random, or random with/without the outcome). The heart
   is set up from that row with the same rules as `src/patient_twin.py`; an ejection fraction
   column, when present, sets pumping strength directly, otherwise the model's risk lowers it.
5. **Change the row's values**: every value the model uses is a control (slider, yes/no, or
   category list). Each change re-runs the model and the heart at once; the cards show the change
   against the row as recorded, and the main heart inputs are marked with a heart.

Checked against every dataset in the project:

| Dataset | Rows | Outcome found | Heart inputs matched | Held-out AUC |
|---|---|---|---|---|
| UCI heart disease (920, 4 hospitals) | 920 | `num` | 8 | 0.90 |
| Kaggle `heart.csv` / Heart Attack Data Set | 1,025 / 303 | `target` | 8 | 0.92 / 0.94 |
| Framingham | 4,240 | `TenYearCHD` | 7 | 0.74 |
| cardio_train (`;`, age in days) | 70,000 (60k sample) | `cardio` | 6 | 0.76 |
| BRFSS 2015 indicators | 253,680 (60k sample) | `HeartDiseaseorAttack` | 5 | 0.85 |
| CDC 2020 survey (Yes/No, age bands) | 319,795 (60k sample) | `HeartDisease` | 3 | 0.84 |
| heart_attack_prediction (`158/88` BP) | 8,763 | `Heart Attack Risk` | 7 | 0.51 |
| ECG beats (no header, 140 samples) | 4,998 | last column | 0 | 0.99 |

The 0.51 on heart_attack_prediction is the data, not the reader: its outcome is close to random
with respect to its columns. The ECG file has no heart inputs to match, so the heart stays at
typical values and only the risk (and so pumping strength) follows the model.

## Valve conditions

**Narrowed aortic valve** (aortic stenosis) and **Leaky mitral valve** (mitral regurgitation) sliders
add valve disease on top of either source. Both now act in the simulation itself, not just on the
picture:

- Stenosis raises the resistance across the aortic valve with the square of the narrowing, so the
  left pump has to build pressure well above the aorta to empty. At rest (HR 72, BP 120/80): mild
  (0.3) peak LV 111 mmHg; moderate (0.6) 137 mmHg, EF 50%; severe (1) 167 mmHg against an aortic
  pressure near 100, EF 41%, cardiac output 5.8 -> 4.4 L/min. The aortic leaflets open less and turn
  a calcified yellow-grey.
- Regurgitation leaks blood back into the left atrium while the LV squeezes, drawn as an orange jet
  through the mitral valve, whose leaflets no longer close fully. The atrium's pressure and size
  rise with a systolic v wave, the LV dilates (EDV 150 -> 168 mL at severe), EF looks high (69%)
  because it counts the leaked blood, and forward output falls (stroke volume 81 -> 74 mL). This is
  the textbook pattern of chronic mitral regurgitation.
- Valve leaflets swing open and shut over a few tens of milliseconds instead of snapping, so the
  valves view shows real opening and closing motion.

## Direct link (no install)

`npm run build:standalone` builds `dist-standalone/`: the whole twin with relative paths and no
backend (offline model, dataset mode, samples), which works from any static host. That build is
what the shared link in the project thread serves.

## Live heart (real time)

Every view follows the data shown: change an input or pick / edit a dataset row and the risk model,
heart physiology, beating 3D hearts (all views), valves, chambers, blood flow, ECG, pressure trace,
cards and the viewer's read-outs update together. The **Live heart** panel
([`LiveControls.tsx`](src/components/Dashboard/LiveControls.tsx)) starts at the data's heart rate,
systolic pressure and pumping strength and moves the beating heart instantly; a later data edit takes
over for any value it sets, and *Back to the data values* drops the hand changes. The simulation runs
in real time at any frame rate (long frames are integrated in small steps), and a heart-rate change
speeds up or slows the beat smoothly from where it is. In the holographic views the coronary arteries
and mesh nodes take the predicted risk's colour (green, amber, red).

## Cordia AI dashboard (`cordia.html`)

A fourth page, [`cordia.html`](cordia.html) ([`src/cordia/`](src/cordia/)), is the Cordia AI dashboard:
the project's sidebar (Overview, Digital Twin Heart, Patient Monitoring, Risk Assessment, Dataset &
Model, Heart Simulation, Patient History, Preventive Care, About Project), the heart in the middle
and the live rail on the right (Live Parameters with heart rate, blood pressure, SpO2 and breathing
rate, a simulation status, the ECG and the Simulation Controls).

The heart is the realistic twin in an illustrated look (`RealisticHeart look="illustrated"`): glossy
red muscle, blue venae cavae and pulmonary arteries, a violet right atrium and golden coronary
arteries, on a glowing platform. It turns 360°, beats with the simulation, and carries leader-line
labels (aorta, superior vena cava, pulmonary artery, atria, ventricles, coronary artery) that follow
each structure as it turns ([`ShowcaseHeart.tsx`](src/cordia/ShowcaseHeart.tsx)). Rotate, Zoom,
Reset View, Toggle Labels and the Front / Left / Right / Back thumbnails (`public/cordia/`) move the
camera. Heart rate and blood pressure are the patient's baseline (shared with the risk pages); the
activity state and intensity add exercise on top, and SpO2 sets the blood oxygen. The other pages
reuse the workstation's risk models and team-dataset loader, so any CSV row becomes the patient here
too.

## Heart model on its own (`heart.html`)

[`heart.html`](heart.html) ([`src/heartpage/`](src/heartpage/)) shows just the heart, in the
`reference` look matched to the Cordia dashboard render: its own great vessels
(`REFERENCE_VESSELS`: a tall red aortic arch with three branches, the blue pulmonary trunk crossing in
front and ending in an open cut end, the superior vena cava behind, the inferior vena cava running
down beneath the heart, open pulmonary vein stubs), a larger maroon right atrium, warm fat across the
front and broad golden coronary arteries. It turns 360° on the platform with the dashboard's labels,
tools and Front / Left / Right / Back views (`public/heart/`); *Look Inside* cuts the front away to
show the four valves opening and closing and the chambers filling, with the labels switching to the
valves. The heart-rate chip sets how fast it beats.

## Cordia Heart Twin workstation (`workstation.html`)

A third page, [`workstation.html`](workstation.html), is the Cordia Heart Twin workstation (patient
inputs | live heart | model read-outs) with this 3D twin as its heart
([`src/workstation/`](src/workstation/)). It keeps the workstation's layout, wording and behaviour:
the ~30 patient fields in their groups, *Apply healthy targets* / *Reset to patient*, the four browser
risk models (UCI angiography, Framingham 10-year CHD, the 70k cardiovascular exams, CDC BRFSS) as
cards with the change against the loaded patient, the main drivers, the 10-year trajectory, the
time machine (*Play 10 years*), cut-away depth, wall opacity, animation speed, structure toggles and
the Anterior / Posterior / Left lateral / 4-chamber views
([`models.ts`](src/workstation/models.ts) is a direct port of its models).

The centre is the twin: chips switch between Hologram, Low-poly, Realistic, Red + blue, Inside,
Valves, Electrical and Risk map. The patient's heart rate, blood pressure, age, diabetes and resting
ECG set the simulated heart; its pumping strength is tuned until the simulated EF settles on the
patient's EF, the ectopic-beat % makes that share of beats come early, and the coronary colour and
risk map follow the highest model risk. The HUD shows the workstation's line (rate, BP, EF, visual
stenosis, years) plus the twin's live EF, output and LV volume.

**Team dataset**: *Load CSV…* (or the UCI / Framingham samples) reads any heart CSV in the browser.
Each patient field is matched to a column by name, with the usual spellings of the UCI, Framingham,
cardio-70k, BRFSS and Kaggle heart files, and units / codings are converted (age in days, mmol/L,
0-based Kaggle codes, text such as `ATA`, `Flat`, `M`/`F`, BMI to weight). Matched fields are tagged
*data*; ◀ ▶, a row number or *Random* picks the patient, and fields the file lacks keep the example
patient's values ([`wsData.ts`](src/workstation/wsData.ts)).

## Holographic twin (`digital.html`)

A second page, [`digital.html`](digital.html), is the same twin (same panels, example patient,
**Your dataset (CSV)** mode, valve conditions, ECG and pressure charts) with a dark holographic theme
and three extra views, built on the realistic heart's anatomy
([`DigitalHeart.tsx`](src/components/Heart3D/DigitalHeart.tsx)):

- **Hologram**: a glowing see-through blue shell traced with fine muscle fibres; coronary arteries in
  red with a pulse wave running down them each beat, cardiac veins in cyan; aorta glowing red with
  the pressure.
- **Low-poly mesh**: the heart as a faceted blue mesh with its wireframe and glowing nodes.
- **Red + blue**: the realistic heart and its holographic twin side by side, each turning in place.

All three show the four valves inside (cusps open and shut with the simulation, the ring flashes as
a valve closes, a narrowed aortic valve turns amber, a leaky mitral valve stays apart and turns red),
blood flowing through the great vessels at the simulated rates (red oxygen-rich, blue oxygen-poor),
and live read-outs (heart rate, pressure, EF, output, valve states) over the viewer. The page ends
with the design reference images (`public/references/`), each with a button to show that look.
In dev open `http://localhost:5173/digital.html`; both pages are in every build.

**Look inside** works in every view, including the model hearts (realistic, hologram, low-poly,
red + blue): *See-through wall* fades the heart's wall, and *Cut away front* slices it open from the
side facing you ([`shellCut.ts`](src/components/Heart3D/shellCut.ts)) while the valves inside stay
whole. The four chambers ([`ModelChambers.tsx`](src/components/Heart3D/ModelChambers.tsx)) sit inside
every model heart, right side blue and left side red, each filling and emptying with its own
simulated volume (glowing in the hologram, faceted in the low-poly mesh, glossy blood-filled
cavities in the realistic heart once you look inside). The valves ([`ModelValves.tsx`](src/components/Heart3D/ModelValves.tsx)) are cupped cusps on
a ring, with chordae and papillary muscles under the mitral and tricuspid valves that tighten as
they shut; a valve's ring glows briefly as it closes. The hologram also has a slow scan line, the
low-poly nodes swell with each beat, and each view grows in gently when chosen. Labels start off on
this page and are one click away (*Show: Labels*).

## Realistic heart and 360° rotation

The **Realistic heart** view shows the closed heart as an anatomical model on a dark grey studio
backdrop ([`RealisticHeart.tsx`](src/components/Heart3D/RealisticHeart.tsx)), styled on a sculpted
anatomy render: rich red ventricles with the muscle bundles showing as fine grooves winding toward
the apex, pale peach atria with flap-like scalloped auricles, a procedurally grown branching tree of
coronary arteries (dark red) and cardiac veins (pale grey) laid over the surface, and the great
vessels with open, hollow ends (aortic arch with its three branches, pulmonary trunk and arteries,
venae cavae, pulmonary veins). Colour is painted per vertex from 3D noise in
[`tissue.ts`](src/components/Heart3D/tissue.ts), so it looks right from every side. It still beats
with the simulation: the ventricles (with their vessels) squeeze and twist with the simulated
ventricular volume, the atria and auricles with atrial volume.

The **Rotate 360°** button (top right of the viewer, any view) turns the heart round continuously;
the Realistic view starts with it on. Dragging still turns it by hand in any direction.

## Looking inside the heart

Under the heart, **Look inside** has two sliders that work together with the view buttons:

- **See-through wall** (Outside and Risk views) fades the outer heart muscle from solid to
  glass-like, revealing the four chambers, the valves opening and closing, the papillary muscles and
  chordae, the septum and the blood inside, all still beating.
- **Cut away front** (every view) slices the whole heart with a front-to-back plane, like a
  dissection: the chamber walls show in cross-section (the LV wall visibly ~2.5x the RV's). The
  plane is fixed to the heart, so orbiting shows the cut from any side.

**Close up** puts both back. The outer heart also carries the coronary arteries (left anterior
descending with a diagonal branch, circumflex, right coronary with a marginal branch), following the
common right-dominant pattern.

## Risk view

The **Risk** view colours the heart by the predicted chance of heart disease, from the server's
model or the offline estimate: green below 35%, amber around the 50% decision point, red from 65%
(the same display zones as `src/risk_reading.py`). The coronary arteries take the colour and pulse
more strongly with each beat at higher risk, and a shading over the ventricles grows with the
estimate, strongest over the left anterior descending artery's territory (front of the LV, the
front of the septum and the apex), where coronary disease most often shows. A legend on the canvas
gives the figure and verdict. Every input change updates it within a moment. The shading marks
where disease commonly appears; it is not a perfusion scan of this patient's heart.

## Per-patient digital twin (inside CordiaAI)

Opened from a patient's report in the CordiaAI web app (the **Digital twin** button), the twin loads
that patient's record and sets every simulation parameter from it. `npm run build` produces `dist/`,
which the FastAPI backend serves at `/twin/` — the same origin as the main app — so it reuses the
logged-in session. `?patient=<id>` loads one of a doctor's patients; `?self=1` loads a patient's own
record. The record-to-physiology mapping lives server-side in
[`src/patient_twin.py`](../src/patient_twin.py) (`GET /api/patients/{id}/twin`, `GET /api/me/twin`).

Each patient gets two scenarios, switchable in the panel:

| Parameter | At rest | Peak exercise (stress test) |
|---|---|---|
| Heart rate | Typical 72 BPM — the record's `thalach` is the stress-test peak, not a resting rate | From record: `thalach` |
| Systolic BP | From record: `trestbps` | Estimated: resting value plus a typical rise with heart rate |
| Diastolic BP | Typical 80 mmHg (the record holds systolic only) | Same |
| Stroke volume | Estimated: typical adult value by sex | Same |
| Vascular resistance | Estimated from BP, age (arterial stiffening) and fasting blood sugar | Lowered for exercise vasodilation |
| Afterload | Estimated from BP; raised for LV hypertrophy on the resting ECG | Same |
| Contractility | Estimated: reduced with the risk model's predicted severity and a resting ST-T abnormality | +20 for exercise inotropy, minus exercise ischemia (`exang`, `oldpeak`) |
| Preload | Typical value | Raised for exercise venous return |
| Oxygen saturation | Typical value | Same |

The panel badges every value as **From record**, **Estimated** or **Typical value**, and hovering a
row shows its exact rule. Cholesterol has no direct hemodynamic parameter, so it informs the risk
prediction but not the simulation. Run against the stored patients, the mapping separates them the
way the records say it should — e.g. a patient with exercise angina and 4.3 mm ST depression drops to
contractility 12 at peak exercise, while one with a clean stress test rises from 55 to 67.

Stated plainly: these records contain no echo, chamber-volume or pressure-volume data, which is what a
clinically calibrated cardiac digital twin is fitted to. This twin is patient-specific in its inputs,
not a validated model of that patient's heart.

### What-if controls and graphs

The controls are the same ten fields the risk model is trained on, in everyday wording with the
clinical term alongside ([`src/patient/inputFields.ts`](src/patient/inputFields.ts)). Each change is
debounced and sent to `POST /api/twin/what-if`, which re-runs the real model and re-derives both
scenarios from the changed inputs; responses to superseded requests are dropped, so a fast slider
drag can't apply an out-of-date result. Inputs that directly set part of the heart are marked with
a heart; the rest (chest pain type, cholesterol) change the risk estimate and reach the heart only
through the predicted severity. Changing an exercise-test value switches to the exercising heart.
The saved record is never modified; "Reset to record" restores it. Two charts track the effect — the
severity breakdown, record vs. now, and the chance of heart disease after each change.

### Making changes visible on the heart

Most inputs change the heart by a few percent, which was invisible: chamber size was drawn with a
cube root of volume and the wall followed it at 28%, so a recorded heart and one at 180 mm Hg moved
their LV wall 4% vs. 3% per beat. Three changes make a what-if readable:

- **Change vs. record.** On load and after every change, both scenarios are run to steady state
  off-screen ([`src/simulation/steadyState.ts`](src/simulation/steadyState.ts), 20 s of simulated
  time). Each vital card shows ▲/▼ against the record, coloured by whether the direction is better
  or worse for the heart (heart rate is neutral); under 1 unit reads "Same as record".
- **Stronger squeeze.** Chamber size is `clamp((V / Vref)^0.6, 0.5, 1.3)` and the wall follows 65% of
  it; the intact (Outside) heart pulses with total ventricular volume. The same comparison now moves
  the LV wall 15% vs. 11% per beat.
- **Strain glow.** The muscle glows red with intensity `min(0.9, 2.2 × (workload / record workload − 1))`,
  eased in over about a second — e.g. blood pressure 139 → 180 raises workload 29% and gives a clear glow.

### Patient view (`?self=1`)

Opened from a patient's own account, the twin uses plain wording ("Resting" / "Being active", "your
results"), hides the severity breakdown, and offers only blood pressure (with its everyday band) and
high blood sugar as controls; everything else stays as the clinic recorded it. "Back to my results"
restores the record.

## 1. How the physiological model works

Everything in the app — the 3D heart, blood-flow particles, ECG, pressure graph, and dashboard —
reads from **one shared simulation state**, computed in [`src/simulation/heartModel.ts`](src/simulation/heartModel.ts)
and advanced every animation frame by [`SimulationDriver`](src/components/Heart3D/SimulationDriver.tsx).
Nothing is independently animated; every visual is a function of that live state.

The core of the model is a **time-varying elastance simulation**, run separately for the left
ventricle (systemic circuit) and right ventricle (pulmonary circuit) — the same family of
equations used throughout cardiovascular modelling (the Suga-Sagawa formulation) and the same
approach already validated in this project's standalone Python prototype
(`digital_twin_heart_simulation.py`), ported to TypeScript and extended to a genuine two-ventricle
simulation:

- **Elastance** `E(t)` oscillates every beat between a relaxed value (`Emin`) and a contracted value
  (`Emax`), and instantaneous ventricular pressure is `P = E(t) x (V - V_dead)`.
- **Valves** (mitral, tricuspid, aortic, pulmonic) are modelled as one-way flow resistors:
  `Q = max(0, ΔP / R)` — blood only flows the correct direction, only when the pressure gradient
  favors it. This is what makes the valve indicators open/close correctly and what makes
  regurgitation and stenosis mechanically real rather than cosmetic.
- Each ventricle drains into a **Windkessel arterial compartment** (aortic for the LV, pulmonary
  for the RV): `dP_arterial/dt = (Q_out - P_arterial / R_systemic) / C_arterial`.
- The whole state is integrated with **RK4** (4th-order Runge-Kutta), the same integrator used in
  the Python prototype, re-run every frame with the real elapsed time (`delta`) from Three.js's
  render loop, capped and clamped for numerical stability under extreme slider combinations (see
  §3).
- **Atrial systole** is modelled as a brief pressure "kick" added to atrial pressure shortly after
  the P wave — this produces the real late-diastolic filling boost and the "a wave" seen in real
  atrial pressure tracings, without a full second elastance chamber.
- **Electrical timing**: `tFrac = 0` is defined as SA-node firing (start of the P wave) — the
  standard ECG reference point. Every other event (AV node, Bundle of His, bundle branches,
  Purkinje fibers, QRS, ventricular contraction onset, T wave) is timed relative to it in
  [`CONDUCTION_TIMING`](src/simulation/heartModel.ts), so the ECG, the electrical-conduction
  animation, and the mechanical contraction all stay genuinely synchronized — not just sharing a
  period.

### Validated behaviour

The model was run headlessly to steady state (60 s of simulated time) for each preset and the
settled values checked against normal physiological ranges. Results are stable across timestep
sizes from 1/120 s to 1/50 s (EDV varies by 0 mL, cardiac output by 0.1 L/min), so the explicit
RK4 integrator is well-behaved at real browser frame rates.

| Case | EDV (mL) | ESV (mL) | SV (mL) | EF (%) | CO (L/min) | Peak LV (mmHg) |
|---|---|---|---|---|---|---|
| Normal | 150 | 69 | 81 | 54 | 5.8 | 106 |
| Tachycardia (150 BPM) | 129 | 74 | 55 | 43 | 8.3 | 89 |
| Bradycardia (45 BPM) | 158 | 60 | 98 | 62 | 4.4 | 109 |
| Reduced contractility | 152 | 115 | 36 | 24 | 2.6 | 69 |
| High preload | 181 | 83 | 99 | 54 | 7.1 | 129 |
| Low preload | 123 | 58 | 65 | 53 | 4.7 | 85 |
| Increased afterload | 151 | 89 | 61 | 41 | 4.4 | 140 |
| Hypertension | 151 | 98 | 53 | 35 | 3.8 | 154 |
| Heart failure | 173 | 147 | 25 | 15 | 2.4 | 85 |

Normal values sit inside the textbook ranges (EDV 120-170, ESV 50-70, EF 50-70%, CO 4-8 L/min),
and every preset moves in the physiologically correct direction: raising heart rate shortens
filling and drops stroke volume while still raising cardiac output; raising preload raises
end-diastolic volume (Frank-Starling); raising afterload raises the pressure the ventricle must
generate while reducing what it ejects; reducing contractility raises end-systolic volume and
collapses ejection fraction.

These numbers are a calibration check on the model's internal consistency, not a claim of
clinical accuracy.

## 2. How each physiological parameter affects the simulation

These are the simulation's internal parameters. In the current app they are not user sliders:
they are derived from the patient's ten inputs (see "Per-patient digital twin" above) and shown,
with their source, under "How the heart model is set up". Every dependent value is recalculated
from them every frame — nothing is hard-coded independently.

| Slider | What it changes mechanically |
|---|---|
| **Heart rate** | Sets the cardiac cycle duration (`60 / HR`). A shorter cycle means less time for ventricular filling and a visibly faster beat, ECG, and particle flow. |
| **Stroke volume** | Baseline pump strength (scales `Emax`). The *displayed* stroke volume is always the simulation's actual `EDV - ESV` from the live beat, not this raw slider value — see §3. |
| **Systolic / diastolic BP** | Combined into a target mean arterial pressure (`MAP = DBP + (SBP-DBP)/3`), which is used to solve for the baseline systemic resistance via `MAP = CO x TPR` — a real hemodynamic identity, not an arbitrary mapping. |
| **Contractility** | Scales `Emax` directly (0.5x-2x) — the inotropic state. Higher contractility raises peak ventricular pressure and ejection fraction for the same filling volume. |
| **Preload** | Scales atrial pressure (`P_LA`, `P_RA`), which drives filling via the mitral/tricuspid resistor equations — higher preload means more filling, i.e. Frank-Starling behavior, without hard-coding EDV directly. |
| **Afterload** | Scales the resistance the ventricle ejects against, on top of vascular resistance — higher afterload makes ejection harder, which (via the same ODEs) reduces stroke volume unless compensated by contractility. |
| **Oxygen saturation** | Currently a display/legend value (visualized via the blood-flow particle color legend); it does not yet feed back into the mechanical simulation — an explicit simplification (see §3). |
| **Vascular resistance** | Scales systemic resistance directly — combines with cardiac output to set arterial pressure, exactly as in real hemodynamics. |

The model also has valve-disease terms — **aortic stenosis** narrows the aortic valve's effective
resistance (`R_effective = R / (1 - severity x 0.85)`), which mechanically forces LV pressure well
above aortic pressure during ejection (the real transvalvular gradient), and **mitral
regurgitation** adds a backward leak term (`Q_regurg`) active only while LV pressure exceeds LA
pressure during systole. They are not exposed in the current app, because the patient records hold
no valve data; the earlier disease presets that used them were removed with the physiology sliders.

## 3. What's scientifically simplified

Stated honestly, the same way every other part of this project documents its limitations:

- **Two ventricles are fully simulated; the atria are not.** Atrial "contraction" is a pressure
  kick, not a separate elastance chamber with its own volume ODE — a deliberate simplification to
  keep the app real-time and maintainable, not a hidden shortcut.
- **Numerical stability clamps.** The valve-flow resistors are deliberately small (real valve
  orifices are low-resistance), which makes the explicit RK4 integrator stiff under extreme slider
  combinations (e.g. severe aortic stenosis at high contractility). Rather than a fully adaptive
  substepping integrator, ventricular volume and pressure are hard-clamped to the outer edge of
  plausible human physiology (e.g. LV/RV pressure capped at 350 mmHg, arterial pressure at 300
  mmHg). This is both a numerical-stability guard and a literal implementation of spec requirement
  "prevent impossible combinations where practical."
- **Oxygen saturation is not yet mechanically coupled** — it's visualized (blood-flow particle
  color legend, dashboard card) but doesn't currently feed back into contractility or resistance
  the way real hypoxia would.
- **Right-heart pressures use the same elastance/valve/Windkessel structure as the left heart**,
  just with different (lower-pressure, higher-compliance) constants — a legitimate simplification,
  not a separately-tuned/hand-animated system, but real right-heart afterload is dominated by
  pulmonary vascular resistance in ways this model only coarsely approximates (fixed ratio of
  systemic resistance rather than an independently modeled pulmonary circulation).
- **Anatomy is procedurally generated**, not a segmented CT/MRI mesh. No licensed or public-domain
  high-fidelity heart mesh was available to bundle, so each chamber is its own parametric
  thick-walled shell ([`heartGeometry.ts`](src/components/Heart3D/heartGeometry.ts)), opened at the
  front so the cut rim shows wall thickness (LV 0.14 vs RV 0.055 scene units, as in real anatomy).
  The chambers are kept as separate structures so each one reads distinctly. It is a stylized
  approximation, the same honesty policy applied to this project's earlier matplotlib prototype.
  Coronary vessels are not modeled.
- **Great-vessel layout** follows the textbook anterior view: the aorta arches posteriorly toward
  the patient's left (brachiocephalic, left common carotid, left subclavian branching in that
  order), and the pulmonary trunk crosses anterior to the ascending aorta before bifurcating.
  Vessels are separate tubes plugged into the mass, not part of the distance field, so blood-flow
  particles can travel visibly inside them.
- **Blood-flow particles** move along one fixed, pre-authored closed-loop curve through all four
  chambers and the great vessels; their *speed* through each zone is driven by the live valve flow
  rates (so they visibly slow/stall when a valve is closed), but the path geometry itself is not
  derived from the vessel meshes.
- **Cardiac cycle phase proportions are fixed fractions of the beat period**, not derived from
  first-principles electrophysiology — so at very high heart rates, both systole and diastole
  shrink proportionally, whereas real hearts shorten diastole disproportionately more than systole.
  This is the same simplification already disclosed in the earlier Python prototype.
- **Presets are simplified educational patterns**, explicitly labeled in the UI as "not medically
  validated patient models" — they demonstrate a pathophysiological *direction* (e.g. "aortic
  stenosis raises LV pressure"), not a diagnosed patient's actual hemodynamics.

## 4. Connecting to real patient data (future direction)

A first step is already built: the parameters are set from each patient's stored record and the
risk model's prediction (see "Per-patient digital twin"), not from manual sliders. `HeartParams` is
a plain, serializable object and the visual layer is a pure function of `HeartOutputs`, so going
further means feeding better measurements into the same mapping:

1. **Use measured values where the record only allows estimates.** Resting heart rate, diastolic
   pressure and oxygen saturation are typical values today because the record doesn't hold them; a
   bedside monitor or wearable feed could supply them directly through `src/patient_twin.py`.
2. **Fit `contractility`/`preload`/`afterload` from echocardiographic measurements** (e.g.
   ejection fraction, E/A ratio) instead of estimating them by rule — this project's
   planned Echo branch (EchoNet-Dynamic, still blocked on the user's own Stanford AIMI
   registration per the main README) would be the natural source.
3. **Use the ECG branch's diagnosis too.** The predicted severity already lowers contractility;
   the ECG ViT's classes (e.g. hypertrophy, conduction disturbance) could adjust the relevant
   parameters in the same way once an uploaded ECG is linked to a patient record.
4. **Replace the procedural mesh with a real segmented anatomy** once a licensed or public-domain
   patient-specific mesh (e.g. derived from cardiac MRI segmentation) is available, keeping the
   same `HeartOutputs`-driven scaling/coloring logic.
5. **Add per-patient calibration** — right now `HeartSimulation`'s constants (`V_dead`, `Emax`
   baseline, resistances) are population-typical, not patient-specific; a real digital twin would
   fit these from that patient's own measured PV loop or imaging data, the same direction already
   explored in this project's Synthea-based longitudinal digital twin (`build_digital_twin_synthea.py`).

Items 1–5 are not implemented; they describe the extension path beyond the record-based
parameterisation that is.
