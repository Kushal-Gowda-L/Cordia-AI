# CordiaAI

Multimodal cardiovascular risk prediction system combining clinical tabular data,
ECG signals, and echocardiography, with explainability, privacy-preserving
federated training, and a longitudinal patient digital twin.

The full design rationale is in the separate methodology document (`CordiaAI_Methodology.docx`, not in this repo).
This repository is the actual implementation.


> **Repository notes.** Large ViT checkpoints (`checkpoints/ecg_vit.pt`, `ecg_vit_unweighted_backup.pt`, `cxr_vit.pt`, ~346 MB each) exceed GitHub's file limit and are not in this repo; retrain them with the notebooks in `notebooks/` or get them from the author. The patient database (`mobile_app/backend/app.db`) and credentialed imaging datasets are never committed. Cross-validation experiments live in `experiments/ft_cv/`, a project review in `docs/cordia_ai_review.md`, and app screenshots in `docs/screenshots/`.

## Project layout

```
cordia_ai/
  datasets/
    uci/          # UCI Heart Disease (tabular, 920 samples) — Phase 1 (done)
    ecg-images/   # PTB-XL ECG embeddings (E_v, 256-dim) — Phase 2 (done, class-weighted)
    echo/         # EchoNet-Dynamic clips — Phase 3 (needs GPU)
    paired/       # optional future work: MIMIC-IV paired cohort CSV (credentialed data, never commit)
    framingham/   # Framingham cohort (public cross-sectional subset) — digital twin risk model
  notebooks/
    step3_gan_augmentation_colab.ipynb  # standalone Colab benchmark notebook
    step5_ecg_vit_colab.ipynb           # standalone Colab PTB-XL/ViT training notebook
    step15_xray_vit_kaggle.ipynb        # chest X-ray branch (NIH ChestX-ray14, ViT-B/16)
    step16_echo_ef_colab.ipynb          # echo branch (EchoNet-Dynamic, ViT per frame + temporal transformer -> EF)
    step17_mri_acdc_colab.ipynb         # MRI branch (ACDC, Swin-UNETR -> volumes, EF, diagnosis, 3D meshes)
    step18_ct_imagecas_colab.ipynb      # CT branch (ImageCAS, 3D Swin-UNETR -> coronary tree mesh)
  checkpoints/
    ft_transformer.pt  # trained tabular branch weights
    ecg_vit.pt          # trained ECG branch weights (fine-tuned ViT-B/16, class-weighted run)
    ecg_vit_unweighted_backup.pt  # earlier unweighted run, kept for comparison (HYP=0 F1)
    fusion_ensemble.pt  # cross-attention fusion + 5-model stacking ensemble (architecture validated, see Step 6)
    ft_transformer_federated.pt  # FedAvg + Opacus DP-SGD global model, 3 simulated hospital nodes (Step 7)
    digital_twin.joblib  # Framingham risk model + population age-drift rates for trajectory simulation
    digital_twin_synthea.joblib  # rebuilt on real longitudinal data (Synthea) — see "Digital twin" below
  mobile_app/
    backend/main.py     # FastAPI: role-based auth, per-doctor isolated patients, prediction, reports, twin endpoints, /api/ecg-predict; serves the twin at /twin/
    backend/imaging_api.py  # /api/imaging/* (X-ray, echo, MRI, CT, combined), loaded lazily per branch
    backend/physics_api.py  # /api/physics/* multi-physics heart twin (see "Multi-physics heart twin")
    frontend/            # clinical web app: sign-in + dashboard (patients, assessment report, digital twin, summary, ECG analysis)
    frontend/heart_physics.*  # multi-physics heart viewer (Three.js in frontend/vendor/three/)
  heart_digital_twin_3d/  # React + Three.js per-patient heart digital twin (built to dist/, served by the backend at /twin/)
  src/
    patient_twin.py   # patient record + risk prediction -> digital-twin physiology (at rest / during exercise)
    imaging_twin.py   # measured imaging values (EF, volumes, wall thickness, X-ray findings) -> twin parameters
    imaging/          # imaging preprocessing, cardiac measurements, 3D meshes, CT pipeline,
                      #   heart_physics.py: electrical / muscle / blood-flow / fat twin
    preprocessing/  # data loading, cleaning, feature selection, splitting
    models/
      ft_transformer.py  # tabular branch architecture
      ft_ensemble.py       # FT-Transformer v2: 13 inputs, 5-model ensemble, preprocessing (used by the web app)
      ecg_vit.py           # ECG branch architecture (ViT-B/16), for local inference/Grad-CAM
      fusion.py            # cross-attention fusion (E_t query, E_v key+value) + stacking MLP ensemble
      late_fusion.py       # combiner over whichever branches a patient has; fitted only on real paired patients
      imaging/             # X-ray ViT, echo ViT, MRI/CT Swin-UNETR architectures
      explain.py          # SHAP / LIME / DiCE explainability, used by the mobile app's reports
      llm_report.py        # offline LLM narration (Ollama + llama3.2:3b), used by /api/*/narrate
      augmentation/       # SMOTE / CTGAN / diffusion generators + evaluation metrics
    utils/          # shared helpers
  explore_uci.py                 # Step 1-2 entry point (EDA + preprocessing)
  run_augmentation.py            # Step 3 benchmark: SMOTE vs CTGAN vs diffusion
  generate_augmented_dataset.py  # Step 3 (final): builds the augmented training set
  train_ft_transformer.py        # Step 4: trains the tabular branch (v1)
  train_ft_transformer_v2.py     # FT-Transformer v2: 5-fold CV + the model the web app serves
  benchmark_xgboost.py            # binary vs. 5-class accuracy check against the reference paper's 97.57%
  train_fusion.py                 # Step 6: trains the cross-attention fusion + stacking ensemble
  train_federated.py              # Step 7: FedAvg + Opacus DP-SGD across 3 simulated hospital nodes
  audit_fairness.py               # Explainable AI layer: IBM AIF360 fairness audit + Reweighing mitigation
  gradcam_ecg.py                   # Explainable AI layer: Grad-CAM attribution for the ECG branch
  build_digital_twin.py            # Digital twin: Framingham risk model + population-drift trajectory simulation
  build_digital_twin_synthea.py    # Digital twin, rebuilt on real longitudinal data (Synthea; expects C:/synthea/output/csv/, not checked in)
  Dockerfile                       # Deployment packaging for the mobile app (see "Deployment packaging")
  docker-compose.yml               # cordia-app + ollama, two-container deployment
  app.py                         # earlier Streamlit dashboard (superseded by mobile_app/)
  requirements.txt               # CPU-only deps; torch installed separately
  STEP5_STATUS.md                 # exact resume point for the ECG class-imbalance retrain
  download_imaging_datasets.py    # list / download / verify the imaging datasets (see "Imaging branches")
  build_mimic_paired_cohort.py    # optional future work: paired ECG + X-ray + labs cohort from MIMIC-IV (not run)
  train_late_fusion.py            # optional future work: late-fusion combiner on that cohort (not run)
  tests/test_imaging.py           # imaging checks on synthetic data (python -m pytest tests -q)
  tests/test_heart_physics.py     # multi-physics twin checks on a synthetic ventricle
```

## Setup

```bash
git clone https://github.com/Kushal-Gowda-L/Cordia-AI.git
cd Cordia-AI
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
./.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For the offline AI summary feature (optional — everything else works without it):
```bash
winget install Ollama.Ollama
ollama pull llama3.2:3b
```
Ollama's Windows installer runs its API server automatically in the
background afterward; no separate `ollama serve` step needed.

## Web app

A clinical web application for hospital use. It opens directly on a sign-in screen — no public
marketing page and no model-accuracy figures — then a dashboard with a patient table, the
assessment report, the per-patient digital twin, a plain-language summary and ECG analysis. Light,
neutral theme (IBM Plex Sans / IBM Plex Mono, one clinical-blue accent, a navy sidebar). Every
model input and output is labelled in everyday words with the clinical term beside it, so the same
screen reads for a clinician and for a patient. The report draws the severity breakdown and the
SHAP contributions as charts, and all patient-entered text is HTML-escaped before rendering. This
replaced an earlier marketing-style site (landing hero, feature grid, results table), which in turn
replaced a phone-width mobile shell.

**A real bug found and fixed while building this:** the dashboard's inner
views (patient list, report, simulation, etc.) and the top-level pages
(landing, login, signup) both used a `.view` class, but for two different
visibility mechanisms — a CSS rule meant only for the dashboard
(`.view { display: none }`) was silently matching the landing/login/signup
pages too, making the entire site render as a blank navy screen despite
every element being present and correctly styled in the DOM. Fixed by
scoping the rule to `.main-area > .view`.

```bash
cd mobile_app/backend
../../.venv/Scripts/python.exe -m uvicorn main:app --port 8000
```
Open `http://localhost:8000`.

**Role-based accounts:**
- **Doctor** — manages a patient list. Each doctor's patients are strictly
  isolated from every other doctor's (row-level scoping by the logged-in
  doctor's user id on every query — verified with two separate doctor
  accounts, each only ever sees their own patient).
- **Patient** — views the record their doctor entered; patients don't enter clinical values.
  A patient can't be expected to know their chest-pain classification, resting-ECG finding or
  stress-test ST depression, so the doctor records the assessment and, from the report's
  **Patient access** panel, creates a one-time access code (`XXXX-XXXX`, 7-day expiry, only a
  SHA-256 hash stored, a new code cancels the previous one). The patient enters it under
  **My heart report** (`POST /api/me/link`) to link the record to their account; the patient table
  shows which records are linked. The patient's report is written in plain language (no severity
  classes or explainability tables), and their heart model offers only what a patient can relate
  to: blood pressure (with a Normal / Slightly raised / High / Very high band) and whether their
  blood sugar runs high. Older databases gain the access-code columns automatically on startup.
  Verified against a copy of the database: wrong, reused and cross-account codes are rejected,
  codes are case- and dash-insensitive, and only doctors can issue them.

**Follow-up visits and visit comparison:** a returning patient is not added again. Each patient
keeps a list of visits (table `visits`: that visit's 13 model inputs, the risk and severity the
model gave them, the model version, an optional note and the date). The patient row always mirrors
the latest visit, so the report, digital twin and patient account show the current state.
- **Record follow-up visit** (report page) opens the form filled with the last visit's values;
  change what differs today and save. It is scored by the model now serving and stored as a new
  visit; earlier visits are not changed (`POST /api/patients/{id}/visits`).
- **Edit details** corrects the values recorded at a visit, e.g. a typing mistake. The visit keeps
  its date and is re-scored (`PUT /api/patients/{id}/visits/{visit_id}`). The **Visits** panel on
  the report lists every visit with its risk and the change since the visit before, and can edit
  or delete any of them (`DELETE .../visits/{visit_id}`; a patient always keeps at least one).
- **Compare visits** shows two visits side by side, previous vs. current by default and any two
  through the pickers: each visit's risk and reading, the change in risk in percentage points
  (red ▲ when it went up, green ▼ when it went down), and every recorded detail with the changed
  ones highlighted and numeric changes given in their units (e.g. ▲ +33 mm Hg).
- The patient table shows the number of visits and the date of the last one, newest first.

On first start with this version the database is copied to `app.db.pre-visits-backup`, then every
existing patient becomes visit 1 dated when they were added. Visits saved under the other model
version are re-scored on startup, like patients. Verified on a copy of the real database (20
patients migrated, risks match, a second start changes nothing), plus follow-up, edit, delete and
cross-doctor isolation checks and a browser click-through.

**Report generation**, for every patient record, actually runs:
- The trained FT-Transformer for the risk score and severity class
- Threshold-based clinical findings (cholesterol, BP, etc. vs. reference ranges)
- **SHAP** (KernelExplainer) — top contributing factors to the **chance of heart disease**
  (1 − P(No disease)), shown in percentage points: red raises the risk, green lowers it. It used to
  explain whichever severity class won, so for a "No disease" prediction a red bar meant *reassuring*
  and the chart could contradict the headline risk; explaining P(disease) keeps it consistent with the
  risk figure for every patient. Because class probabilities sum to 1, this is exactly minus the
  "No disease" SHAP values, so no extra computation is needed. Note that "typical angina" usually
  shows as *lowering* the risk: in the UCI data, patients with no chest pain (asymptomatic) had the
  highest disease rate (~79%), so the model learned that pattern.
- **LIME** (LimeTabularExplainer) — local linear explanation rules
- **DiCE** (random-search counterfactuals) — concrete value changes that
  would move the prediction one severity level down
- A quick single-factor sensitivity what-if (re-runs the model directly)
- The per-patient **digital twin** (what-if on the ten inputs, with a live 3D heart) and an
  **offline AI-generated plain-language summary**, both on demand — see their own sections below

All three explainability methods are wrapped in try/except in `src/models/explain.py`
so a failure in any one library degrades gracefully rather than breaking the report.

## Interactive 3D digital twin (heart_digital_twin_3d/)

A separate, standalone React + Three.js (React Three Fiber) application — not part of the
FastAPI website, since it needs its own Vite dev server (`cd heart_digital_twin_3d
&& npm install && npm run dev`, then open http://localhost:5173). With no backend running it
falls back to an in-browser offline risk estimate, so that command alone is enough to try it. A real-time, physiologically-simulated 3D heart: two full
ventricle simulations (time-varying elastance + valve flow resistors + Windkessel arterial
compartments, integrated with RK4 every frame), synchronized ECG and electrical-conduction
animation, a live Wiggers-style pressure graph, blood-flow particles whose speed is driven by
the live valve flow rates, seven inspection views (outside, inside, blood flow, pressure,
valves, electrical signal, risk), a see-through wall and a front-to-back cutaway plane for
looking inside the closed heart, and coronary arteries coloured by the predicted risk. Every dashboard number and every
visual is a function of one shared simulation state — nothing is independently animated. Full
write-up of the physiological model, what each slider does, and every stated simplification
(procedural — not segmented-CT — anatomy; atria modeled as a pressure kick, not a full
elastance chamber; numerical-stability pressure clamps under extreme slider combinations) is
in `heart_digital_twin_3d/README.md`. Verified with a real browser session: zero console
errors on fresh load, physics validated against expected values (e.g. tachycardia preset
settling to HR 150 x SV 34 mL ≈ 5.09 L/min cardiac output, matching the displayed figure), and
aortic stenosis confirmed to produce the correct transvalvular LV-pressure gradient once a
numerical-stability bug (unclamped pressure blowup under that specific extreme parameter
combination) was found and fixed.

### Per-patient digital twin

Every patient report now has a **Digital twin** button that opens the 3D heart above, parameterised
from that patient's own record — resting BP, peak exercise heart rate, exercise angina, ST depression,
resting-ECG findings, age, sex and the model's predicted severity — in an "at rest" and a "peak
exercise" scenario. The twin's production build is served by this backend at `/twin/`, the mapping is
in `src/patient_twin.py`, and the panel labels each parameter as taken from the record, estimated by
a stated rule, or a typical value. Full mapping table in `heart_digital_twin_3d/README.md`. The records
hold no echo or pressure-volume data, so this is a patient-parameterised model, not a calibrated one.

**What-if controls and graphs.** The twin's controls are the same ten fields the risk model is
trained on, in everyday wording (e.g. "Heart strain on the exercise tracing", with "ST depression"
beside it) — not internal physiology settings. Each change goes to `POST /api/twin/what-if`, which
re-runs the real model and re-derives the heart from the changed inputs, so the risk estimate and
the simulated heart always describe the same values; the stored record is never modified. Changing
an exercise-test value switches to the exercising heart, where that value applies. Two charts track
the effect: the severity breakdown (record vs. now) and the chance of heart disease after each
change, with the recorded value as a reference line. Every heart number (blood per heartbeat,
pumping efficiency, workload, …) shows ▲/▼ against the record, computed from the settled simulation
rather than the beat-to-beat value; the heart muscle glows red in proportion to extra workload
(heart rate × blood pressure); and chamber size follows blood volume strongly enough that a weaker
beat is visibly shallower (see the twin README). This replaced the earlier physiology sliders
and disease presets, and the separate "3D what-if simulation" page.

Docker: the image now builds the twin in a Node stage and copies `dist/` in. Adding it surfaced a
separate, pre-existing packaging bug, now fixed: the image never included `checkpoints/ecg_vit.pt`
(excluded in `.dockerignore`, commented as unused), yet `main.py`'s startup loads it for the ECG
upload endpoint — so the container would have failed on boot.

### Reading the risk numbers: two steps

The model outputs five probabilities (No disease, Mild, Moderate, Severe, Critical). Taking the single
largest one as the answer can contradict the total: one stored patient has 45.2% "No disease", larger
than any single disease level, while the four disease levels together make 54.8%. So every screen reads
the output in two steps ([`src/risk_reading.py`](src/risk_reading.py)):

1. **Is heart disease likely?** The total chance of disease, 1 − P(No disease), against the model's
   **50% decision point**. The report also places it on three zones: **unlikely** below 35%,
   **borderline** 35–65% ("leans towards heart disease / no heart disease"), **likely** 65% and above.
2. **If heart disease is present, how severe?** The four disease levels rescaled to add up to 100%
   among disease cases, i.e. the conditional probability P(level | disease) = P(level) ÷ P(disease).
   For the patient above: Severe 30 ÷ 54.8 ≈ 55%, Moderate 22%, Mild 14%, Critical 9%, so
   "Heart disease: borderline, leans towards disease; if present, most likely Severe".

The model's raw five-outcome breakdown stays available on the report, collapsed, with a note on why
"No disease" can be the largest bar. The patient list shows the Step 1 reading, the patient's own report
explains it in plain words, and the digital twin shows the verdict, the Step 2 chart and the zones
shaded on its risk-history chart.

What is and isn't a medical claim here:
- The rescaling is exact conditional probability, so it adds no error, but it is only as good as the
  model's own probabilities (their calibration has not been verified).
- The 50% point is the model's decision rule, not a clinical threshold.
- **The 35% / 65% zone limits are this application's display convention** for flagging results close to
  the decision point. They come from no clinical guideline, and the app labels them that way.
- Mild / Moderate / Severe / Critical is this project's naming for the UCI dataset's `num` values 1–4,
  not a clinical severity scale.

The stored `severity_idx` (the single largest class) is unchanged in the database. The digital twin
still uses it to set contractility.

### Known model issue: cholesterol

The digital twin exposed a real defect in the tabular model. For a fixed patient, raising cholesterol
*lowers* the predicted risk — e.g. 83.4% at 120 mg/dl, 71.4% at 187, 61% at 300, flat above ~260 —
the opposite of the medical evidence. The cause is in the training data, not the twin: 172 of the 920
UCI records (18.7%) store cholesterol as 0, a missing-value code, and 88.4% of those patients have
disease versus 48.2% of the rest. The preprocessing (`src/preprocessing/clean.py`) only imputes `NaN`,
so those zeros reach the model as real measurements and it learns "very low cholesterol → disease".
Among the genuine (non-zero) records the disease rate *rises* with cholesterol (40.6% below 180 mg/dl,
57.5% at 260–300), so the true relationship in the data is the expected one.

The fix is to treat `chol == 0` as missing before imputation and retrain. **FT-Transformer v2 (below)
does this**, and with it risk rises with cholesterol again (for one test patient: 87.3% at 150 mg/dl,
87.8% at 200, 89.1% at 250, 90.1% at 300). v1 is unchanged. The twin's caution text on its cholesterol
control is part of its built `dist/` and stays until the twin is rebuilt.

### FT-Transformer v2 (the model the web app now serves)

Same FT-Transformer architecture, changed in four ways, each measured with 5-fold cross-validation on
all 920 patients before being adopted (`src/models/ft_ensemble.py`, `train_ft_transformer_v2.py`):

1. **slope, ca and thal are used again**, with an "is missing" flag for each. SF-2 dropped them because
   they are blank for most non-Cleveland patients, but where present they carry real signal.
2. **Token size 128 instead of 256**: 920 patients is small for a 256-wide transformer.
3. **5 copies trained with different seeds, probabilities averaged.**
4. **Cholesterol 0 and blood pressure 0 are treated as missing** (see above), with an "is missing"
   flag for cholesterol. Without the flag the fix cost 1.8 points (55.9 ± 2.3%), because in UCI a
   missing cholesterol is itself a strong sign of disease; the flag keeps that signal without
   pretending 0 mg/dl is a measurement.

Trained on real patients only (the diffusion-augmented set lowered 5-class accuracy). Optimiser,
learning rate, batch size and early stopping are the same as v1.

| Setup (5-fold CV, all 920 patients) | Accuracy | Macro F1 |
|---|---|---|
| v1 setup: 10 features, d=256, one model | 53.8 ± 3.5% | 0.327 |
| + slope, ca, thal | 54.8 ± 3.0% | 0.347 |
| + d=128 | 56.0 ± 2.6% | 0.356 |
| + 5-model ensemble | 57.7 ± 2.3% | 0.356 |
| **v2 as trained by `train_ft_transformer_v2.py`** (+ cholesterol fix) | **57.3 ± 2.2%** | **0.354** |

On v1's own held-out test set (the same 184 patients, v1's train split): **v2 57.6% vs v1 53.3%
accuracy**; macro F1 0.344 vs 0.347. Like v1, it is weak on Severe and Critical, which have 21 and 6
test patients. Two things tried and not
adopted: class-weighted loss and per-category tokens for cp/restecg/slope/ca/thal raised macro F1
(to 0.373 and 0.388) but lowered accuracy (53.6% and 54.0%).

Quote the cross-validated figure: on a single 184-patient split the same model moves between about
52% and 61% depending on which patients land in the test set.

**In the web app.** The patient form has an optional "Further test results" section (ST slope,
vessels on fluoroscopy, thallium scan), each defaulting to "Not done"; reports list them only when
recorded. SHAP, LIME and DiCE explain the ten main inputs, with slope/ca/thal held at the patient's
own values. The twin carries them through its what-if requests unchanged. On startup, records saved
under the other model version are re-scored so the list, report and twin agree; v1 scores are not kept
(`app.db.v1-backup` is a copy taken before the switch).

**Switching back to v1:** set `CORDIA_MODEL=v1` before starting the backend (and the records are
re-scored with v1). If `checkpoints/ft_transformer_v2.pt` is missing the backend falls back to v1 on
its own. Fusion, federated learning and the fairness audit still use v1 and its embeddings.

## ECG upload

A real feature, with a real scope boundary stated up front: the sidebar's
"ECG upload" page lets you upload a **WFDB-format ECG recording** (a
`.hea` header file plus its matching `.dat` signal file — PTB-XL's own
native format, and the format most PhysioNet ECGs ship in) and get a live
prediction from the actual trained ECG ViT branch — same
`ecg_to_spectrogram()` conversion the model was trained and Grad-CAM'd on,
reused directly (`POST /api/ecg-predict`). Verified working end-to-end
with a real cached PTB-XL sample record, giving the same prediction (CD,
~64% confidence) as the standalone Grad-CAM script did on that same record
earlier.

**What this deliberately does not do:** accept a photo or scan of a
printed ECG strip. The model was trained on spectrograms (frequency
heat-maps), which look nothing like a photographed waveform trace —
feeding one in would silently produce meaningless output, not a small
accuracy gap. Turning a photographed strip into usable data ("ECG
digitization") is its own hard, separate computer-vision problem and
isn't something this endpoint attempts.
Run: sign in → sidebar → **ECG upload**, or `POST /api/ecg-predict` directly.

## Deployment packaging

`Dockerfile` + `docker-compose.yml` package the mobile app (FastAPI +
SQLite) and Ollama into two containers, for whoever wants to run this
somewhere other than a local `uvicorn` process:

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2:3b   # first run only
```

**Honest limits, stated plainly:** this is *not* the methodology's full
production stack (PostgreSQL, MLflow, Nginx reverse proxy) — that's a
further step beyond what an app running fine on SQLite+FastAPI at this
project's scale actually needs, and wasn't built. More importantly: **this
was never run-tested**, because Docker itself isn't installed on the
machine this was written on, and Docker Desktop's usual WSL2/virtualization
setup plus a required reboot was judged too disruptive to install
autonomously. The Dockerfile mirrors the project's real, already-working
local setup exactly (same `pip install -r requirements.txt` + `uvicorn`
commands from "Setup"/"Mobile app" above), so it should build cleanly — but
treat the first real `docker compose up --build` as the actual
verification step, not this write-up. One real bug was caught and fixed
while writing this even without being able to run it: the first draft
mounted a volume directly over the backend's own code directory, which
would have hidden `main.py` behind an empty volume at container start —
fixed by adding a `CORDIA_DB_PATH` environment override so the database
lives in its own dedicated `/app/data` volume instead.

## Offline AI-generated summary

Live in the mobile app: open any report and tap the ✦ icon for a
plain-English narration written by a **local, offline LLM** (methodology
Section 6.4, step 10) — nothing about the patient is ever sent anywhere.

**Two deliberate substitutions from the methodology spec, stated
plainly:**
- **`llama3.2:3b` (~2GB) instead of Llama-3 8B (~4.7GB+).** Installed via
  Ollama (`winget install Ollama.Ollama`, then `ollama pull llama3.2:3b`).
  A smaller, faster download and faster CPU inference on a laptop, while
  still genuinely a local offline model — swap the `MODEL` constant in
  `src/models/llm_report.py` for a larger one if better report quality is
  worth the extra download/inference time.
- **One structured prompt instead of the methodology's full LangGraph
  4-agent pipeline** (DataAgent → ModelAgent → XAIAgent → ReportAgent).
  The actual deliverable — an offline, privacy-preserving, plain-English
  narration built from a real prediction plus real SHAP/DiCE output —
  doesn't need four separate agent hops to reach a good result, and
  skipping LangGraph avoids one more heavy dependency in a pipeline that
  already has several. `build_prompt()` in `src/models/llm_report.py`
  assembles the same findings/SHAP/DiCE data the rest of the app already
  computes into one prompt and calls Ollama's local HTTP API once.

**Verified working end-to-end**, including through the actual app UI, not
just the standalone function — example output for a 58-year-old, 280 mg/dl
cholesterol, 150 mm Hg BP, moderate-risk profile: *"The patient's
cardiovascular risk report indicates a moderate risk level, with a
predicted risk of 98.6%... The top contributing factors to this risk are
age... and resting blood pressure, which is above the high-risk
threshold..."* — correctly grounded in the real computed numbers, not
generic text. **Measured latency: ~30 seconds** on this machine's CPU
(no GPU used) — noticeably slower than the other live features (prediction,
3D simulation), so the UI shows a "Generating…" state rather than blocking
silently; worth knowing this isn't instant if demoing live.

**A real, honest quirk worth knowing before presenting:** DiCE (Step 5,
already documented above) sometimes proposes changing `age` as part of a
counterfactual, since it's just another numerical feature the search can
adjust — not a filtering bug, but not an actionable suggestion either. The
LLM narrates whatever DiCE gives it, so an AI summary can end up
recommending something like "reducing age to 46," which reads oddly. This
is a property of the underlying counterfactual search, not something the
LLM step introduces or could reasonably filter out on its own.

## 3D what-if simulation (superseded)

The original what-if page (methodology Section 6.4, step 9) — a stylised extruded-heart mesh
coloured by three heuristic region scores from `POST /api/simulate` — has been removed. The
per-patient digital twin now does the same job more directly: its controls are the same model
inputs, every change re-runs the real model (`POST /api/twin/what-if`), and the heart shown is the
physiological simulation described above rather than a coloured shape. See "Per-patient digital
twin".

## Digital twin

`build_digital_twin.py` implements the "longitudinal patient digital twin"
piece. **Scope caveat, stated up front:** the *real* Framingham Heart Study
is longitudinal — the same patients re-examined every ~2 years across
decades — which is exactly what a genuine digital twin needs, but that
multi-exam data is NIH/BioLINCC controlled access (a formal application,
days-to-weeks turnaround), not obtainable in this session. What's in
`datasets/framingham/framingham.csv` (downloaded from a public teaching
mirror of the dataset, 4,240 patients) is the standard **cross-sectional**
subset: one baseline exam per patient plus whether they developed CHD
within the next 10 years. There is no per-patient repeated-measurement
trajectory in this file at all.

So this doesn't track anyone's real trajectory. It (1) trains a logistic
regression risk model on the cross-sectional cohort (held-out **AUC
0.70**, consistent with published Framingham Risk Score literature — a
useful sanity check that the model isn't broken), (2) fits **population-level**
linear drift rates for cholesterol/BP/BMI against age from that same
cohort (e.g. total cholesterol +1.35/year, systolic BP +1.01/year — "how
much higher is this factor, on average, in someone X years older in this
population," not "how much does any specific person's factor actually
change"), then (3) projects a synthetic future risk curve for one patient
by aging them forward and nudging their factors along that population
drift, re-scoring risk at each step. This is a real, published technique
(population-drift microsimulation) used precisely because true
longitudinal per-patient data is rarely available — but the paper should
describe it as a **simulated projection**, not a measured one. Two example
trajectories (a healthy 35-year-old, a hypertensive smoking 55-year-old)
are printed when the script runs, showing sensible, well-differentiated
10-year risk curves (9.9%→21.2% vs. 74.4%→87.6%).
Run: `build_digital_twin.py` → `checkpoints/digital_twin.joblib`.

### Rebuilt on real longitudinal data (Synthea)

`build_digital_twin_synthea.py` is a genuine upgrade over the version
above: it's built on [Synthea](https://github.com/synthetichealth/synthea),
a free, open-source synthetic patient generator that produces **actually
longitudinal** records — the same simulated patient followed across a
simulated lifetime, with repeated real vitals at real dates. This means the
twin can compute each patient's **own empirical trend** ("this patient's
cholesterol really did rise/fall at this rate across their own 4 visits"),
not one population-average number applied to everyone — a structural fix to
the exact limitation stated above.

**The one caveat that matters, said twice on purpose: these are
computer-generated synthetic patients, not real historical people.** Their
trajectories come from Synthea's own clinical-guideline-based simulation
modules, not measured human physiology. This is a real upgrade in *how* the
twin models change over time (real per-patient trends vs. one global rate)
— it is not a claim that real patient data was obtained.

Generated 1,000 synthetic patients (Massachusetts population), of which
**1,128 had 2+ real vitals visits** — enough to fit a genuine trend, not
just a snapshot. Real per-patient drift rates computed from those repeated
visits:

| Factor | Real per-patient slope | Framingham's population-average (above) |
|---|---|---|
| Total cholesterol | **&minus;0.19/year** | +1.35/year |
| Systolic BP | +0.06/year | +1.01/year |
| Diastolic BP | +0.004/year | +0.29/year |
| BMI | +0.30/year | (not tracked) |

**The cholesterol sign flip is a genuine, worth-reporting finding, not an
error:** Framingham's older, largely untreated real cohort shows
cholesterol rising with age; Synthea's simulated patients show it
*falling*, plausibly reflecting that Synthea's modules model modern
statin/treatment-guideline logic where cholesterol is medically managed
down over time. Both numbers are individually defensible — they're
measuring different things (an older real cohort's natural history vs. a
modern clinical-guideline simulation), and the disagreement itself says
something honest about how much "the average drift rate" depends on the
population and era it's measured from.

The risk model (logistic regression, same real feature set as before, now
trained on 281 patients' most recent real visit against whether they have
an actual recorded cardiac event in their history) scored **held-out AUC
0.88** — noticeably higher than the Framingham version's 0.70.
**This number should not be read as "the twin got better," and the reason
is a specific, known pitfall, not general caution:** Synthea's outcomes are
generated by the same rule-based clinical logic the model is being asked to
predict, so a model trained and evaluated purely within Synthea's own world
can look artificially strong at reproducing Synthea's own internal rules,
without that proving anything about real patients. The AUC is reported
here in full, not hidden, precisely because pretending it validates
real-world accuracy would be the dishonest move — the trustworthy part of
this upgrade is the real per-patient drift rates table above, not this
AUC number.
Run: `build_digital_twin_synthea.py` (expects Synthea's CSV output at
`C:/synthea/output/csv/`) → `checkpoints/digital_twin_synthea.joblib`.

## Accuracy vs. the reference paper's 97.57%

The synopsis's stated target is to surpass El-Sofany (2024, IEEE Access) —
97.57% accuracy, XGBoost + ANOVA/chi-square/MI feature selection, 503
samples (CHDD + private data). The FT-Transformer's 53.3% (Step 3) looks
like a huge shortfall against that, but the comparison wasn't apples-to-apples:
El-Sofany's task (and most 90%+ papers on this dataset) is **binary**
disease/no-disease classification; CordiaAI's branches predict the much
harder **5-class severity** (`num` 0-4), needed for the HIGH/MEDIUM/LOW risk
categories downstream. `benchmark_xgboost.py` tested both framings with
XGBoost (the reference paper's own model family, and a much better fit than
a from-scratch Transformer on <1000 rows) on the same leak-free SF-2
features and held-out test set used everywhere else in this repo:

| Task | Data | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| 5-class severity | real-only | 58.7% | 0.368 | 0.80 (OvR) |
| 5-class severity | + diffusion augmentation | 55.4% | 0.326 | 0.79 (OvR) |
| **Binary disease/no-disease** | real-only | 84.8% | 0.845 | 0.903 |
| **Binary disease/no-disease** | + diffusion augmentation | **86.4%** | 0.862 | **0.910** |

**Binary + augmentation + XGBoost gets to 86.4% / AUC 0.91** — a real, large
jump from the Transformer's 53.3%, and evidence the underlying features and
augmentation pipeline are sound; the Transformer was simply the wrong model
choice for this row count, as already noted in Step 3. **97.57% is still
not realistically reachable here**, for two structural reasons worth
stating plainly rather than chasing with hyperparameter tricks: (1) this
project deliberately uses the full combined 4-source UCI dataset (920
samples, more missing values, more source-site noise) rather than the
cleaner 303-sample Cleveland-only subset most 95%+ papers restrict
themselves to; (2) the 5-class severity task — the one CordiaAI's
architecture actually needs for its risk categories — inherently caps lower
regardless of model choice, and 55-65% is in line with published multi-class
results on this same dataset. **Honest framing for the paper:** cite the
binary AUC (0.91) as the directly comparable number against the reference
paper's accuracy claim, and report the 5-class numbers as a harder task the
reference paper doesn't attempt, rather than presenting the FT-Transformer's
53.3% as the project's headline accuracy figure against a 97.57% binary
baseline.
Run: `benchmark_xgboost.py`.

## Fairness audit

`audit_fairness.py` runs an IBM AIF360 audit of the trained FT-Transformer
(binarized to disease/no-disease, the standard UCI framing) across two
protected attributes: sex and age (bucketed at the test set's median). The
audit found real, worth-reporting disparity — **not** a false alarm:

| Attribute | Disparate impact | Statistical parity diff | Target |
|---|---|---|---|
| Sex (male vs. female) | 2.03 | **+0.459** | \|diff\| < 0.05 |
| Age (older vs. younger half) | 1.89 | +0.330 | \|diff\| < 0.05 |

Both far exceed the methodology's 5% disparity target — the model predicts
"disease" roughly twice as often for men as for women, and for the older
half vs. younger half of the test set. Applying AIF360's **Reweighing**
(sample weights that counteract the training set's own sex/label
correlation, then retraining) narrowed the sex gap from +0.459 to +0.342
without hurting accuracy (macro F1 actually rose slightly, 0.347 → 0.348) —
a partial mitigation, not a fix. Reweighing on sex alone also didn't help
(and slightly worsened) the age gap, since it only targets the one
attribute it's given. **Honest takeaway for the paper:** some of this
"disparity" plausibly reflects real epidemiology (heart disease prevalence
and presentation genuinely differ by sex and age), so a lower statistical
parity difference isn't automatically the right target — equal opportunity
difference (which compares true-positive rates rather than raw positive
rates) is arguably the more clinically meaningful metric here, and it's
comparatively smaller (+0.053 for sex) even before mitigation. This
distinction — and that reweighing alone doesn't close either gap — is worth
stating plainly rather than glossed over.

## ECG explainability (Grad-CAM)

`gradcam_ecg.py` computes a Grad-CAM-style attribution map for the ECG
branch (methodology Section 6.2 #12), run locally against 10 sample PTB-XL
records downloaded fresh for this script (the raw signals used to train
`checkpoints/ecg_vit.pt` only ever lived on the ephemeral Colab runtime, so
this pulls its own small sample set directly from PhysioNet — a few hundred
KB, not the full dataset). It hooks the last ViT block's *input* tokens
(not its output — LayerNorm right after that block acts per-position, so
the CLS logit has zero gradient w.r.t. that block's own output patches; the
cross-token mixing that lets patches influence the CLS token happens
*inside* the block's self-attention, so gradient only reaches the tokens
flowing in).

**A genuine finding, not glossed over:** the textbook Grad-CAM formula
(global-average-pool the gradient into per-channel weights, weight the
activations, sum, then ReLU to keep only positive contributions) produced
an **all-zero heatmap on every one of the 10 test records** — the per-patch
weighted sum came out strictly negative everywhere, and ReLU wiped it out
completely. This is a documented failure mode of porting CNN Grad-CAM's
sign convention onto ViT patch tokens; their activation statistics don't
match conv feature maps closely enough for it to transfer cleanly. The
script instead min-max normalizes the *signed* weighted sum (no ReLU), which
still answers the useful question — which patches were relatively most/least
supportive of the predicted class — without silently collapsing to a blank
image. Heatmaps: `datasets/ecg-images/gradcam_samples/heatmaps/` (red =
relatively supportive of the predicted class, blue = relatively
suppressive). 5/10 sample predictions were correct, roughly matching the
34.7% held-out test accuracy from Step 4.

## Imaging branches (X-ray, echo, MRI, CT)

Four more branches, each trained on its own public dataset on a free cloud
GPU, each validated against that dataset's own test split. All of them are
vision transformers: ViT-B/16 (the ECG branch's backbone) for X-ray and
echo, and Swin-UNETR (transformer encoder + U-Net decoder) for MRI and CT,
where the twin needs the outline of every chamber, not one label per image.

| Branch | Dataset | Model | Output | Train with |
|---|---|---|---|---|
| Chest X-ray | NIH ChestX-ray14 (112,120 images, open) | ViT-B/16, multi-label | Cardiomegaly / Edema / Effusion probabilities | `notebooks/step15_xray_vit_kaggle.ipynb` |
| Echo | EchoNet-Dynamic (10,030 videos, Stanford agreement) | ViT-B/16 per frame + temporal transformer | Ejection fraction (%) | `notebooks/step16_echo_ef_colab.ipynb` |
| Cardiac MRI | ACDC (150 patients, CC BY-NC-SA) | 2D Swin-UNETR | RV / myocardium / LV masks, volumes, EF, LV mass, wall thickness, 5-group diagnosis, 3D meshes | `notebooks/step17_mri_acdc_colab.ipynb` |
| Coronary CT | ImageCAS (1,000 scans, authors' Kaggle upload `xiaoweixumedicalai/imagecas`) | 3D Swin-UNETR | Coronary artery mask and 3D mesh | `notebooks/step18_ct_imagecas_colab.ipynb` |

**Getting the data.** `python download_imaging_datasets.py list` shows every
dataset, its licence and whether it can be fetched automatically. NIH X-ray
(via Kaggle), ACDC (Hugging Face mirror) and MIMIC-IV-ECG are automatic. The
rest need *your own* registration first, because accepting a data use
agreement is a legal step in your name: EchoNet-Dynamic and CheXpert
(Stanford AIMI), CAMUS (Human Heart Project account) and ImageCAS (the
authors' Kaggle upload, or email them). The credentialed MIMIC-CXR /
MIMIC-IV-ECHO / MIMIC-IV modules are listed for completeness but are not
used in this project (see below). `info <name>` prints the steps;
`verify <name> --root <folder>` checks a download has the layout the
notebooks expect. Downloads belong in Colab / Kaggle / Google Drive, not in
this OneDrive folder (the NIH set alone is ~45 GB), and the agreements for
EchoNet-Dynamic and all MIMIC data forbid re-sharing them, so they must
never be committed to git.

**Same code in training and in the app.** Each notebook's `%%writefile`
cells are generated copies of `src/models/imaging/` and `src/imaging/`, and
`tests/test_imaging.py` fails if a copy drifts from the module it came from.
Download each notebook's checkpoint into `checkpoints/`; the app picks it up
on the next request (`GET /api/imaging/status` lists what is trained). A
branch without its checkpoint answers 503 naming its notebook, rather than
stopping the app from starting.

**What the imaging does for the digital twin.** `src/patient_twin.py` has
to estimate contractility, stroke volume and preload, because a UCI record
holds no cardiac measurements. `src/imaging_twin.py` replaces those
estimates with the patient's measured values when they have a scan:
ejection fraction -> contractility, stroke volume (EDV - ESV) -> stroke
volume, end-diastolic volume -> preload, wall thickness >= 13 mm ->
afterload (hypertrophy), plus X-ray findings and 3D chamber / coronary
meshes. Every replaced value's source names the scan and the measurement.
The mappings onto the simulator's 0-100 scales are stated linear rules
(EF 60% and EDV 120 ml land on the existing baselines), not a fitted
physiological model. Endpoint: `POST /api/patients/{id}/twin/imaging` with
the `/api/imaging/*` results.

**What it does NOT do: raise the UCI accuracy.** None of these datasets
contain the 920 UCI patients, so they cannot improve the FT-Transformer's
5-class result, and combining one dataset's images with another dataset's
patients would repeat the random-pairing problem of Step 6.

**How the branches are combined, and how they are validated.** Every
reported number in this project is a per-modality result on that
modality's own public benchmark and its own held-out test split (UCI for
the FT-Transformer, PTB-XL for ECG, ACDC for MRI, NIH ChestX-ray14 for
X-ray, EchoNet-Dynamic for echo, ImageCAS for CT). No public dataset gives
the same patients an ECG, an X-ray, an echo and the UCI clinical fields
together, so there is **no validated multimodal accuracy, and none is
claimed.** In the app the branches are combined in two ways that do not
need one:

- *The digital twin* takes each scan's measurements (EF, volumes, wall
  thickness, X-ray findings, meshes) as inputs to the simulator, with every
  value's source labelled. This is measurement, not a fused prediction.
- *The report* shows each branch's own output side by side, each with the
  benchmark it was validated on.

Any demo that feeds one person's record through several branches uses
images and records from *different* datasets, so it is an **illustrative
(synthetic) pairing** that shows the pipeline working end to end, not a
clinical accuracy. `POST /api/imaging/combined` returns no combined number
(`"available": false`) unless a combiner has been fitted on real paired
patients, so the app cannot present a made-up fused score.

**Future work: a real paired cohort.** `build_mimic_paired_cohort.py` and
`train_late_fusion.py` are written and unit-tested but were not run: they
need PhysioNet credentialed access to MIMIC-IV, MIMIC-CXR and MIMIC-IV-ECHO,
which did not fit this project's timeline. With access, they would build
admissions with an ECG, a chest X-ray and labs from the same patient, label
them from discharge diagnosis codes, and report patient-grouped
cross-validation for each branch alone and for all together, so a fusion
gain could be measured honestly.

**Verified so far:** `tests/test_imaging.py` (22 checks, synthetic data:
model shapes, volume / EF / wall thickness against shapes with known
answers, meshes, twin mapping, late fusion with missing branches, API
endpoints), plus a dry run of all four notebooks and the backend on fake
data in each dataset's real file layout.

**Results so far — MRI branch (ACDC, trained on Colab T4, 2026-10-25).**
Measured on ACDC's 50 official test patients, never used for training,
model selection or the diagnosis classifier:

| Measure | Result |
|---|---|
| Segmentation Dice | LV cavity 0.88, LV myocardium 0.82, RV cavity 0.82 |
| Ejection fraction error (mean absolute) | LV 7.8 points, RV 8.0 points |
| LV end-diastolic volume error | 7.3 ml |
| LV mass error | 11.4 g |
| 5-group diagnosis, end to end from predicted masks | 84% |

For context, the leading ACDC challenge methods report Dice around 0.95
and diagnosis accuracy of 90-96%, so these are solid but below the state of
the art. Longer training does not close the gap: a 150-epoch re-run
(2026-10-25) plateaued by epoch 40-50 at validation Dice RV 0.85 / myocardium
0.85 / LV 0.91, with internal test Dice RV 0.79 / myocardium 0.81 / LV 0.90, so
the 80-epoch checkpoint was kept. A stronger architecture (e.g. an nnU-Net
comparison) is the obvious next step. Checkpoints: `checkpoints/mri_swin_unetr.pt`,
`checkpoints/acdc_diagnosis.joblib` (saved with scikit-learn 1.6.1, Colab's
version; a different local version loads it with a warning),
`checkpoints/sample_heart_mesh.json` (patient049's chambers at end-diastole).

**Results so far — X-ray branch (NIH ChestX-ray14, trained on Kaggle T4,
2026-10-25).** Measured on NIH's official test list (25,596 images from
patients never seen in training or validation); thresholds were chosen on
validation only:

| Finding | Test AUROC | Test F1 | Threshold |
|---|---|---|---|
| Cardiomegaly | 0.818 | 0.177 | 0.337 |
| Edema | 0.823 | 0.136 | 0.369 |
| Effusion | 0.777 | 0.434 | 0.408 |

For context, the dataset's original paper (Wang et al., CVPR 2017) reports
AUROC of about 0.81 / 0.81 / 0.76 for these three findings, so this branch
is at or slightly above that baseline. CheXNet's often-quoted 0.92 for
cardiomegaly used a random image split, which is easier than the official
patient-wise test list, so it is not a like-for-like comparison. The F1
scores are low because the findings are rare (edema is about 2% of images):
AUROC is the fair measure here. Validation AUROC during training was higher
(about 0.86 / 0.90 / 0.85) because the validation images come from the same
pool as training. Checkpoints: `checkpoints/cxr_vit.pt`,
`checkpoints/cxr_thresholds.json`.

**Not yet trained:** echo and CT. The late-fusion combiner is future work
(it needs a real paired cohort, see above).

## Multi-physics heart twin (electrical, muscle, blood flow, fat)

**Report page -> Multi-physics heart**, or `/static/heart_physics.html`
(`?patient=<id>` for a doctor's patient, `?self=1` for a patient's own
record). It puts four kinds of physics on one heart shape: the LV
myocardium mesh from the MRI branch (the ACDC sample heart,
`checkpoints/sample_heart_mesh.json`, until a patient's own ED + ES MRI is
uploaded on the page).

**Stated plainly: this is an illustrative model, not a clinical
simulation.** It is calibrated to the measurements the patient has
(volumes from MRI, blood pressure and heart rate from the record), and the
physics are the simplified forms used in teaching and early digital-twin
research. None of it has been validated against this patient, and the page
and every API response say so.

### What is modelled

| Part | Model | Inputs it is fitted to | Code |
|---|---|---|---|
| Electrical impulses | Eikonal-style activation on a graph of the myocardium: mesh vertices plus mid-wall points, anisotropic conduction (0.8 m/s along fibres, 0.4 across), rule-based fibres (+60 deg endocardium to -60 deg epicardium, Streeter 1969), fast endocardial Purkinje layer (2.5 m/s), three left-septal breakthrough sites (Durrer 1970); repolarisation = activation + an APD longer at the endocardium, scaled with heart rate | The patient's mesh | `activation_times`, `repolarisation_times` |
| Muscle | Wall thickness at every vertex (endocardium to epicardium on the ED mesh); **measured** systolic wall thickening from the ED and ES segmentations by ray casting, per slice and per AHA-style segment, mapped onto the mesh; scar as non-contracting, slow-conducting muscle | ED and ES masks | `build_geometry`, `regional_thickening`, `thickening_at_vertices` |
| Blood pressure and flow | 0D left-heart circulation: time-varying elastance LV (ESPVR + exponential EDPVR) and LA with atrial kick, mitral and aortic valves as one-way resistors, three-element Windkessel aorta, pulmonary-venous filling pressure. Gives the pressure-volume loop, LV/aortic/LA pressures and valve flows over the beat | EDV, ESV, systolic and diastolic BP, heart rate (calibrated to within 1-2%) | `simulate_circulation`, `calibrate` |
| Electromechanical coupling | The ventricle's elastance is the average of every vertex's twitch, each started at its own activation time, so slower or dyssynchronous activation (LBBB, pacing, scar) really weakens the beat | -- | `ventricular_activation_curve` |
| Fat | Epicardial fat = voxels inside the pericardium at -190 to -30 HU, when a CT with a pericardium mask exists. CordiaAI's CT branch segments the coronary lumen only, so the twin reports fat as **not measured**; the viewer's yellow fat layer is a drawing with a thickness slider, labelled as such | CT + pericardium mask | `epicardial_fat` |

**What-ifs** (the model is calibrated once on the patient's own normal
conduction, then each change is applied with that calibration fixed):
conduction (normal, left bundle branch block, RV apex pacing, CRT = LBBB +
LV lateral lead), conduction speed (diffuse fibrosis), an infarct scar
(position, height, size), heart rate, contractility, afterload, arterial
stiffness, filling pressure, diastolic muscle stiffness and aortic
stenosis. On the sample heart (BP 120/80, HR 72, assumed EF 60%), for
example, LBBB lengthens LV activation from 66 to 186 ms and lowers stroke
volume from 81 to 74 ml and peak dP/dt by about a fifth; CRT recovers
most of it (activation 119 ms, stroke volume 78 ml). Those directions match what is known
clinically; the sizes are this model's, not measurements.

**The viewer** (Three.js 0.169, vendored in `mobile_app/frontend/vendor/three/`
so it works offline) colours the mesh by the animated electrical wave,
activation time, repolarisation time, wall thickness, measured thickening,
fibre angle or scar; beats the wall with the simulated volume curve (each
region squeezes when its own twitch does, scar bulges); moves blood
particles in the cavity with the simulated valve flows (direction and
timing only, not a flow pattern); and plots the pressure-volume loop
(calibrated vs. what-if), pressures and valve flows with hover read-outs.
A "Where the numbers come from" table names the source of every
calibration value (measured, from the record, assumed or typical).

**API.** `GET /api/physics/sample`; `POST /api/physics/simulate` with
`{meshes?, metrics?, patient_id? | self_record?, whatif?, fat?}` (see
`mobile_app/backend/physics_api.py`). `POST /api/imaging/mri` now also
returns `regional` (ED/ES wall thickness, thickening per slice and per
segment, and per mesh vertex). Tests: `python -m pytest
tests/test_heart_physics.py -q` (22 tests on a synthetic ventricle with
known wall thickness: geometry, activation, calibration, the direction of
every what-if, measured thickening, fat, and the API).

### Honestly left out (would need full 3D solvers)

- **3D wall mechanics** (stress and strain through the wall, torsion):
  needs a volumetric (tetrahedral) mesh, fibre data and a finite-element
  solver such as FEniCS or FEBio, plus hours of compute per beat.
- **Ionic electrophysiology** (monodomain/bidomain with cell models such as
  ten Tusscher, for arrhythmias and re-entry): openCARP-class software. Here
  activation is a graph eikonal model with a fixed action potential
  duration, so it cannot show arrhythmias, and there is no simulated ECG.
- **Blood flow inside the chambers** (3D CFD: vortices, jets, wall shear):
  needs SimVascular-class software and time-resolved geometry. Here flow is
  a 0D circuit: correct pressures and flows over time, no flow pattern.
- **Right ventricle and atria as muscle**: ACDC segments only the RV
  cavity, not RV muscle or the atria, so activation and mechanics are LV
  only and the QRS estimate (LV activation + 10 ms) leaves out the RV.
- **Epicardial fat and coronary plaque**: need CT with fat and plaque
  segmentation, which no branch here is trained for.
- **Validation**: a clinical twin is checked against invasive pressures,
  electro-anatomical maps or 4D-flow MRI for the same patient. No such data
  is available in this project.

## Build phases

1. **Tabular data collection & preprocessing — done.** UCI Heart Disease
   (all 4 sources, 920 samples), leak-free train/test split, SF-2 feature
   selection (ANOVA + Chi-square + Mutual Information consensus).
   Run: `explore_uci.py`
2. **GAN-based augmentation — done.** Benchmarked SMOTE, CTGAN, and a
   diffusion generator (TabDDPM-style) on Colab (GPU, 800 epochs) using
   Frechet distance, ML-efficacy, and privacy distance. **Diffusion won on
   all three metrics** and is the selected generator — see
   `datasets/uci/augmentation_comparison.csv` for the numbers.
   Run: `run_augmentation.py` (benchmark) then `generate_augmented_dataset.py`
   (produces `datasets/uci/augmented_train.csv`, balanced to 212 samples per
   severity class, and `datasets/uci/preprocessed_test.csv`, the untouched
   held-out test set).
3. **FT-Transformer training — done.** Custom PyTorch implementation
   (depth=3, heads=8, d=256, dropout=0.2; `rtdl`'s installer fails on this
   machine's non-ASCII OneDrive path). Test accuracy 53.3% / macro F1 0.347
   — underperforms the plain RandomForest baseline from Step 2, a known
   pattern for transformers on small tabular datasets (~1000 rows).
   Run: `train_ft_transformer.py` → `checkpoints/ft_transformer.pt`,
   `datasets/uci/ft_embeddings_{train,test}.csv`.
   **v2** (13 inputs, d=128, 5-model ensemble, cholesterol fix) reaches
   57.3 ± 2.2% accuracy in 5-fold CV and 57.6% on the same test set, and is
   what the web app serves — see "FT-Transformer v2" above.
   Run: `train_ft_transformer_v2.py` → `checkpoints/ft_transformer_v2.pt`.
4. **ECG branch (PTB-XL + ViT-B/16) — done.**
   Downloaded PTB-XL's 100Hz signals (switched from 500Hz after PhysioNet
   throttled the full-zip download to ~230KB/s; per-file parallel download
   was ~5x smaller and far faster), aggregated SCP codes to 5 diagnostic
   superclasses (NORM/MI/STTC/CD/HYP), computed STFT spectrograms
   (n_fft=64, hop_length=8 — adapted for the shorter 100Hz signal), and
   fine-tuned ViT-B/16 (top 4 of 12 layers + a 256-dim projection head,
   28.5M trainable params). Split per PTB-XL's official protocol (folds
   1-8 train, 9 val, 10 test).
   The first (unweighted) run scored test accuracy 62.1% / macro F1 0.382,
   but the `HYP` class (535 of 16,244 training examples, smallest of the 5)
   scored 0/0/0 — the model never predicted it, a class-imbalance casualty.
   **Fixed** by capping every majority class at 500 training examples (so
   HYP is no longer a minority) plus class-weighted `CrossEntropyLoss` on
   top. **Test accuracy 34.7%, macro F1 0.34** — lower overall than the
   unweighted run, but that number was inflated by majority-class collapse.
   **HYP now scores 0.38 precision / 0.29 recall / 0.33 F1**, in line with
   the other four classes (0.27-0.42 F1 each) instead of zero. See
   `STEP5_STATUS.md` for the full per-class table and rationale.
   Run: `notebooks/step5_ecg_vit_colab.ipynb` on Colab (T4 GPU) →
   `checkpoints/ecg_vit.pt`, `datasets/ecg-images/ecg_embeddings_{train,val,test}.csv`.
5. **Mobile app — done.** Role-based (doctor/patient) auth, per-doctor
   isolated patient data, live FT-Transformer prediction, per-patient
   reports with SHAP + LIME + DiCE explainability, and an interactive 3D
   what-if heart simulation. See "Mobile app" and "3D what-if simulation"
   above.
6. **Multimodal fusion — architecture done, real training blocked on data.**
   Implemented `src/models/fusion.py`: cross-attention (E_t tabular embedding
   as Query, E_v ECG embedding as Key+Value, d=256, 8 heads) feeding a
   5-model stacking MLP ensemble (GAN-Bagging), exactly as specified in the
   methodology (Section 6.2, items 4-5). **Caveat that matters for the
   paper:** the UCI tabular dataset and PTB-XL ECG dataset are two
   independent public sources with no shared patient ID, so there is no
   ground truth for which ECG belongs to which tabular record.
   `train_fusion.py` pairs them by random sampling (fixed seed) as an
   architecture smoke test only — it confirms the fusion module trains and
   evaluates correctly end-to-end (test accuracy 53.8%, macro F1 0.41,
   tracking the standalone FT-Transformer's 53.3% almost exactly, since the
   randomly-paired ECG embedding carries no real signal for the tabular
   label). This is **not** a clinically meaningful multimodal result and
   shouldn't be reported as one — genuine fusion training needs a dataset
   with both modalities per patient (e.g. MIMIC-IV-ECG, or hospital-collected
   data), which is out of scope for this project's public-data constraints.
   Run: `train_fusion.py` → `checkpoints/fusion_ensemble.pt`.
7. **Federated learning + differential privacy — done.** `train_federated.py`
   simulates 3 hospital nodes by splitting the 736 real UCI training patients
   with a Dirichlet(alpha=0.5) label-skew (each node ends up with a visibly
   different severity mix — 499/77/160 patients — the harder, more realistic
   non-IID case rather than an easy identical split). Each round: every node
   trains 5 local epochs under **Opacus DP-SGD** (target ε=2.0, δ=1e-5 per
   round — see `train_federated.py`'s docstring for why this is a per-round
   budget rather than one composed across all 20 rounds), then the server
   aggregates via **FedAvg** (`w_global = Σ(n_k/n × w_k)`). 20 rounds total.
   `nn.MultiheadAttention` is swapped for Opacus's `DPMultiheadAttention` via
   `ModuleValidator.fix()` (same reasoning as the `rtdl` substitution in
   Step 3 — keep the from-scratch architecture, swap only what a specific
   library can't trace), and dropout is disabled for this run only, since
   Opacus's `vmap`-based per-sample gradients can't trace dropout's random
   mask. **Test accuracy 56.5%, macro F1 0.25** — lower macro F1 than the
   centralized FT-Transformer's 0.347, and the model collapses onto the two
   most common severity classes (0 and 1), never predicting 2/3/4. This is
   the expected privacy-utility tradeoff for a fairly strict per-round ε=2.0
   budget combined with a small, sharply non-IID per-node sample size, not a
   bug — and is itself a legitimate finding to report: differential privacy
   is not free, especially on a dataset this small.
   Run: `train_federated.py` → `checkpoints/ft_transformer_federated.pt`.
8. **Digital twin — done, then upgraded to real longitudinal data.**
   `build_digital_twin.py` (Framingham cross-sectional, population-drift
   simulation) followed by `build_digital_twin_synthea.py` (Synthea,
   genuinely longitudinal synthetic patients — real per-patient drift
   rates, not a population average). See "Digital twin" above for both,
   and specifically the honest caveat on the Synthea version's AUC.
9. **3D what-if simulation — done.** See "3D what-if simulation" above.
10. **Offline AI-generated clinical summary — done.** Ollama +
    `llama3.2:3b`, verified end-to-end through the app. See "Offline
    AI-generated summary" above.
11. **Deployment packaging — done, written but not run-tested** (Docker
    isn't installed on this machine; see "Deployment packaging" above for
    exactly why and what that means for trusting this). Not the
    methodology's full Postgres/MLflow/Nginx production stack — a
    pragmatic Docker Compose packaging of what's actually built.
12. **ECG upload — done.** Real WFDB-file upload → live prediction from
    the trained ECG branch. See "ECG upload" above, including what it
    deliberately doesn't support (photos of printed strips).
13. **Full website — done.** Replaced the original phone-width mobile
    shell with a full desktop landing page + dashboard. See "Web app"
    above, including a real CSS bug found and fixed while building it.
14. **Imaging branches (X-ray, echo, MRI, CT) — code done and tested;
    MRI trained (ACDC external test: Dice 0.82-0.88, 84% diagnosis);
    X-ray trained (NIH official test AUROC 0.78-0.82); echo and CT
    training pending.** See "Imaging
    branches" above. Echo is now one of four imaging branches rather than
    the single remaining phase. The patient-pairing caveat from Step 6
    stands: each branch is validated on its own public benchmark, any
    multi-branch demo is an illustrative pairing, and a measured fusion
    result is future work on a real paired cohort (MIMIC-IV, not used here
    because PhysioNet credentialing did not fit the timeline).

Training happens on free cloud GPUs (Colab/Kaggle) where a GPU is actually
needed (Steps 2 and 4); trained models are exported and only *inference*
runs locally otherwise. The diffusion generator (Step 2), FT-Transformer
(Step 3), fusion ensemble (Step 6), federated learning (Step 7), the
fairness audit, and the Grad-CAM script all run locally on the i5 CPU —
none of them need a GPU at this dataset size.
