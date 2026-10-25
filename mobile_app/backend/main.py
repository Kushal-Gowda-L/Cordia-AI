"""CordiaAI mobile app backend.

FastAPI serving: role-based auth (doctor / patient, signup/login), per-doctor
isolated patient data, live prediction via the trained FT-Transformer, and
per-patient report generation with SHAP + LIME + DiCE explainability plus a
direct sensitivity-analysis what-if section. Each patient keeps a history of
visits, so a follow-up is recorded against the same patient and can be
compared with earlier visits.

Everything here is real: the model is the actual trained checkpoint, the
preprocessing is the actual fitted pipeline, and every number in a report
comes from actually re-running the model or a real explainability library —
nothing is canned text.
"""
import hashlib
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import wfdb
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parents[1]  # cordia_ai/
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
sys.path.insert(0, str(REPO_ROOT))

from src.models.ecg_vit import CLASSES as ECG_CLASSES  # noqa: E402
from src.models.ecg_vit import ECGViT  # noqa: E402
from src.models.explain import Explainer  # noqa: E402
from src.models.ft_ensemble import EXTRA_COLS, load_v2  # noqa: E402
from src.models.ft_transformer import FTTransformer  # noqa: E402
from src.models.llm_report import generate_report as generate_llm_report  # noqa: E402
from src.preprocessing.clean import Preprocessor, split_features_target  # noqa: E402
from src.preprocessing.load_data import fetch_uci_heart_disease  # noqa: E402
from src.preprocessing.split import stratified_split  # noqa: E402
from src.imaging_twin import apply_imaging  # noqa: E402
from src.patient_twin import build_twin  # noqa: E402
from src.risk_reading import read_risk, zone  # noqa: E402

# Overridable so Docker Compose can point this at a mounted volume instead
# of the image's writable layer (which is lost when the container is
# recreated) -- see docker-compose.yml.
DB_PATH = Path(os.environ["CORDIA_DB_PATH"]) if "CORDIA_DB_PATH" in os.environ else BACKEND_DIR / "app.db"
FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]
# Which tabular model serves predictions: "v2" (checkpoints/ft_transformer_v2.pt,
# 5-model ensemble that also uses slope/ca/thal -- see src/models/ft_ensemble.py)
# or "v1" (checkpoints/ft_transformer.pt). Falls back to v1 if the v2 checkpoint
# hasn't been trained yet. Set CORDIA_MODEL=v1 to switch back.
MODEL_VERSION = os.environ.get("CORDIA_MODEL", "v2")
V2_CHECKPOINT = REPO_ROOT / "checkpoints" / "ft_transformer_v2.pt"
CLASSES = ["No disease", "Mild", "Moderate", "Severe", "Critical"]
# The model inputs stored for each visit (and mirrored on the patient row).
VISIT_COLS = FEATURE_COLS + EXTRA_COLS

app = FastAPI(title="CordiaAI Mobile API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: dict[str, int] = {}  # token -> user_id


# --------------------------------------------------------------- Database
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _backup_before_visits() -> None:
    """Copy the database aside the first time it is opened by a version that
    keeps visit histories, before anything in it is changed."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    has_visits = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'visits'").fetchone()
    conn.close()
    backup = DB_PATH.with_name(DB_PATH.name + ".pre-visits-backup")
    if not has_visits and not backup.exists():
        shutil.copy2(DB_PATH, backup)
        print(f"Backed up the database to {backup.name} before adding visit histories")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _backup_before_visits()
    conn = get_db()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'doctor',
            created_at REAL NOT NULL
        )"""
    )
    # patients.user_id is always a *doctor*'s id, whether the row was added
    # by a doctor for one of their patients, or is a patient-role user's own
    # self-assessment (self_user_id links it back to that account). This is
    # what gives each doctor an isolated set of patients: every query below
    # filters on user_id = the logged-in doctor.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            self_user_id INTEGER,
            name TEXT NOT NULL,
            age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER, chol INTEGER,
            fbs INTEGER, restecg INTEGER, thalach INTEGER, exang INTEGER, oldpeak REAL,
            slope INTEGER, ca INTEGER, thal INTEGER,
            risk_prob REAL, severity_idx INTEGER, model_version TEXT,
            created_at REAL NOT NULL,
            access_code_hash TEXT, access_code_expires REAL
        )"""
    )
    # Databases created before patient access codes, slope/ca/thal or model
    # versions existed gain the columns here.
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(patients)")}
    for column, decl in (
        ("access_code_hash", "TEXT"), ("access_code_expires", "REAL"),
        ("slope", "INTEGER"), ("ca", "INTEGER"), ("thal", "INTEGER"), ("model_version", "TEXT"),
    ):
        if column not in existing:
            conn.execute(f"ALTER TABLE patients ADD COLUMN {column} {decl}")
    # One row per visit: the values recorded at that visit and the risk the
    # model gave them. The patients row always mirrors the patient's latest
    # visit, so the report, digital twin and patient account show the current
    # state without having to know about visits.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER, chol INTEGER,
            fbs INTEGER, restecg INTEGER, thalach INTEGER, exang INTEGER, oldpeak REAL,
            slope INTEGER, ca INTEGER, thal INTEGER,
            risk_prob REAL, severity_idx INTEGER, model_version TEXT,
            note TEXT,
            created_at REAL NOT NULL,
            updated_at REAL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS visits_patient ON visits (patient_id, created_at)")
    # Patients recorded before visits existed become visit 1, dated when the
    # patient was added.
    cols = ", ".join(VISIT_COLS)
    migrated = conn.execute(
        f"""INSERT INTO visits (patient_id, {cols}, risk_prob, severity_idx, model_version, created_at)
            SELECT id, {cols}, risk_prob, severity_idx, model_version, created_at FROM patients
            WHERE id NOT IN (SELECT patient_id FROM visits)"""
    ).rowcount
    conn.commit()
    conn.close()
    if migrated:
        print(f"Recorded {migrated} existing patient(s) as their first visit")


# ------------------------------------------------------------------- Auth
def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def current_user(authorization: str | None) -> sqlite3.Row:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.removeprefix("Bearer ")
    user_id = SESSIONS.get(token)
    if user_id is None:
        raise HTTPException(401, "Invalid or expired session")
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(401, "Invalid session")
    return row


class AuthRequest(BaseModel):
    username: str
    password: str
    role: str = "doctor"


@app.post("/api/signup")
def signup(req: AuthRequest):
    if len(req.username.strip()) < 3 or len(req.password) < 4:
        raise HTTPException(400, "Username needs 3+ chars, password 4+ chars")
    if req.role not in ("doctor", "patient"):
        raise HTTPException(400, "role must be 'doctor' or 'patient'")
    conn = get_db()
    salt = secrets.token_hex(16)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (req.username.strip(), hash_password(req.password, salt), salt, req.role, time.time()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Username already taken")
    finally:
        conn.close()
    return login(AuthRequest(username=req.username, password=req.password))


@app.post("/api/login")
def login(req: AuthRequest):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (req.username.strip(),)).fetchone()
    conn.close()
    if row is None or hash_password(req.password, row["salt"]) != row["password_hash"]:
        raise HTTPException(401, "Incorrect username or password")
    token = secrets.token_hex(24)
    SESSIONS[token] = row["id"]
    return {"token": token, "username": row["username"], "role": row["role"]}


# -------------------------------------------------------------- ML model
_prep: Preprocessor | None = None
_model: FTTransformer | None = None
_explainer: Explainer | None = None
_ecg_model: ECGViT | None = None


def load_model() -> None:
    global _prep, _model, _explainer, MODEL_VERSION
    df = fetch_uci_heart_disease()
    X, y = split_features_target(df)
    X_train, _, y_train, _ = stratified_split(X, y)

    if MODEL_VERSION == "v2" and not V2_CHECKPOINT.exists():
        print(f"{V2_CHECKPOINT.name} not found (run train_ft_transformer_v2.py); using v1")
        MODEL_VERSION = "v1"

    if MODEL_VERSION == "v2":
        _prep, _model = load_v2(V2_CHECKPOINT)
        extra_cols = EXTRA_COLS
        # Cholesterol 0 means "not measured" in UCI; keep it out of the explainers' reference data.
        reference = X_train[FEATURE_COLS].replace({"chol": {0: np.nan}, "trestbps": {0: np.nan}})
    else:
        _prep = Preprocessor()
        _prep.fit(X_train, y_train)
        _model = FTTransformer(n_features=len(FEATURE_COLS), n_classes=5)
        state = torch.load(REPO_ROOT / "checkpoints" / "ft_transformer.pt", map_location="cpu", weights_only=True)
        _model.load_state_dict(state)
        _model.eval()
        extra_cols = []
        reference = X_train[FEATURE_COLS]

    X_train_imputed = reference.fillna(reference.median())
    _explainer = Explainer(_model, _prep, FEATURE_COLS, CLASSES, X_train_imputed, extra_cols=extra_cols)
    print(f"Tabular model: FT-Transformer {MODEL_VERSION}")


def rescore_patients() -> None:
    """Stored risk figures come from whichever model was live when a record was
    saved; re-score records saved under the other version so the patient list,
    reports and twin all use the model now serving."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM patients WHERE model_version IS NULL OR model_version != ?", (MODEL_VERSION,)
    ).fetchall()
    for row in rows:
        risk_prob, severity_idx, _ = predict(_raw_from_row(row))
        conn.execute(
            "UPDATE patients SET risk_prob = ?, severity_idx = ?, model_version = ? WHERE id = ?",
            (risk_prob, severity_idx, MODEL_VERSION, row["id"]),
        )
    visits = conn.execute(
        "SELECT * FROM visits WHERE model_version IS NULL OR model_version != ?", (MODEL_VERSION,)
    ).fetchall()
    for row in visits:
        risk_prob, severity_idx, _ = predict(_raw_from_row(row))
        conn.execute(
            "UPDATE visits SET risk_prob = ?, severity_idx = ?, model_version = ? WHERE id = ?",
            (risk_prob, severity_idx, MODEL_VERSION, row["id"]),
        )
    conn.commit()
    conn.close()
    if rows:
        print(f"Re-scored {len(rows)} stored patient(s) with FT-Transformer {MODEL_VERSION}")
    if visits:
        print(f"Re-scored {len(visits)} stored visit(s) with FT-Transformer {MODEL_VERSION}")


def load_ecg_model() -> None:
    global _ecg_model
    _ecg_model = ECGViT()
    state = torch.load(REPO_ROOT / "checkpoints" / "ecg_vit.pt", map_location="cpu", weights_only=True)
    _ecg_model.load_state_dict(state)
    _ecg_model.eval()


# ---------------------------------------------------------- ECG upload
ECG_LEAD_II_IDX = 1  # PTB-XL lead order: I, II, III, AVR, AVL, AVF, V1-V6


def wfdb_bytes_to_spectrogram(hea_bytes: bytes, dat_bytes: bytes, record_name: str) -> np.ndarray:
    """Same conversion the ECG branch was trained and Grad-CAM'd on
    (see gradcam_ecg.py): read the WFDB pair, take Lead II, STFT with the
    same n_fft/hop_length adapted for PTB-XL's 100Hz signal, normalise to
    a 224x224 3-channel image."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / f"{record_name}.hea").write_bytes(hea_bytes)
        (Path(tmp) / f"{record_name}.dat").write_bytes(dat_bytes)
        signal, _ = wfdb.rdsamp(str(Path(tmp) / record_name))

    lead = signal[:, ECG_LEAD_II_IDX].astype(np.float32)
    stft = librosa.stft(lead, n_fft=64, hop_length=8, window="hann")
    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    normalised = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-8)
    img = Image.fromarray((normalised * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.stack([np.array(img)] * 3, axis=-1)


@app.post("/api/ecg-predict")
async def ecg_predict(
    hea: UploadFile = File(...), dat: UploadFile = File(...), authorization: str | None = Header(None)
):
    """Upload a real WFDB-format ECG record (a .hea header + .dat signal
    file pair -- PTB-XL's native format, and the format PhysioNet ECGs
    generally ship in) and get a live prediction from the actual trained
    ECG ViT branch. This does NOT accept a photo of a printed ECG strip --
    that would need a fundamentally different "digitize a scanned
    waveform" step this endpoint doesn't do; feeding one in would silently
    produce meaningless output, so it isn't offered as an option."""
    current_user(authorization)
    hea_bytes, dat_bytes = await hea.read(), await dat.read()
    record_name = Path(hea.filename).stem
    try:
        image = wfdb_bytes_to_spectrogram(hea_bytes, dat_bytes, record_name)
    except Exception as e:
        raise HTTPException(400, f"Could not read this as a WFDB record (.hea + matching .dat): {e}")

    x = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
    x_t = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_ecg_model(x_t), dim=1)[0]
    pred_idx = int(probs.argmax())
    return {
        "predicted_class": ECG_CLASSES[pred_idx],
        "confidence_percent": round(float(probs[pred_idx]) * 100, 1),
        "class_probabilities": {ECG_CLASSES[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)},
    }


def predict(raw_row: dict) -> tuple[float, int, list[float]]:
    X = _prep.transform(pd.DataFrame([raw_row]))
    with torch.no_grad():
        logits = _model(torch.tensor(X.values, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return float(1 - probs[0]), int(probs.argmax()), [float(p) for p in probs]


def build_report(raw_row: dict, risk_prob: float, severity_idx: int, probs: list[float]) -> dict:
    findings = []
    if raw_row["chol"] >= 240:
        findings.append(f"Cholesterol is {raw_row['chol']} mg/dl — above the high-risk threshold (240 mg/dl).")
    elif raw_row["chol"] >= 200:
        findings.append(f"Cholesterol is {raw_row['chol']} mg/dl — borderline-high (200-239 mg/dl).")
    if raw_row["trestbps"] >= 140:
        findings.append(f"Resting blood pressure is {raw_row['trestbps']} mm Hg — Stage 2 hypertension range.")
    elif raw_row["trestbps"] >= 130:
        findings.append(f"Resting blood pressure is {raw_row['trestbps']} mm Hg — Stage 1 hypertension range.")
    if raw_row["fbs"] == 1:
        findings.append("Fasting blood sugar is elevated (>120 mg/dl).")
    if raw_row["exang"] == 1:
        findings.append("Patient experiences exercise-induced angina.")
    if raw_row["oldpeak"] >= 2.0:
        findings.append(f"ST depression (oldpeak) of {raw_row['oldpeak']} indicates significant exercise-induced cardiac stress.")

    what_if = []
    for feature, delta, label in [
        ("oldpeak", -1.0, "reducing ST depression by 1.0"),
        ("trestbps", -15, "lowering blood pressure by 15 mm Hg"),
        ("chol", -40, "lowering cholesterol by 40 mg/dl"),
    ]:
        modified = dict(raw_row)
        modified[feature] = max(0, modified[feature] + delta)
        new_risk, _, _ = predict(modified)
        change = risk_prob - new_risk
        if change > 0.01:
            what_if.append(
                f"{label.capitalize()} would change predicted risk from {risk_prob:.0%} to {new_risk:.0%} "
                f"({change:.0%} reduction), holding everything else constant."
            )

    shap_factors = _explainer.shap_disease_factors(raw_row)
    lime_factors = _explainer.lime_top_factors(raw_row, severity_idx)
    dice_cfs = _explainer.dice_counterfactuals(raw_row, severity_idx)

    return {
        "risk_percent": round(risk_prob * 100, 1),
        "severity": CLASSES[severity_idx],
        "class_probabilities": {CLASSES[i]: round(p * 100, 1) for i, p in enumerate(probs)},
        "reading": read_risk(probs, CLASSES),
        "findings": findings or ["No individual values fall outside common reference ranges."],
        "what_if": what_if or ["No single-factor adjustment in this simulation produced a >1% risk reduction."],
        "shap_factors": shap_factors,
        "lime_factors": lime_factors,
        "dice_counterfactuals": dice_cfs,
        "raw": raw_row,
    }


# --------------------------------------------------------------- Patients
class PatientIn(BaseModel):
    name: str
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    # Optional: results of the slope of the exercise ST segment, the number of
    # major vessels seen on fluoroscopy and the thallium scan. Used by v2 when
    # given; blank is fine (most UCI patients have them blank too).
    slope: int | None = Field(None, ge=1, le=3)
    ca: int | None = Field(None, ge=0, le=3)
    thal: int | None = Field(None)
    # Optional free text kept with the visit, e.g. "3-month follow-up, on statins".
    note: str | None = Field(None, max_length=500)


def _row_dict(patient: PatientIn) -> dict:
    row = patient.model_dump(exclude={"name", "note"})
    if row["thal"] not in (None, 3, 6, 7):
        raise HTTPException(422, "thal must be 3 (normal), 6 (fixed defect) or 7 (reversible defect)")
    return row


def _raw_from_row(row: sqlite3.Row) -> dict:
    """A stored patient's model inputs: the ten form fields plus slope/ca/thal (may be blank)."""
    return {c: row[c] for c in VISIT_COLS}


def _note(patient: PatientIn) -> str | None:
    return (patient.note or "").strip() or None


def _insert_patient(user_id: int, patient: PatientIn, self_user_id: int | None = None) -> tuple[int, float, int]:
    raw_row = _row_dict(patient)
    risk_prob, severity_idx, _ = predict(raw_row)
    now = time.time()
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO patients
           (user_id, self_user_id, name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
            slope, ca, thal, risk_prob, severity_idx, model_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, self_user_id, patient.name, *(raw_row[c] for c in VISIT_COLS),
         risk_prob, severity_idx, MODEL_VERSION, now),
    )
    patient_id = cur.lastrowid
    _insert_visit(conn, patient_id, raw_row, risk_prob, severity_idx, _note(patient), now)
    conn.commit()
    conn.close()
    return patient_id, risk_prob, severity_idx


# --------------------------------------------------------------- Visits
def _insert_visit(conn: sqlite3.Connection, patient_id: int, raw_row: dict, risk_prob: float,
                  severity_idx: int, note: str | None, created_at: float) -> int:
    cols = ", ".join(VISIT_COLS)
    cur = conn.execute(
        f"""INSERT INTO visits (patient_id, {cols}, risk_prob, severity_idx, model_version, note, created_at)
            VALUES (?, {", ".join("?" for _ in VISIT_COLS)}, ?, ?, ?, ?, ?)""",
        (patient_id, *(raw_row[c] for c in VISIT_COLS), risk_prob, severity_idx, MODEL_VERSION, note, created_at),
    )
    return cur.lastrowid


def _visits_of(conn: sqlite3.Connection, patient_id: int) -> list[sqlite3.Row]:
    """A patient's visits, oldest first (visit 1 is the first assessment)."""
    return conn.execute(
        "SELECT * FROM visits WHERE patient_id = ? ORDER BY created_at, id", (patient_id,)
    ).fetchall()


def _sync_patient_to_latest_visit(conn: sqlite3.Connection, patient_id: int) -> None:
    """Make the patient row show the latest visit's values and risk."""
    latest = _visits_of(conn, patient_id)[-1]
    conn.execute(
        f"""UPDATE patients SET {", ".join(f"{c} = ?" for c in VISIT_COLS)},
            risk_prob = ?, severity_idx = ?, model_version = ? WHERE id = ?""",
        (*(latest[c] for c in VISIT_COLS), latest["risk_prob"], latest["severity_idx"],
         latest["model_version"], patient_id),
    )


def _doctor_patient(conn: sqlite3.Connection, user: sqlite3.Row, patient_id: int) -> sqlite3.Row:
    if user["role"] != "doctor":
        conn.close()
        raise HTTPException(403, "Only doctor accounts can manage visits")
    row = conn.execute("SELECT * FROM patients WHERE id = ? AND user_id = ?", (patient_id, user["id"])).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(404, "Patient not found")
    return row


def _visit_json(row: sqlite3.Row, number: int) -> dict:
    risk_percent = round(row["risk_prob"] * 100, 1)
    return {
        "id": row["id"], "number": number, "created_at": row["created_at"], "updated_at": row["updated_at"],
        "note": row["note"], "inputs": _raw_from_row(row),
        "risk_percent": risk_percent, "severity": CLASSES[row["severity_idx"]], **zone(risk_percent),
    }


# -- Doctor flow: manage many patients, strictly scoped to this doctor's own data --
@app.post("/api/patients")
def add_patient(patient: PatientIn, authorization: str | None = Header(None)):
    user = current_user(authorization)
    if user["role"] != "doctor":
        raise HTTPException(403, "Only doctor accounts can add patients")
    patient_id, risk_prob, severity_idx = _insert_patient(user["id"], patient)
    return {"id": patient_id, "risk_percent": round(risk_prob * 100, 1), "severity": CLASSES[severity_idx]}


@app.get("/api/patients")
def list_patients(authorization: str | None = Header(None)):
    user = current_user(authorization)
    if user["role"] != "doctor":
        raise HTTPException(403, "Only doctor accounts have a patient list")
    conn = get_db()
    rows = conn.execute(
        """SELECT p.id, p.name, p.age, p.sex, p.risk_prob, p.severity_idx, p.created_at, p.self_user_id,
                  COUNT(v.id) AS visit_count, COALESCE(MAX(v.created_at), p.created_at) AS last_visit_at
           FROM patients p LEFT JOIN visits v ON v.patient_id = p.id
           WHERE p.user_id = ? GROUP BY p.id ORDER BY last_visit_at DESC""",
        (user["id"],),
    ).fetchall()
    conn.close()
    return [
        {"id": r["id"], "name": r["name"], "age": r["age"], "sex": "Male" if r["sex"] == 1 else "Female",
         "risk_percent": round(r["risk_prob"] * 100, 1), "severity": CLASSES[r["severity_idx"]],
         "created_at": r["created_at"], "last_visit_at": r["last_visit_at"], "visit_count": r["visit_count"],
         "linked": r["self_user_id"] is not None,
         **zone(r["risk_prob"] * 100)}
        for r in rows
    ]


@app.get("/api/patients/{patient_id}/report")
def get_report(patient_id: int, authorization: str | None = Header(None)):
    user = current_user(authorization)
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id = ? AND user_id = ?", (patient_id, user["id"])).fetchone()
    linked = None
    visits = []
    if row is not None:
        visits = _visits_of(conn, patient_id)
        if row["self_user_id"] is not None:
            linked = conn.execute("SELECT username FROM users WHERE id = ?", (row["self_user_id"],)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, "Patient not found")
    raw_row = _raw_from_row(row)
    _, _, probs = predict(raw_row)
    report = build_report(raw_row, row["risk_prob"], row["severity_idx"], probs)
    report["name"] = row["name"]
    report["age"] = row["age"]
    report["patient_access"] = {"linked": linked is not None, "username": linked["username"] if linked else None}
    # The report describes the latest visit; the history lets the page offer comparisons.
    report["visits"] = [_visit_json(v, i + 1) for i, v in enumerate(visits)]
    return report


@app.post("/api/patients/{patient_id}/narrate")
def narrate_patient(patient_id: int, authorization: str | None = Header(None)):
    user = current_user(authorization)
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id = ? AND user_id = ?", (patient_id, user["id"])).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, "Patient not found")
    raw_row = _raw_from_row(row)
    _, _, probs = predict(raw_row)
    report = build_report(raw_row, row["risk_prob"], row["severity_idx"], probs)
    report["name"] = row["name"]
    report["age"] = row["age"]
    return generate_llm_report(report)


@app.delete("/api/patients/{patient_id}")
def delete_patient(patient_id: int, authorization: str | None = Header(None)):
    user = current_user(authorization)
    conn = get_db()
    deleted = conn.execute("DELETE FROM patients WHERE id = ? AND user_id = ?", (patient_id, user["id"])).rowcount
    if deleted:
        conn.execute("DELETE FROM visits WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


# -- Visits: a returning patient's follow-up is recorded against the same patient --
@app.get("/api/patients/{patient_id}/visits")
def list_visits(patient_id: int, authorization: str | None = Header(None)):
    user = current_user(authorization)
    conn = get_db()
    row = _doctor_patient(conn, user, patient_id)
    visits = _visits_of(conn, patient_id)
    conn.close()
    return {"id": row["id"], "name": row["name"], "visits": [_visit_json(v, i + 1) for i, v in enumerate(visits)]}


@app.post("/api/patients/{patient_id}/visits")
def add_visit(patient_id: int, patient: PatientIn, authorization: str | None = Header(None)):
    """Record a follow-up visit: new values, re-scored by the model now serving.
    Earlier visits are kept unchanged for comparison."""
    user = current_user(authorization)
    raw_row = _row_dict(patient)
    risk_prob, severity_idx, _ = predict(raw_row)
    conn = get_db()
    _doctor_patient(conn, user, patient_id)
    visit_id = _insert_visit(conn, patient_id, raw_row, risk_prob, severity_idx, _note(patient), time.time())
    conn.execute("UPDATE patients SET name = ? WHERE id = ?", (patient.name, patient_id))
    _sync_patient_to_latest_visit(conn, patient_id)
    conn.commit()
    conn.close()
    return {"id": patient_id, "visit_id": visit_id, "risk_percent": round(risk_prob * 100, 1),
            "severity": CLASSES[severity_idx]}


@app.put("/api/patients/{patient_id}/visits/{visit_id}")
def update_visit(patient_id: int, visit_id: int, patient: PatientIn, authorization: str | None = Header(None)):
    """Correct the values recorded at a visit (for example a typing mistake).
    The visit keeps its date and is re-scored."""
    user = current_user(authorization)
    raw_row = _row_dict(patient)
    risk_prob, severity_idx, _ = predict(raw_row)
    conn = get_db()
    _doctor_patient(conn, user, patient_id)
    updated = conn.execute(
        f"""UPDATE visits SET {", ".join(f"{c} = ?" for c in VISIT_COLS)},
            risk_prob = ?, severity_idx = ?, model_version = ?, note = ?, updated_at = ?
            WHERE id = ? AND patient_id = ?""",
        (*(raw_row[c] for c in VISIT_COLS), risk_prob, severity_idx, MODEL_VERSION, _note(patient), time.time(),
         visit_id, patient_id),
    ).rowcount
    if not updated:
        conn.close()
        raise HTTPException(404, "Visit not found")
    conn.execute("UPDATE patients SET name = ? WHERE id = ?", (patient.name, patient_id))
    _sync_patient_to_latest_visit(conn, patient_id)
    conn.commit()
    conn.close()
    return {"id": patient_id, "visit_id": visit_id, "risk_percent": round(risk_prob * 100, 1),
            "severity": CLASSES[severity_idx]}


@app.delete("/api/patients/{patient_id}/visits/{visit_id}")
def delete_visit(patient_id: int, visit_id: int, authorization: str | None = Header(None)):
    """Remove a visit recorded by mistake. A patient always keeps at least one
    visit; to remove the last one, delete the patient."""
    user = current_user(authorization)
    conn = get_db()
    _doctor_patient(conn, user, patient_id)
    visits = _visits_of(conn, patient_id)
    if not any(v["id"] == visit_id for v in visits):
        conn.close()
        raise HTTPException(404, "Visit not found")
    if len(visits) == 1:
        conn.close()
        raise HTTPException(400, "This is the patient's only visit. Delete the patient instead.")
    conn.execute("DELETE FROM visits WHERE id = ?", (visit_id,))
    _sync_patient_to_latest_visit(conn, patient_id)
    conn.commit()
    conn.close()
    return {"ok": True}


# -- Patient access: the doctor records the clinical results; the patient views them --
# Patients can't be expected to know their chest-pain classification, ECG or
# stress-test values, so they don't enter them. Instead a doctor issues a
# one-time access code for a record and the patient enters it in their own
# account to link it. Only a hash of the code is stored, and codes expire.
ACCESS_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I look-alikes
ACCESS_CODE_TTL_SECONDS = 7 * 24 * 3600


def _hash_access_code(code: str) -> str:
    normalised = code.replace("-", "").replace(" ", "").upper()
    return hashlib.sha256(normalised.encode()).hexdigest()


@app.post("/api/patients/{patient_id}/access-code")
def create_access_code(patient_id: int, authorization: str | None = Header(None)):
    user = current_user(authorization)
    if user["role"] != "doctor":
        raise HTTPException(403, "Only doctor accounts can give patients access")
    conn = get_db()
    row = conn.execute("SELECT id FROM patients WHERE id = ? AND user_id = ?", (patient_id, user["id"])).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(404, "Patient not found")
    raw = "".join(secrets.choice(ACCESS_CODE_ALPHABET) for _ in range(8))
    code = f"{raw[:4]}-{raw[4:]}"
    expires = time.time() + ACCESS_CODE_TTL_SECONDS
    # Issuing a new code replaces (and so cancels) any earlier one.
    conn.execute(
        "UPDATE patients SET access_code_hash = ?, access_code_expires = ? WHERE id = ?",
        (_hash_access_code(code), expires, patient_id),
    )
    conn.commit()
    conn.close()
    return {"code": code, "expires_at": expires}


class LinkIn(BaseModel):
    code: str = Field(min_length=8, max_length=20)


# ------------------------------------------------ Digital twin what-if
# The digital twin's controls are the same ten fields the risk model uses.
# Each change re-runs the real model and re-derives the simulated heart from
# the changed inputs (src/patient_twin.py), so the risk estimate and the heart
# always describe the same inputs. Ranges match the data-entry form; nothing
# is saved -- the patient's stored record is untouched.
class TwinInputs(BaseModel):
    age: int = Field(ge=28, le=77)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: int = Field(ge=90, le=200)
    chol: int = Field(ge=100, le=600)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=60, le=202)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-2, le=6)
    # Not twin controls: carried over unchanged from the patient's record.
    slope: int | None = None
    ca: int | None = None
    thal: int | None = None


@app.post("/api/twin/what-if")
def twin_what_if(inputs: TwinInputs, authorization: str | None = Header(None)):
    current_user(authorization)
    raw_row = inputs.model_dump()
    risk_prob, severity_idx, probs = predict(raw_row)
    return build_twin("What-if", raw_row, risk_prob, severity_idx, probs, CLASSES)


# -- Patient account: views the record a doctor linked to it --
def _my_record(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    # A record a doctor linked comes before any older self-entered one
    # (patients entered their own values before access codes existed).
    return conn.execute(
        "SELECT * FROM patients WHERE self_user_id = ? ORDER BY (user_id = self_user_id), created_at DESC",
        (user_id,),
    ).fetchone()


@app.get("/api/me")
def get_my_assessment(authorization: str | None = Header(None)):
    user = current_user(authorization)
    if user["role"] != "patient":
        raise HTTPException(403, "Only patient accounts have a linked record")
    conn = get_db()
    row = _my_record(conn, user["id"])
    conn.close()
    if row is None:
        return None
    return {"name": row["name"], "risk_percent": round(row["risk_prob"] * 100, 1), "severity": CLASSES[row["severity_idx"]]}


@app.post("/api/me/link")
def link_my_record(req: LinkIn, authorization: str | None = Header(None)):
    user = current_user(authorization)
    if user["role"] != "patient":
        raise HTTPException(403, "Only patient accounts can link a record")
    conn = get_db()
    row = conn.execute(
        "SELECT id, self_user_id, access_code_expires FROM patients WHERE access_code_hash = ?",
        (_hash_access_code(req.code),),
    ).fetchone()
    if row is None or row["access_code_expires"] is None or row["access_code_expires"] < time.time():
        conn.close()
        raise HTTPException(400, "That access code isn't valid or has expired. Ask your clinic for a new one.")
    if row["self_user_id"] not in (None, user["id"]):
        conn.close()
        raise HTTPException(409, "This record is already linked to another account.")
    # Codes work once: linking clears the stored hash.
    conn.execute(
        "UPDATE patients SET self_user_id = ?, access_code_hash = NULL, access_code_expires = NULL WHERE id = ?",
        (user["id"], row["id"]),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/me/report")
def get_my_report(authorization: str | None = Header(None)):
    user = current_user(authorization)
    if user["role"] != "patient":
        raise HTTPException(403, "Only patient accounts have a self-report")
    conn = get_db()
    row = _my_record(conn, user["id"])
    conn.close()
    if row is None:
        raise HTTPException(404, "No record is linked to this account yet")
    raw_row = _raw_from_row(row)
    _, _, probs = predict(raw_row)
    report = build_report(raw_row, row["risk_prob"], row["severity_idx"], probs)
    report["name"] = row["name"]
    report["age"] = row["age"]
    return report


@app.post("/api/me/narrate")
def narrate_my_report(authorization: str | None = Header(None)):
    user = current_user(authorization)
    if user["role"] != "patient":
        raise HTTPException(403, "Only patient accounts have a self-report")
    conn = get_db()
    row = _my_record(conn, user["id"])
    conn.close()
    if row is None:
        raise HTTPException(404, "No record is linked to this account yet")
    raw_row = _raw_from_row(row)
    _, _, probs = predict(raw_row)
    report = build_report(raw_row, row["risk_prob"], row["severity_idx"], probs)
    report["name"] = row["name"]
    report["age"] = row["age"]
    return generate_llm_report(report)


# ---------------------------------------------------------- Digital twin
# Parameters for the 3D heart digital twin (heart_digital_twin_3d/), derived
# from the patient's stored record -- see src/patient_twin.py for exactly
# which values come from the record and which are estimated.
def _twin_for_row(row: sqlite3.Row) -> dict:
    raw_row = _raw_from_row(row)
    _, _, probs = predict(raw_row)
    return build_twin(row["name"], raw_row, row["risk_prob"], row["severity_idx"], probs, CLASSES)


# Imaging (X-ray, echo, MRI, CT) and late fusion -- see imaging_api.py. Each
# branch loads its checkpoint on first use, so the app still starts before
# the imaging branches have been trained.
sys.path.insert(0, str(BACKEND_DIR))
from imaging_api import build_router as build_imaging_router  # noqa: E402

app.include_router(build_imaging_router(current_user))


class ImagingIn(BaseModel):
    """Results of the /api/imaging/* endpoints for this patient; any may be left out."""
    mri: dict | None = None
    echo: dict | None = None
    xray: dict | None = None
    meshes: dict | None = None


@app.get("/api/patients/{patient_id}/twin")
def get_patient_twin(patient_id: int, authorization: str | None = Header(None)):
    user = current_user(authorization)
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id = ? AND user_id = ?", (patient_id, user["id"])).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, "Patient not found")
    return _twin_for_row(row)


@app.get("/api/me/twin")
def get_my_twin(authorization: str | None = Header(None)):
    user = current_user(authorization)
    if user["role"] != "patient":
        raise HTTPException(403, "Only patient accounts have a self-record")
    conn = get_db()
    row = _my_record(conn, user["id"])
    conn.close()
    if row is None:
        raise HTTPException(404, "No record is linked to this account yet")
    return _twin_for_row(row)


@app.post("/api/patients/{patient_id}/twin/imaging")
def get_patient_twin_with_imaging(patient_id: int, imaging: ImagingIn, authorization: str | None = Header(None)):
    """The patient's twin with measured imaging values (EF, volumes, wall thickness,
    X-ray findings, chamber meshes) replacing the estimated ones -- see src/imaging_twin.py."""
    user = current_user(authorization)
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id = ? AND user_id = ?", (patient_id, user["id"])).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, "Patient not found")
    return apply_imaging(_twin_for_row(row), imaging.model_dump())


# Multi-physics heart twin on the MRI mesh (electrical activation, muscle, blood
# pressure and flow, fat) -- see physics_api.py and src/imaging/heart_physics.py.
# Blood pressure and heart rate come from the same record-derived twin as above.
from physics_api import build_router as build_physics_router  # noqa: E402

app.include_router(build_physics_router(
    current_user,
    lambda authorization, patient_id: get_my_twin(authorization) if patient_id is None
    else get_patient_twin(patient_id, authorization),
))


# ---------------------------------------------------------------- Static
@app.on_event("startup")
def startup():
    init_db()
    load_model()
    rescore_patients()
    load_ecg_model()


# Browsers otherwise keep using a cached app.js / style.css / twin page after
# an update and show the old screens. "no-cache" still lets them reuse a cached
# file, but only after checking with the server (a cheap 304 when unchanged).
@app.middleware("http")
async def revalidate_frontend_files(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith(("/static/", "/twin")):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# The 3D digital twin's production build (`npm run build` in
# heart_digital_twin_3d/). Served from this same origin so it can reuse the
# main app's login token; skipped if it hasn't been built.
TWIN_DIST = REPO_ROOT / "heart_digital_twin_3d" / "dist"
if TWIN_DIST.exists():
    app.mount("/twin", StaticFiles(directory=str(TWIN_DIST), html=True), name="twin")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
