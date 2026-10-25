"""Late fusion: one heart-disease prediction from whichever tests a patient has.

Each branch (ECG, X-ray, echo, MRI) is trained on its own public dataset.
For one patient, every branch they have data for runs on that patient, and
this combiner turns the branches' outputs into one prediction. A patient
with only an ECG and an X-ray still gets a prediction: missing branches are
filled with the training median and flagged by a has_<branch> indicator,
so the combiner learns how much to trust what is present.

What this deliberately does NOT do is train on rows stitched together from
different datasets' patients (the random-pairing problem train_fusion.py
documents). The combiner is only fitted on a cohort where every branch's
input comes from the SAME real patient -- build_mimic_paired_cohort.py
builds one from MIMIC-IV (ECG + chest X-ray + echo measurements + labs).
Until such a fit exists, the app shows each branch's findings side by side
and no combined number.

The UCI tabular branch is not an input here: MIMIC-IV does not record the
UCI stress-test features (cp, thalach, exang, oldpeak), so the FT-Transformer
cannot be run on MIMIC patients. Age, sex and blood labs take its place.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.ecg_vit import CLASSES as ECG_CLASSES
from src.models.imaging.cxr_vit import CXR_FINDINGS

BRANCH_FEATURES = {
    "clinical": ["age", "sex"],
    "ecg": [f"ecg_{c}" for c in ECG_CLASSES],
    "xray": [f"cxr_{f}" for f in CXR_FINDINGS],
    "echo": ["echo_lv_ef_percent"],
    "mri": ["mri_lv_ef_percent", "mri_lv_edv_ml", "mri_lv_mass_g", "mri_max_wall_thickness_mm"],
    "labs": ["lab_troponin_t", "lab_ntprobnp", "lab_creatinine", "lab_hemoglobin", "lab_glucose", "lab_cholesterol"],
}


def feature_columns(branches: list[str]) -> list[str]:
    cols = [c for b in branches for c in BRANCH_FEATURES[b]]
    return cols + [f"has_{b}" for b in branches if b != "clinical"]


def add_indicators(df: pd.DataFrame, branches: list[str]) -> pd.DataFrame:
    df = df.copy()
    for b in branches:
        cols = [c for c in BRANCH_FEATURES[b] if c in df]
        if b != "clinical":
            df[f"has_{b}"] = df[cols].notna().any(axis=1).astype(float) if cols else 0.0
        for c in BRANCH_FEATURES[b]:
            if c not in df:
                df[c] = np.nan
    return df


class LateFusionCombiner:
    def __init__(self, branches: list[str], label_names: list[str]):
        self.branches = branches
        self.label_names = label_names
        self.columns = feature_columns(branches)
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        self.meta: dict = {}

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_indicators(df, self.branches)[self.columns]

    def cross_validate(self, df: pd.DataFrame, y: np.ndarray, groups=None, folds: int = 5, seed: int = 42) -> np.ndarray:
        """Out-of-fold class probabilities, for an honest accuracy estimate. Pass
        groups=subject_id so one patient's admissions never sit in two folds."""
        if groups is None:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        else:
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return cross_val_predict(self.pipeline, self._matrix(df), y, cv=cv, groups=groups, method="predict_proba")

    def fit(self, df: pd.DataFrame, y: np.ndarray, **meta) -> "LateFusionCombiner":
        self.pipeline.fit(self._matrix(df), y)
        self.meta = {"n_patients": int(len(df)), **meta}
        return self

    def predict_proba(self, outputs: dict) -> dict[str, float]:
        """outputs: flat dict of any of the feature columns for ONE patient."""
        probs = self.pipeline.predict_proba(self._matrix(pd.DataFrame([outputs])))[0]
        return {self.label_names[i]: float(p) for i, p in enumerate(probs)}

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "LateFusionCombiner":
        return joblib.load(path)


def flatten_branch_outputs(ecg: dict | None = None, xray: dict | None = None, echo: dict | None = None,
                           mri: dict | None = None, clinical: dict | None = None, labs: dict | None = None) -> dict:
    """Branch outputs as the app produces them -> the combiner's flat feature dict.
    ecg / xray: {class: probability 0-1}; echo / mri: metric dicts from src/imaging."""
    row: dict = {}
    for c, p in (ecg or {}).items():
        row[f"ecg_{c}"] = p
    for f, p in (xray or {}).items():
        row[f"cxr_{f}"] = p
    if echo and echo.get("lv_ef_percent") is not None:
        row["echo_lv_ef_percent"] = echo["lv_ef_percent"]
    for k in ("lv_ef_percent", "lv_edv_ml", "lv_mass_g", "max_wall_thickness_mm"):
        if mri and mri.get(k) is not None:
            row[f"mri_{k}"] = mri[k]
    row.update(clinical or {})
    row.update({f"lab_{k}": v for k, v in (labs or {}).items()})
    return row
