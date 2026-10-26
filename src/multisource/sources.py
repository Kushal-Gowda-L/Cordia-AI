"""The datasets, one encoder each, and how a patient profile maps into each one.

Every `Source` says three things:
- which file holds it and how to turn that file into model columns + a 0/1 target
  (`load`),
- the model columns themselves (`features`), each tagged with the profile
  fields it is derived from (`concepts`), which is how explanations are
  reported in the profile's words,
- how to build one row of those columns from a patient profile (`from_profile`).
  A column the profile cannot supply is left out of the row and fed to the
  encoder as "not provided" (a learned mask token), never guessed.

Columns with no profile equivalent (race, income, education, health-insurance
status) are deliberately not model inputs: they could never be supplied at
prediction time, so they would only inflate the test-set numbers.

Datasets that are NOT used, and why:
- `heart.csv` (1025 rows) and `Heart Attack Data Set.csv` (303 rows) from Kaggle
  are both the UCI Cleveland subset. heart.csv is the same 302 unique patients
  repeated about 3.4 times, so a random split puts copies of a test patient in
  the training set. In both, `target` is inverted relative to UCI's original
  `num` (target 1 = no disease; checked by joining on age, sex, blood pressure,
  cholesterol, max heart rate and ST depression). The `uci` source uses the
  original four-site UCI file instead, which contains Cleveland with the correct
  labels plus Hungary, Switzerland and VA Long Beach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.multisource import profile as P

REPO_DATASETS = Path(__file__).resolve().parents[2] / "datasets"


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "num" or "cat"
    concepts: tuple[str, ...]
    categories: tuple = ()


@dataclass
class Source:
    name: str
    title: str
    outcome: str
    files: tuple[str, ...]
    features: list[Feature]
    _load: Callable[[Path], pd.DataFrame]
    _from_profile: Callable[[dict], dict]
    kind: str = "tabular"  # or "beat"
    notes: str = ""
    extra_dirs: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def find_file(self, data_dir: Path | None) -> Path:
        """First match of any of `files` (or the same name + .zip) in data_dir,
        its subfolders, or the repo's own datasets folders."""
        dirs = ([Path(data_dir)] if data_dir else []) + list(self.extra_dirs)
        for d in dirs:
            if not d.exists():
                continue
            for name in self.files:
                for candidate in (name, name + ".zip"):
                    direct = d / candidate
                    if direct.is_file():
                        return direct
                    found = sorted(f for f in d.rglob(candidate) if f.is_file())
                    if found:
                        return found[0]
        raise FileNotFoundError(
            f"{self.name}: none of {list(self.files)} (or .zip) found under "
            f"{[str(d) for d in dirs]}"
        )

    def load(self, data_dir: Path | None) -> pd.DataFrame:
        df = self._load(self.find_file(data_dir))
        missing = [c for c in self.feature_names + ["target"] if c not in df]
        if missing:
            raise ValueError(f"{self.name}: loader did not produce {missing}")
        return df[self.feature_names + ["target"]].reset_index(drop=True)

    def from_profile(self, profile: dict) -> dict:
        """Profile (already normalized) -> {column: value}; underivable columns omitted."""
        row = self._from_profile(profile)
        return {k: v for k, v in row.items() if k in self.feature_names and v is not None
                and not (isinstance(v, float) and np.isnan(v))}

    def provided_concepts(self, profile: dict) -> dict[str, list[str]]:
        """column -> the profile fields this patient actually supplied for it."""
        row = self.from_profile(profile)
        return {f.name: [c for c in f.concepts if c in profile] for f in self.features if f.name in row}


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _yes_no(series) -> pd.Series:
    return series.map({"Yes": 1, "No": 0})


def _sex01(p: dict):
    return {"male": 1, "female": 0}.get(p.get("sex"))


def _get(p: dict, key: str):
    return p.get(key)


# ------------------------------------------------------------------ UCI
_CP = {"typical": 1, "atypical": 2, "non_anginal": 3, "asymptomatic": 4}
_RESTECG = {"normal": 0, "st_t_abnormality": 1, "lvh": 2}
_SLOPE = {"up": 1, "flat": 2, "down": 3}
_THAL = {"normal": 3, "fixed_defect": 6, "reversible_defect": 7}


def _load_uci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("trestbps", "chol"):  # 0 is how UCI stores a missing value here
        df.loc[df[c] == 0, c] = np.nan
    df["target"] = (df["num"] > 0).astype(int)
    return df


def _uci_from_profile(p: dict) -> dict:
    return {
        "age": p.get("age"), "sex": _sex01(p), "cp": _CP.get(p.get("chest_pain_type")),
        "trestbps": p.get("systolic_bp"), "chol": p.get("total_cholesterol"), "fbs": P.fasting_sugar_high(p),
        "restecg": _RESTECG.get(p.get("resting_ecg")), "thalach": p.get("max_heart_rate"),
        "exang": p.get("exercise_angina"), "oldpeak": p.get("st_depression"),
        "slope": _SLOPE.get(p.get("st_slope")), "ca": p.get("major_vessels"),
        "thal": _THAL.get(p.get("thalassemia")),
    }


UCI = Source(
    name="uci",
    title="UCI Heart Disease (Cleveland, Hungary, Switzerland, VA; 920 patients)",
    outcome="Coronary artery disease on angiography (>50% narrowing in any major vessel)",
    files=("heart_disease_raw.csv",),
    extra_dirs=(REPO_DATASETS / "uci",),
    features=[
        Feature("age", "num", ("age",)),
        Feature("sex", "cat", ("sex",), (0, 1)),
        Feature("cp", "cat", ("chest_pain_type",), (1, 2, 3, 4)),
        Feature("trestbps", "num", ("systolic_bp",)),
        Feature("chol", "num", ("total_cholesterol",)),
        Feature("fbs", "cat", ("glucose", "diabetes"), (0, 1)),
        Feature("restecg", "cat", ("resting_ecg",), (0, 1, 2)),
        Feature("thalach", "num", ("max_heart_rate",)),
        Feature("exang", "cat", ("exercise_angina",), (0, 1)),
        Feature("oldpeak", "num", ("st_depression",)),
        Feature("slope", "cat", ("st_slope",), (1, 2, 3)),
        Feature("ca", "num", ("major_vessels",)),
        Feature("thal", "cat", ("thalassemia",), (3, 6, 7)),
    ],
    _load=_load_uci,
    _from_profile=_uci_from_profile,
    notes="slope, ca and thal are blank for most non-Cleveland patients; blanks get their own "
          "learned 'not recorded' token, separate from the 'not provided' token used at prediction time.",
)


# ------------------------------------------------------------------ Framingham
def _load_framingham(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target"] = df["TenYearCHD"].astype(int)
    return df


def _framingham_from_profile(p: dict) -> dict:
    cigs = p.get("cigs_per_day")
    if cigs is None and p.get("smoker") == 0:
        cigs = 0.0
    return {
        "male": _sex01(p), "age": p.get("age"), "currentSmoker": p.get("smoker"), "cigsPerDay": cigs,
        "BPMeds": p.get("bp_meds"), "prevalentStroke": p.get("stroke_history"), "prevalentHyp": P.high_bp(p),
        "diabetes": p.get("diabetes"), "totChol": p.get("total_cholesterol"), "sysBP": p.get("systolic_bp"),
        "diaBP": p.get("diastolic_bp"), "BMI": p.get("bmi"), "heartRate": p.get("resting_heart_rate"),
        "glucose": p.get("glucose"),
    }


FRAMINGHAM = Source(
    name="framingham",
    title="Framingham Heart Study teaching subset (4,240 people)",
    outcome="Coronary heart disease within 10 years",
    files=("framingham.csv",),
    extra_dirs=(REPO_DATASETS / "framingham",),
    features=[
        Feature("male", "cat", ("sex",), (0, 1)),
        Feature("age", "num", ("age",)),
        Feature("currentSmoker", "cat", ("smoker",), (0, 1)),
        Feature("cigsPerDay", "num", ("cigs_per_day", "smoker")),
        Feature("BPMeds", "cat", ("bp_meds",), (0, 1)),
        Feature("prevalentStroke", "cat", ("stroke_history",), (0, 1)),
        Feature("prevalentHyp", "cat", ("systolic_bp", "diastolic_bp", "bp_meds"), (0, 1)),
        Feature("diabetes", "cat", ("diabetes",), (0, 1)),
        Feature("totChol", "num", ("total_cholesterol",)),
        Feature("sysBP", "num", ("systolic_bp",)),
        Feature("diaBP", "num", ("diastolic_bp",)),
        Feature("BMI", "num", ("bmi",)),
        Feature("heartRate", "num", ("resting_heart_rate",)),
        Feature("glucose", "num", ("glucose",)),
    ],
    _load=_load_framingham,
    _from_profile=_framingham_from_profile,
)


# ------------------------------------------------------------------ Cardio (70k)
def _load_cardio(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["age_years"] = df["age"] / 365.25
    df["sex"] = (df["gender"] == 2).astype(int)  # 1 = women, 2 = men in this dataset
    # The raw file has typos such as ap_hi 16020 or negative pressures; treat
    # physiologically impossible values as not recorded rather than dropping rows.
    df.loc[~df["ap_hi"].between(70, 250), "ap_hi"] = np.nan
    df.loc[~df["ap_lo"].between(40, 160), "ap_lo"] = np.nan
    df.loc[df["ap_lo"] > df["ap_hi"], ["ap_hi", "ap_lo"]] = np.nan
    df.loc[~df["height"].between(120, 220), "height"] = np.nan
    df.loc[~df["weight"].between(30, 200), "weight"] = np.nan
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    df["target"] = df["cardio"].astype(int)
    return df


def _chol_level(p: dict):
    c = p.get("total_cholesterol")
    return None if c is None else (1 if c < 200 else 2 if c < 240 else 3)


def _gluc_level(p: dict):
    g = p.get("glucose")
    if g is not None:
        return 1 if g < 100 else 2 if g < 126 else 3
    return {1.0: 3, 0.0: 1}.get(p.get("diabetes"))


def _cardio_from_profile(p: dict) -> dict:
    return {
        "age_years": p.get("age"), "sex": _sex01(p), "height": p.get("height_cm"), "weight": p.get("weight_kg"),
        "bmi": p.get("bmi"), "ap_hi": p.get("systolic_bp"), "ap_lo": p.get("diastolic_bp"),
        "cholesterol": _chol_level(p), "gluc": _gluc_level(p), "smoke": p.get("smoker"),
        "alco": p.get("heavy_alcohol"), "active": p.get("physically_active"),
    }


CARDIO = Source(
    name="cardio",
    title="Cardiovascular Disease dataset (cardio_train, 70,000 examinations)",
    outcome="Cardiovascular disease present at examination",
    files=("cardio_train.csv",),
    features=[
        Feature("age_years", "num", ("age",)),
        Feature("sex", "cat", ("sex",), (0, 1)),
        Feature("height", "num", ("height_cm",)),
        Feature("weight", "num", ("weight_kg",)),
        Feature("bmi", "num", ("bmi",)),
        Feature("ap_hi", "num", ("systolic_bp",)),
        Feature("ap_lo", "num", ("diastolic_bp",)),
        Feature("cholesterol", "cat", ("total_cholesterol",), (1, 2, 3)),
        Feature("gluc", "cat", ("glucose", "diabetes"), (1, 2, 3)),
        Feature("smoke", "cat", ("smoker",), (0, 1)),
        Feature("alco", "cat", ("heavy_alcohol",), (0, 1)),
        Feature("active", "cat", ("physically_active",), (0, 1)),
    ],
    _load=_load_cardio,
    _from_profile=_cardio_from_profile,
    notes="Cholesterol and glucose are only recorded as normal / above normal / well above normal.",
)


# ------------------------------------------------------------------ BRFSS 2015
def _load_brfss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target"] = df["HeartDiseaseorAttack"].astype(int)
    return df


def _brfss_age_code(age):
    """BRFSS _AGEG5YR: 1 = 18-24, 2 = 25-29, ... 12 = 75-79, 13 = 80+."""
    if age is None:
        return None
    return float(np.clip((age - 25) // 5 + 2, 1, 13)) if age >= 25 else 1.0


def _brfss_diabetes(p: dict):
    g = p.get("glucose")
    if p.get("diabetes") == 1 or (g is not None and g >= 126):
        return 2
    if g is not None and g >= 100:
        return 1
    if p.get("diabetes") == 0 or g is not None:
        return 0
    return None


def _general_health_code(p: dict):
    gh = p.get("general_health")
    return None if gh is None else float(P.GENERAL_HEALTH_LEVELS.index(gh) + 1)


def _brfss_from_profile(p: dict) -> dict:
    return {
        "HighBP": P.high_bp(p), "HighChol": P.high_chol(p), "BMI": p.get("bmi"), "Smoker": p.get("smoker"),
        "Stroke": p.get("stroke_history"), "Diabetes": _brfss_diabetes(p), "PhysActivity": p.get("physically_active"),
        "Fruits": p.get("eats_fruit_daily"), "Veggies": p.get("eats_veg_daily"),
        "HvyAlcoholConsump": p.get("heavy_alcohol"), "GenHlth": _general_health_code(p),
        "MentHlth": p.get("mental_health_days"), "PhysHlth": p.get("physical_health_days"),
        "DiffWalk": p.get("difficulty_walking"), "Sex": _sex01(p), "Age": _brfss_age_code(p.get("age")),
    }


BRFSS = Source(
    name="brfss2015",
    title="CDC BRFSS 2015 heart-disease health indicators (253,680 survey responses)",
    outcome="Self-reported coronary heart disease or heart attack",
    files=("heart_disease_health_indicators_BRFSS2015.csv",),
    features=[
        Feature("HighBP", "cat", ("systolic_bp", "diastolic_bp", "bp_meds"), (0, 1)),
        Feature("HighChol", "cat", ("total_cholesterol",), (0, 1)),
        Feature("BMI", "num", ("bmi",)),
        Feature("Smoker", "cat", ("smoker",), (0, 1)),
        Feature("Stroke", "cat", ("stroke_history",), (0, 1)),
        Feature("Diabetes", "cat", ("diabetes", "glucose"), (0, 1, 2)),
        Feature("PhysActivity", "cat", ("physically_active",), (0, 1)),
        Feature("Fruits", "cat", ("eats_fruit_daily",), (0, 1)),
        Feature("Veggies", "cat", ("eats_veg_daily",), (0, 1)),
        Feature("HvyAlcoholConsump", "cat", ("heavy_alcohol",), (0, 1)),
        Feature("GenHlth", "num", ("general_health",)),
        Feature("MentHlth", "num", ("mental_health_days",)),
        Feature("PhysHlth", "num", ("physical_health_days",)),
        Feature("DiffWalk", "cat", ("difficulty_walking",), (0, 1)),
        Feature("Sex", "cat", ("sex",), (0, 1)),
        Feature("Age", "num", ("age",)),
    ],
    _load=_load_brfss,
    _from_profile=_brfss_from_profile,
    notes="Smoker means 100+ cigarettes in a lifetime; HighBP/HighChol mean ever told so by a health "
          "professional. Income, education and healthcare-access columns are not used.",
)


# ------------------------------------------------------------------ Personal Key Indicators 2020
_AGE_BINS_2020 = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
                  "60-64", "65-69", "70-74", "75-79", "80 or older"]
_GEN_HEALTH_2020 = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
_DIABETIC_2020 = ("No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)")


def _load_key2020(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity", "Asthma", "KidneyDisease"):
        df[c] = _yes_no(df[c])
    df["Sex"] = (df["Sex"] == "Male").astype(int)
    df["AgeCode"] = df["AgeCategory"].map({b: i + 1 for i, b in enumerate(_AGE_BINS_2020)}).astype(float)
    df["GenHealth"] = df["GenHealth"].map(_GEN_HEALTH_2020).astype(float)
    df["Diabetic"] = df["Diabetic"].map({v: i for i, v in enumerate(_DIABETIC_2020)})
    df["target"] = _yes_no(df["HeartDisease"]).astype(int)
    return df


def _key2020_from_profile(p: dict) -> dict:
    b = _brfss_diabetes(p)  # 0 no, 1 borderline, 2 yes -> same order as _DIABETIC_2020's first three
    return {
        "BMI": p.get("bmi"), "Smoking": p.get("smoker"), "AlcoholDrinking": p.get("heavy_alcohol"),
        "Stroke": p.get("stroke_history"), "PhysicalHealth": p.get("physical_health_days"),
        "MentalHealth": p.get("mental_health_days"), "DiffWalking": p.get("difficulty_walking"), "Sex": _sex01(p),
        "AgeCode": _brfss_age_code(p.get("age")), "GenHealth": _general_health_code(p),
        "Diabetic": b, "PhysicalActivity": p.get("physically_active"), "SleepTime": p.get("sleep_hours"),
        "Asthma": p.get("asthma"), "KidneyDisease": p.get("kidney_disease"),
    }


KEY2020 = Source(
    name="key_indicators_2020",
    title="Personal Key Indicators of Heart Disease, CDC BRFSS 2020 (319,795 survey responses)",
    outcome="Self-reported coronary heart disease or heart attack",
    files=("heart_2020_cleaned.csv",),
    features=[
        Feature("BMI", "num", ("bmi",)),
        Feature("Smoking", "cat", ("smoker",), (0, 1)),
        Feature("AlcoholDrinking", "cat", ("heavy_alcohol",), (0, 1)),
        Feature("Stroke", "cat", ("stroke_history",), (0, 1)),
        Feature("PhysicalHealth", "num", ("physical_health_days",)),
        Feature("MentalHealth", "num", ("mental_health_days",)),
        Feature("DiffWalking", "cat", ("difficulty_walking",), (0, 1)),
        Feature("Sex", "cat", ("sex",), (0, 1)),
        Feature("AgeCode", "num", ("age",)),
        Feature("GenHealth", "num", ("general_health",)),
        Feature("Diabetic", "cat", ("diabetes", "glucose"), (0, 1, 2, 3)),
        Feature("PhysicalActivity", "cat", ("physically_active",), (0, 1)),
        Feature("SleepTime", "num", ("sleep_hours",)),
        Feature("Asthma", "cat", ("asthma",), (0, 1)),
        Feature("KidneyDisease", "cat", ("kidney_disease",), (0, 1)),
    ],
    _load=_load_key2020,
    _from_profile=_key2020_from_profile,
    notes="Race and skin cancer columns are not used.",
)


# ------------------------------------------------------------------ Kaggle heart-attack risk
def _load_hapd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bp = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic"] = _num(bp[0])
    df["Diastolic"] = _num(bp[1])
    df["Sex"] = (df["Sex"] == "Male").astype(int)
    df["target"] = df["Heart Attack Risk"].astype(int)
    return df


def _hapd_from_profile(p: dict) -> dict:
    return {
        "Age": p.get("age"), "Sex": _sex01(p), "Cholesterol": p.get("total_cholesterol"),
        "Systolic": p.get("systolic_bp"), "Diastolic": p.get("diastolic_bp"), "Heart Rate": p.get("resting_heart_rate"),
        "Diabetes": p.get("diabetes"), "Family History": p.get("family_history"), "Smoking": p.get("smoker"),
        "Obesity": P.obese(p), "Alcohol Consumption": p.get("heavy_alcohol"),
        "Exercise Hours Per Week": p.get("exercise_hours_week"), "Previous Heart Problems": p.get("previous_heart_problems"),
        "Stress Level": p.get("stress_level"), "BMI": p.get("bmi"), "Triglycerides": p.get("triglycerides"),
        "Sleep Hours Per Day": p.get("sleep_hours"),
    }


HAPD = Source(
    name="heart_attack_risk",
    title="Kaggle Heart Attack Risk Prediction dataset (8,763 rows)",
    outcome="'Heart Attack Risk' label",
    files=("heart_attack_prediction_dataset.csv",),
    features=[
        Feature("Age", "num", ("age",)),
        Feature("Sex", "cat", ("sex",), (0, 1)),
        Feature("Cholesterol", "num", ("total_cholesterol",)),
        Feature("Systolic", "num", ("systolic_bp",)),
        Feature("Diastolic", "num", ("diastolic_bp",)),
        Feature("Heart Rate", "num", ("resting_heart_rate",)),
        Feature("Diabetes", "cat", ("diabetes",), (0, 1)),
        Feature("Family History", "cat", ("family_history",), (0, 1)),
        Feature("Smoking", "cat", ("smoker",), (0, 1)),
        Feature("Obesity", "cat", ("bmi",), (0, 1)),
        Feature("Alcohol Consumption", "cat", ("heavy_alcohol",), (0, 1)),
        Feature("Exercise Hours Per Week", "num", ("exercise_hours_week",)),
        Feature("Previous Heart Problems", "cat", ("previous_heart_problems",), (0, 1)),
        Feature("Stress Level", "num", ("stress_level",)),
        Feature("BMI", "num", ("bmi",)),
        Feature("Triglycerides", "num", ("triglycerides",)),
        Feature("Sleep Hours Per Day", "num", ("sleep_hours",)),
    ],
    _load=_load_hapd,
    _from_profile=_hapd_from_profile,
    notes="Widely reported to be synthetically generated, with labels unrelated to the inputs. It is "
          "trained and evaluated like the others; if its held-out AUROC is near 0.5 the fusion step "
          "gives it zero weight automatically.",
)


# ------------------------------------------------------------------ ECG5000 heartbeats
ECG_LEN = 140


def _load_ecg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    out = pd.DataFrame(df.iloc[:, :ECG_LEN].values, columns=[f"t{i}" for i in range(ECG_LEN)])
    out["target"] = (df.iloc[:, ECG_LEN] == 0).astype(int)  # label 1 = normal beat
    return out


def _ecg_from_profile(p: dict) -> dict:
    beat = p.get("ecg_beat")
    return {} if beat is None else {f"t{i}": v for i, v in enumerate(beat)}


ECG = Source(
    name="ecg5000",
    title="ECG5000 heartbeats (4,998 beats, 140 samples each)",
    outcome="Abnormal heartbeat",
    files=("ecg.csv",),
    kind="beat",
    features=[Feature(f"t{i}", "num", ("ecg_beat",)) for i in range(ECG_LEN)],
    _load=_load_ecg,
    _from_profile=_ecg_from_profile,
    notes="All beats come from one 20-hour recording of a single patient with severe heart failure, "
          "so this branch recognises abnormal beat shapes; it says nothing about a person's long-term "
          "risk and its weight in the fusion should be read with that in mind.",
)


SOURCES: dict[str, Source] = {s.name: s for s in (UCI, FRAMINGHAM, CARDIO, BRFSS, KEY2020, HAPD, ECG)}
