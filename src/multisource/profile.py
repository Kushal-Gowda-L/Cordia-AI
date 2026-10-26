"""The patient profile: one dataset-independent description of a patient.

Every field is optional. Each dataset encoder maps the profile into its own
input columns (sources.py) and treats anything it cannot derive as unknown,
so a profile with only age, sex and blood pressure still gets a prediction,
just from fewer encoders and with less weight on each.

Units are the everyday clinical ones: years, mm Hg, mg/dL, kg, cm, beats/min.
Yes/no fields take 1/0 or True/False.
"""
from __future__ import annotations

import math

# name -> (kind, description). kind: "num", "bool", or a tuple of allowed values.
FIELDS: dict[str, tuple] = {
    "age": ("num", "Age in years"),
    "sex": (("male", "female"), "Sex"),
    "height_cm": ("num", "Height (cm)"),
    "weight_kg": ("num", "Weight (kg)"),
    "bmi": ("num", "Body-mass index (kg/m^2); derived from height and weight if omitted"),
    "systolic_bp": ("num", "Resting systolic blood pressure (mm Hg)"),
    "diastolic_bp": ("num", "Resting diastolic blood pressure (mm Hg)"),
    "bp_meds": ("bool", "Takes blood-pressure medication"),
    "total_cholesterol": ("num", "Total cholesterol (mg/dL)"),
    "triglycerides": ("num", "Triglycerides (mg/dL)"),
    "glucose": ("num", "Fasting blood glucose (mg/dL)"),
    "diabetes": ("bool", "Diagnosed diabetes"),
    "smoker": ("bool", "Currently smokes"),
    "cigs_per_day": ("num", "Cigarettes per day"),
    "heavy_alcohol": ("bool", "Heavy alcohol use"),
    "physically_active": ("bool", "Physically active in the past month"),
    "exercise_hours_week": ("num", "Exercise hours per week"),
    "sleep_hours": ("num", "Sleep hours per night"),
    "stress_level": ("num", "Stress level, 1 (low) to 10 (high)"),
    "eats_fruit_daily": ("bool", "Eats fruit at least once a day"),
    "eats_veg_daily": ("bool", "Eats vegetables at least once a day"),
    "general_health": (("excellent", "very_good", "good", "fair", "poor"), "Self-rated general health"),
    "physical_health_days": ("num", "Days of poor physical health in the past 30"),
    "mental_health_days": ("num", "Days of poor mental health in the past 30"),
    "difficulty_walking": ("bool", "Serious difficulty walking or climbing stairs"),
    "stroke_history": ("bool", "Has had a stroke"),
    "kidney_disease": ("bool", "Kidney disease"),
    "asthma": ("bool", "Asthma"),
    "family_history": ("bool", "Family history of heart disease"),
    "previous_heart_problems": ("bool", "Previous heart problems"),
    "resting_heart_rate": ("num", "Resting heart rate (beats/min)"),
    "chest_pain_type": (("typical", "atypical", "non_anginal", "asymptomatic"), "Chest-pain type"),
    "max_heart_rate": ("num", "Maximum heart rate on exercise test (beats/min)"),
    "exercise_angina": ("bool", "Chest pain brought on by exercise"),
    "st_depression": ("num", "ST depression on exercise relative to rest (mm)"),
    "st_slope": (("up", "flat", "down"), "Slope of the peak-exercise ST segment"),
    "resting_ecg": (("normal", "st_t_abnormality", "lvh"), "Resting ECG"),
    "major_vessels": ("num", "Major vessels coloured by fluoroscopy (0-3)"),
    "thalassemia": (("normal", "fixed_defect", "reversible_defect"), "Thallium stress-test result"),
    "ecg_beat": ("beat", "One heartbeat, 140 samples, as in ECG5000"),
}

GENERAL_HEALTH_LEVELS = FIELDS["general_health"][0]


def _present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return True


def normalize(profile: dict) -> dict:
    """Validated copy with booleans as 0/1, numbers as float, and bmi derived
    from height and weight when absent. Unknown keys raise, so a typo is not
    silently ignored; values None/NaN are dropped."""
    out: dict = {}
    for key, value in profile.items():
        if key not in FIELDS:
            raise KeyError(f"unknown profile field {key!r}; see src/multisource/profile.py")
        if not _present(value):
            continue
        kind = FIELDS[key][0]
        if kind == "num":
            out[key] = float(value)
        elif kind == "bool":
            if isinstance(value, str):
                value = value.strip().lower() in ("1", "yes", "true", "y")
            out[key] = float(bool(value))
        elif kind == "beat":
            beat = [float(x) for x in value]
            if len(beat) != 140:
                raise ValueError("ecg_beat needs exactly 140 samples")
            out[key] = beat
        else:
            if value not in kind:
                raise ValueError(f"{key} must be one of {kind}, got {value!r}")
            out[key] = value
    if "bmi" not in out and "height_cm" in out and "weight_kg" in out:
        out["bmi"] = out["weight_kg"] / (out["height_cm"] / 100) ** 2
    return out


def high_bp(p: dict):
    """Ever told blood pressure is high (BRFSS-style): on medication or >= 140/90."""
    if p.get("bp_meds") == 1:
        return 1.0
    known = [k for k in ("systolic_bp", "diastolic_bp") if k in p]
    if not known:
        return p.get("bp_meds")  # 0 or None
    return float(p.get("systolic_bp", 0) >= 140 or p.get("diastolic_bp", 0) >= 90)


def high_chol(p: dict):
    return float(p["total_cholesterol"] >= 240) if "total_cholesterol" in p else None


def fasting_sugar_high(p: dict):
    if "glucose" in p:
        return float(p["glucose"] > 120)
    return p.get("diabetes")


def obese(p: dict):
    return float(p["bmi"] >= 30) if "bmi" in p else None
