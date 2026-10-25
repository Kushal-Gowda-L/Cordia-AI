"""Per-patient parameters for the 3D cardiac digital twin.

Maps one stored patient record -- the UCI heart-disease feature set the
risk model uses (age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak) plus that model's prediction -- onto the physiological
parameters of the heart simulation in heart_digital_twin_3d/
(src/simulation/heartModel.ts), for two scenarios: at rest, and at the
peak of the exercise stress test that the record's thalach / exang /
oldpeak values come from.

Stated plainly: these records hold none of the measurements a clinically
calibrated cardiac digital twin is fitted to (echo ejection fraction,
chamber volumes, pressure-volume loops). So each parameter is either taken
from the record where the record has it, or estimated by a stated rule,
and every value carries its source so the UI can show which is which. The
twin is patient-specific in its inputs, not a validated model of that
patient's heart.
"""
from __future__ import annotations

from src.risk_reading import read_risk

# Contractility reduction by the risk model's predicted severity class
# (No disease .. Critical): a heuristic link -- coronary disease, reduced
# myocardial perfusion, weaker contraction. Deliberately modest, since
# resting contractility is often preserved in coronary disease without a
# prior infarction.
SEVERITY_CONTRACTILITY_PENALTY = [0, 4, 8, 13, 18]

SIM_HR_MAX = 180  # upper end of the simulator's heart-rate range


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _rest_scenario(r: dict, severity_idx: int) -> tuple[dict, dict]:
    params: dict[str, float] = {}
    sources: dict[str, str] = {}

    def put(key: str, value: float, source: str) -> None:
        params[key] = round(float(value), 1)
        sources[key] = source

    sbp = r["trestbps"]
    male = r["sex"] == 1
    put(
        "heartRate",
        72,
        "Not in record: the record's heart rate (thalach) is the stress-test peak, so a typical resting value is used",
    )
    put("systolicBP", sbp, f"From record: resting blood pressure {sbp} mm Hg")
    put(
        "diastolicBP",
        80 if sbp >= 110 else sbp - 30,
        "Not in record: the record holds systolic pressure only, so a typical diastolic value is used",
    )
    put(
        "baseStrokeVolume",
        72 if male else 62,
        f"Estimated: typical resting stroke volume for an adult {'male' if male else 'female'}",
    )

    vr = 50 + 0.6 * (sbp - 120) + 0.3 * max(0, r["age"] - 50) + (6 if r["fbs"] else 0)
    vr_note = f"Estimated from resting BP {sbp} mm Hg and age {r['age']} (arterial stiffening)"
    if r["fbs"]:
        vr_note += ", raised for elevated fasting blood sugar"
    put("vascularResistance", _clamp(vr, 15, 95), vr_note)

    afterload = 50 + 0.5 * (sbp - 120) + (10 if r["restecg"] == 2 else 0)
    al_note = f"Estimated from resting BP {sbp} mm Hg"
    if r["restecg"] == 2:
        al_note += "; raised for left ventricular hypertrophy on the resting ECG (chronic pressure overload)"
    put("afterload", _clamp(afterload, 15, 95), al_note)

    contractility = 55 - SEVERITY_CONTRACTILITY_PENALTY[severity_idx] - (5 if r["restecg"] == 1 else 0)
    c_note = "Estimated: typical baseline, reduced in step with the risk model's predicted severity"
    if r["restecg"] == 1:
        c_note += " and for ST-T wave abnormality on the resting ECG"
    put("contractility", _clamp(contractility, 10, 90), c_note)

    put("preload", 50, "Not in record: typical value")
    put("oxygenSaturation", 98, "Not in record: typical value")
    put("aorticStenosis", 0, "Not in record: no valve disease assumed")
    put("mitralRegurgitation", 0, "Not in record: no valve disease assumed")
    return params, sources


def _stress_scenario(r: dict, rest: dict, rest_sources: dict) -> tuple[dict, dict]:
    params = dict(rest)
    sources = dict(rest_sources)

    hr = r["thalach"]
    params["heartRate"] = float(min(hr, SIM_HR_MAX))
    sources["heartRate"] = f"From record: peak exercise heart rate (thalach) {hr} BPM" + (
        f", capped at the simulator's {SIM_HR_MAX} BPM" if hr > SIM_HR_MAX else ""
    )

    params["systolicBP"] = round(min(220.0, r["trestbps"] + 0.35 * (hr - 72)), 1)
    sources["systolicBP"] = f"Estimated: resting {r['trestbps']} mm Hg plus a typical exercise rise with heart rate"

    ischemia = (15 if r["exang"] else 0) + min(25.0, 6 * max(0.0, r["oldpeak"]))
    params["contractility"] = round(_clamp(rest["contractility"] + 20 - ischemia, 10, 95), 1)
    note = "Estimated: resting value +20 for exercise inotropy"
    if ischemia > 0:
        causes = []
        if r["exang"]:
            causes.append("exercise-induced angina")
        if r["oldpeak"] > 0:
            causes.append(f"ST depression {r['oldpeak']}")
        note += f", reduced for exercise ischemia ({', '.join(causes)})"
    sources["contractility"] = note

    params["vascularResistance"] = round(_clamp(rest["vascularResistance"] - 15, 10, 95), 1)
    sources["vascularResistance"] = "Estimated: resting value, lowered for exercise vasodilation"
    params["preload"] = 65.0
    sources["preload"] = "Estimated: raised for increased venous return during exercise"
    return params, sources


def build_twin(
    name: str, record: dict, risk_prob: float, severity_idx: int, probs: list[float], classes: list[str]
) -> dict:
    rest, rest_sources = _rest_scenario(record, severity_idx)
    stress, stress_sources = _stress_scenario(record, rest, rest_sources)
    return {
        "patient": {"name": name, "age": record["age"], "sex": "Male" if record["sex"] == 1 else "Female"},
        "prediction": {
            "risk_percent": round(risk_prob * 100, 1),
            "severity": classes[severity_idx],
            "severity_percent": round(probs[severity_idx] * 100, 1),
            "class_probabilities": {classes[i]: round(p * 100, 1) for i, p in enumerate(probs)},
            "reading": read_risk(probs, classes),
        },
        "record": record,
        "scenarios": {
            "rest": {"label": "At rest", "params": rest, "sources": rest_sources},
            "stress": {"label": "Peak exercise (stress test)", "params": stress, "sources": stress_sources},
        },
    }
