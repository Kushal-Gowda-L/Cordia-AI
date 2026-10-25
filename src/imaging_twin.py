"""Imaging measurements -> digital twin parameters.

src/patient_twin.py builds the twin from the UCI record alone, and says
plainly that the record holds none of the measurements a cardiac twin is
normally fitted to (ejection fraction, chamber volumes). This module fills
that gap when the patient has imaging: values measured from their own echo,
MRI or X-ray replace the estimated ones, and each replaced value's source
says which scan and which measurement it came from.

The mappings from a measurement (EF in %, volume in ml) onto the
simulator's 0-100 control scales are linear rules chosen so that typical
normal values land on patient_twin.py's existing baselines (EF 60% ->
contractility 55, LV end-diastolic volume 120 ml -> preload 50). They are
stated rules, not a fitted physiological model.
"""
from __future__ import annotations

import copy

NORMAL_EF = 60.0  # %, lands on patient_twin's contractility baseline of 55
NORMAL_LV_EDV = 120.0  # ml, lands on the preload baseline of 50
HYPERTROPHY_WALL_MM = 13.0  # ESC threshold for LV hypertrophy
CARDIOMEGALY_THRESHOLD = 0.5


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _pick(imaging: dict, key: str):
    """First available value of `key`, MRI before echo (MRI volumes are the reference standard)."""
    for modality in ("mri", "echo"):
        value = (imaging.get(modality) or {}).get(key)
        if value is not None:
            return value, modality
    return None, None


SOURCE_NAMES = {"mri": "cardiac MRI", "echo": "echocardiogram"}


def apply_imaging(twin: dict, imaging: dict) -> dict:
    """imaging: {"mri": {...metrics}, "echo": {...metrics}, "xray": {"Cardiomegaly": p, ...}}.
    Any modality may be missing; only the parameters it measures change."""
    twin = copy.deepcopy(twin)
    rest = twin["scenarios"]["rest"]
    stress = twin["scenarios"]["stress"]
    findings: list[str] = []

    ef, src = _pick(imaging, "lv_ef_percent")
    if ef is not None:
        old = rest["params"]["contractility"]
        new = round(_clamp(55 + 1.2 * (ef - NORMAL_EF), 10, 90), 1)
        rest["params"]["contractility"] = new
        rest["sources"]["contractility"] = (
            f"From {SOURCE_NAMES[src]}: measured ejection fraction {ef:.0f}% "
            f"(mapped linearly, EF {NORMAL_EF:.0f}% = baseline 55)"
        )
        stress["params"]["contractility"] = round(_clamp(stress["params"]["contractility"] + new - old, 10, 95), 1)
        stress["sources"]["contractility"] = (
            f"From {SOURCE_NAMES[src]}: resting value from measured EF {ef:.0f}%, "
            "with the same exercise and ischemia adjustments as before"
        )
        if ef < 40:
            findings.append(f"Reduced ejection fraction ({ef:.0f}%, below 40%) on {SOURCE_NAMES[src]}")
        elif ef < 50:
            findings.append(f"Mildly reduced ejection fraction ({ef:.0f}%) on {SOURCE_NAMES[src]}")

    sv, src = _pick(imaging, "lv_sv_ml")
    if sv is not None:
        rest["params"]["baseStrokeVolume"] = round(_clamp(sv, 20, 150), 1)
        rest["sources"]["baseStrokeVolume"] = f"From {SOURCE_NAMES[src]}: measured stroke volume {sv:.0f} ml (EDV - ESV)"
        stress["params"]["baseStrokeVolume"] = rest["params"]["baseStrokeVolume"]
        stress["sources"]["baseStrokeVolume"] = rest["sources"]["baseStrokeVolume"]

    edv, src = _pick(imaging, "lv_edv_ml")
    if edv is not None:
        rest["params"]["preload"] = round(_clamp(50 + 0.25 * (edv - NORMAL_LV_EDV), 10, 95), 1)
        rest["sources"]["preload"] = (
            f"From {SOURCE_NAMES[src]}: measured LV end-diastolic volume {edv:.0f} ml "
            f"(mapped linearly, {NORMAL_LV_EDV:.0f} ml = baseline 50)"
        )

    wall = (imaging.get("mri") or {}).get("max_wall_thickness_mm")
    if wall is not None and wall >= HYPERTROPHY_WALL_MM:
        rest["params"]["afterload"] = round(_clamp(rest["params"]["afterload"] + 10, 15, 95), 1)
        rest["sources"]["afterload"] += f"; raised for LV wall thickness {wall:.0f} mm on cardiac MRI (hypertrophy, >= 13 mm)"
        findings.append(f"Left ventricular wall thickness {wall:.0f} mm (hypertrophy threshold 13 mm)")

    diagnosis = (imaging.get("mri") or {}).get("diagnosis")
    if diagnosis:
        findings.append(f"MRI pattern most like: {diagnosis}")

    xray = imaging.get("xray") or {}
    for finding, label in (("Cardiomegaly", "Enlarged heart"), ("Edema", "Fluid in the lungs"), ("Effusion", "Fluid around the lungs")):
        p = xray.get(finding)
        if p is not None and p >= CARDIOMEGALY_THRESHOLD:
            findings.append(f"{label} ({finding.lower()}) suggested on chest X-ray ({p * 100:.0f}%)")

    twin["imaging"] = {
        "measurements": {k: v for k, v in imaging.items() if k in ("mri", "echo", "xray", "ct") and v},
        "findings": findings,
        "meshes": imaging.get("meshes") or {},
    }
    return twin
