"""Clinical measurements from segmentation masks.

These are the numbers a cardiologist reads off an MRI or echo report, and
the numbers the digital twin is fitted to: chamber volumes, ejection
fraction, myocardial mass and wall thickness. All formulas are standard:

- Volume (MRI): voxel count x voxel volume (Simpson's method of discs over
  the short-axis stack, which is what summing voxels slice by slice is).
- Ejection fraction: (EDV - ESV) / EDV.
- LV mass: myocardial volume x 1.05 g/ml (myocardial tissue density).
- Wall thickness: twice the largest inscribed-circle radius of the
  myocardium on any slice (distance transform), an upper-bound estimate.
- Volume (2D echo): single-plane area-length, V = 8 A^2 / (3 pi L).
- Body surface area: Mosteller, sqrt(height_cm x weight_kg / 3600).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

RV, MYO, LV = 1, 2, 3  # ACDC label values
MYOCARDIUM_DENSITY = 1.05  # g/ml


def volume_ml(mask: np.ndarray, label: int, spacing_mm) -> float:
    return float((mask == label).sum() * np.prod(spacing_mm) / 1000.0)


def ejection_fraction(edv: float, esv: float) -> float:
    return float(100.0 * (edv - esv) / edv) if edv > 0 else float("nan")


def max_wall_thickness_mm(mask: np.ndarray, spacing_mm, label: int = MYO) -> float:
    """mask: (H, W, slices). In-plane distance transform on each slice."""
    best = 0.0
    for k in range(mask.shape[-1]):
        myo = mask[..., k] == label
        if myo.any():
            dist = ndimage.distance_transform_edt(myo, sampling=spacing_mm[:2])
            best = max(best, float(dist.max()))
    return 2.0 * best


def bsa_mosteller(height_cm: float, weight_kg: float) -> float:
    return float(np.sqrt(height_cm * weight_kg / 3600.0))


def mri_function(ed_mask: np.ndarray, es_mask: np.ndarray, spacing_mm,
                 height_cm: float | None = None, weight_kg: float | None = None) -> dict:
    """Both ventricles' volumes and EF, LV mass and wall thickness from ED + ES masks."""
    lv_edv, lv_esv = volume_ml(ed_mask, LV, spacing_mm), volume_ml(es_mask, LV, spacing_mm)
    rv_edv, rv_esv = volume_ml(ed_mask, RV, spacing_mm), volume_ml(es_mask, RV, spacing_mm)
    myo_ed = volume_ml(ed_mask, MYO, spacing_mm)
    out = {
        "lv_edv_ml": lv_edv,
        "lv_esv_ml": lv_esv,
        "lv_sv_ml": lv_edv - lv_esv,
        "lv_ef_percent": ejection_fraction(lv_edv, lv_esv),
        "rv_edv_ml": rv_edv,
        "rv_esv_ml": rv_esv,
        "rv_ef_percent": ejection_fraction(rv_edv, rv_esv),
        "lv_mass_g": myo_ed * MYOCARDIUM_DENSITY,
        "myo_es_ml": volume_ml(es_mask, MYO, spacing_mm),
        "max_wall_thickness_mm": max_wall_thickness_mm(ed_mask, spacing_mm),
        "rv_lv_edv_ratio": rv_edv / lv_edv if lv_edv > 0 else float("nan"),
        "mass_edv_ratio": myo_ed * MYOCARDIUM_DENSITY / lv_edv if lv_edv > 0 else float("nan"),
    }
    if height_cm and weight_kg:
        bsa = bsa_mosteller(height_cm, weight_kg)
        out["bsa_m2"] = bsa
        for key in ("lv_edv_ml", "lv_esv_ml", "rv_edv_ml", "rv_esv_ml", "lv_mass_g"):
            out[f"{key}_per_m2"] = out[key] / bsa
    return out


# Features for the ACDC 5-group diagnosis classifier, in a fixed order.
# Indexed volumes need height + weight, which ACDC provides for every patient.
ACDC_FEATURES = [
    "lv_edv_ml_per_m2", "lv_esv_ml_per_m2", "lv_ef_percent",
    "rv_edv_ml_per_m2", "rv_esv_ml_per_m2", "rv_ef_percent",
    "lv_mass_g_per_m2", "max_wall_thickness_mm", "rv_lv_edv_ratio", "mass_edv_ratio",
]


def acdc_feature_vector(metrics: dict) -> np.ndarray:
    return np.array([metrics[k] for k in ACDC_FEATURES], dtype=np.float32)


# ---------------------------------------------------------------- 2D echo
def long_axis_length_mm(mask2d: np.ndarray, spacing_mm) -> float:
    """Extent of the cavity along its principal axis (PCA of pixel coordinates)."""
    ys, xs = np.nonzero(mask2d)
    if len(xs) < 3:
        return 0.0
    pts = np.stack([ys * spacing_mm[0], xs * spacing_mm[1]], axis=1)
    pts -= pts.mean(0)
    axis = np.linalg.svd(pts, full_matrices=False)[2][0]
    proj = pts @ axis
    return float(proj.max() - proj.min())


def echo_area_length_volume_ml(mask2d: np.ndarray, spacing_mm, label: int = 1) -> float:
    cavity = mask2d == label
    area_mm2 = cavity.sum() * spacing_mm[0] * spacing_mm[1]
    length = long_axis_length_mm(cavity, spacing_mm)
    if length == 0:
        return 0.0
    return float(8 * area_mm2**2 / (3 * np.pi * length) / 1000.0)


def echo_function(ed_mask2d: np.ndarray, es_mask2d: np.ndarray, spacing_mm, label: int = 1) -> dict:
    edv = echo_area_length_volume_ml(ed_mask2d, spacing_mm, label)
    esv = echo_area_length_volume_ml(es_mask2d, spacing_mm, label)
    return {"lv_edv_ml": edv, "lv_esv_ml": esv, "lv_sv_ml": edv - esv, "lv_ef_percent": ejection_fraction(edv, esv)}
