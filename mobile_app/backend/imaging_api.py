"""Imaging endpoints: chest X-ray, echo video, cardiac MRI, coronary CT, and late fusion.

Each branch is loaded on first use, from its checkpoint in cordia_ai/checkpoints/.
A branch whose checkpoint is not there yet answers 503 with the notebook
that trains it, instead of the whole app failing to start -- the imaging
branches are trained on Colab/Kaggle one at a time (notebooks/step15-18).
"""
from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from PIL import Image

from src.imaging.cardiac_metrics import acdc_feature_vector, echo_function, mri_function
from src.imaging.heart_physics import regional_thickening, thickening_at_vertices
from src.imaging.mesh import label_meshes, mask_to_mesh
from src.imaging.preprocess import clip_to_tensor, read_video_bytes, sample_clip, segment_mri_volume, xray_to_tensor
from src.models.imaging.cardiac_seg import ACDC_GROUP_NAMES, ACDC_LABELS, build_ct_segmenter, build_mri_segmenter
from src.models.imaging.cxr_vit import CXR_FINDINGS, CXRViT
from src.models.imaging.echo_vit import N_FRAMES, EchoViT
from src.models.late_fusion import LateFusionCombiner, flatten_branch_outputs

CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"

BRANCHES = {
    "xray": ("cxr_vit.pt", "notebooks/step15_xray_vit_kaggle.ipynb"),
    "echo": ("echo_vit.pt", "notebooks/step16_echo_ef_colab.ipynb"),
    "mri": ("mri_swin_unetr.pt", "notebooks/step17_mri_acdc_colab.ipynb"),
    "mri_diagnosis": ("acdc_diagnosis.joblib", "notebooks/step17_mri_acdc_colab.ipynb"),
    "ct": ("ct_swin_unetr.pt", "notebooks/step18_ct_imagecas_colab.ipynb"),
    "fusion": ("late_fusion.joblib", "build_mimic_paired_cohort.py + train_late_fusion.py"),
}


def _require(branch: str) -> Path:
    name, trainer = BRANCHES[branch]
    path = CHECKPOINTS / name
    if not path.exists():
        raise HTTPException(503, f"The {branch} branch is not trained yet: run {trainer} and put {name} in checkpoints/.")
    return path


def _state(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


@lru_cache(maxsize=None)
def _xray_model():
    model = CXRViT(pretrained=False)
    model.load_state_dict(_state(_require("xray")))
    thresholds_path = CHECKPOINTS / "cxr_thresholds.json"
    thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {f: 0.5 for f in CXR_FINDINGS}
    return model.eval(), thresholds


@lru_cache(maxsize=None)
def _echo_model():
    model = EchoViT(pretrained=False)
    model.load_state_dict(_state(_require("echo")))
    return model.eval()


@lru_cache(maxsize=None)
def _mri_model():
    model = build_mri_segmenter()
    model.load_state_dict(_state(_require("mri")))
    return model.eval()


@lru_cache(maxsize=None)
def _mri_diagnosis():
    import joblib

    path = CHECKPOINTS / BRANCHES["mri_diagnosis"][0]
    return joblib.load(path) if path.exists() else None


@lru_cache(maxsize=None)
def _ct_model():
    model = build_ct_segmenter(use_checkpoint=False)
    model.load_state_dict(_state(_require("ct")))
    return model.eval()


@lru_cache(maxsize=None)
def _combiner():
    path = CHECKPOINTS / BRANCHES["fusion"][0]
    return LateFusionCombiner.load(path) if path.exists() else None


def _nifti(data: bytes):
    import nibabel as nib

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.nii.gz"
        path.write_bytes(data)
        img = nib.load(str(path))
        return np.asarray(img.dataobj), tuple(float(z) for z in img.header.get_zooms()[:3])


def ef_category(ef: float) -> str:
    # ESC heart failure guideline bands
    if ef < 40:
        return "Reduced (below 40%)"
    if ef < 50:
        return "Mildly reduced (40-49%)"
    return "Normal (50% or above)"


def build_router(current_user: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/imaging")

    @router.get("/status")
    def status(authorization: str | None = Header(None)):
        current_user(authorization)
        return {b: {"trained": (CHECKPOINTS / name).exists(), "checkpoint": name, "trained_by": trainer}
                for b, (name, trainer) in BRANCHES.items()}

    @router.post("/xray")
    async def xray(image: UploadFile = File(...), authorization: str | None = Header(None)):
        """A frontal chest X-ray (PNG/JPG) -> probability of each heart-related finding."""
        current_user(authorization)
        model, thresholds = _xray_model()
        try:
            img = Image.open(image.file)
            x = xray_to_tensor(img).unsqueeze(0)
        except Exception as e:
            raise HTTPException(400, f"Could not read this as an image: {e}")
        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].numpy()
        return {
            "findings": {f: round(float(p), 4) for f, p in zip(CXR_FINDINGS, probs)},
            "flagged": [f for f, p in zip(CXR_FINDINGS, probs) if p >= thresholds[f]],
            "thresholds": thresholds,
        }

    @router.post("/echo")
    async def echo(video: UploadFile = File(...), authorization: str | None = Header(None)):
        """An apical 4-chamber echo video (AVI/MP4) -> ejection fraction. Averages three
        16-frame windows across the clip, as a sonographer averages several beats."""
        current_user(authorization)
        model = _echo_model()
        try:
            frames = read_video_bytes(await video.read(), Path(video.filename or "x.avi").suffix or ".avi")
        except Exception as e:
            raise HTTPException(400, f"Could not read this video: {e}")
        span = N_FRAMES * 2
        starts = np.linspace(0, max(0, len(frames) - span), 3).astype(int)
        clips = torch.stack([clip_to_tensor(sample_clip(frames, N_FRAMES, 2, int(s))) for s in starts])
        with torch.no_grad():
            efs = model(clips).numpy()
        ef = float(np.clip(efs.mean(), 5, 90))
        return {"lv_ef_percent": round(ef, 1), "category": ef_category(ef), "per_window": [round(float(e), 1) for e in efs]}

    @router.post("/mri")
    async def mri(
        ed: UploadFile = File(..., description="End-diastolic short-axis stack, NIfTI"),
        es: UploadFile = File(..., description="End-systolic short-axis stack, NIfTI"),
        height_cm: float | None = Form(None),
        weight_kg: float | None = Form(None),
        meshes: bool = Form(True),
        authorization: str | None = Header(None),
    ):
        """Cine MRI at end-diastole and end-systole -> chamber volumes, EF, LV mass,
        wall thickness and regional thickening, ACDC-group pattern, and 3D meshes of the patient's chambers."""
        current_user(authorization)
        model = _mri_model()
        try:
            (ed_vol, spacing), (es_vol, _) = _nifti(await ed.read()), _nifti(await es.read())
        except Exception as e:
            raise HTTPException(400, f"Could not read these as NIfTI volumes: {e}")
        ed_mask, es_mask = segment_mri_volume(model, ed_vol), segment_mri_volume(model, es_vol)
        metrics = mri_function(ed_mask, es_mask, spacing, height_cm, weight_kg)
        result = {"metrics": {k: round(float(v), 2) for k, v in metrics.items()},
                  "ef_category": ef_category(metrics["lv_ef_percent"])}

        diag = _mri_diagnosis()
        if diag and height_cm and weight_kg:
            probs = diag["model"].predict_proba(acdc_feature_vector(metrics)[None])[0]
            result["diagnosis"] = {ACDC_GROUP_NAMES[g]: round(float(p), 3) for g, p in zip(diag["groups"], probs)}
            result["metrics"]["diagnosis"] = max(result["diagnosis"], key=result["diagnosis"].get)
        elif diag:
            result["diagnosis_note"] = "Give height and weight: the diagnosis uses body-size-indexed volumes."
        # Regional wall thickening, ED -> ES, for the multi-physics twin (src/imaging/heart_physics.py)
        result["regional"] = regional_thickening(ed_mask, es_mask, spacing)
        if meshes:
            result["meshes"] = label_meshes(ed_mask, spacing, ACDC_LABELS)
            if "LV myocardium" in result["meshes"]:
                result["regional"]["thickening_at_vertices"] = thickening_at_vertices(
                    result["meshes"]["LV myocardium"]["vertices"], result["regional"], spacing)
        return result

    @router.post("/echo-masks")
    async def echo_masks(ed_mask: UploadFile = File(...), es_mask: UploadFile = File(...),
                         authorization: str | None = Header(None)):
        """Already-segmented 2D echo frames (e.g. CAMUS ground truth, NIfTI) -> volumes and EF
        by the single-plane area-length method."""
        current_user(authorization)
        (ed_m, sp), (es_m, _) = _nifti(await ed_mask.read()), _nifti(await es_mask.read())
        out = echo_function(np.squeeze(ed_m), np.squeeze(es_m), sp[:2])
        return {k: round(float(v), 2) for k, v in out.items()}

    @router.post("/ct")
    async def ct(scan: UploadFile = File(...), authorization: str | None = Header(None)):
        """Coronary CT angiogram (NIfTI) -> 3D mesh of the coronary tree. Slow on a CPU (minutes)."""
        current_user(authorization)
        from src.imaging.ct import segment_ct_file

        model = _ct_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.nii.gz"
            path.write_bytes(await scan.read())
            mask, spacing = segment_ct_file(model, str(path))
        mesh = mask_to_mesh(mask, spacing, step=1, smooth_sigma=0.5)
        return {"coronary_volume_ml": round(float(mask.sum() * np.prod(spacing) / 1000), 2),
                "meshes": {"Coronary arteries": mesh} if mesh else {}}

    @router.post("/combined")
    def combined(outputs: dict, authorization: str | None = Header(None)):
        """Branch outputs for ONE patient ({"ecg": {...}, "xray": {...}, "echo": {...}, "mri": {...},
        "clinical": {"age":..,"sex":..}, "labs": {...}}) -> one combined prediction, if a combiner has
        been fitted on a real paired cohort. Otherwise says so and returns no number."""
        current_user(authorization)
        combiner = _combiner()
        if combiner is None:
            return {"available": False,
                    "reason": "No combiner fitted on real paired patients yet (see train_late_fusion.py). "
                              "Each branch's own result is still valid on its own."}
        row = flatten_branch_outputs(**{k: outputs.get(k) for k in ("ecg", "xray", "echo", "mri", "clinical", "labs")})
        return {"available": True, "probabilities": combiner.predict_proba(row),
                "trained_on": {k: combiner.meta.get(k) for k in ("cohort", "target", "n_patients", "n_subjects")},
                "cross_validation": combiner.meta.get("cv_results")}

    return router
