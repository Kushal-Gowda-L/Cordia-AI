"""Multi-physics heart twin endpoints: electrical activation, muscle, blood
pressure/flow and fat on the patient's MRI mesh -- see src/imaging/heart_physics.py.

GET  /api/physics/sample    the ACDC sample heart (checkpoints/sample_heart_mesh.json)
POST /api/physics/simulate  run the twin with what-if changes

The twin is calibrated to measured values when it has them and says where
every value came from: the patient's MRI (volumes), their record (blood
pressure), or a typical adult value. Every response carries the
illustrative-model note.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.imaging import heart_physics as hp

CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"
SAMPLE_MESH = CHECKPOINTS / "sample_heart_mesh.json"
TYPICAL = {"sbp_mmhg": 120.0, "dbp_mmhg": 80.0, "hr_bpm": 72.0, "ef_percent": 60.0}


class SimulateIn(BaseModel):
    meshes: dict | None = None  # label_meshes output from /api/imaging/mri; the sample heart if left out
    metrics: dict | None = None  # /api/imaging/mri "metrics" (lv_edv_ml, lv_esv_ml, ...)
    patient_id: int | None = None  # take blood pressure and heart rate from this patient's record
    self_record: bool = False  # ... or from the logged-in patient's own record
    whatif: dict | None = None
    fat: dict | None = None  # an epicardial_fat() result, if a CT with pericardium mask was measured


class _Cache(OrderedDict):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def get_or(self, key, make):
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = self[key] = make()
        if len(self) > self.size:
            self.popitem(last=False)
        return value


_geometries = _Cache(8)
_calibrations = _Cache(32)


def _mesh_key(meshes: dict) -> str:
    return hashlib.sha1(json.dumps(meshes["LV myocardium"]["vertices"]).encode()).hexdigest()


def load_sample() -> dict:
    if not SAMPLE_MESH.exists():
        raise HTTPException(503, "checkpoints/sample_heart_mesh.json is missing: it is written by "
                                 "notebooks/step17_mri_acdc_colab.ipynb.")
    return json.loads(SAMPLE_MESH.read_text())


def resolve_targets(meshes: dict, metrics: dict | None, twin: dict | None) -> tuple[dict, dict]:
    """Calibration targets and where each came from."""
    metrics = metrics or {}
    targets, sources = {}, {}
    if metrics.get("lv_edv_ml"):
        targets["edv_ml"], sources["edv_ml"] = float(metrics["lv_edv_ml"]), "Measured: cardiac MRI segmentation"
    else:
        targets["edv_ml"] = hp.mesh_volume_ml(meshes["LV cavity"])
        sources["edv_ml"] = "Measured: volume of the end-diastolic LV cavity mesh"
    if metrics.get("lv_esv_ml"):
        targets["esv_ml"], sources["esv_ml"] = float(metrics["lv_esv_ml"]), "Measured: cardiac MRI segmentation"
    else:
        targets["esv_ml"] = targets["edv_ml"] * (1 - TYPICAL["ef_percent"] / 100)
        sources["esv_ml"] = "Assumed: no end-systolic scan, typical EF of 60%"
    params = ((twin or {}).get("scenarios", {}).get("rest", {}) or {}).get("params", {})
    for key, name in (("sbp_mmhg", "systolicBP"), ("dbp_mmhg", "diastolicBP"), ("hr_bpm", "heartRate")):
        if params.get(name):
            targets[key] = float(params[name])
            sources[key] = "From the patient's twin at rest (" + (
                (twin["scenarios"]["rest"].get("sources") or {}).get(name, "record") + ")")
        else:
            targets[key], sources[key] = TYPICAL[key], "Typical adult value"
    if targets["esv_ml"] >= targets["edv_ml"] - 5:
        raise HTTPException(400, "End-systolic volume must be smaller than end-diastolic volume")
    return targets, sources


def build_router(current_user: Callable, twin_lookup: Callable | None = None) -> APIRouter:
    """twin_lookup(authorization, patient_id | None) -> the record-derived twin (patient_id None =
    the logged-in patient's own record)."""
    router = APIRouter(prefix="/api/physics")

    @router.get("/sample")
    def sample(authorization: str | None = Header(None)):
        current_user(authorization)
        data = load_sample()
        return {"pid": data.get("pid"), "spacing_mm": data.get("spacing_mm"), "meshes": data["meshes"],
                "source": "ACDC sample heart (end-diastole), segmented by the MRI branch"}

    @router.post("/simulate")
    def simulate(body: SimulateIn, authorization: str | None = Header(None)):
        current_user(authorization)
        meshes = body.meshes or load_sample()["meshes"]
        if not {"LV myocardium", "LV cavity"} <= set(meshes):
            raise HTTPException(400, "Needs the 'LV myocardium' and 'LV cavity' meshes from /api/imaging/mri")
        twin = None
        if twin_lookup and (body.patient_id is not None or body.self_record):
            twin = twin_lookup(authorization, body.patient_id)
        targets, sources = resolve_targets(meshes, body.metrics, twin)

        key = _mesh_key(meshes)
        try:
            geo = _geometries.get_or(key, lambda: hp.build_geometry(meshes))
        except ValueError as e:
            raise HTTPException(400, str(e))
        tkey = (key, *(round(targets[k], 1) for k in ("edv_ml", "esv_ml", "sbp_mmhg", "dbp_mmhg", "hr_bpm")))

        def calibrate():
            at0 = hp.activation_times(geo, "normal")
            return hp.calibrate(targets, hp.ventricular_activation_curve(at0, np.ones(len(at0)), targets["hr_bpm"], 0.0005), 0.0005)

        whatif = dict(body.whatif or {})
        if whatif.get("conduction", "normal") not in hp.CONDUCTION_MODES:
            raise HTTPException(400, f"conduction must be one of {hp.CONDUCTION_MODES}")
        unknown = set(whatif) - set(hp.DEFAULT_WHATIF)
        if unknown:
            raise HTTPException(400, f"Unknown what-if setting(s): {sorted(unknown)}")
        calibration = _calibrations.get_or(tkey, calibrate)
        out = hp.run_twin(meshes, targets, whatif, geometry=geo, calibration=calibration)
        out["targets"] = {k: {"value": round(v, 1), "source": sources[k]} for k, v in targets.items()}
        out["fat"] = body.fat if body.fat and body.fat.get("measured") else hp.FAT_NOT_MEASURED
        out["out_of_scope"] = OUT_OF_SCOPE
        return out

    return router


OUT_OF_SCOPE = [
    "3D finite-element wall mechanics (stress and strain in the muscle): needs a volumetric mesh, fibre "
    "data and a solver such as FEniCS or FEBio.",
    "Ionic electrophysiology (monodomain/bidomain with cell models): needs openCARP-class software; "
    "here activation is an eikonal-style graph model with a fixed action potential duration.",
    "Blood flow inside the chambers (3D CFD): needs SimVascular-class software and time-resolved "
    "geometry; here flow is a 0D circuit, so it gives pressures and flows over time, not flow patterns.",
    "Right ventricle and atria as 3D muscle: ACDC does not segment RV muscle or the atria.",
    "Epicardial fat and coronary plaque: need CT with fat and plaque segmentation, not trained here.",
]
