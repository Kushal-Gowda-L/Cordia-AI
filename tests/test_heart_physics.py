"""Checks for the multi-physics heart twin (src/imaging/heart_physics.py) and its API.

    python -m pytest tests/test_heart_physics.py -q

Runs on a synthetic left ventricle (a truncated ellipsoidal shell with known
wall thickness), so the geometry checks have answers from maths, and the
physics checks test directions every physiology textbook agrees on (a blocked
left bundle slows activation and weakens the beat, raising resistance raises
blood pressure, ...). They do not, and cannot, show the model is accurate
for a patient.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mobile_app" / "backend"))

from src.imaging import heart_physics as hp  # noqa: E402
from src.imaging.mesh import label_meshes, mask_to_mesh  # noqa: E402

SPACING = (1.5, 1.5, 4.0)
WALL_MM = 10.0
DT = 0.0005


def synthetic_heart(wall_mm: float = WALL_MM, cavity_a: float = 22.0) -> np.ndarray:
    """ACDC-style labels (1 RV, 2 myocardium, 3 LV cavity) for a half-ellipsoid LV, base at z = 0."""
    shape = (100, 100, 20)
    i, j, k = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, SPACING)], indexing="ij")
    x, y, z = i - 75.0, j - 75.0, k - 4.0  # base plane at z = 4 mm, apex towards +z
    c = 55.0

    def inside(a, cc):
        return (x / a) ** 2 + (y / a) ** 2 + (z / cc) ** 2 <= 1

    mask = np.zeros(shape, np.uint8)
    lv = inside(cavity_a, c) & (z >= 0)
    myo = inside(cavity_a + wall_mm, c + wall_mm) & (z >= 0) & ~lv
    rv = ((x + 40) / 14) ** 2 + (y / 26) ** 2 + (z / 45) ** 2 <= 1
    mask[rv & (z >= 0) & ~myo & ~lv] = 1
    mask[myo] = 2
    mask[lv] = 3
    return mask


@pytest.fixture(scope="module")
def meshes():
    return label_meshes(synthetic_heart(), SPACING, {1: "RV cavity", 2: "LV myocardium", 3: "LV cavity"}, step=1)


@pytest.fixture(scope="module")
def geo(meshes):
    return hp.build_geometry(meshes)


TARGETS = {"edv_ml": 130.0, "esv_ml": 55.0, "sbp_mmhg": 130.0, "dbp_mmhg": 80.0, "hr_bpm": 70.0}


@pytest.fixture(scope="module")
def calibration(geo):
    at = hp.activation_times(geo, "normal")
    return hp.calibrate(TARGETS, hp.ventricular_activation_curve(at, np.ones(len(at)), TARGETS["hr_bpm"], DT), DT)


def twin(meshes, geo, calibration, **whatif):
    return hp.run_twin(meshes, TARGETS, whatif, geometry=geo, calibration=calibration)


# ---------------------------------------------------------------- geometry
def test_mesh_volume_of_a_sphere():
    r = 20.0
    g = np.arange(64) - 32.0
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    mesh = mask_to_mesh(x**2 + y**2 + z**2 <= r**2, (1.0, 1.0, 1.0), step=1, smooth_sigma=0)
    assert hp.mesh_volume_ml(mesh) == pytest.approx(4 / 3 * np.pi * r**3 / 1000, rel=0.03)


def test_geometry_axis_wall_thickness_and_fibres(geo):
    assert geo.axis[2] == pytest.approx(-1.0)  # base (z = 0, wide end) is at -z, so apex -> base points -z
    n = geo.extra["n_surface"]
    epi = geo.transmural[:n] >= 1
    # mid-ventricle, away from the base cap and the apex
    mid = epi & (geo.level[:n] > 0.4) & (geo.level[:n] < 0.85)
    assert np.median(geo.thickness_mm[:n][mid]) == pytest.approx(WALL_MM, abs=2.5)
    assert np.allclose(np.linalg.norm(geo.fibre, axis=1), 1.0)
    assert geo.helix_deg.max() == pytest.approx(60) and geo.helix_deg.min() == pytest.approx(-60)
    # the septum faces the RV, which sits at -x
    septal = np.abs(geo.angle_deg[:n]) < 30
    assert geo.vertices[:n][septal, 0].mean() < geo.vertices[:n][~septal, 0].mean()


# ---------------------------------------------------------------- electrical
def test_normal_activation_is_fast_and_starts_at_the_septum(geo):
    at = hp.activation_times(geo, "normal")
    assert np.isfinite(at).all() and at.min() == 0
    assert 40 < at.max() < 120  # healthy LV total activation is roughly 60-80 ms
    septal, lateral = np.abs(geo.angle_deg) < 45, np.abs(geo.angle_deg) > 135
    assert at[septal].mean() < at[lateral].mean()


def test_lbbb_is_slow_and_crt_resynchronises(geo):
    normal = hp.activation_times(geo, "normal").max()
    lbbb = hp.activation_times(geo, "lbbb")
    crt = hp.activation_times(geo, "crt").max()
    assert lbbb.max() > normal + 50
    assert crt < lbbb.max()
    lateral = np.abs(geo.angle_deg) > 135
    assert lbbb[lateral].mean() > lbbb[~lateral].mean()  # lateral wall activates last in LBBB


def test_slower_conduction_lengthens_activation(geo):
    fast = hp.activation_times(geo, "normal", cv_scale=1.0).max()
    slow = hp.activation_times(geo, "normal", cv_scale=0.5).max()
    assert slow == pytest.approx(2 * fast, rel=0.05)


def test_repolarisation_follows_activation(geo):
    at = hp.activation_times(geo, "normal")
    rt = hp.repolarisation_times(geo, at, 60)
    apd = rt - at
    assert (apd > 200).all()
    assert apd[geo.transmural <= 0].mean() > apd[geo.transmural >= 1].mean()  # endocardium longer


# ---------------------------------------------------------------- blood flow
def test_calibration_matches_measured_values(calibration):
    _, fit = calibration
    for key, v in fit.items():
        assert v["model"] == pytest.approx(v["target"], rel=0.03), key


def test_volumes_and_pressures_are_physiological(meshes, geo, calibration):
    out = twin(meshes, geo, calibration)
    s = out["hemodynamics"]["summary"]
    assert s["ef_percent"] == pytest.approx(100 * (130 - 55) / 130, abs=2)
    assert s["lv_peak_pressure_mmhg"] >= s["sbp_mmhg"] - 2  # the LV must exceed the aorta to eject
    assert 2 < s["mean_la_pressure_mmhg"] < 20
    assert s["cardiac_output_l_min"] == pytest.approx(75 * 70 / 1000, rel=0.05)
    c = out["hemodynamics"]["curves"]
    # one-way valves: no flow backwards, and never both open at once
    q_av, q_mv = np.array(c["q_av"]), np.array(c["q_mv"])
    assert (q_av >= 0).all() and (q_mv >= 0).all() and not ((q_av > 1) & (q_mv > 1)).any()


def test_dyssynchrony_weakens_the_beat(meshes, geo, calibration):
    """Electromechanical coupling: same muscle, same calibration, slower spread -> weaker beat."""
    base = twin(meshes, geo, calibration)["hemodynamics"]["summary"]
    lbbb = twin(meshes, geo, calibration, conduction="lbbb")["hemodynamics"]["summary"]
    crt = twin(meshes, geo, calibration, conduction="crt")["hemodynamics"]["summary"]
    assert lbbb["sv_ml"] < base["sv_ml"] and lbbb["dpdt_max_mmhg_s"] < base["dpdt_max_mmhg_s"]
    assert lbbb["sv_ml"] < crt["sv_ml"] <= base["sv_ml"] + 0.5


@pytest.mark.parametrize("whatif, key, direction", [
    ({"contractility": 0.6}, "ef_percent", -1),
    ({"resistance": 1.5}, "sbp_mmhg", +1),
    ({"arterial_stiffness": 2.0}, "sbp_mmhg", +1),
    ({"diastolic_stiffness": 2.5}, "edv_ml", -1),
    ({"filling_pressure": 1.5}, "edv_ml", +1),
    ({"aortic_stenosis": 0.8}, "lv_peak_pressure_mmhg", +1),
    ({"scar": {"angle_deg": 180, "level": 0.5, "radius_mm": 20}}, "sv_ml", -1),
])
def test_what_if_directions(meshes, geo, calibration, whatif, key, direction):
    base = twin(meshes, geo, calibration)["hemodynamics"]["summary"][key]
    changed = twin(meshes, geo, calibration, **whatif)["hemodynamics"]["summary"][key]
    assert np.sign(changed - base) == direction


def test_scar_does_not_contract(meshes, geo, calibration):
    out = twin(meshes, geo, calibration, scar={"angle_deg": 180, "level": 0.5, "radius_mm": 15})
    contracts = np.array(out["muscle"]["contracts"])
    assert 0 < (contracts == 0).sum() < len(contracts) / 2
    assert len(out["electrical"]["activation_ms"]) == len(meshes["LV myocardium"]["vertices"])


# ---------------------------------------------------------------- muscle (measured) and fat
def test_regional_thickening_from_ed_and_es_masks():
    ed = synthetic_heart(wall_mm=8.0, cavity_a=24.0)
    es = synthetic_heart(wall_mm=12.0, cavity_a=18.0)
    out = hp.regional_thickening(ed, es, SPACING)
    mid = [s for s in out["slices"] if s is not None][3]
    assert np.median(mid["ed_mm"]) == pytest.approx(8.0, abs=1.6)
    assert np.median(mid["thickening_pct"]) == pytest.approx(50, abs=15)
    assert len(out["segments"]) == 16
    meshes = label_meshes(ed, SPACING, {2: "LV myocardium"}, step=1)
    vals = hp.thickening_at_vertices(meshes["LV myocardium"]["vertices"], out, SPACING)
    assert len(vals) == len(meshes["LV myocardium"]["vertices"])
    assert np.nanmedian([v for v in vals if v is not None]) > 20


def test_epicardial_fat_counts_only_fat_inside_the_pericardium():
    hu = np.full((20, 20, 20), 40.0)  # soft tissue
    hu[5:10, 5:10, 5:10] = -100.0  # 125 voxels of fat inside
    hu[0:2, 0:2, 0:2] = -100.0  # fat outside the sac
    hu[12:14, 12:14, 12:14] = -900.0  # air: below the fat window
    peric = np.zeros_like(hu, bool)
    peric[3:18, 3:18, 3:18] = True
    out = hp.epicardial_fat(hu, peric, (2.0, 2.0, 2.0))
    assert out["eat_volume_ml"] == pytest.approx(125 * 8 / 1000, abs=0.05)
    assert out["eat_mean_hu"] == -100.0


# ---------------------------------------------------------------- API
def _client(twin_lookup=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import physics_api

    app = FastAPI()
    app.include_router(physics_api.build_router(lambda authorization: {"id": 1}, twin_lookup))
    return TestClient(app), physics_api


def test_api_simulate_with_posted_meshes_and_record_vitals(meshes):
    record = {"scenarios": {"rest": {"params": {"systolicBP": 150, "diastolicBP": 90, "heartRate": 65},
                                     "sources": {"systolicBP": "From record: trestbps"}}}}
    client, _ = _client(lambda authorization, pid: record)
    body = {"meshes": meshes, "metrics": {"lv_edv_ml": 140, "lv_esv_ml": 70}, "patient_id": 3}
    r = client.post("/api/physics/simulate", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["targets"]["sbp_mmhg"]["value"] == 150 and "trestbps" in out["targets"]["sbp_mmhg"]["source"]
    assert out["targets"]["esv_ml"]["source"].startswith("Measured")
    assert out["hemodynamics"]["summary"]["sbp_mmhg"] == pytest.approx(150, rel=0.03)
    assert out["fat"]["measured"] is False and "Illustrative" in out["note"]

    r = client.post("/api/physics/simulate", json={**body, "whatif": {"conduction": "lbbb"}})
    assert r.json()["electrical"]["lv_total_activation_ms"] > out["electrical"]["lv_total_activation_ms"]
    assert client.post("/api/physics/simulate", json={**body, "whatif": {"warp_drive": 1}}).status_code == 400
    assert client.post("/api/physics/simulate", json={**body, "whatif": {"conduction": "magic"}}).status_code == 400


def test_api_without_measurements_says_what_was_assumed(meshes):
    client, _ = _client()
    out = client.post("/api/physics/simulate", json={"meshes": meshes}).json()
    assert out["targets"]["esv_ml"]["source"].startswith("Assumed")
    assert out["targets"]["sbp_mmhg"]["source"] == "Typical adult value"
    assert out["targets"]["edv_ml"]["value"] == pytest.approx(hp.mesh_volume_ml(meshes["LV cavity"]), abs=0.1)


@pytest.mark.skipif(not (ROOT / "checkpoints" / "sample_heart_mesh.json").exists(), reason="sample mesh not present")
def test_api_sample_heart():
    client, _ = _client()
    sample = client.get("/api/physics/sample").json()
    assert {"LV myocardium", "LV cavity"} <= set(sample["meshes"])
    out = client.post("/api/physics/simulate", json={}).json()
    assert len(out["electrical"]["activation_ms"]) == len(sample["meshes"]["LV myocardium"]["vertices"])
    json.dumps(out)  # serialisable for the browser
