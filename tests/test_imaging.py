"""Checks for the imaging branches, run on synthetic data (no datasets or GPU needed).

    python -m pytest tests/test_imaging.py -q

Geometry tests use shapes with known answers (an ellipsoid's volume, a
ring's thickness), so the measurement formulas are checked against maths,
not against themselves.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import ViTConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.imaging import cardiac_metrics as cm  # noqa: E402
from src.imaging.mesh import mask_to_mesh, mesh_to_obj  # noqa: E402
from src.imaging.preprocess import clip_to_tensor, mask_slices_to_tensor, sample_clip, segment_mri_volume, xray_to_tensor  # noqa: E402
from src.models.imaging.cardiac_seg import build_mri_segmenter  # noqa: E402
from src.models.imaging.cxr_vit import CXR_FINDINGS, CXRViT  # noqa: E402
from src.models.imaging.echo_vit import EchoViT  # noqa: E402
from src.models.late_fusion import LateFusionCombiner, flatten_branch_outputs  # noqa: E402

TINY_VIT = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)


# ---------------------------------------------------------------- models
def test_cxr_vit_shapes_and_freezing():
    model = CXRViT(config=TINY_VIT, train_from_layer=1)
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, len(CXR_FINDINGS))
    assert model.embed(torch.randn(1, 3, 224, 224)).shape == (1, 256)
    trainable = {n for n, p in model.vit.named_parameters() if p.requires_grad}
    assert trainable and all(".1." in n for n in trainable)  # only encoder layer 1 is trainable


def test_echo_vit_predicts_one_ef_per_clip():
    model = EchoViT(config=TINY_VIT, n_frames=4, train_from_layer=1)
    assert model(torch.randn(2, 4, 3, 224, 224)).shape == (2,)


def test_mri_segmenter_output_matches_input_size():
    model = build_mri_segmenter(feature_size=12).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 224, 224)).shape == (1, 4, 224, 224)
    mask = segment_mri_volume(model, np.random.rand(150, 170, 3).astype(np.float32))
    assert mask.shape == (150, 170, 3) and mask.dtype == np.uint8


# ---------------------------------------------------------------- preprocessing
def test_xray_any_image_mode_becomes_vit_input():
    from PIL import Image

    for mode in ("L", "RGB", "I;16"):
        t = xray_to_tensor(Image.new(mode, (512, 400)))
        assert t.shape == (3, 224, 224) and -1.0 <= float(t.min()) <= float(t.max()) <= 1.0


def test_clip_sampling_loops_short_videos():
    frames = np.arange(10)[:, None, None].repeat(112, 1).repeat(112, 2).astype(np.uint8)
    clip = sample_clip(frames, 16, 2, start=0)
    assert len(clip) == 16 and clip[5, 0, 0] == 0  # index 10 wraps to frame 0
    assert clip_to_tensor(clip).shape == (16, 3, 224, 224)


def test_mask_resize_keeps_label_values():
    mask = np.random.randint(0, 4, (100, 120, 3)).astype(np.uint8)
    assert set(torch.unique(mask_slices_to_tensor(mask)).tolist()) <= {0, 1, 2, 3}


# ---------------------------------------------------------------- measurements
def ellipsoid(shape, centre, radii, spacing):
    grid = np.indices(shape).astype(float)
    d = sum(((grid[i] - centre[i]) * spacing[i] / radii[i]) ** 2 for i in range(3))
    return d <= 1


def test_lv_volume_and_ef_match_the_ellipsoid_formula():
    spacing = (1.0, 1.0, 2.0)
    ed = ellipsoid((80, 80, 50), (40, 40, 25), (25, 25, 40), spacing).astype(np.uint8) * cm.LV
    es = ellipsoid((80, 80, 50), (40, 40, 25), (18, 18, 35), spacing).astype(np.uint8) * cm.LV
    edv_true = 4 / 3 * np.pi * 25 * 25 * 40 / 1000
    esv_true = 4 / 3 * np.pi * 18 * 18 * 35 / 1000
    m = cm.mri_function(ed, es, spacing, height_cm=175, weight_kg=70)
    assert m["lv_edv_ml"] == pytest.approx(edv_true, rel=0.03)
    assert m["lv_ef_percent"] == pytest.approx(100 * (edv_true - esv_true) / edv_true, abs=1.5)
    assert m["bsa_m2"] == pytest.approx(np.sqrt(175 * 70 / 3600))
    assert len(cm.acdc_feature_vector(m)) == len(cm.ACDC_FEATURES)


def test_wall_thickness_of_a_ring():
    yy, xx = np.indices((100, 100))
    r = np.hypot(yy - 50, xx - 50)
    ring = ((r >= 20) & (r <= 30)).astype(np.uint8)[..., None] * cm.MYO  # 10 px wall
    assert cm.max_wall_thickness_mm(ring, (1.2, 1.2, 8.0)) == pytest.approx(10 * 1.2, rel=0.12)


def test_echo_area_length_is_exact_for_a_prolate_ellipsoid():
    yy, xx = np.indices((300, 300))
    a, b = 90, 40  # long and short semi-axes, px
    ellipse = ((((yy - 150) / a) ** 2 + ((xx - 150) / b) ** 2) <= 1).astype(np.uint8)
    v = cm.echo_area_length_volume_ml(ellipse, (0.3, 0.3))
    assert v == pytest.approx(4 / 3 * np.pi * (a * 0.3) * (b * 0.3) ** 2 / 1000, rel=0.03)


def test_sphere_mesh_is_closed_and_in_millimetres():
    spacing = (1.0, 1.0, 1.0)
    sphere = ellipsoid((40, 40, 40), (20, 20, 20), (12, 12, 12), spacing)
    mesh = mask_to_mesh(sphere, spacing, step=1, smooth_sigma=0)
    v = np.array(mesh["vertices"])
    radius = np.linalg.norm(v - v.mean(0), axis=1)
    assert radius.mean() == pytest.approx(12, rel=0.08)
    assert mesh_to_obj(mesh).count("\nf ") + 1 >= len(mesh["faces"])
    assert mask_to_mesh(np.zeros((5, 5, 5)), spacing) is None


# ---------------------------------------------------------------- digital twin
def sample_twin():
    from src.patient_twin import build_twin

    record = {"age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 230, "fbs": 0, "restecg": 2,
              "thalach": 150, "exang": 1, "oldpeak": 1.5}
    classes = ["No disease", "Mild", "Moderate", "Severe", "Critical"]
    return build_twin("Test", record, 0.6, 1, [0.4, 0.3, 0.15, 0.1, 0.05], classes)


def test_imaging_replaces_estimates_and_says_so():
    from src.imaging_twin import apply_imaging

    base = sample_twin()
    twin = apply_imaging(base, {"mri": {"lv_ef_percent": 35.0, "lv_sv_ml": 50.0, "lv_edv_ml": 180.0,
                                        "max_wall_thickness_mm": 15.0, "diagnosis": "Dilated cardiomyopathy"},
                                "xray": {"Cardiomegaly": 0.8, "Edema": 0.1, "Effusion": 0.2}})
    rest = twin["scenarios"]["rest"]
    assert rest["params"]["contractility"] == 55 + 1.2 * (35 - 60)
    assert rest["params"]["baseStrokeVolume"] == 50.0
    assert rest["params"]["preload"] == 65.0
    assert rest["params"]["afterload"] == base["scenarios"]["rest"]["params"]["afterload"] + 10
    assert rest["sources"]["contractility"].startswith("From cardiac MRI")
    assert any("Enlarged heart" in f for f in twin["imaging"]["findings"])
    assert any("Reduced ejection fraction" in f for f in twin["imaging"]["findings"])
    assert base["scenarios"]["rest"]["params"]["baseStrokeVolume"] != 50.0  # input not mutated


def test_echo_used_when_no_mri_and_nothing_changes_without_imaging():
    from src.imaging_twin import apply_imaging

    twin = apply_imaging(sample_twin(), {"echo": {"lv_ef_percent": 62.0}})
    assert twin["scenarios"]["rest"]["sources"]["contractility"].startswith("From echocardiogram")
    same = apply_imaging(sample_twin(), {})
    assert same["scenarios"] == sample_twin()["scenarios"] and same["imaging"]["findings"] == []


# ---------------------------------------------------------------- late fusion
def synthetic_cohort(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "subject_id": rng.integers(0, n // 2, n),
        "age": rng.normal(60, 10, n), "sex": rng.integers(0, 2, n),
        "ecg_MI": np.clip(0.3 + 0.3 * y + rng.normal(0, 0.15, n), 0, 1),
        "cxr_Cardiomegaly": np.clip(0.2 + 0.3 * y + rng.normal(0, 0.2, n), 0, 1),
    })
    df.loc[rng.random(n) < 0.3, "cxr_Cardiomegaly"] = np.nan  # some patients had no X-ray
    return df, y


def test_fusion_beats_a_single_branch_on_paired_data(tmp_path):
    df, y = synthetic_cohort()
    from sklearn.metrics import roc_auc_score

    single = LateFusionCombiner(["clinical", "xray"], ["No", "Yes"]).cross_validate(df, y, groups=df["subject_id"])
    fused = LateFusionCombiner(["clinical", "ecg", "xray"], ["No", "Yes"]).cross_validate(df, y, groups=df["subject_id"])
    assert roc_auc_score(y, fused[:, 1]) > roc_auc_score(y, single[:, 1])

    combiner = LateFusionCombiner(["clinical", "ecg", "xray"], ["No", "Yes"]).fit(df, y, cohort="synthetic")
    combiner.save(tmp_path / "lf.joblib")
    loaded = LateFusionCombiner.load(tmp_path / "lf.joblib")
    probs = loaded.predict_proba(flatten_branch_outputs(ecg={"MI": 0.9}, clinical={"age": 60, "sex": 1}))
    assert set(probs) == {"No", "Yes"} and probs["Yes"] > 0.5  # works with the X-ray missing


# ---------------------------------------------------------------- notebooks + download script
@pytest.mark.parametrize("nb", sorted((ROOT / "notebooks").glob("step1[5-8]_*.ipynb")))
def test_notebook_module_copies_match_the_project(nb):
    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    copies = [c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in cells]
    copies = [s for s in copies if s.startswith("%%writefile ")]
    assert copies
    for src in copies:
        path, body = src.split("\n", 1)
        assert (ROOT / path.split()[1]).read_text(encoding="utf-8").strip("\n") == body.strip("\n"), path


def test_download_script_verify(tmp_path, capsys):
    sys.path.insert(0, str(ROOT))
    import download_imaging_datasets as dl

    (tmp_path / "Videos").mkdir()
    (tmp_path / "FileList.csv").write_text("FileName,EF,Split\n")
    assert not dl.verify(dl.DATASETS["echonet_dynamic"], tmp_path)
    (tmp_path / "Videos" / "0X1.avi").write_bytes(b"")
    assert dl.verify(dl.DATASETS["echonet_dynamic"], tmp_path)


# ---------------------------------------------------------------- API
@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(ROOT / "mobile_app" / "backend"))
    import imaging_api

    monkeypatch.setattr(imaging_api, "CHECKPOINTS", tmp_path)
    for f in (imaging_api._xray_model, imaging_api._echo_model, imaging_api._mri_model,
              imaging_api._mri_diagnosis, imaging_api._ct_model, imaging_api._combiner):
        f.cache_clear()
    app = FastAPI()
    app.include_router(imaging_api.build_router(lambda authorization: {"id": 1}))
    return TestClient(app), tmp_path, imaging_api


def test_untrained_branch_answers_503_with_the_notebook(client):
    c, _, _ = client
    assert c.get("/api/imaging/status").json()["xray"]["trained"] is False
    r = c.post("/api/imaging/xray", files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 503 and "step15" in r.json()["detail"]
    assert c.post("/api/imaging/combined", json={}).json()["available"] is False


def test_xray_endpoint_end_to_end_with_random_weights(client):
    from PIL import Image

    c, ckpt, _ = client
    torch.save(CXRViT(pretrained=False).state_dict(), ckpt / "cxr_vit.pt")
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(256, 256) * 255).astype(np.uint8)).save(buf, format="PNG")
    r = c.post("/api/imaging/xray", files={"image": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and set(r.json()["findings"]) == set(CXR_FINDINGS)
    bad = c.post("/api/imaging/xray", files={"image": ("x.png", b"garbage", "image/png")})
    assert bad.status_code == 400


def test_mri_endpoint_end_to_end_with_random_weights(client, tmp_path):
    import nibabel as nib

    c, ckpt, _ = client
    torch.save(build_mri_segmenter().state_dict(), ckpt / "mri_swin_unetr.pt")
    vol = (np.random.rand(96, 96, 4) * 500).astype(np.float32)
    path = tmp_path / "v.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.diag([1.5, 1.5, 10, 1])), str(path))
    data = path.read_bytes()
    r = c.post("/api/imaging/mri", files={"ed": ("ed.nii.gz", data), "es": ("es.nii.gz", data)},
               data={"height_cm": "170", "weight_kg": "70"})
    assert r.status_code == 200, r.text
    assert "lv_ef_percent" in r.json()["metrics"] and "meshes" in r.json()


def test_tiny_structure_gives_no_mesh_instead_of_crashing():
    tiny = np.zeros((20, 20, 3), bool)
    tiny[10, 10, 1] = True
    mesh = mask_to_mesh(tiny, (1.5, 1.5, 10.0))
    assert mesh is None or len(mesh["faces"]) > 0
