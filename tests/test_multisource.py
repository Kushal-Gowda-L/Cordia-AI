"""Checks for the multi-source heart-risk model, on synthetic data (no dataset files needed).

    python -m pytest tests/test_multisource.py -q
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predict_multisource import EXAMPLE  # noqa: E402
from src.multisource import profile as P  # noqa: E402
from src.multisource.explain import shapley_sampling, tabular_shapley  # noqa: E402
from src.multisource.fusion import HeartRiskModel, LoadedEncoder  # noqa: E402
from src.multisource.model import MASK, NA, OBSERVED, FTEncoder, TabularPreprocessor, build  # noqa: E402
from src.multisource.report import plain_language_report  # noqa: E402
from src.multisource.sources import SOURCES  # noqa: E402
from src.multisource.train import train_source  # noqa: E402


def synthetic_uci(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """UCI-shaped rows where age, exang and oldpeak drive the label."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(55, 9, n), "sex": rng.integers(0, 2, n), "cp": rng.integers(1, 5, n),
        "trestbps": rng.normal(132, 18, n), "chol": rng.normal(230, 45, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.normal(140, 22, n), "exang": rng.integers(0, 2, n),
        "oldpeak": rng.gamma(1.2, 0.8, n), "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n),
    })
    df.loc[rng.random(n) < 0.3, "ca"] = np.nan  # blanks, as in the real data
    z = 0.08 * (df["age"] - 55) + 1.5 * df["exang"] + 0.9 * (df["oldpeak"] - 1) - 0.2
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def test_profile_normalize_derives_bmi_and_rejects_typos():
    p = P.normalize({"height_cm": 180, "weight_kg": 81, "smoker": "yes", "age": None})
    assert p["bmi"] == pytest.approx(25.0)
    assert p["smoker"] == 1.0 and "age" not in p
    with pytest.raises(KeyError):
        P.normalize({"systolic": 120})
    with pytest.raises(ValueError):
        P.normalize({"sex": "M"})


@pytest.mark.parametrize("name", list(SOURCES))
def test_from_profile_produces_valid_columns(name):
    source = SOURCES[name]
    beat = {"ecg_beat": list(np.sin(np.linspace(0, 6, 140)))}
    row = source.from_profile(P.normalize({**EXAMPLE, **beat}))
    assert row, f"{name}: example profile reaches none of its columns"
    for f in source.features:
        if f.name in row and f.kind == "cat":
            assert int(row[f.name]) in f.categories, (name, f.name, row[f.name])


def test_age_codes_match_survey_bins():
    brfss = SOURCES["brfss2015"]
    assert brfss.from_profile(P.normalize({"age": 22}))["Age"] == 1
    assert brfss.from_profile(P.normalize({"age": 25}))["Age"] == 2
    assert brfss.from_profile(P.normalize({"age": 79}))["Age"] == 12
    assert brfss.from_profile(P.normalize({"age": 92}))["Age"] == 13


def test_preprocessor_separates_blank_from_not_provided():
    source = SOURCES["uci"]
    prep = TabularPreprocessor(source).fit(synthetic_uci(100))
    df = pd.DataFrame([{"age": 60, "ca": np.nan, "cp": 4, "thal": 5}])  # thal 5 is not a valid code
    enc = prep.encode(df)
    num = [f.name for f in prep.num]
    cat = [f.name for f in prep.cat]
    assert enc["num_state"][0, num.index("age")] == OBSERVED
    assert enc["num_state"][0, num.index("ca")] == NA
    assert enc["num_state"][0, num.index("chol")] == MASK  # column absent -> not provided
    assert enc["x_cat"][0, cat.index("cp")] > MASK
    assert enc["x_cat"][0, cat.index("thal")] == NA
    assert enc["x_cat"][0, cat.index("sex")] == MASK


def test_shapley_sampling_exact_for_additive_game():
    weights = np.array([1.0, -2.0, 0.5])
    phi = shapley_sampling(lambda present: present.astype(float) @ weights, 3, n_permutations=8)
    np.testing.assert_allclose(phi, weights)


def test_tabular_shapley_sums_to_logit_difference():
    torch.manual_seed(0)
    source = SOURCES["uci"]
    prep, model = build(source, {"d_token": 16, "depth": 1, "heads": 2, "dropout": 0.0})
    prep.fit(synthetic_uci(100))
    model.eval()
    row = {"age": 61, "sex": 1, "cp": 4, "oldpeak": 2.0, "exang": 1}
    phi = tabular_shapley(prep, model, row, n_permutations=16)
    with torch.no_grad():
        full = model(**prep.encode(pd.DataFrame([row]))).item()
        empty = model(**prep.encode(pd.DataFrame([{}]))).item()
    assert sum(phi.values()) == pytest.approx(full - empty, abs=1e-4)


def test_encoder_handles_any_subset_of_columns():
    model = FTEncoder(n_num=3, cardinalities=[2, 4], d_token=8, depth=1, heads=2)
    out = model(torch.zeros(5, 3), torch.tensor([[0, 1, 2]] * 5), torch.tensor([[3, 2]] * 5))
    assert out.shape == (5,) and torch.isfinite(out).all()


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("ckpt")
    ckpt = train_source(SOURCES["uci"], synthetic_uci(), {"max_epochs": 25, "d_token": 32, "depth": 2,
                                                          "lr": 1e-3, "patience": 25}, log=lambda *_: None)
    torch.save(ckpt, out / "uci.pt")
    return out, ckpt


def test_training_learns_signal_and_beats_chance(trained_dir):
    _, ckpt = trained_dir
    m = ckpt["metrics"]
    assert m["test"]["auroc"] > 0.7
    assert 0 <= m["test"]["brier"] < 0.25
    assert set(m["baselines"]) == {"logistic_regression", "gradient_boosting"}
    # The columns that carry the signal should matter most.
    top = sorted(ckpt["importance"], key=ckpt["importance"].get, reverse=True)[:3]
    assert {"exang", "oldpeak"} & set(top)


def test_fused_prediction_and_explanation(trained_dir):
    out, _ = trained_dir
    model = HeartRiskModel.load(out)
    res = model.predict({"age": 66, "sex": "male", "exercise_angina": 1, "st_depression": 2.5,
                         "systolic_bp": 150, "total_cholesterol": 250})
    assert 0 < res["risk"] < 1 and res["band"]
    uci = next(s for s in res["sources"] if s["name"] == "uci")
    assert uci["used"] and uci["share"] == 1.0
    # Factors are in log-odds and add back up to the fused log-odds.
    total = res["baseline_log_odds"] + sum(f["log_odds"] for f in res["factors"])
    assert total == pytest.approx(res["log_odds"], abs=1e-3)
    text = plain_language_report(res)
    assert "heart-risk index" in text and "UCI" in text

    low = model.predict({"age": 40, "sex": "female", "exercise_angina": 0, "st_depression": 0,
                         "systolic_bp": 115, "total_cholesterol": 180}, explain=False)
    assert low["risk"] < res["risk"]


def test_what_if_and_insufficient_profile(trained_dir):
    out, _ = trained_dir
    model = HeartRiskModel.load(out)
    base = {"age": 66, "sex": "male", "exercise_angina": 1, "st_depression": 2.5}
    wi = model.what_if(base, {"exercise_angina": 0, "st_depression": 0.0})
    assert wi["risk_change"] < 0
    assert model.predict({"sleep_hours": 7})["risk"] is None


def test_checkpoint_loads_with_weights_only(trained_dir):
    out, _ = trained_dir
    enc = LoadedEncoder(torch.load(out / "uci.pt", map_location="cpu", weights_only=True))
    assert enc.coverage({"age": 50}) < enc.coverage({"age": 50, "exang": 1, "oldpeak": 1.0})


def test_loaders_read_zip_and_fix_labels(tmp_path):
    # ECG5000: last column 1 = normal beat -> target 0.
    ecg = pd.DataFrame(np.random.default_rng(0).normal(size=(4, 141)))
    ecg[140] = [1, 0, 1, 0]
    with zipfile.ZipFile(tmp_path / "ecg.csv.zip", "w") as z:
        z.writestr("ecg.csv", ecg.to_csv(header=False, index=False))
    df = SOURCES["ecg5000"].load(tmp_path)
    assert df["target"].tolist() == [0, 1, 0, 1]

    # cardio_train: ';'-separated, age in days, impossible blood pressures blanked.
    cardio = pd.DataFrame({"id": [0, 1], "age": [20000, 15000], "gender": [2, 1], "height": [170, 160],
                           "weight": [80.0, 60.0], "ap_hi": [16020, 120], "ap_lo": [80, 80], "cholesterol": [1, 3],
                           "gluc": [1, 1], "smoke": [0, 1], "alco": [0, 0], "active": [1, 0], "cardio": [1, 0]})
    cardio.to_csv(tmp_path / "cardio_train.csv", sep=";", index=False)
    df = SOURCES["cardio"].load(tmp_path)
    assert df["age_years"].iloc[0] == pytest.approx(54.76, abs=0.01)
    assert np.isnan(df["ap_hi"].iloc[0]) and df["sex"].tolist() == [1, 0]
