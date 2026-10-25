"""FT-Transformer v2: the same FTTransformer architecture, used as a 5-model
ensemble on 13 inputs instead of 10.

Changes from v1 (checkpoints/ft_transformer.pt), each measured with 5-fold
cross-validation on all 920 UCI patients before being adopted:
- slope, ca and thal are used again, with an "is missing" flag for each. They
  are blank for most non-Cleveland patients, which is why SF-2 dropped them,
  but where present they carry real signal.
- token size 128 instead of 256 (920 patients is small for a 256-wide model).
- 5 copies trained with different seeds; predictions are their averaged
  probabilities.
- cholesterol 0 and resting blood pressure 0 are treated as missing, not as
  real measurements (172 UCI records store a missing cholesterol as 0), with an
  "is missing" flag for cholesterol, since in UCI a missing cholesterol is
  itself a strong sign of disease.

`V2Preprocessor` and `FTEnsemble` mirror v1's `Preprocessor.transform()` and
`FTTransformer.forward()`, so code that does `model(prep.transform(df))` and
takes a softmax works unchanged with either version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.ft_transformer import FTTransformer

BASE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]
EXTRA_COLS = ["slope", "ca", "thal"]
INPUT_COLS = BASE_COLS + EXTRA_COLS
FLAG_COLS = EXTRA_COLS + ["chol"]
MODEL_COLS = INPUT_COLS + [f"{c}_missing" for c in FLAG_COLS]
ZERO_MEANS_MISSING = ["chol", "trestbps"]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Raw inputs -> numeric frame with all 13 input columns; blanks become NaN."""
    out = pd.DataFrame(index=df.index)
    for c in INPUT_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c in ZERO_MEANS_MISSING:
        out.loc[out[c] == 0, c] = np.nan
    return out


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FLAG_COLS:
        df[f"{c}_missing"] = df[c].isna().astype(float)
    return df


class V2Preprocessor:
    """Median imputation + standard scaling, fitted on training rows only."""

    def __init__(self, medians: dict | None = None, mean: list | None = None, scale: list | None = None):
        self.medians_ = pd.Series(medians) if medians is not None else None
        self.mean_ = np.asarray(mean, dtype=np.float64) if mean is not None else None
        self.scale_ = np.asarray(scale, dtype=np.float64) if scale is not None else None

    def fit(self, X_raw: pd.DataFrame) -> "V2Preprocessor":
        X = add_missing_flags(clean_raw(X_raw))[MODEL_COLS]
        self.medians_ = X.median()
        filled = X.fillna(self.medians_)
        self.mean_ = filled.mean().values
        self.scale_ = filled.std(ddof=0).replace(0, 1).values
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = add_missing_flags(clean_raw(X_raw))[MODEL_COLS].fillna(self.medians_)
        return pd.DataFrame((X.values - self.mean_) / self.scale_, columns=MODEL_COLS, index=X_raw.index)

    def state(self) -> dict:
        return {
            "medians": {k: float(v) for k, v in self.medians_.items()},
            "mean": [float(v) for v in self.mean_],
            "scale": [float(v) for v in self.scale_],
        }


class FTEnsemble(nn.Module):
    """Averages the softmax outputs of several FTTransformers.

    forward() returns log(mean probability), so softmax(forward(x)) is exactly
    the ensemble's averaged probabilities, as with a single model's logits."""

    def __init__(self, members: list[FTTransformer]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([torch.softmax(m(x), dim=1) for m in self.members]).mean(dim=0)
        return torch.log(probs.clamp_min(1e-12))


def build_member(config: dict) -> FTTransformer:
    return FTTransformer(
        n_features=len(MODEL_COLS), n_classes=5,
        d_token=config["d_token"], depth=config["depth"], heads=config["heads"], dropout=config["dropout"],
    )


def load_v2(path) -> tuple[V2Preprocessor, FTEnsemble]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    prep = V2Preprocessor(**ckpt["preprocessor"])
    members = []
    for state in ckpt["states"]:
        m = build_member(ckpt["config"])
        m.load_state_dict(state)
        members.append(m)
    model = FTEnsemble(members)
    model.eval()
    return prep, model
