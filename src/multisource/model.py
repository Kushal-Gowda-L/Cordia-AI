"""The per-dataset encoders: an FT-Transformer for tabular datasets and a
patch transformer for single ECG beats.

The tabular encoder follows Gorishniy et al. (2021): every column becomes its
own d-dimensional token (numerical value * W_i + b_i, or a categorical
embedding), a [CLS] token is prepended, and a transformer encoder mixes them.
On top of the original design, each column has two extra learned tokens:

- NA: the value is blank in the dataset itself (e.g. UCI's `ca` for most
  Hungarian patients). Blankness can carry information about where a record
  came from, so it gets its own token.
- MASK: the value was not provided. Training randomly hides observed values
  behind this token, so the encoder learns to predict from whatever subset of
  columns it is given. At prediction time every column the patient profile
  cannot supply is fed as MASK, and the explainer uses MASK as the "feature
  absent" baseline.

Value states are 0 = observed, 1 = NA, 2 = MASK.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.multisource.sources import Source

OBSERVED, NA, MASK = 0, 1, 2


class TabularPreprocessor:
    """Standardizes numerical columns and indexes categories (fitted on train rows)."""

    def __init__(self, source: Source, state: dict | None = None):
        self.num = [f for f in source.features if f.kind == "num"]
        self.cat = [f for f in source.features if f.kind == "cat"]
        self.mean = np.zeros(len(self.num))
        self.std = np.ones(len(self.num))
        if state:
            self.mean = np.asarray(state["mean"], dtype=np.float64)
            self.std = np.asarray(state["std"], dtype=np.float64)

    @property
    def cardinalities(self) -> list[int]:
        return [len(f.categories) for f in self.cat]

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        values = df[[f.name for f in self.num]].astype(float)
        self.mean = values.mean().fillna(0).values
        std = values.std(ddof=0).fillna(1).values
        self.std = np.where(std > 0, std, 1.0)
        return self

    def state(self) -> dict:
        return {"mean": [float(v) for v in self.mean], "std": [float(v) for v in self.std]}

    def encode(self, df: pd.DataFrame, masked: set[str] | None = None) -> dict[str, torch.Tensor]:
        """Columns missing from df, or named in `masked`, are encoded as MASK;
        NaN values in df are encoded as NA."""
        masked = masked or set()
        n = len(df)
        x_num = np.zeros((n, len(self.num)), dtype=np.float32)
        num_state = np.zeros((n, len(self.num)), dtype=np.int64)
        for j, f in enumerate(self.num):
            if f.name not in df or f.name in masked:
                num_state[:, j] = MASK
                continue
            v = pd.to_numeric(df[f.name], errors="coerce").to_numpy(dtype=np.float64)
            na = np.isnan(v)
            x_num[:, j] = np.where(na, 0.0, (v - self.mean[j]) / self.std[j])
            num_state[:, j] = np.where(na, NA, OBSERVED)
        x_cat = np.zeros((n, len(self.cat)), dtype=np.int64)
        for j, f in enumerate(self.cat):
            if f.name not in df or f.name in masked:
                x_cat[:, j] = MASK
                continue
            index = {c: i + 3 for i, c in enumerate(f.categories)}
            v = pd.to_numeric(df[f.name], errors="coerce")
            x_cat[:, j] = [index.get(int(x), NA) if not pd.isna(x) else NA for x in v]
        return {"x_num": torch.from_numpy(x_num), "num_state": torch.from_numpy(num_state),
                "x_cat": torch.from_numpy(x_cat)}


def random_mask(batch: dict[str, torch.Tensor], max_rate: float, generator=None) -> dict[str, torch.Tensor]:
    """Hide a random share (0..max_rate, drawn per row) of each row's observed values behind MASK."""
    out = dict(batch)
    n = batch["x_num"].shape[0] if batch["x_num"].numel() else batch["x_cat"].shape[0]
    rate = torch.rand(n, 1, generator=generator) * max_rate
    if batch["num_state"].numel():
        hide = (torch.rand(batch["num_state"].shape, generator=generator) < rate) & (batch["num_state"] == OBSERVED)
        out["num_state"] = torch.where(hide, torch.full_like(batch["num_state"], MASK), batch["num_state"])
    if batch["x_cat"].numel():
        hide = (torch.rand(batch["x_cat"].shape, generator=generator) < rate) & (batch["x_cat"] > MASK)
        out["x_cat"] = torch.where(hide, torch.full_like(batch["x_cat"], MASK), batch["x_cat"])
    return out


class FTEncoder(nn.Module):
    """FT-Transformer with NA and MASK tokens; forward() returns one logit per row."""

    def __init__(self, n_num: int, cardinalities: list[int], d_token: int = 64, depth: int = 3,
                 heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_num = n_num
        bound = 1 / max(n_num, 1) ** 0.5
        self.num_weight = nn.Parameter(torch.empty(n_num, d_token).uniform_(-bound, bound))
        self.num_bias = nn.Parameter(torch.empty(n_num, d_token).uniform_(-bound, bound))
        self.num_special = nn.Parameter(torch.randn(n_num, 2, d_token) * 0.02)  # NA, MASK
        # One embedding table for all categorical columns: per column, NA, MASK, then its categories.
        sizes = [c + 3 for c in cardinalities]
        self.register_buffer("cat_offsets", torch.tensor([0] + sizes[:-1], dtype=torch.long).cumsum(0))
        self.cat_embed = nn.Embedding(max(sum(sizes), 1), d_token)
        nn.init.normal_(self.cat_embed.weight, std=0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        layer = nn.TransformerEncoderLayer(d_token, heads, d_token * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def tokens(self, x_num, num_state, x_cat) -> torch.Tensor:
        parts = []
        if self.n_num:
            value_tok = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
            special = self.num_special.unsqueeze(0).expand(x_num.shape[0], -1, -1, -1)
            na_tok, mask_tok = special[:, :, 0], special[:, :, 1]
            state = num_state.unsqueeze(-1)
            parts.append(torch.where(state == NA, na_tok, torch.where(state == MASK, mask_tok, value_tok)))
        if x_cat.shape[1]:
            parts.append(self.cat_embed(x_cat + self.cat_offsets))
        return torch.cat(parts, dim=1)

    def embed(self, x_num, num_state, x_cat) -> torch.Tensor:
        tok = self.tokens(x_num, num_state, x_cat)
        seq = torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], dim=1)
        return self.norm(self.encoder(seq)[:, 0])

    def forward(self, x_num, num_state, x_cat) -> torch.Tensor:
        return self.head(self.embed(x_num, num_state, x_cat)).squeeze(-1)


class BeatPreprocessor:
    """Per-beat z-normalisation is already done in ECG5000; this keeps the
    training mean beat, used as the 'absent' baseline when explaining."""

    def __init__(self, source: Source, state: dict | None = None):
        self.cols = source.feature_names
        self.mean_beat = np.asarray(state["mean_beat"], dtype=np.float32) if state else np.zeros(len(self.cols), np.float32)

    def fit(self, df: pd.DataFrame) -> "BeatPreprocessor":
        self.mean_beat = df[self.cols].astype(float).mean().values.astype(np.float32)
        return self

    def state(self) -> dict:
        return {"mean_beat": [float(v) for v in self.mean_beat]}

    def encode(self, df: pd.DataFrame, masked=None) -> dict[str, torch.Tensor]:
        return {"beat": torch.tensor(df[self.cols].astype(float).values, dtype=torch.float32)}


class BeatTransformer(nn.Module):
    """ViT-style encoder for a 1-D heartbeat: non-overlapping patches -> tokens."""

    def __init__(self, length: int = 140, patch: int = 10, d_token: int = 64, depth: int = 3,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert length % patch == 0
        self.patch = patch
        self.n_patches = length // patch
        self.proj = nn.Linear(patch, d_token)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, d_token) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        layer = nn.TransformerEncoderLayer(d_token, heads, d_token * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, beat: torch.Tensor) -> torch.Tensor:
        tok = self.proj(beat.view(beat.shape[0], self.n_patches, self.patch))
        seq = torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], dim=1) + self.pos
        return self.head(self.norm(self.encoder(seq)[:, 0])).squeeze(-1)


def build(source: Source, config: dict, prep_state: dict | None = None):
    """(preprocessor, model) for a source; config holds d_token/depth/heads/dropout."""
    arch = {k: config[k] for k in ("d_token", "depth", "heads", "dropout")}
    if source.kind == "beat":
        return BeatPreprocessor(source, prep_state), BeatTransformer(length=len(source.features), **arch)
    prep = TabularPreprocessor(source, prep_state)
    return prep, FTEncoder(len(prep.num), prep.cardinalities, **arch)
