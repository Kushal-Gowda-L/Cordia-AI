"""Explanations for one patient: Shapley values of each encoder's inputs.

The value of a set S of inputs is the encoder's logit when only S is shown
and every other input is fed as MASK ("not provided") -- the same state the
encoder was trained to handle by random masking. So the Shapley values of a
patient's inputs add up exactly to logit(patient) - logit(nothing provided),
i.e. they split how far this patient's log-odds sit from the encoder's
prior. No background dataset or kernel approximation is needed.

For an ECG beat the inputs are 14 consecutive 10-sample segments, and an
absent segment is replaced by the training-set mean beat.

Values are estimated by permutation sampling (antithetic pairs); with the
few inputs per encoder here, 64 permutations give stable rankings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.multisource.model import MASK


def shapley_sampling(value_fn, n_players: int, n_permutations: int = 64, seed: int = 0) -> np.ndarray:
    """value_fn(present: bool array (k, n_players)) -> (k,) values. Returns (n_players,) Shapley values."""
    if n_players == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(max(1, n_permutations // 2)):
        p = rng.permutation(n_players)
        perms += [p, p[::-1]]
    # For each permutation, coalitions of size 0..n along that order.
    present = np.zeros((len(perms), n_players + 1, n_players), dtype=bool)
    for i, p in enumerate(perms):
        for k in range(1, n_players + 1):
            present[i, k] = present[i, k - 1]
            present[i, k, p[k - 1]] = True
    values = value_fn(present.reshape(-1, n_players)).reshape(len(perms), n_players + 1)
    phi = np.zeros(n_players)
    for i, p in enumerate(perms):
        phi[p] += np.diff(values[i])
    return phi / len(perms)


def tabular_shapley(prep, model, row: dict, n_permutations: int = 64) -> dict[str, float]:
    """Shapley values (log-odds units, before calibration) of the columns present in `row`."""
    players = list(row)
    base = prep.encode(pd.DataFrame([row]))

    def value_fn(present: np.ndarray) -> np.ndarray:
        k = len(present)
        batch = {name: t.repeat(k, 1) for name, t in base.items()}
        num_names = [f.name for f in prep.num]
        cat_names = [f.name for f in prep.cat]
        for j, name in enumerate(players):
            hide = torch.from_numpy(~present[:, j])
            if name in num_names:
                col = num_names.index(name)
                batch["num_state"][:, col] = torch.where(hide, MASK, batch["num_state"][:, col])
            else:
                col = cat_names.index(name)
                batch["x_cat"][:, col] = torch.where(hide, MASK, batch["x_cat"][:, col])
        with torch.no_grad():
            return model(**batch).numpy()

    phi = shapley_sampling(value_fn, len(players), n_permutations)
    return {name: float(v) for name, v in zip(players, phi)}


def beat_shapley(prep, model, beat: list[float], n_permutations: int = 64) -> dict[str, float]:
    """Shapley values of each 10-sample segment of a heartbeat (keys 'samples 0-9', ...)."""
    beat = np.asarray(beat, dtype=np.float32)
    patch = model.patch
    n = len(beat) // patch

    def value_fn(present: np.ndarray) -> np.ndarray:
        mask = np.repeat(present, patch, axis=1)
        x = np.where(mask, beat[None, :], prep.mean_beat[None, :]).astype(np.float32)
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()

    phi = shapley_sampling(value_fn, n, n_permutations)
    return {f"samples {i * patch}-{(i + 1) * patch - 1}": float(v) for i, v in enumerate(phi)}
