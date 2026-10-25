"""SMOTE baseline — the augmentation method used by the base paper (paper 1),
kept here as the comparison point for CTGAN and the diffusion generator."""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE


def smote_augment(
    X_train: pd.DataFrame, y_train: pd.Series, target_counts: dict[int, int], random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample minority classes up to `target_counts` via SMOTE.
    k_neighbors is capped below the smallest class size actually present.
    """
    smallest_class_size = y_train.value_counts().min()
    k_neighbors = max(1, min(5, smallest_class_size - 1))

    sampling_strategy = {cls: n for cls, n in target_counts.items() if n > (y_train == cls).sum()}
    if not sampling_strategy:
        return X_train.iloc[0:0], y_train.iloc[0:0]

    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # SMOTE returns the full resampled set (real + synthetic); keep only the
    # newly synthesized rows so callers can treat all three generators the
    # same way (real data + a synthetic top-up).
    n_original = len(X_train)
    return X_res.iloc[n_original:].reset_index(drop=True), y_res.iloc[n_original:].reset_index(drop=True)
