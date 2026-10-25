"""CTGAN-based minority-class augmentation.

Trains a separate CTGAN per minority class on that class's real rows only
(already SF-2-selected and standardized), then samples the deficit needed to
reach a target count. Features are treated as continuous since they arrive
already scaled by `Preprocessor`; the CTGAN library still uses its
mode-specific normalization internally, so this is a reasonable
simplification rather than declaring discrete columns on z-scored data.
"""
from __future__ import annotations

import pandas as pd
from ctgan import CTGAN


def ctgan_augment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_counts: dict[int, int],
    epochs: int = 300,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    synthetic_X_parts, synthetic_y_parts = [], []

    for cls, target in target_counts.items():
        class_rows = X_train[y_train == cls]
        deficit = target - len(class_rows)
        if deficit <= 0:
            continue

        # pac=1 and a small batch_size avoid CTGAN's default assumptions
        # (batch_size=500, pac=10), which break down on classes this small
        # (as few as ~22 rows for the rarest severity level).
        model = CTGAN(epochs=epochs, batch_size=10, pac=1, verbose=False)
        model.fit(class_rows)
        synthetic_rows = model.sample(deficit)

        synthetic_X_parts.append(synthetic_rows)
        synthetic_y_parts.append(pd.Series([cls] * deficit))

    if not synthetic_X_parts:
        return X_train.iloc[0:0], y_train.iloc[0:0]

    X_synth = pd.concat(synthetic_X_parts, ignore_index=True)
    y_synth = pd.concat(synthetic_y_parts, ignore_index=True)
    return X_synth, y_synth
