"""Stratified train/test split, kept separate from preprocessing so the test
set is never touched by anything fit on the training data (scalers, feature
selectors, generators)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
