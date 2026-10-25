"""Benchmarking utilities for comparing synthetic-data generators: a tabular
Frechet-style distance, an ML-efficacy test, and nearest-neighbour privacy
distance. All operate on already-scaled numeric feature matrices.

Note: true Frechet Inception Distance uses a pretrained Inception network's
embedding space, which has no tabular equivalent. Following common practice
in tabular-GAN evaluation, we compute the Frechet distance directly in the
standardized feature space instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors


def frechet_distance(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    mu1, mu2 = real.mean().values, synthetic.mean().values
    sigma1, sigma2 = np.cov(real.values, rowvar=False), np.cov(synthetic.values, rowvar=False)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def ml_efficacy(
    X_train_real: pd.DataFrame,
    y_train_real: pd.Series,
    X_synthetic: pd.DataFrame,
    y_synthetic: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> dict[str, float]:
    """Compare a classifier trained on real-only data vs real+synthetic data,
    both evaluated on the same held-out real test set."""
    clf_real = RandomForestClassifier(random_state=random_state).fit(X_train_real, y_train_real)
    real_only_acc = accuracy_score(y_test, clf_real.predict(X_test))
    real_only_f1 = f1_score(y_test, clf_real.predict(X_test), average="macro")

    X_combined = pd.concat([X_train_real, X_synthetic], ignore_index=True)
    y_combined = pd.concat([y_train_real, y_synthetic], ignore_index=True)
    clf_aug = RandomForestClassifier(random_state=random_state).fit(X_combined, y_combined)
    augmented_acc = accuracy_score(y_test, clf_aug.predict(X_test))
    augmented_f1 = f1_score(y_test, clf_aug.predict(X_test), average="macro")

    return {
        "real_only_accuracy": real_only_acc,
        "real_only_macro_f1": real_only_f1,
        "augmented_accuracy": augmented_acc,
        "augmented_macro_f1": augmented_f1,
    }


def nearest_neighbour_privacy_distance(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """For each synthetic row, distance to its nearest real neighbour.
    Very small distances suggest the generator is memorising real rows
    rather than learning the underlying distribution."""
    nn = NearestNeighbors(n_neighbors=1).fit(real.values)
    distances, _ = nn.kneighbors(synthetic.values)
    return {
        "min_distance": float(distances.min()),
        "mean_distance": float(distances.mean()),
        "5th_percentile_distance": float(np.percentile(distances, 5)),
    }
