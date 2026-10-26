"""Binary-classification metrics reported for every encoder and baseline."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score, roc_curve)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (chosen on validation data)."""
    fpr, tpr, thr = roc_curve(y, p)
    best = int(np.argmax(tpr - fpr))
    return float(min(max(thr[best], 1e-6), 1 - 1e-6))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int = 200, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float, ci: bool = True) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    if ci:
        out["auroc_ci95"] = bootstrap_auc_ci(y, p)
    return out
