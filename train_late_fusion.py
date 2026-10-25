"""Fit the late-fusion combiner on a real paired cohort, and measure what fusion adds.

Input: the CSV from build_mimic_paired_cohort.py (one row per admission,
every branch's output from the same patient). Output:
checkpoints/late_fusion.joblib, which the app then uses to give one
combined prediction.

Reported honestly:
- 5-fold cross-validation grouped by patient (no patient in two folds).
- An ablation: each branch alone vs. all together, so the fusion gain is a
  measured number rather than a claim.

    python train_late_fusion.py --cohort datasets/paired/mimic_paired_cohort.csv --target heart_disease
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.late_fusion import BRANCH_FEATURES, LateFusionCombiner

ROOT = Path(__file__).resolve().parent


def score(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    out = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
    out["auroc"] = roc_auc_score(y, probs[:, 1]) if probs.shape[1] == 2 else roc_auc_score(y, probs, multi_class="ovr")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, default=ROOT / "datasets" / "paired" / "mimic_paired_cohort.csv")
    ap.add_argument("--target", default="heart_disease")
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    y = df[args.target].astype(int).values
    groups = df["subject_id"].values
    present = [b for b, cols in BRANCH_FEATURES.items() if any(c in df and df[c].notna().any() for c in cols)]
    labels = ["No", "Yes"] if set(np.unique(y)) <= {0, 1} else [str(v) for v in np.unique(y)]
    print(f"{len(df)} admissions, {df['subject_id'].nunique()} patients, target '{args.target}' "
          f"positive rate {y.mean():.3f}\nBranches present: {present}\n")

    rows = []
    for name, branches in [*[(b, ["clinical", b] if b != "clinical" else ["clinical"]) for b in present],
                           ("ALL (fusion)", present)]:
        probs = LateFusionCombiner(branches, labels).cross_validate(df, y, groups=groups)
        rows.append({"inputs": name, **{k: round(v, 3) for k, v in score(y, probs).items()}})
    table = pd.DataFrame(rows)
    print("5-fold patient-grouped cross-validation (each single branch also gets age + sex):")
    print(table.to_string(index=False))

    combiner = LateFusionCombiner(present, labels).fit(
        df, y, cohort=str(args.cohort.name), target=args.target,
        n_subjects=int(df["subject_id"].nunique()), cv_results=rows)
    out = ROOT / "checkpoints" / "late_fusion.joblib"
    combiner.save(out)
    print(f"\nSaved {out} (fitted on all {len(df)} admissions).")


if __name__ == "__main__":
    main()
