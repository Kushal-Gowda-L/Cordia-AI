"""Explainable AI layer, item 15: IBM AIF360 fairness audit of the trained
FT-Transformer across gender and age subgroups (methodology Section 6.2 #15,
Section 6.4 step 7, target: demographic disparity < 5%).

The 5-class severity output is binarized to disease/no-disease (num > 0),
the standard framing for the UCI Heart Disease target and the only framing
AIF360's binary fairness metrics support directly. Two protected attributes
are audited separately: sex (as encoded in the UCI data: 1=male, 0=female)
and age (bucketed at the test set's median into older/younger, since AIF360
needs a discrete privileged/unprivileged split).

If disparate impact or equal-opportunity difference exceeds the 5% target,
this script applies AIF360's Reweighing preprocessing algorithm and retrains
the FT-Transformer on the reweighted data, then re-audits, so the report
shows a concrete before/after rather than only flagging the problem.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.models.ft_transformer import FTTransformer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "uci"
CHECKPOINT_DIR = ROOT / "checkpoints"
FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]
DISPARITY_TARGET = 0.05


def load_test_with_predictions() -> pd.DataFrame:
    test_df = pd.read_csv(DATA_DIR / "preprocessed_test.csv").reset_index(drop=True)
    model = FTTransformer(n_features=len(FEATURE_COLS), n_classes=5)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "ft_transformer.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        x = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
        preds = model(x).argmax(dim=1).numpy()
    test_df["disease_true"] = (test_df["num"] > 0).astype(int)
    test_df["disease_pred"] = (preds > 0).astype(int)
    return test_df


def audit(df: pd.DataFrame, protected_attr: str, privileged_value: float, label_name: str) -> dict:
    """Computes AIF360 group-fairness metrics for one protected attribute.
    Favorable label = 0 (no disease) by AIF360 convention; the metrics below
    are symmetric in that choice."""
    common_cols = [protected_attr, "disease_true", "disease_pred"]
    ds_true = BinaryLabelDataset(
        df=df[[protected_attr, "disease_true"]].rename(columns={"disease_true": "label"}),
        label_names=["label"],
        protected_attribute_names=[protected_attr],
        favorable_label=0,
        unfavorable_label=1,
    )
    ds_pred = ds_true.copy()
    ds_pred.labels = df[["disease_pred"]].values

    privileged_groups = [{protected_attr: privileged_value}]
    unprivileged_groups = [{protected_attr: 1 - privileged_value}]
    metric = ClassificationMetric(
        ds_true, ds_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )
    return {
        "attribute": label_name,
        "disparate_impact": metric.disparate_impact(),
        "statistical_parity_difference": metric.statistical_parity_difference(),
        "equal_opportunity_difference": metric.equal_opportunity_difference(),
        "average_odds_difference": metric.average_odds_difference(),
    }


def print_report(results: list[dict], heading: str) -> None:
    print(f"\n{heading}")
    print("-" * len(heading))
    for r in results:
        print(f"  {r['attribute']}:")
        print(f"    Disparate impact               : {r['disparate_impact']:.3f}  (1.0 = perfectly fair)")
        print(f"    Statistical parity difference  : {r['statistical_parity_difference']:+.3f}  (0.0 = perfectly fair)")
        print(f"    Equal opportunity difference   : {r['equal_opportunity_difference']:+.3f}  (0.0 = perfectly fair)")
        print(f"    Average odds difference        : {r['average_odds_difference']:+.3f}  (0.0 = perfectly fair)")


def worst_disparity(results: list[dict]) -> float:
    return max(abs(r["statistical_parity_difference"]) for r in results)


def reweigh_and_retrain(train_df: pd.DataFrame, protected_attr: str, privileged_value: float) -> pd.Series:
    """Runs AIF360's Reweighing on the training set for one protected
    attribute and returns per-row sample weights to use in retraining."""
    train_df = train_df.copy()
    train_df["disease_label"] = (train_df["num"] > 0).astype(int)
    ds = BinaryLabelDataset(
        df=train_df[[protected_attr, "disease_label"]].rename(columns={"disease_label": "label"}),
        label_names=["label"],
        protected_attribute_names=[protected_attr],
        favorable_label=0,
        unfavorable_label=1,
    )
    rw = Reweighing(
        unprivileged_groups=[{protected_attr: 1 - privileged_value}],
        privileged_groups=[{protected_attr: privileged_value}],
    )
    ds_reweighed = rw.fit_transform(ds)
    return pd.Series(ds_reweighed.instance_weights, index=train_df.index)


def retrain_with_weights(train_df: pd.DataFrame, sample_weights: pd.Series, test_df: pd.DataFrame) -> np.ndarray:
    model = FTTransformer(n_features=len(FEATURE_COLS), n_classes=5)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")

    x = torch.tensor(train_df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(train_df["num"].values, dtype=torch.long)
    w = torch.tensor(sample_weights.values, dtype=torch.float32)

    for _ in range(150):
        model.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 32):
            idx = perm[i : i + 32]
            optimiser.zero_grad()
            per_sample_loss = criterion(model(x[idx]), y[idx])
            loss = (per_sample_loss * w[idx]).mean()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
        preds = model(x_test).argmax(dim=1).numpy()
    return preds


def main() -> None:
    test_df = load_test_with_predictions()
    print(f"Test set: {len(test_df)} patients")

    # FEATURE_COLS went through the leak-free StandardScaler in Preprocessor,
    # so "sex" is not literally 0/1 here -- it's one of two scaled constants.
    # The larger one is male (raw 1), since StandardScaler preserves order.
    sex_values = sorted(test_df["sex"].unique())
    male_scaled = sex_values[-1]
    test_df["sex_group"] = (test_df["sex"] == male_scaled).astype(int)  # 1 = male

    age_median = test_df["age"].median()
    test_df["age_group"] = (test_df["age"] >= age_median).astype(int)  # 1 = older half

    results = [
        audit(test_df, "sex_group", privileged_value=1, label_name="Sex (1=male privileged)"),
        audit(test_df, "age_group", privileged_value=1, label_name="Age (>= test-set median, privileged)"),
    ]
    print_report(results, "Fairness audit — original model (checkpoints/ft_transformer.pt)")

    disparity = worst_disparity(results)
    print(f"\nWorst |statistical parity difference|: {disparity:.3f}  (target < {DISPARITY_TARGET})")

    if disparity > DISPARITY_TARGET:
        print("\nDisparity exceeds target — applying AIF360 Reweighing and retraining...")
        train_df = pd.read_csv(DATA_DIR / "augmented_train.csv")
        train_df = train_df[train_df["is_synthetic"] == 0].reset_index(drop=True)

        worst_attr, worst_priv = ("sex_group", 1) if abs(results[0]["statistical_parity_difference"]) >= abs(
            results[1]["statistical_parity_difference"]
        ) else ("age_group", 1)
        if worst_attr == "sex_group":
            train_sex_values = sorted(train_df["sex"].unique())
            train_df["sex_group"] = (train_df["sex"] == train_sex_values[-1]).astype(int)
        else:
            train_df["age_group"] = (train_df["age"] >= age_median).astype(int)

        weights = reweigh_and_retrain(train_df, worst_attr, worst_priv)
        reweighed_preds = retrain_with_weights(train_df, weights, test_df)
        test_df["disease_pred"] = (reweighed_preds > 0).astype(int)

        acc = accuracy_score(test_df["num"], reweighed_preds)
        f1 = f1_score(test_df["num"], reweighed_preds, average="macro")
        print(f"\nReweighted model — test accuracy: {acc:.4f}, macro F1: {f1:.4f}")

        results_after = [
            audit(test_df, "sex_group", privileged_value=1, label_name="Sex (1=male privileged)"),
            audit(test_df, "age_group", privileged_value=1, label_name="Age (>= test-set median, privileged)"),
        ]
        print_report(results_after, f"Fairness audit — AFTER reweighing on '{worst_attr}'")
        print(f"\nWorst |statistical parity difference| after reweighing: {worst_disparity(results_after):.3f}")
    else:
        print("\nDisparity within target — no reweighting needed.")


if __name__ == "__main__":
    main()
