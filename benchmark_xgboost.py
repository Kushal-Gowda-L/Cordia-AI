"""Benchmark: how close can this dataset get to the reference paper's 97.57%
accuracy (El-Sofany, 2024, IEEE Access -- XGBoost + ANOVA/chi-square/MI
feature selection), and is that number even reachable on the *same task*
CordiaAI actually predicts (5-class severity, not binary disease/no-disease)?

Runs XGBoost -- the same model family as the reference paper, and a much
stronger fit than a from-scratch Transformer on <1000 rows -- on both
framings, using the project's own leak-free SF-2 features:

1. Binary (disease vs. no-disease, num>0): the task most 90%+ papers on
   this dataset, including El-Sofany's, actually solve.
2. 5-class severity (num 0-4): the harder task CordiaAI's FT-Transformer
   and ECG branch actually predict, needed for consistent severity-based
   risk categories (HIGH/MEDIUM/LOW) downstream.

Each is evaluated on real-only training data and on the diffusion-augmented
training set, against the same untouched held-out test set used everywhere
else in this repo.
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "uci"
FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]


def fit_eval(X_train, y_train, X_test, y_test, n_classes: int) -> dict:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    result = {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "predictions": preds,
    }
    if n_classes == 2:
        result["auc"] = roc_auc_score(y_test, proba[:, 1])
    else:
        result["auc_ovr"] = roc_auc_score(y_test, proba, multi_class="ovr")
    return result


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "augmented_train.csv")
    real_train = train_df[train_df["is_synthetic"] == 0]
    test_df = pd.read_csv(DATA_DIR / "preprocessed_test.csv")

    print("=" * 70)
    print("5-CLASS SEVERITY (num 0-4) -- the task CordiaAI's branches predict")
    print("=" * 70)
    for label, X_tr, y_tr in [
        ("real-only", real_train[FEATURE_COLS], real_train["num"]),
        ("diffusion-augmented", train_df[FEATURE_COLS], train_df["num"]),
    ]:
        r = fit_eval(X_tr, y_tr, test_df[FEATURE_COLS], test_df["num"], n_classes=5)
        print(f"\nXGBoost ({label}, n={len(X_tr)}): accuracy={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}  AUC(ovr)={r['auc_ovr']:.4f}")
    print("\n" + classification_report(test_df["num"], r["predictions"], zero_division=0))

    print("\n" + "=" * 70)
    print("BINARY (disease vs. no-disease, num>0) -- the task most 90%+ papers")
    print("on this dataset, incl. the reference paper, actually solve")
    print("=" * 70)
    for label, X_tr, y_tr in [
        ("real-only", real_train[FEATURE_COLS], (real_train["num"] > 0).astype(int)),
        ("diffusion-augmented", train_df[FEATURE_COLS], (train_df["num"] > 0).astype(int)),
    ]:
        y_test_bin = (test_df["num"] > 0).astype(int)
        r = fit_eval(X_tr, y_tr, test_df[FEATURE_COLS], y_test_bin, n_classes=2)
        print(f"\nXGBoost ({label}, n={len(X_tr)}): accuracy={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}  AUC={r['auc']:.4f}")
    print("\n" + classification_report(y_test_bin, r["predictions"], zero_division=0))

    print("\nReference: El-Sofany (2024) reports 97.57% accuracy (XGBoost, CHDD + private data, 503 samples).")


if __name__ == "__main__":
    main()
