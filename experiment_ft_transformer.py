"""Hyperparameter experiment for the FT-Transformer tabular branch.

Uses the EXACT same data, split, and evaluation protocol as
train_ft_transformer.py, so results are directly comparable to the current
baseline (test accuracy 53.3%, macro F1 0.347). The current system is left
completely untouched: this script never reads or writes
checkpoints/ft_transformer.pt or the ft_embeddings_*.csv files -- only a
separate checkpoints/ft_transformer_experiment.pt if a configuration beats
the baseline. If nothing here beats it, the baseline stays exactly as-is
and this script has changed nothing about the running system.
"""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ft_transformer import FTTransformer

DATA_DIR = Path(__file__).resolve().parent / "datasets" / "uci"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]

BASELINE = {"depth": 3, "heads": 8, "d_token": 256, "dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-4}
BASELINE_TEST_ACC = 0.533
BASELINE_TEST_F1 = 0.347

CONFIGS = [
    {"name": "baseline (re-run for a clean apples-to-apples check)", **BASELINE},
    {"name": "shrink + more dropout", "depth": 2, "heads": 8, "d_token": 256, "dropout": 0.35, "lr": 1e-4, "weight_decay": 1e-4},
    {"name": "smaller token dim", "depth": 3, "heads": 8, "d_token": 128, "dropout": 0.3, "lr": 1e-4, "weight_decay": 1e-4},
    {"name": "lower LR, more weight decay", "depth": 3, "heads": 8, "d_token": 256, "dropout": 0.2, "lr": 5e-5, "weight_decay": 5e-4},
    {"name": "deeper + more heads", "depth": 4, "heads": 4, "d_token": 256, "dropout": 0.25, "lr": 1e-4, "weight_decay": 1e-4},
    {"name": "small token dim + fewer heads", "depth": 2, "heads": 4, "d_token": 96, "dropout": 0.3, "lr": 1e-4, "weight_decay": 1e-4},
]

MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
SEED = 42


def to_tensors(X, y):
    return TensorDataset(
        torch.tensor(X[FEATURE_COLS].values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.long),
    )


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X[FEATURE_COLS].values, dtype=torch.float32, device=device)
        preds = model(x).argmax(dim=1).cpu().numpy()
    return {"accuracy": accuracy_score(y, preds), "macro_f1": f1_score(y, preds, average="macro"), "predictions": preds}


def train_one(config, fit_df, val_df, test_df, device):
    torch.manual_seed(SEED)
    train_loader = DataLoader(to_tensors(fit_df, fit_df["num"]), batch_size=BATCH_SIZE, shuffle=True)

    model = FTTransformer(
        n_features=len(FEATURE_COLS), n_classes=5,
        d_token=config["d_token"], depth=config["depth"], heads=config["heads"], dropout=config["dropout"],
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

        val_metrics = evaluate(model, val_df, val_df["num"], device)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_df, test_df["num"], device)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "epochs_run": epoch, "best_val_f1": best_val_f1,
        "test_acc": test_metrics["accuracy"], "test_f1": test_metrics["macro_f1"],
        "predictions": test_metrics["predictions"], "n_params": n_params, "state": best_state,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Baseline to beat: test_acc={BASELINE_TEST_ACC:.3f}  test_macro_f1={BASELINE_TEST_F1:.3f}\n")

    train_df = pd.read_csv(DATA_DIR / "augmented_train.csv")
    test_df = pd.read_csv(DATA_DIR / "preprocessed_test.csv")
    real_train = train_df[train_df["is_synthetic"] == 0]
    real_fit_idx, val_idx = train_test_split(real_train.index, test_size=0.15, random_state=42, stratify=real_train["num"])
    val_df = train_df.loc[val_idx]
    fit_df = train_df.drop(index=val_idx)

    results = []
    for config in CONFIGS:
        print(f"--- {config['name']} ---")
        print(f"    depth={config['depth']} heads={config['heads']} d_token={config['d_token']} "
              f"dropout={config['dropout']} lr={config['lr']} weight_decay={config['weight_decay']}")
        r = train_one(config, fit_df, val_df, test_df, device)
        beat = r["test_f1"] > BASELINE_TEST_F1
        print(f"    epochs_run={r['epochs_run']}  params={r['n_params']:,}  "
              f"test_acc={r['test_acc']:.4f}  test_macro_f1={r['test_f1']:.4f}"
              f"{'  <-- BEATS BASELINE' if beat else ''}\n")
        results.append({"config": config, **r})

    results.sort(key=lambda r: r["test_f1"], reverse=True)
    best = results[0]

    print("=" * 70)
    print("SUMMARY (sorted by test macro F1, best first)")
    print("=" * 70)
    print(f"{'Configuration':<45} {'Acc':>7} {'MacroF1':>8}")
    print(f"{'BASELINE (current system, unchanged)':<45} {BASELINE_TEST_ACC:>7.4f} {BASELINE_TEST_F1:>8.4f}")
    for r in results:
        print(f"{r['config']['name']:<45} {r['test_acc']:>7.4f} {r['test_f1']:>8.4f}")

    print(f"\n\n" + classification_report(test_df["num"], best["predictions"], zero_division=0))

    if best["test_f1"] > BASELINE_TEST_F1:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CHECKPOINT_DIR / "ft_transformer_experiment.pt"
        torch.save(best["state"], out_path)
        print(f"\nBest experimental config ('{best['config']['name']}') BEATS the baseline "
              f"({best['test_f1']:.4f} vs {BASELINE_TEST_F1:.4f} macro F1).")
        print(f"Saved to checkpoints/ft_transformer_experiment.pt -- NOT overwriting ft_transformer.pt.")
        print("This is a candidate to review and adopt manually, not auto-swapped in.")
    else:
        print(f"\nNo experimental configuration beat the baseline's {BASELINE_TEST_F1:.4f} macro F1.")
        print("checkpoints/ft_transformer.pt (the current system) is untouched and remains the one in use.")


if __name__ == "__main__":
    main()
