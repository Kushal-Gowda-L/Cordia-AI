"""Train FT-Transformer v2 (see src/models/ft_ensemble.py for what changed
from v1 and why). v1 -- checkpoints/ft_transformer.pt, train_ft_transformer.py
-- is not touched.

Two parts:
1. 5-fold stratified cross-validation on all 920 real patients, reported as
   mean +- std. A single 184-patient test set moves by several points
   depending on which patients land in it, so this is the number to quote.
2. The deployable model: trained on the same 80% train split v1 used
   (stratified_split, random_state=42) and scored on the same held-out 20%,
   so the single-split figure is directly comparable to v1's 53.3%.
   Saved to checkpoints/ft_transformer_v2.pt.

Trained on real patients only: the diffusion-augmented set lowered 5-class
accuracy in benchmark_xgboost.py. Same optimiser, learning rate, batch size and
early stopping as train_ft_transformer.py.

Run: python train_ft_transformer_v2.py            (CV + final model)
     python train_ft_transformer_v2.py --skip-cv  (final model only)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ft_ensemble import INPUT_COLS, FTEnsemble, V2Preprocessor, build_member
from src.preprocessing.load_data import fetch_uci_heart_disease
from src.preprocessing.split import stratified_split

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "ft_transformer_v2.pt"
CONFIG = {"d_token": 128, "depth": 3, "heads": 8, "dropout": 0.2}
N_MEMBERS = 5
MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4


def train_member(X: np.ndarray, y: np.ndarray, seed: int):
    """One FT-Transformer; 15% of the given rows are held out for early stopping."""
    fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
    torch.manual_seed(seed)
    as_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(as_tensor(X[fit_idx]), torch.tensor(y[fit_idx], dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    model = build_member(CONFIG)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(as_tensor(X[val_idx])).argmax(1).numpy()
        val_f1 = f1_score(y[val_idx], val_pred, average="macro")
        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, epoch


def train_ensemble(X_raw: pd.DataFrame, y: np.ndarray, seed_base: int):
    prep = V2Preprocessor().fit(X_raw)
    X = prep.transform(X_raw).values
    members = []
    for s in range(N_MEMBERS):
        member, epochs = train_member(X, y, seed=seed_base + s)
        print(f"    member {s + 1}/{N_MEMBERS}: {epochs} epochs", flush=True)
        members.append(member)
    ensemble = FTEnsemble(members)
    ensemble.eval()
    return prep, ensemble


def predict(prep: V2Preprocessor, model: FTEnsemble, X_raw: pd.DataFrame) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(prep.transform(X_raw).values, dtype=torch.float32)).argmax(1).numpy()


def cross_validate(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"5-fold cross-validation, all {len(df)} patients")
    print("=" * 60)
    y = df["num"].values
    scores = []
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y)):
        t0 = time.time()
        print(f"Fold {k + 1}/5", flush=True)
        prep, model = train_ensemble(df.iloc[tr][INPUT_COLS], y[tr], seed_base=k * 100)
        pred = predict(prep, model, df.iloc[te][INPUT_COLS])
        acc, f1 = accuracy_score(y[te], pred), f1_score(y[te], pred, average="macro")
        scores.append((acc, f1))
        print(f"  accuracy={acc:.4f}  macro F1={f1:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    s = np.array(scores)
    print(f"\nCV accuracy: {s[:, 0].mean():.4f} +- {s[:, 0].std():.4f}")
    print(f"CV macro F1: {s[:, 1].mean():.4f} +- {s[:, 1].std():.4f}\n")


def main() -> None:
    df = fetch_uci_heart_disease()
    if "--skip-cv" not in sys.argv:
        cross_validate(df)

    print("=" * 60)
    print("Final model: v1's train split, scored on v1's held-out test set")
    print("=" * 60)
    X, y = df[INPUT_COLS], df["num"]
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    prep, model = train_ensemble(X_train, y_train.values, seed_base=1000)
    pred = predict(prep, model, X_test)
    print(f"\nHeld-out test accuracy: {accuracy_score(y_test, pred):.4f}  (v1: 0.533)")
    print(f"Held-out test macro F1: {f1_score(y_test, pred, average='macro'):.4f}  (v1: 0.347)")
    print("\n" + classification_report(y_test, pred, zero_division=0))

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "config": CONFIG,
            "preprocessor": prep.state(),
            "states": [m.state_dict() for m in model.members],
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved: {CHECKPOINT_PATH.relative_to(CHECKPOINT_PATH.parents[1])}")


if __name__ == "__main__":
    main()
