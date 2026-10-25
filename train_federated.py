"""Step 8: federated learning across 3 simulated hospital nodes, with
per-round differential privacy (Opacus DP-SGD), per the methodology
(Section 6.2 items 13-14; Section 6.4 step 8).

Design notes / honest simplifications, worth reading before citing the
numbers below:

- **Non-IID split.** The 736 real (non-synthetic) UCI training rows are
  partitioned across 3 nodes with a Dirichlet(alpha=0.5) label-skew split,
  so each simulated hospital has a visibly different severity-class mix —
  more realistic than an IID split, and the harder, more interesting case
  for FedAvg to handle.
- **Dropout disabled (dropout=0.0) for the federated model only.** Opacus
  computes per-sample gradients via `torch.vmap`, which cannot trace
  dropout's random mask without extra plumbing this version of Opacus
  doesn't expose simply. This is a federated-training-only constraint —
  the centralized FT-Transformer (`train_ft_transformer.py`) keeps its
  normal dropout=0.2.
- **`nn.MultiheadAttention` is swapped for Opacus's `DPMultiheadAttention`**
  via `ModuleValidator.fix()` — a structurally-equivalent drop-in Opacus
  ships specifically for this (same reasoning as the `rtdl` substitution in
  Step 4: keep the from-scratch architecture, swap only what a specific
  library can't trace).
- **Per-round DP budget, not a composed end-to-end epsilon.** Each node
  calls `make_private_with_epsilon(target_epsilon=2.0, target_delta=1e-5)`
  fresh every round for that round's 5 local epochs — i.e. epsilon=2.0 is
  the privacy cost of *one round's* local training, not a formally composed
  bound across all 20 rounds (that would need an accountant tracked across
  rounds, a natural next refinement). This still faithfully implements the
  methodology's Gaussian-mechanism/DP-SGD requirement per round.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ft_transformer import FTTransformer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "uci"
CHECKPOINT_DIR = ROOT / "checkpoints"
FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]

N_NODES = 3
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
TARGET_EPSILON = 2.0
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0
DIRICHLET_ALPHA = 0.5
SEED = 42


def dirichlet_partition(y: np.ndarray, n_nodes: int, alpha: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Splits sample indices across n_nodes with a Dirichlet label-skew, so
    each node ends up with a different class mix (a realistic multi-hospital
    simulation) instead of an identical IID split."""
    node_indices: list[list[int]] = [[] for _ in range(n_nodes)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha=[alpha] * n_nodes)
        split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        for node_id, chunk in enumerate(np.split(cls_idx, split_points)):
            node_indices[node_id].extend(chunk.tolist())
    return [np.array(idx) for idx in node_indices]


def make_model() -> nn.Module:
    """FT-Transformer with dropout disabled and MultiheadAttention swapped
    for Opacus's DP-compatible replacement (see module docstring)."""
    model = FTTransformer(n_features=len(FEATURE_COLS), n_classes=5, dropout=0.0)
    return ModuleValidator.fix(model)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    return {
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "predictions": preds,
    }


def train_node(global_state: dict, x_node: np.ndarray, y_node: np.ndarray) -> dict:
    model = make_model()
    model.load_state_dict(global_state)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_node, dtype=torch.float32), torch.tensor(y_node, dtype=torch.long)),
        batch_size=min(BATCH_SIZE, len(y_node)),
        shuffle=True,
    )
    privacy_engine = PrivacyEngine()
    model, optimiser, loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimiser,
        data_loader=loader,
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        epochs=LOCAL_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
    return model._module.state_dict()


def fed_avg(local_states: list[dict], node_sizes: list[int]) -> dict:
    total = sum(node_sizes)
    keys = local_states[0].keys()
    return {
        k: sum(local_states[i][k].float() * (node_sizes[i] / total) for i in range(len(local_states)))
        for k in keys
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    train_df = pd.read_csv(DATA_DIR / "augmented_train.csv")
    real_train = train_df[train_df["is_synthetic"] == 0].reset_index(drop=True)
    test_df = pd.read_csv(DATA_DIR / "preprocessed_test.csv")

    X_all = real_train[FEATURE_COLS].values.astype(np.float32)
    y_all = real_train["num"].values.astype(np.int64)
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["num"].values.astype(np.int64)

    node_idx = dirichlet_partition(y_all, N_NODES, DIRICHLET_ALPHA, rng)
    print(f"Simulated hospital nodes ({len(y_all)} real patients total):")
    for i, idx in enumerate(node_idx):
        counts = pd.Series(y_all[idx]).value_counts().sort_index().to_dict()
        print(f"  Node {i+1}: {len(idx)} patients, severity distribution {counts}")
    print()

    global_model = make_model()
    global_state = global_model.state_dict()

    for round_num in range(1, ROUNDS + 1):
        local_states, node_sizes = [], []
        for idx in node_idx:
            state = train_node(global_state, X_all[idx], y_all[idx])
            local_states.append(state)
            node_sizes.append(len(idx))

        global_state = fed_avg(local_states, node_sizes)
        global_model.load_state_dict(global_state)

        if round_num % 2 == 0 or round_num == 1:
            metrics = evaluate(global_model, X_test, y_test)
            print(
                f"Round {round_num:2d}/{ROUNDS}  test_acc={metrics['accuracy']:.4f}  "
                f"test_macro_f1={metrics['macro_f1']:.4f}"
            )

    final_metrics = evaluate(global_model, X_test, y_test)
    print("\n" + "=" * 60)
    print("Federated global model — held-out test set performance")
    print("=" * 60)
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Macro F1: {final_metrics['macro_f1']:.4f}")
    print("\n" + classification_report(y_test, final_metrics["predictions"], zero_division=0))
    print(
        f"Per-round privacy budget: target_epsilon={TARGET_EPSILON}, "
        f"target_delta={TARGET_DELTA} (see module docstring — this is per "
        f"round, not composed across all {ROUNDS} rounds)."
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(global_state, CHECKPOINT_DIR / "ft_transformer_federated.pt")
    print("Saved: checkpoints/ft_transformer_federated.pt")


if __name__ == "__main__":
    main()
