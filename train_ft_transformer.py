"""Step 4: train the FT-Transformer on the SF-2 tabular branch.

Validation is computed on real rows only (is_synthetic == 0), so early
stopping never rewards the model for memorising synthetic patients — only
the final held-out preprocessed_test.csv (100% real, never touched by
augmentation or preprocessing fitting) is used to report final performance.
"""
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

MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4


def to_tensors(X: pd.DataFrame, y: pd.Series) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X[FEATURE_COLS].values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.long),
    )


def evaluate(model: FTTransformer, X: pd.DataFrame, y: pd.Series, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X[FEATURE_COLS].values, dtype=torch.float32, device=device)
        preds = model(x).argmax(dim=1).cpu().numpy()
    return {
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "predictions": preds,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_df = pd.read_csv(DATA_DIR / "augmented_train.csv")
    test_df = pd.read_csv(DATA_DIR / "preprocessed_test.csv")

    real_train = train_df[train_df["is_synthetic"] == 0]
    real_fit_idx, val_idx = train_test_split(
        real_train.index, test_size=0.15, random_state=42, stratify=real_train["num"]
    )
    val_df = train_df.loc[val_idx]
    fit_df = train_df.drop(index=val_idx)  # remaining real rows + all synthetic rows

    print(f"Fit set: {len(fit_df)} rows ({(fit_df['is_synthetic'] == 1).sum()} synthetic)")
    print(f"Validation set: {len(val_df)} rows (all real)")
    print(f"Held-out test set: {len(test_df)} rows (all real)\n")

    train_loader = DataLoader(to_tensors(fit_df, fit_df["num"]), batch_size=BATCH_SIZE, shuffle=True)

    model = FTTransformer(n_features=len(FEATURE_COLS), n_classes=5).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(xb)

        val_metrics = evaluate(model, val_df, val_df["num"], device)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}  train_loss={total_loss / len(fit_df):.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['macro_f1']:.4f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"\nBest validation macro F1: {best_val_f1:.4f}")

    test_metrics = evaluate(model, test_df, test_df["num"], device)
    print("\n" + "=" * 60)
    print("Held-out test set performance")
    print("=" * 60)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print("\n" + classification_report(test_df["num"], test_metrics["predictions"], zero_division=0))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CHECKPOINT_DIR / "ft_transformer.pt")
    print(f"Saved model weights: checkpoints/ft_transformer.pt")

    save_embeddings(model, train_df, test_df, device)


def save_embeddings(model: FTTransformer, train_df: pd.DataFrame, test_df: pd.DataFrame, device: torch.device) -> None:
    """Persist E_t (the 256-dim CLS embedding) for real rows, for later fusion with ECG/echo branches."""
    model.eval()
    real_train = train_df[train_df["is_synthetic"] == 0]

    with torch.no_grad():
        train_emb = model.embed(torch.tensor(real_train[FEATURE_COLS].values, dtype=torch.float32, device=device))
        test_emb = model.embed(torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32, device=device))

    emb_cols = [f"embed_{i}" for i in range(train_emb.shape[1])]
    train_out = pd.DataFrame(train_emb.cpu().numpy(), columns=emb_cols)
    train_out["num"] = real_train["num"].values
    test_out = pd.DataFrame(test_emb.cpu().numpy(), columns=emb_cols)
    test_out["num"] = test_df["num"].values

    train_out.to_csv(DATA_DIR / "ft_embeddings_train.csv", index=False)
    test_out.to_csv(DATA_DIR / "ft_embeddings_test.csv", index=False)
    print("Saved: datasets/uci/ft_embeddings_train.csv, ft_embeddings_test.csv")


if __name__ == "__main__":
    main()
