"""Step 6: multimodal fusion (cross-attention) + stacking MLP ensemble.

Important limitation, read before trusting the printed metrics: the tabular
branch (UCI Heart Disease, 920 samples) and the ECG branch (PTB-XL, 21,837
recordings) are two independent public datasets with no shared patient ID —
there is no ground-truth pairing between a UCI record and a PTB-XL ECG. This
script pairs them by random sampling (fixed seed) purely to validate that the
fusion architecture (Section 6.2 of the methodology: E_t as Query, E_v as
Key+Value, then a 5-model stacking ensemble) runs correctly end-to-end. The
accuracy/F1 below are an architecture smoke test, not a clinically meaningful
multimodal result — genuine fusion training needs a dataset where both
modalities exist for the same patient (e.g. MIMIC-IV-ECG).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.models.fusion import FusionEnsemble

ROOT = Path(__file__).resolve().parent
SEED = 42
N_EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4


def load_split(tab_path: Path, ecg_path: Path, rng: np.random.Generator):
    tab = pd.read_csv(tab_path)
    ecg = pd.read_csv(ecg_path)
    e_t = tab[[c for c in tab.columns if c.startswith("embed_")]].values.astype(np.float32)
    y = tab["num"].values.astype(np.int64)
    ecg_feats = ecg[[c for c in ecg.columns if c.startswith("embed_")]].values.astype(np.float32)
    pair_idx = rng.integers(0, len(ecg_feats), size=len(e_t))
    e_v = ecg_feats[pair_idx]
    return torch.from_numpy(e_t), torch.from_numpy(e_v), torch.from_numpy(y)


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    e_t_train, e_v_train, y_train = load_split(
        ROOT / "datasets/uci/ft_embeddings_train.csv",
        ROOT / "datasets/ecg-images/ecg_embeddings_train.csv",
        rng,
    )
    e_t_test, e_v_test, y_test = load_split(
        ROOT / "datasets/uci/ft_embeddings_test.csv",
        ROOT / "datasets/ecg-images/ecg_embeddings_test.csv",
        rng,
    )
    print(f"Train pairs: {len(y_train)}  Test pairs: {len(y_test)}\n")

    model = FusionEnsemble(n_classes=5)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.NLLLoss()

    n = len(y_train)
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            probs, _ = model(e_t_train[idx], e_v_train[idx])
            loss = criterion(torch.log(probs + 1e-8), y_train[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(idx)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                probs, _ = model(e_t_test, e_v_test)
                preds = probs.argmax(dim=1)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            print(f"Epoch {epoch:3d}  train_loss={total_loss / n:.4f}  test_acc={acc:.4f}  test_macro_f1={f1:.4f}")

    model.eval()
    with torch.no_grad():
        probs, _ = model(e_t_test, e_v_test)
        preds = probs.argmax(dim=1)
    print("\n" + classification_report(y_test, preds, zero_division=0))

    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "fusion_ensemble.pt")
    print("Saved: checkpoints/fusion_ensemble.pt")

    print(
        "\nNOTE: E_t/E_v pairs above were built by random sampling (no shared "
        "patient ID between UCI and PTB-XL). This confirms the fusion "
        "architecture trains and evaluates correctly end-to-end; it is NOT a "
        "clinically meaningful multimodal result. See README.md Step 6."
    )


if __name__ == "__main__":
    main()
