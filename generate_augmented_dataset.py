"""Step 3 (final): generate the augmented training set using the selected
generator.

Diffusion was benchmarked against SMOTE and CTGAN on Colab (GPU, 800 epochs
each) and won on all three axes — see
datasets/uci/augmentation_comparison.csv for the numbers. This script
reproduces that generation locally (the diffusion model is a small MLP, so
it trains quickly even on CPU, unlike CTGAN) and writes the final augmented
training set that Step 4 (FT-Transformer) will consume.
"""
from pathlib import Path

import pandas as pd

from src.models.augmentation.diffusion_generator import diffusion_augment
from src.preprocessing.clean import Preprocessor, split_features_target
from src.preprocessing.load_data import fetch_uci_heart_disease
from src.preprocessing.split import stratified_split

EPOCHS = 800  # matches the Colab benchmark run
OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "uci"


def main() -> None:
    df = fetch_uci_heart_disease()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    prep = Preprocessor()
    X_train_p = prep.fit_transform(X_train, y_train)
    X_test_p = prep.transform(X_test)

    counts = y_train.value_counts().sort_index()
    target = int(counts.sort_values(ascending=False).iloc[1])
    target_counts = {cls: target for cls in [2, 3, 4]}

    print(f"Train class counts before augmentation:\n{counts}\n")
    print(f"Generating synthetic samples up to {target} per minority class (diffusion, {EPOCHS} epochs)...")

    X_synth, y_synth = diffusion_augment(X_train_p, y_train, target_counts, epochs=EPOCHS)
    print(f"Generated {len(X_synth)} synthetic rows.\n")

    X_augmented = pd.concat([X_train_p, X_synth], ignore_index=True)
    y_augmented = pd.concat([y_train, y_synth], ignore_index=True)

    print(f"Augmented train class counts:\n{y_augmented.value_counts().sort_index()}\n")

    is_synthetic = [0] * len(X_train_p) + [1] * len(X_synth)
    train_out = X_augmented.assign(num=y_augmented, is_synthetic=is_synthetic)
    # X_test_p and y_test still share the original (non-reset) index from the
    # train/test split, so this assign aligns correctly by index.
    test_out = X_test_p.assign(num=y_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(OUTPUT_DIR / "augmented_train.csv", index=False)
    test_out.to_csv(OUTPUT_DIR / "preprocessed_test.csv", index=False)

    print(f"Saved: datasets/uci/augmented_train.csv {train_out.shape}")
    print(f"Saved: datasets/uci/preprocessed_test.csv {test_out.shape}")


if __name__ == "__main__":
    main()
