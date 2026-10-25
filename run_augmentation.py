"""Step 3: GAN-based data augmentation.

Benchmarks SMOTE (the base-paper's method), CTGAN, and a diffusion-based
generator (TabDDPM-style) at bringing the minority severity classes (2, 3, 4)
up to the size of the second-largest class, then evaluates each with a
tabular Frechet distance, an ML-efficacy test, and nearest-neighbour privacy
distance. Epoch counts here are kept low for a quick end-to-end run on a
CPU-only laptop — raise them for a production-quality comparison.
"""
from src.models.augmentation.ctgan_generator import ctgan_augment
from src.models.augmentation.diffusion_generator import diffusion_augment
from src.models.augmentation.evaluate import (
    frechet_distance,
    ml_efficacy,
    nearest_neighbour_privacy_distance,
)
from src.models.augmentation.smote_baseline import smote_augment
from src.preprocessing.clean import Preprocessor, split_features_target
from src.preprocessing.load_data import fetch_uci_heart_disease
from src.preprocessing.split import stratified_split

QUICK_EPOCHS = 100  # raise to 300+ for a real comparison, once this runs cleanly


def main() -> None:
    df = fetch_uci_heart_disease()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    prep = Preprocessor()
    X_train_p = prep.fit_transform(X_train, y_train)
    X_test_p = prep.transform(X_test)

    counts = y_train.value_counts().sort_index()
    target = int(counts.sort_values(ascending=False).iloc[1])  # 2nd-largest class size
    target_counts = {cls: target for cls in [2, 3, 4]}
    print(f"Train class counts:\n{counts}\n")
    print(f"Augmentation target per minority class: {target}\n")

    generators = {
        "SMOTE": lambda: smote_augment(X_train_p, y_train, target_counts),
        "CTGAN": lambda: ctgan_augment(X_train_p, y_train, target_counts, epochs=QUICK_EPOCHS),
        "Diffusion": lambda: diffusion_augment(X_train_p, y_train, target_counts, epochs=QUICK_EPOCHS),
    }

    results = {}
    for name, generate in generators.items():
        print(f"--- {name} ---")
        X_synth, y_synth = generate()
        if len(X_synth) == 0:
            print("No synthetic samples generated, skipping.\n")
            continue

        fd = frechet_distance(X_train_p, X_synth)
        privacy = nearest_neighbour_privacy_distance(X_train_p, X_synth)
        efficacy = ml_efficacy(X_train_p, y_train, X_synth, y_synth, X_test_p, y_test)

        results[name] = {"frechet_distance": fd, **privacy, **efficacy}
        for k, v in results[name].items():
            print(f"  {k}: {v:.4f}")
        print()

    print("=" * 60)
    print("Summary (lower Frechet distance is better; higher privacy")
    print("distance and higher augmented accuracy/F1 are better)")
    print("=" * 60)
    for name, metrics in results.items():
        print(
            f"{name:10s} FD={metrics['frechet_distance']:.3f}  "
            f"privacy(min)={metrics['min_distance']:.3f}  "
            f"aug_acc={metrics['augmented_accuracy']:.3f}  "
            f"aug_f1={metrics['augmented_macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
