"""Phase 1 entry point: load, explore, split, and preprocess the UCI Heart
Disease dataset without leaking the test set into imputation/scaling/feature
selection."""
from src.preprocessing.clean import Preprocessor, split_features_target
from src.preprocessing.load_data import fetch_uci_heart_disease
from src.preprocessing.split import stratified_split


def main() -> None:
    df = fetch_uci_heart_disease()

    print("=" * 60)
    print("UCI Heart Disease — raw dataset")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nTarget ('num') distribution (0 = no disease, 1-4 = severity):")
    print(df["num"].value_counts().sort_index())

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    prep = Preprocessor()
    X_train_p = prep.fit_transform(X_train, y_train)
    X_test_p = prep.transform(X_test)

    print("\n" + "=" * 60)
    print("After Step 2 preprocessing (fit on train only)")
    print("=" * 60)
    print(f"SF-2 feature subset ({len(prep.sf2_features_)} features): {prep.sf2_features_}")
    print(f"Train shape: {X_train_p.shape}  |  Test shape: {X_test_p.shape}")
    print(f"\nTrain class balance:\n{y_train.value_counts(normalize=True).sort_index().round(3)}")
    print(f"\nTest class balance:\n{y_test.value_counts(normalize=True).sort_index().round(3)}")


if __name__ == "__main__":
    main()
