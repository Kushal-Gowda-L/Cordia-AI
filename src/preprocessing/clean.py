"""Step 2: Preprocessing — imputation, scaling, and SF-2 feature selection.

Fit on training data only; `transform()` reuses those fitted statistics on
any other split, so the test set never leaks into imputation medians, the
scaler, or the feature-selection ranking.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TARGET_COL = "num"
N_SF2_FEATURES = 10


def _select_sf2(X: pd.DataFrame, y: pd.Series, k: int = N_SF2_FEATURES) -> list[str]:
    """SF-2 feature subset via consensus ranking across ANOVA F-test,
    Chi-square, and Mutual Information (lower combined rank = kept)."""
    X_nonneg = MinMaxScaler().fit_transform(X)  # chi2 requires non-negative input

    anova_scores = f_classif(X, y)[0]
    chi2_scores = chi2(X_nonneg, y)[0]
    mi_scores = mutual_info_classif(X, y, random_state=42)

    ranks = pd.DataFrame(
        {
            "anova_rank": pd.Series(-anova_scores).rank(),
            "chi2_rank": pd.Series(-chi2_scores).rank(),
            "mi_rank": pd.Series(-mi_scores).rank(),
        },
        index=X.columns,
    )
    ranks["consensus"] = ranks.sum(axis=1)
    return ranks.sort_values("consensus").head(k).index.tolist()


class Preprocessor:
    """Fit medians/scaler/feature-selection on training data; apply the same
    fitted transform to any other split."""

    def __init__(self, k: int = N_SF2_FEATURES):
        self.k = k
        self.medians_: pd.Series | None = None
        self.sf2_features_: list[str] | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Preprocessor":
        numeric_cols = X.select_dtypes(include="number").columns
        self.medians_ = X[numeric_cols].median()

        X_imputed = X[numeric_cols].fillna(self.medians_)
        self.sf2_features_ = _select_sf2(X_imputed, y, self.k)

        self.scaler_ = StandardScaler().fit(X_imputed[self.sf2_features_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Only requires the SF-2 columns to be present — the other numeric
        columns seen at fit time (e.g. slope/ca/thal) are never used
        downstream, so callers doing single-patient inference shouldn't need
        to supply them just to satisfy this interface."""
        if self.medians_ is None:
            raise RuntimeError("Preprocessor must be fit() before transform()")
        X_selected = X[self.sf2_features_].fillna(self.medians_[self.sf2_features_])
        return pd.DataFrame(
            self.scaler_.transform(X_selected),
            columns=self.sf2_features_,
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL]).select_dtypes(include="number")
    return X, y
