"""SHAP, LIME, and DiCE explainability for the tabular FT-Transformer branch.

Each method wraps the same underlying prediction function — raw (unscaled)
SF-2 feature values in -> the fitted Preprocessor -> the trained model ->
softmax probabilities — so all three explain in the same, clinically
readable feature space (age in years, cholesterol in mg/dl, etc.) rather
than the model's internal standardized space.

Every method is wrapped in try/except and returns an empty result on
failure rather than raising: a report should still render if one
explainability library breaks, rather than taking down the whole request.
This matters for a live demo, and these three libraries — each with their
own dependency chain and API quirks — are exactly the kind of thing that
can break in ways unrelated to the core model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


class Explainer:
    """`extra_cols` are model inputs that are held at the patient's own values
    (possibly blank) rather than explained -- v2's slope/ca/thal, which most
    records leave blank and which SHAP/LIME/DiCE can't perturb meaningfully."""

    def __init__(
        self, model, prep, feature_cols: list[str], classes: list[str], X_train_raw: pd.DataFrame,
        extra_cols: list[str] | None = None,
    ):
        self.model = model
        self.prep = prep
        self.feature_cols = feature_cols
        self.extra_cols = extra_cols or []
        self.classes = classes
        self.X_train_raw = X_train_raw[feature_cols].reset_index(drop=True)

        self._shap_background = None
        self._lime_explainer = None
        self._dice_data = None

    def predict_proba(self, X_raw: np.ndarray, extras: dict | None = None) -> np.ndarray:
        df = pd.DataFrame(np.asarray(X_raw, dtype=float), columns=self.feature_cols)
        for c in self.extra_cols:
            value = (extras or {}).get(c)
            df[c] = np.nan if value is None else float(value)
        X_scaled = self.prep.transform(df)
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled.values, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def _predict_fn(self, raw_row: dict):
        """predict_proba with this patient's extra_cols values held fixed."""
        extras = {c: raw_row.get(c) for c in self.extra_cols}
        return lambda X: self.predict_proba(X, extras)

    # ------------------------------------------------------------- SHAP
    def shap_disease_factors(self, raw_row: dict, top_n: int = 3) -> list[dict]:
        """SHAP contributions to the chance of any heart disease (1 - P(No disease)).

        Explaining whichever class won makes the chart flip meaning from patient
        to patient -- for a "No disease" prediction a bar pointing "towards it"
        is reassuring, for "Severe" it is alarming. Explaining P(disease) always
        matches the headline risk: positive raises it, negative lowers it.
        Class probabilities sum to 1, so SHAP's additivity gives the disease
        contribution exactly as minus the "No disease" contribution.
        """
        factors = self.shap_top_factors(raw_row, predicted_class=0, top_n=top_n)
        return [{**f, "impact": -f["impact"]} for f in factors]

    def shap_top_factors(self, raw_row: dict, predicted_class: int, top_n: int = 3) -> list[dict]:
        try:
            import shap

            if self._shap_background is None:
                self._shap_background = shap.kmeans(self.X_train_raw.values, 20)
            explainer = shap.KernelExplainer(self._predict_fn(raw_row), self._shap_background)

            x = np.array([[raw_row[c] for c in self.feature_cols]])
            shap_values = explainer.shap_values(x, nsamples=100, silent=True)

            if isinstance(shap_values, list):
                vals = np.asarray(shap_values[predicted_class])[0]
            elif shap_values.ndim == 3:
                vals = shap_values[0, :, predicted_class]
            else:
                vals = shap_values[0]

            order = np.argsort(-np.abs(vals))[:top_n]
            return [
                {
                    "feature": self.feature_cols[i],
                    "value": raw_row[self.feature_cols[i]],
                    "impact": round(float(vals[i]), 4),
                }
                for i in order
            ]
        except Exception:
            return []

    # ------------------------------------------------------------- LIME
    def lime_top_factors(self, raw_row: dict, predicted_class: int, top_n: int = 3) -> list[dict]:
        try:
            from lime.lime_tabular import LimeTabularExplainer

            if self._lime_explainer is None:
                self._lime_explainer = LimeTabularExplainer(
                    self.X_train_raw.values,
                    feature_names=self.feature_cols,
                    class_names=self.classes,
                    mode="classification",
                    discretize_continuous=True,
                )
            x = np.array([raw_row[c] for c in self.feature_cols])
            exp = self._lime_explainer.explain_instance(
                x, self._predict_fn(raw_row), num_features=top_n, num_samples=500, labels=[predicted_class]
            )
            return [
                {"rule": rule, "weight": round(float(weight), 4)}
                for rule, weight in exp.as_list(label=predicted_class)
            ]
        except Exception:
            return []

    # ------------------------------------------------------------- DiCE
    def dice_counterfactuals(self, raw_row: dict, predicted_class: int, total_cfs: int = 2) -> list[dict]:
        if predicted_class == 0:
            return []  # already the healthiest class, nothing to counterfactual toward
        try:
            import dice_ml

            if self._dice_data is None:
                ref_df = self.X_train_raw.copy()
                ref_df["num"] = 0  # placeholder outcome column DiCE's Data object requires
                self._dice_data = dice_ml.Data(dataframe=ref_df, continuous_features=self.feature_cols, outcome_name="num")

            predict_fn = self._predict_fn(raw_row)

            class _SklearnLikeWrapper:
                def predict_proba(self, X):
                    return predict_fn(np.asarray(X))

                def predict(self, X):
                    return predict_fn(np.asarray(X)).argmax(axis=1)

            model = dice_ml.Model(model=_SklearnLikeWrapper(), backend="sklearn")
            dice = dice_ml.Dice(self._dice_data, model, method="random")

            target_class = predicted_class - 1  # aim one severity level down, not straight to "no disease"
            query = pd.DataFrame([raw_row])[self.feature_cols]
            cf = dice.generate_counterfactuals(
                query, total_CFs=total_cfs, desired_class=target_class, verbose=False
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df
            results = []
            for _, r in cf_df.iterrows():
                changes = {
                    c: round(float(r[c]), 2)
                    for c in self.feature_cols
                    if abs(float(r[c]) - raw_row[c]) > 1e-6
                }
                if changes:
                    results.append({"changes": changes, "target_class": self.classes[target_class]})
            return results
        except Exception:
            return []
