"""Late fusion of the per-dataset encoders for one patient profile.

For every encoder whose inputs the profile reaches:
  coverage = share of the encoder's input importance (validation AUROC lost
             when a column is hidden) that the profile actually supplies
  skill    = 2 * validation AUROC - 1
  weight   = skill * coverage, or 0 when coverage < MIN_COVERAGE or skill < MIN_SKILL

The fused risk is the weighted mean of the encoders' calibrated log-odds.
The weights come from each encoder's own held-out data, not from a model fitted
on combined patients, because no patient appears in more than one dataset.

The datasets measure different outcomes (angiographic coronary disease,
10-year coronary events, self-reported heart disease, abnormal beats), so the
fused figure is a composite cardiovascular-risk index, not a calibrated
probability of any single one of them. Each encoder's own calibrated
probability and outcome are returned beside it, together with the patient's
percentile among that dataset's held-out patients.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.multisource import profile as P
from src.multisource.explain import beat_shapley, tabular_shapley
from src.multisource.model import build
from src.multisource.sources import SOURCES

MIN_COVERAGE = 0.2
MIN_SKILL = 0.1  # validation AUROC 0.55; below this an encoder is treated as chance
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "multisource"


def _sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


class LoadedEncoder:
    def __init__(self, ckpt: dict):
        self.ckpt = ckpt
        self.source = SOURCES[ckpt["source"]]
        self.prep, self.model = build(self.source, ckpt["config"], ckpt["prep"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.cal = ckpt["calibration"]
        self.skill = float(ckpt["skill"])
        imp = ckpt["importance"]
        # A small floor so a column the encoder barely uses still counts a little.
        self.importance = {f: imp.get(f, 0.0) + 1e-3 for f in self.source.feature_names}
        self.reference = np.asarray(ckpt["reference_scores"], dtype=np.float64)

    def coverage(self, row: dict) -> float:
        total = sum(self.importance.values())
        return sum(v for f, v in self.importance.items() if f in row) / total

    def logit(self, row: dict) -> float:
        """Calibrated log-odds for one row (missing columns fed as MASK)."""
        with torch.no_grad():
            raw = self.model(**self.prep.encode(pd.DataFrame([row]))).item()
        return self.cal["a"] * raw + self.cal["b"]

    def empty_logit(self) -> float:
        if self.source.kind == "beat":
            return self.logit({f: v for f, v in zip(self.source.feature_names, self.prep.mean_beat)})
        return self.logit({})

    def percentile(self, p: float) -> float:
        return float(100 * np.searchsorted(self.reference, p) / max(len(self.reference), 1))

    def shapley(self, row: dict, profile: dict, n_permutations: int) -> dict[str, float]:
        """Shapley values in calibrated log-odds."""
        if self.source.kind == "beat":
            phi = beat_shapley(self.prep, self.model, profile["ecg_beat"], n_permutations)
        else:
            phi = tabular_shapley(self.prep, self.model, row, n_permutations)
        return {k: self.cal["a"] * v for k, v in phi.items()}


class HeartRiskModel:
    def __init__(self, encoders: list[LoadedEncoder]):
        self.encoders = encoders

    @classmethod
    def load(cls, directory: Path | str = DEFAULT_DIR) -> "HeartRiskModel":
        directory = Path(directory)
        encoders = [LoadedEncoder(torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True))
                    for name in SOURCES if (directory / f"{name}.pt").exists()]
        if not encoders:
            raise FileNotFoundError(f"no encoder checkpoints in {directory}; run train_multisource.py")
        return cls(encoders)

    def predict(self, profile: dict, explain: bool = True, n_permutations: int = 64) -> dict:
        p = P.normalize(profile)
        rows, sources = {}, []
        for enc in self.encoders:
            row = enc.source.from_profile(p)
            cov = enc.coverage(row) if row else 0.0
            entry = {
                "name": enc.source.name, "title": enc.source.title, "outcome": enc.source.outcome,
                "coverage": round(cov, 3), "skill": round(enc.skill, 3),
                "test_auroc": round(enc.ckpt["metrics"]["test"]["auroc"], 3),
                "inputs_used": sorted(row) if enc.source.kind == "tabular" else (["ecg_beat"] if row else []),
            }
            if cov < MIN_COVERAGE:
                entry.update(weight=0.0, used=False,
                             reason="profile supplies too little of this dataset's inputs" if row else "no matching inputs")
            elif enc.skill < MIN_SKILL:
                entry.update(weight=0.0, used=False, reason="no better than chance on its own held-out data")
            else:
                entry.update(weight=enc.skill * cov, used=True)
            if row:
                z = enc.logit(row)
                prob = _sigmoid(z)
                entry.update(logit=z, probability=round(prob, 4), percentile=round(enc.percentile(prob), 1))
            rows[enc.source.name] = row
            sources.append(entry)

        used = [s for s in sources if s["used"]]
        total_w = sum(s["weight"] for s in used)
        result: dict = {"profile": p, "sources": sources}
        if not total_w:
            result.update(risk=None, percentile=None, band="unknown",
                          message="Not enough information for any dataset model; add age, sex and blood pressure at least.")
            return result
        for s in used:
            s["share"] = round(s["weight"] / total_w, 3)
        fused = sum(s["weight"] * s["logit"] for s in used) / total_w
        pct = sum(s["weight"] * s["percentile"] for s in used) / total_w
        result.update(risk=round(_sigmoid(fused), 4), log_odds=fused, percentile=round(pct, 1), band=risk_band(pct))
        if explain:
            result.update(self._explain(p, rows, used, total_w, n_permutations))
        result["suggested_measurements"] = self._suggestions(p)
        return result

    def _explain(self, p: dict, rows: dict, used: list[dict], total_w: float, n_permutations: int) -> dict:
        by_name = {e.source.name: e for e in self.encoders}
        concept_total: dict[str, float] = {}
        baseline = 0.0
        for s in used:
            enc = by_name[s["name"]]
            share = s["weight"] / total_w
            baseline += share * enc.empty_logit()
            phi = enc.shapley(rows[s["name"]], p, n_permutations)
            s["contributions"] = {k: round(v, 4) for k, v in sorted(phi.items(), key=lambda kv: -abs(kv[1]))}
            if enc.source.kind == "beat":
                concept_total["ecg_beat"] = concept_total.get("ecg_beat", 0.0) + share * sum(phi.values())
                continue
            provided = enc.source.provided_concepts(p)
            for col, v in phi.items():
                concepts = provided.get(col) or ["(derived)"]
                for c in concepts:
                    concept_total[c] = concept_total.get(c, 0.0) + share * v / len(concepts)
        factors = [
            {"field": c, "label": P.FIELDS.get(c, ("", c))[1], "value": _display(c, p.get(c)),
             "log_odds": round(v, 4), "direction": "raises" if v > 0 else "lowers"}
            for c, v in sorted(concept_total.items(), key=lambda kv: -abs(kv[1]))
        ]
        return {"baseline_log_odds": baseline, "factors": factors}

    def _suggestions(self, p: dict, top: int = 5) -> list[dict]:
        """Profile fields not supplied, ranked by how much encoder input they would unlock."""
        gain: dict[str, float] = {}
        for enc in self.encoders:
            if enc.skill < MIN_SKILL or enc.source.kind == "beat":
                continue
            total = sum(enc.importance.values())
            for f in enc.source.features:
                missing = [c for c in f.concepts if c not in p]
                if len(missing) == len(f.concepts):
                    for c in missing:
                        gain[c] = gain.get(c, 0.0) + enc.skill * enc.importance[f.name] / total / len(missing)
        ranked = sorted(gain.items(), key=lambda kv: -kv[1])[:top]
        return [{"field": c, "label": P.FIELDS[c][1]} for c, _ in ranked]

    def what_if(self, profile: dict, changes: dict, explain: bool = False) -> dict:
        """Prediction before and after changing some profile fields (None removes a field)."""
        after_profile = {**profile, **changes}
        after_profile = {k: v for k, v in after_profile.items() if v is not None}
        before = self.predict(profile, explain=explain)
        after = self.predict(after_profile, explain=explain)
        delta = None
        if before["risk"] is not None and after["risk"] is not None:
            delta = round(after["risk"] - before["risk"], 4)
        return {"before": before, "after": after, "risk_change": delta}


def risk_band(percentile: float) -> str:
    if percentile < 40:
        return "below average"
    if percentile < 70:
        return "average"
    if percentile < 90:
        return "above average"
    return "high"


def _display(field: str, v):
    if v is None:
        return None
    if P.FIELDS.get(field, ("",))[0] == "bool":
        return "yes" if v else "no"
    if isinstance(v, list):
        return f"{len(v)}-sample beat"
    if isinstance(v, float):
        return f"{v:g}"
    return v
