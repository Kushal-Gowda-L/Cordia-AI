"""Training, calibration and evaluation of one encoder on its own dataset.

Each dataset is split 70 / 15 / 15 (stratified, fixed seed) into train /
validation / test. Validation is used for early stopping, for the Platt
calibration of the encoder's logit, for the decision threshold and for the
fusion weight; test is touched once, for the reported metrics. The same
split is used for two non-transformer baselines (logistic regression and
gradient-boosted trees) so the transformer's numbers have a reference.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.multisource.metrics import binary_metrics, youden_threshold
from src.multisource.model import build, random_mask
from src.multisource.sources import Source

DEFAULT_CONFIG = {
    "d_token": 64, "depth": 3, "heads": 8, "dropout": 0.1,
    "lr": 3e-4, "weight_decay": 1e-5, "max_epochs": 60, "patience": 8,
    "max_mask_rate": 0.5,  # share of observed values hidden per training row, drawn from U(0, this)
}
BEAT_OVERRIDES = {"heads": 4, "max_mask_rate": 0.0}


def split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, rest = train_test_split(df, test_size=0.30, stratify=df["target"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["target"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def predict_logits(model, batch: dict[str, torch.Tensor], batch_size: int = 8192) -> np.ndarray:
    model.eval()
    n = next(iter(batch.values())).shape[0]
    out = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            out.append(model(**{k: v[i:i + batch_size] for k, v in batch.items()}))
    return torch.cat(out).numpy() if out else np.zeros(0)


def fit_platt(logits: np.ndarray, y: np.ndarray) -> dict:
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(logits.reshape(-1, 1), y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def calibrated(logits: np.ndarray, cal: dict) -> np.ndarray:
    return 1 / (1 + np.exp(-(cal["a"] * logits + cal["b"])))


def train_encoder(source: Source, train: pd.DataFrame, val: pd.DataFrame, config: dict, seed: int = 42, log=print):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    prep, model = build(source, config)
    prep.fit(train)
    tr = prep.encode(train)
    va = prep.encode(val)
    y_tr = torch.tensor(train["target"].values, dtype=torch.float32)
    y_va = val["target"].values
    n = len(train)
    batch_size = int(min(1024, max(64, n // 40)))
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            batch = {k: v[idx] for k, v in tr.items()}
            if config["max_mask_rate"] > 0:
                batch = random_mask(batch, config["max_mask_rate"], gen)
            loss = loss_fn(model(**batch), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        val_auc = roc_auc_score(y_va, predict_logits(model, va))
        history.append({"epoch": epoch, "train_loss": total / n, "val_auroc": float(val_auc)})
        log(f"  [{source.name}] epoch {epoch:3d}  loss {total / n:.4f}  val AUROC {val_auc:.4f}  ({time.time() - t0:.1f}s)")
        if val_auc > best_auc + 1e-4:
            best_auc, best_state, best_epoch = val_auc, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return prep, model, history, best_epoch


def mask_importance(source: Source, prep, model, df: pd.DataFrame) -> dict[str, float]:
    """Drop in validation AUROC when one column is hidden (fed as MASK) for every row.
    This is what the fusion step uses to judge how much of an encoder's input a
    patient profile covers."""
    if source.kind == "beat":
        return {f: 1.0 / len(source.features) for f in source.feature_names}
    y = df["target"].values
    base = roc_auc_score(y, predict_logits(model, prep.encode(df)))
    return {f: float(max(base - roc_auc_score(y, predict_logits(model, prep.encode(df, masked={f}))), 0.0))
            for f in source.feature_names}


def baselines(source: Source, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    num = [f.name for f in source.features if f.kind == "num"]
    cat = [f.name for f in source.features if f.kind == "cat"]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat),
    ])
    models = {
        "logistic_regression": make_pipeline(pre, LogisticRegression(max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0,
            categorical_features=[source.feature_names.index(c) for c in cat] or None),
    }
    out = {}
    cols = source.feature_names
    for name, m in models.items():
        m.fit(train[cols], train["target"])
        thr = youden_threshold(val["target"].values, m.predict_proba(val[cols])[:, 1])
        out[name] = binary_metrics(test["target"].values, m.predict_proba(test[cols])[:, 1], thr, ci=False)
    return out


def train_source(source: Source, df: pd.DataFrame, config: dict | None = None, seed: int = 42,
                 max_train_rows: int | None = None, with_baselines: bool = True, log=print) -> dict:
    """Train + calibrate + evaluate one encoder. Returns the checkpoint dict."""
    config = {**DEFAULT_CONFIG, **(BEAT_OVERRIDES if source.kind == "beat" else {}), **(config or {})}
    train, val, test = split(df, seed)
    if max_train_rows and len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=seed).reset_index(drop=True)
    log(f"[{source.name}] train {len(train)}  val {len(val)}  test {len(test)}  "
        f"positive rate {df['target'].mean():.3f}")
    t0 = time.time()
    prep, model, history, best_epoch = train_encoder(source, train, val, config, seed, log)
    val_logits = predict_logits(model, prep.encode(val))
    cal = fit_platt(val_logits, val["target"].values)
    val_p = calibrated(val_logits, cal)
    threshold = youden_threshold(val["target"].values, val_p)
    test_p = calibrated(predict_logits(model, prep.encode(test)), cal)
    metrics = {
        "val": binary_metrics(val["target"].values, val_p, threshold, ci=False),
        "test": binary_metrics(test["target"].values, test_p, threshold),
        "train_seconds": round(time.time() - t0, 1),
        "best_epoch": best_epoch,
    }
    if with_baselines and source.kind == "tabular":
        metrics["baselines"] = baselines(source, train, val, test)
    importance = mask_importance(source, prep, model, val)
    # Test scores kept so a patient's risk can be placed among that dataset's held-out patients.
    reference = np.sort(test_p).astype(np.float32)
    log(f"[{source.name}] test AUROC {metrics['test']['auroc']:.4f}  "
        f"(95% CI {metrics['test']['auroc_ci95'][0]:.3f}-{metrics['test']['auroc_ci95'][1]:.3f})  "
        f"best epoch {best_epoch}")
    return {
        "source": source.name,
        "config": config,
        "prep": prep.state(),
        "state_dict": model.state_dict(),
        "calibration": cal,
        "threshold": threshold,
        "skill": max(0.0, 2 * metrics["val"]["auroc"] - 1),  # Gini on validation
        "importance": importance,
        "metrics": metrics,
        "history": history,
        "reference_scores": torch.from_numpy(reference[:: max(1, len(reference) // 2000)].copy()),
        "n_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "seed": seed,
    }
