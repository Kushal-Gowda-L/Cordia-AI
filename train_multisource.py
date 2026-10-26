"""Train the multi-source heart-risk model: one FT-Transformer per dataset
(src/multisource/), each calibrated and evaluated on its own held-out
patients, plus logistic-regression and gradient-boosting baselines.

Datasets are read from --data-dir (default datasets/multisource/), as .csv or
as the original Kaggle .csv.zip downloads, searched recursively. UCI and
Framingham are also found in the repo's own datasets/ folder. A dataset whose
file is missing is skipped with a message, so you can train on any subset.

    python train_multisource.py                               # every dataset found
    python train_multisource.py --sources uci framingham      # a subset
    python train_multisource.py --max-train-rows 50000        # cap the large survey sets
    python train_multisource.py --smoke                       # tiny, fast end-to-end check

Writes checkpoints/multisource/<dataset>.pt and metrics.json, and prints a
metrics table (also saved as checkpoints/multisource/METRICS.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from src.multisource.fusion import DEFAULT_DIR
from src.multisource.sources import SOURCES
from src.multisource.train import train_source

ROOT = Path(__file__).resolve().parent


def metrics_table(results: dict) -> str:
    head = ("| Dataset | Outcome | Test n | Prevalence | FT-Transformer AUROC (95% CI) | AUPRC | Brier | ECE | "
            "Sens. | Spec. | LogReg AUROC | GBM AUROC |")
    rows = [head, "|" + "---|" * 12]
    for name, m in results.items():
        t = m["test"]
        b = m.get("baselines", {})
        lr = f"{b['logistic_regression']['auroc']:.3f}" if b else "-"
        gb = f"{b['gradient_boosting']['auroc']:.3f}" if b else "-"
        rows.append(
            f"| {name} | {SOURCES[name].outcome} | {t['n']} | {t['prevalence']:.3f} | "
            f"{t['auroc']:.3f} ({t['auroc_ci95'][0]:.3f}-{t['auroc_ci95'][1]:.3f}) | {t['auprc']:.3f} | "
            f"{t['brier']:.3f} | {t['ece']:.3f} | {t['sensitivity']:.3f} | {t['specificity']:.3f} | {lr} | {gb} |"
        )
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "multisource")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--sources", nargs="*", choices=list(SOURCES), default=list(SOURCES))
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs, at most 2000 training rows, small model: checks the pipeline end to end")
    args = ap.parse_args(argv)

    torch.set_num_threads(os.cpu_count() or 1)
    config = {}
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs
    max_rows = args.max_train_rows
    if args.smoke:
        config.update(max_epochs=2, d_token=32, depth=1)
        max_rows = min(max_rows or 2000, 2000)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    trained = []
    for name in args.sources:
        source = SOURCES[name]
        try:
            df = source.load(args.data_dir)
        except FileNotFoundError as e:
            print(f"[{name}] skipped: {e}")
            continue
        t0 = time.time()
        ckpt = train_source(source, df, config, seed=args.seed, max_train_rows=max_rows,
                            with_baselines=not args.no_baselines)
        torch.save(ckpt, args.out_dir / f"{name}.pt")
        all_metrics[name] = {**ckpt["metrics"], "skill": ckpt["skill"], "importance": ckpt["importance"]
                             if source.kind == "tabular" else None, "n_rows": ckpt["n_rows"],
                             "config": ckpt["config"], "smoke": bool(args.smoke)}
        trained.append(name)
        print(f"[{name}] saved in {time.time() - t0:.0f}s\n")

    if not trained:
        print("No dataset files found. Put the CSV/ZIP files in", args.data_dir)
        return 1
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    table = metrics_table({k: all_metrics[k] for k in SOURCES if k in all_metrics})
    (args.out_dir / "METRICS.md").write_text(
        "# Multi-source model: held-out test metrics\n\n"
        "Each dataset's own 15% test split. Threshold for sensitivity/specificity chosen on validation "
        "(Youden). Probabilities Platt-calibrated on validation.\n\n" + table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
