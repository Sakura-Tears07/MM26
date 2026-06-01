#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import CLASS_NAMES, load_dataset, set_seed
from gmm_model import GMMPerClass
from kde_model import KDEPerClass
from progress import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KDE and GMM baselines per class.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kde-bandwidths",
        type=float,
        nargs="+",
        default=[0.05, 0.08, 0.12, 0.18, 0.25, 0.35],
    )
    parser.add_argument(
        "--gmm-components",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 10, 12],
    )
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-3, help="GMM covariance regularization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_dir)
    train_x = dataset["train_x"]
    train_y = dataset["train_y"]
    log(f"训练基线模型 | train={train_x.shape} | 类别={len(CLASS_NAMES)}")

    log("开始训练 KDE（按类）...")
    kde = KDEPerClass(bandwidth_candidates=list(args.kde_bandwidths))
    kde.fit(train_x, train_y)
    kde_path = ckpt_dir / "kde.pkl"
    kde.save(kde_path)
    log(f"KDE 完成，最优带宽: {kde.best_bandwidth}")

    log(f"开始训练 GMM（按类，reg_covar={args.gmm_reg_covar}）...")
    gmm = GMMPerClass(
        component_candidates=list(args.gmm_components),
        reg_covar=args.gmm_reg_covar,
        random_state=args.seed,
    )
    gmm.fit(train_x, train_y, verbose=True)
    gmm_path = ckpt_dir / "gmm.pkl"
    gmm.save(gmm_path)

    summary = {
        "kde_checkpoint": str(kde_path),
        "gmm_checkpoint": str(gmm_path),
        "kde_best_bandwidth": kde.best_bandwidth,
        "gmm_best_components": gmm.best_components,
    }
    with (metrics_dir / "baseline_train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("Baseline models trained.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
