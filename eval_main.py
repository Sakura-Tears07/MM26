#!/usr/bin/env python3
"""主实验：从已训练模型采样，与测试集对比，输出全套指标（JSON + CSV）。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_utils import CLASS_NAMES, class_counts, load_dataset, set_seed
from diffusion_model import ConditionalDiffusion2D
from gmm_model import GMMPerClass
from kde_model import KDEPerClass
from eval_utils import plot_gen_vs_real
from metrics import compute_all_metrics, macro_average, negative_log_likelihood_kde_on_generated
from progress import log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Main experiment evaluation.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def evaluate_model(
    name: str,
    gen_x: np.ndarray,
    gen_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    nll_fn,
) -> dict:
    per_class: dict[str, dict[str, float]] = {}
    for label, class_name in enumerate(CLASS_NAMES):
        real = test_x[test_y == label]
        fake = gen_x[gen_y == label]
        nll = nll_fn(label, real, fake)
        per_class[class_name] = compute_all_metrics(
            real,
            fake,
            nll=nll,
            include_mode_coverage=(label == 0),
            seed=42 + label,
        )
    result = dict(per_class)
    result["macro_avg"] = macro_average(per_class)
    log(f"[{name}] macro: {result['macro_avg']}")
    return result


def save_metrics_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ckpt_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    sample_dir = args.output_dir / "samples"
    for d in (metrics_dir, sample_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_dir)
    test_x = dataset["test_x"]
    test_y = dataset["test_y"]
    counts = class_counts(test_y)

    all_metrics: dict[str, dict] = {}
    table_rows: list[dict] = []

    log("评估 KDE ...")
    kde = KDEPerClass.load(ckpt_dir / "kde.pkl")
    kde_x, kde_y = kde.sample_from_counts(counts, seed=args.seed + 1)
    np.save(sample_dir / "kde_samples.npy", kde_x)
    np.save(sample_dir / "kde_labels.npy", kde_y)

    def kde_nll(label: int, _real: np.ndarray, _fake: np.ndarray) -> float:
        mask = test_y == label
        pts = np.asarray(test_x[mask], dtype=np.float64)
        lp = kde.models[label].score_samples(pts)
        return float(-np.mean(lp))

    all_metrics["kde"] = evaluate_model("kde", kde_x, kde_y, test_x, test_y, kde_nll)
    if not args.no_figures:
        plot_gen_vs_real(figures_dir / "kde_vs_real.png", test_x, test_y, kde_x, kde_y, "KDE")

    log("评估 GMM ...")
    gmm = GMMPerClass.load(ckpt_dir / "gmm.pkl")
    gmm_x, gmm_y = gmm.sample_from_counts(counts, seed=args.seed + 2)
    np.save(sample_dir / "gmm_samples.npy", gmm_x)
    np.save(sample_dir / "gmm_labels.npy", gmm_y)

    def gmm_nll(label: int, _real: np.ndarray, _fake: np.ndarray) -> float:
        mask = test_y == label
        pts = np.asarray(test_x[mask], dtype=np.float64)
        lp = gmm.models[label].score_samples(pts)
        return float(-np.mean(lp))

    all_metrics["gmm"] = evaluate_model("gmm", gmm_x, gmm_y, test_x, test_y, gmm_nll)
    if not args.no_figures:
        plot_gen_vs_real(figures_dir / "gmm_vs_real.png", test_x, test_y, gmm_x, gmm_y, "GMM")

    log(f"评估 Diffusion ({args.device}) ...")
    diffusion = ConditionalDiffusion2D.load(ckpt_dir / "diffusion.pt", device=args.device)
    diff_x, diff_y = diffusion.sample_from_counts(counts, seed=args.seed + 3)
    np.save(sample_dir / "diffusion_samples.npy", diff_x)
    np.save(sample_dir / "diffusion_labels.npy", diff_y)

    def diff_nll(label: int, real: np.ndarray, fake: np.ndarray) -> float:
        return negative_log_likelihood_kde_on_generated(real, fake, bandwidth=0.12)

    all_metrics["diffusion"] = evaluate_model("diffusion", diff_x, diff_y, test_x, test_y, diff_nll)
    if not args.no_figures:
        plot_gen_vs_real(figures_dir / "diffusion_vs_real.png", test_x, test_y, diff_x, diff_y, "Diffusion")

    for model_name, payload in all_metrics.items():
        for class_name, vals in payload.items():
            if class_name == "macro_avg":
                row = {"model": model_name, "class": "macro_avg", **vals}
            else:
                row = {"model": model_name, "class": class_name, **vals}
            table_rows.append(row)

    with (metrics_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
        f.write("\n")
    save_metrics_table(table_rows, metrics_dir / "evaluation.csv")
    log(f"指标已保存: {metrics_dir / 'evaluation.json'}, {metrics_dir / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
