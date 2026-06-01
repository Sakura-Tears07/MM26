#!/usr/bin/env python3
"""拓展任务二：条件生成 p(x|c) + 与测试集对比的完整指标。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_utils import CLASS_NAMES, class_counts, load_dataset, set_seed
from diffusion_model import ConditionalDiffusion2D
from eval_utils import evaluate_by_class, plot_gen_vs_real
from metrics import compute_all_metrics, macro_average, negative_log_likelihood_kde_on_generated
from progress import log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/diffusion.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ext_conditional"))
    p.add_argument("--samples-per-class", type=int, default=0, help="0 表示与 test 集每类数量一致")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_dir)
    test_x, test_y = dataset["test_x"], dataset["test_y"]
    counts = class_counts(test_y)
    if args.samples_per_class > 0:
        counts = {k: args.samples_per_class for k in counts}

    log(f"加载模型: {args.checkpoint}")
    model = ConditionalDiffusion2D.load(args.checkpoint, device=args.device)

    gen_x_list, gen_y_list = [], []
    per_class_metrics: dict[str, dict] = {}
    for label in range(len(CLASS_NAMES)):
        n = counts[label]
        log(f"条件生成 {CLASS_NAMES[label]} (c={label}), n={n}")
        x = model.sample_by_label(label=label, n_samples=n, seed=args.seed + label)
        y = np.full(n, label, dtype=np.int64)
        gen_x_list.append(x)
        gen_y_list.append(y)
        np.save(args.output_dir / f"samples_class_{label}.npy", x)
        np.save(args.output_dir / f"labels_class_{label}.npy", y)

        real = test_x[test_y == label]
        nll = negative_log_likelihood_kde_on_generated(real, x, bandwidth=0.1 if label == 3 else 0.12)
        m = compute_all_metrics(real, x, nll=nll, include_mode_coverage=(label == 0), seed=args.seed + label)
        per_class_metrics[CLASS_NAMES[label]] = m
        log(f"  {CLASS_NAMES[label]}: mmd={m['mmd']:.4f}, coverage={m['coverage']:.4f}, precision={m['precision']:.4f}")

    gen_x = np.concatenate(gen_x_list)
    gen_y = np.concatenate(gen_y_list)
    np.save(args.output_dir / "samples_all.npy", gen_x)
    np.save(args.output_dir / "labels_all.npy", gen_y)

    metrics = evaluate_by_class(gen_x, gen_y, test_x, test_y, nll_from_generated_kde=True, seed=args.seed)
    with (args.output_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    rows = []
    for class_name in CLASS_NAMES:
        rows.append({"class": class_name, **metrics[class_name]})
    rows.append({"class": "macro_avg", **metrics["macro_avg"]})
    with (args.output_dir / "evaluation.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig_path = args.output_dir / "conditional_vs_real.png"
    plot_gen_vs_real(fig_path, test_x, test_y, gen_x, gen_y, title="Conditional Generation vs Real")

    log(f"指标: {args.output_dir / 'evaluation.json'}")
    log(f"macro: {metrics['macro_avg']}")


if __name__ == "__main__":
    main()
