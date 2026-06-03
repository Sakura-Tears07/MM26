#!/usr/bin/env python3
"""评估：三模型采样 + 指标（支持 test / hidden_test）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config as cfg
from data_utils import CLASS_NAMES, class_counts, get_eval_split, load_dataset, set_seed
from diffusion_model import ConditionalDiffusion2D
from eval_utils import kde_nll_on_generated, plot_gen_vs_real, save_metrics_bundle
from gmm_model import GMMPerClass
from kde_model import KDEPerClass
from metrics import compute_all_metrics, macro_average
from progress import log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate KDE / GMM / Diffusion.")
    p.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=cfg.OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split", type=str, nargs="+", default=["test"], choices=["test", "hidden_test"])
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--diffusion-only",
        action="store_true",
        help="仅评估 Diffusion（不要求 kde.pkl / gmm.pkl）",
    )
    return p.parse_args()


def evaluate_model(
    name: str,
    gen_x: np.ndarray,
    gen_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    nll_fn,
) -> dict:
    per_class: dict[str, dict[str, float]] = {}
    for label, class_name in enumerate(CLASS_NAMES):
        real = eval_x[eval_y == label]
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


def run_split_diffusion_only(
    args: argparse.Namespace,
    dataset: dict,
    split: str,
) -> None:
    from eval_utils import evaluate_diffusion_checkpoint, save_metrics_bundle

    eval_x, eval_y = get_eval_split(dataset, split)
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== Diffusion-only | split={split} ({eval_x.shape[0]} 点) ===")
    fig = (figures_dir / "diffusion_vs_real.png") if split == "test" and not args.no_figures else None
    result = evaluate_diffusion_checkpoint(
        args.output_dir / "checkpoints" / "diffusion.pt",
        args.data_dir,
        split=split,
        device=args.device,
        seed=args.seed,
        save_figure=fig,
        save_samples_dir=args.output_dir / "samples",
    )
    all_metrics = {"diffusion": result}
    table_rows = [{"model": "diffusion", "class": k, **v} for k, v in result.items()]
    json_name = cfg.FILE_EVAL_TEST if split == "test" else cfg.FILE_EVAL_HIDDEN
    csv_name = cfg.FILE_EVAL_TEST_CSV if split == "test" else cfg.FILE_EVAL_HIDDEN_CSV
    alias = cfg.ALIAS_MAIN_EVAL if split == "test" else cfg.ALIAS_HIDDEN_EVAL
    path = save_metrics_bundle(metrics_dir, all_metrics, table_rows, json_name=json_name, csv_name=csv_name, alias_name=alias)
    log(f"指标: {path}")


def run_split(
    args: argparse.Namespace,
    dataset: dict,
    split: str,
) -> None:
    if args.diffusion_only:
        run_split_diffusion_only(args, dataset, split)
        return

    eval_x, eval_y = get_eval_split(dataset, split)
    counts = class_counts(eval_y)
    ckpt_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    sample_dir = args.output_dir / "samples"
    for d in (metrics_dir, sample_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    suffix = "" if split == "test" else f"_{split}"
    all_metrics: dict[str, dict] = {}
    table_rows: list[dict] = []

    log(f"=== 评估 split={split} ({eval_x.shape[0]} 点) ===")

    kde = KDEPerClass.load(ckpt_dir / "kde.pkl")
    kde_x, kde_y = kde.sample_from_counts(counts, seed=args.seed + 1)
    np.save(sample_dir / f"kde_samples{suffix}.npy", kde_x)
    np.save(sample_dir / f"kde_labels{suffix}.npy", kde_y)

    def kde_nll(label: int, _real: np.ndarray, _fake: np.ndarray) -> float:
        pts = np.asarray(eval_x[eval_y == label], dtype=np.float64)
        return float(-np.mean(kde.models[label].score_samples(pts)))

    all_metrics["kde"] = evaluate_model("kde", kde_x, kde_y, eval_x, eval_y, kde_nll)
    if not args.no_figures and split == "test":
        plot_gen_vs_real(figures_dir / "kde_vs_real.png", eval_x, eval_y, kde_x, kde_y, "KDE")

    gmm = GMMPerClass.load(ckpt_dir / "gmm.pkl")
    gmm_x, gmm_y = gmm.sample_from_counts(counts, seed=args.seed + 2)
    np.save(sample_dir / f"gmm_samples{suffix}.npy", gmm_x)
    np.save(sample_dir / f"gmm_labels{suffix}.npy", gmm_y)

    def gmm_nll(label: int, _real: np.ndarray, _fake: np.ndarray) -> float:
        pts = np.asarray(eval_x[eval_y == label], dtype=np.float64)
        return float(-np.mean(gmm.models[label].score_samples(pts)))

    all_metrics["gmm"] = evaluate_model("gmm", gmm_x, gmm_y, eval_x, eval_y, gmm_nll)
    if not args.no_figures and split == "test":
        plot_gen_vs_real(figures_dir / "gmm_vs_real.png", eval_x, eval_y, gmm_x, gmm_y, "GMM")

    diffusion = ConditionalDiffusion2D.load(ckpt_dir / "diffusion.pt", device=args.device)
    diff_x, diff_y = diffusion.sample_from_counts(counts, seed=args.seed + 3)
    np.save(sample_dir / f"diffusion_samples{suffix}.npy", diff_x)
    np.save(sample_dir / f"diffusion_labels{suffix}.npy", diff_y)

    all_metrics["diffusion"] = evaluate_model(
        "diffusion",
        diff_x,
        diff_y,
        eval_x,
        eval_y,
        lambda label, real, fake: kde_nll_on_generated(label, real, fake),
    )
    if not args.no_figures and split == "test":
        plot_gen_vs_real(figures_dir / "diffusion_vs_real.png", eval_x, eval_y, diff_x, diff_y, "Diffusion")

    for model_name, payload in all_metrics.items():
        for class_name, vals in payload.items():
            row = {"model": model_name, "class": class_name, **vals}
            table_rows.append(row)

    json_name = cfg.FILE_EVAL_TEST if split == "test" else cfg.FILE_EVAL_HIDDEN
    csv_name = cfg.FILE_EVAL_TEST_CSV if split == "test" else cfg.FILE_EVAL_HIDDEN_CSV
    alias = None
    if split == "test":
        alias = cfg.ALIAS_MAIN_EVAL
    elif split == "hidden_test":
        alias = cfg.ALIAS_HIDDEN_EVAL

    path = save_metrics_bundle(
        metrics_dir,
        all_metrics,
        table_rows,
        json_name=json_name,
        csv_name=csv_name,
        alias_name=alias,
    )
    log(f"指标: {path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset = load_dataset(args.data_dir)
    for split in args.split:
        run_split(args, dataset, split)


if __name__ == "__main__":
    main()
