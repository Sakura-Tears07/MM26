#!/usr/bin/env python3
"""拓展实验：python extensions.py conditional | sweep | robustness"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import config as cfg
from data_utils import CLASS_NAMES, class_counts, load_dataset, set_seed
from eval_utils import evaluate_by_class, evaluate_diffusion_checkpoint, plot_gen_vs_real, save_metrics_bundle
from progress import ProgressTracker, log

ROOT = Path(__file__).resolve().parent
GRID_KEYS = (
    "batch_sizes",
    "lrs",
    "num_steps_list",
    "ema_decays",
    "accum_steps_list",
    "hidden_dims",
)


def run_command(command: list[str], cwd: Path) -> int:
    log(">> " + " ".join(command))
    return int(subprocess.run(command, cwd=cwd).returncode)


# --- conditional ---


def cmd_conditional(args: argparse.Namespace) -> None:
    from diffusion_model import ConditionalDiffusion2D

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_dir)
    test_x, test_y = dataset["test_x"], dataset["test_y"]
    counts = class_counts(test_y)
    if args.samples_per_class > 0:
        counts = {k: args.samples_per_class for k in counts}

    log(f"加载: {args.checkpoint}")
    model = ConditionalDiffusion2D.load(args.checkpoint, device=args.device)

    gen_parts_x, gen_parts_y = [], []
    for label in range(len(CLASS_NAMES)):
        n = counts[label]
        log(f"条件生成 {CLASS_NAMES[label]} n={n}")
        x = model.sample_by_label(label=label, n_samples=n, seed=args.seed + label)
        y = np.full(n, label, dtype=np.int64)
        gen_parts_x.append(x)
        gen_parts_y.append(y)
        np.save(args.output_dir / f"samples_class_{label}.npy", x)
        np.save(args.output_dir / f"labels_class_{label}.npy", y)

    gen_x = np.concatenate(gen_parts_x)
    gen_y = np.concatenate(gen_parts_y)
    np.save(args.output_dir / "samples_all.npy", gen_x)
    np.save(args.output_dir / "labels_all.npy", gen_y)

    metrics = evaluate_by_class(gen_x, gen_y, test_x, test_y, nll_from_generated_kde=True, seed=args.seed)
    rows = [{"class": cn, **metrics[cn]} for cn in CLASS_NAMES]
    rows.append({"class": "macro_avg", **metrics["macro_avg"]})
    save_metrics_bundle(
        args.output_dir,
        metrics,
        rows,
        alias_name=cfg.ALIAS_CONDITIONAL_EVAL,
    )
    plot_gen_vs_real(
        args.output_dir / "conditional_vs_real.png",
        test_x,
        test_y,
        gen_x,
        gen_y,
        title="Conditional Generation vs Real",
    )
    log(f"macro: {metrics['macro_avg']}")


# --- sweep ---


def load_grid(path: Path) -> dict[str, list]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: raw[k] for k in GRID_KEYS}


def trial_suffix(bs: int, lr: float, ns: int, ema: float, acc: int, hd: int) -> str:
    return f"bs{bs}_lr{lr:.0e}_ns{ns}_ema{ema}_acc{acc}_hd{hd}"


def steps_per_epoch(n_train: int, batch_size: int, gpus: int, accum_steps: int) -> int:
    per_rank = max(1, math.ceil(n_train / max(1, gpus)))
    return max(1, math.ceil(per_rank / max(1, batch_size) / max(1, accum_steps)))


def diffusion_train_cmd(py: str, args: argparse.Namespace, data_dir: Path, out_dir: Path, combo: tuple) -> list[str]:
    bs, lr, ns, ema, acc, hd = combo
    tail = [
        str(ROOT / "train.py"),
        "diffusion",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(bs),
        "--lr",
        str(lr),
        "--num-steps",
        str(ns),
        "--hidden-dim",
        str(hd),
        "--class-emb-dim",
        str(args.class_emb_dim),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--num-workers",
        str(args.num_workers),
        "--accum-steps",
        str(acc),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--ema-decay",
        str(ema),
        "--grad-clip",
        str(args.grad_clip),
    ]
    if args.compile:
        tail.append("--compile")
    if args.gpus > 1:
        return ["torchrun", "--standalone", "--nproc_per_node", str(args.gpus), *tail]
    return [py, *tail]


def cmd_sweep(args: argparse.Namespace) -> None:
    grid_path = args.grid_file if args.grid_file.is_absolute() else ROOT / args.grid_file
    grid = load_grid(grid_path)
    sweep_root = args.sweep_dir.resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    n_train = len(np.load(data_dir / "train.npy"))
    log(f"网格: {grid_path} | train={n_train}")

    combos = list(
        itertools.product(
            grid["batch_sizes"],
            grid["lrs"],
            grid["num_steps_list"],
            grid["ema_decays"],
            grid["accum_steps_list"],
            grid["hidden_dims"],
        )
    )
    valid = [c for c in combos if steps_per_epoch(n_train, c[0], args.gpus, c[4]) >= args.min_steps_per_epoch]
    if args.random_order:
        random.Random(args.random_seed).shuffle(valid)
    if args.max_trials > 0:
        valid = valid[: args.max_trials]

    summary_path = sweep_root / "summary.json"
    summary_rows: list[dict] = []
    if summary_path.exists() and not args.force:
        summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
    done = {r["trial"] for r in summary_rows}

    py = sys.executable
    tracker = ProgressTracker(len(valid), desc="Sweep")
    for run_idx, combo in enumerate(valid, start=1):
        bs, lr, ns, ema, acc, hd = combo
        suffix = trial_suffix(bs, lr, ns, ema, acc, hd)
        tname = f"trial_{run_idx:04d}_{suffix}"
        trial_dir = sweep_root / tname
        ckpt = trial_dir / "checkpoints" / "diffusion.pt"
        eval_path = trial_dir / "metrics" / "diffusion_eval.json"

        if not args.force and eval_path.exists() and tname in done:
            tracker.update(extra="skip")
            continue

        if not args.eval_only and (args.force or not ckpt.exists()):
            if run_command(diffusion_train_cmd(py, args, data_dir, trial_dir, combo), cwd=ROOT) != 0:
                tracker.update(extra="fail")
                continue

        if not ckpt.exists():
            tracker.update(extra="no_ckpt")
            continue

        result = evaluate_diffusion_checkpoint(
            ckpt,
            data_dir,
            device=args.device,
            seed=args.seed,
            save_figure=trial_dir / "figures" / "diffusion_vs_real.png",
            save_samples_dir=trial_dir / "samples",
        )
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        ts = {}
        ts_path = trial_dir / "metrics" / "diffusion_train_summary.json"
        if ts_path.exists():
            ts = json.loads(ts_path.read_text(encoding="utf-8"))
        macro = result["macro_avg"]
        row = {
            "trial": tname,
            "batch_size": bs,
            "lr": lr,
            "num_steps": ns,
            "ema_decay": ema,
            "accum_steps": acc,
            "hidden_dim": hd,
            "steps_per_epoch": ts.get("steps_per_epoch"),
            "final_loss": ts.get("final_loss"),
            "mmd": macro.get("mmd"),
            "coverage": macro.get("coverage"),
            "precision": macro.get("precision"),
            "spiral_precision": result.get("spiral", {}).get("precision"),
            "nll": macro.get("nll"),
        }
        summary_rows = [r for r in summary_rows if r.get("trial") != tname]
        summary_rows.append(row)
        summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
        tracker.update(extra=suffix)

    if not summary_rows:
        return
    with (sweep_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    best = {
        "best_mmd": min(summary_rows, key=lambda r: r["mmd"]),
        "best_precision": max(summary_rows, key=lambda r: r["precision"]),
        "best_spiral_precision": max(summary_rows, key=lambda r: r.get("spiral_precision") or -1.0),
    }
    (sweep_root / "best_trials.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(summary_path, sweep_root / cfg.ALIAS_SWEEP_SUMMARY)
    tracker.done(str(summary_path))


# --- robustness ---


def build_corrupted_data(base_data: Path, target_data: Path, ratio: float, seed: int, outlier_range: float) -> None:
    target_data.mkdir(parents=True, exist_ok=True)
    train_x = np.load(base_data / "train.npy")
    train_y = np.load(base_data / "train_label.npy")
    test_x = np.load(base_data / "test.npy")
    test_y = np.load(base_data / "test_label.npy")
    hidden_x = np.load(base_data / "hidden_test.npy")
    hidden_y = np.load(base_data / "hidden_test_label.npy")

    rng = np.random.default_rng(seed)
    corrupted = train_x.copy()
    k = int(round(len(train_x) * ratio))
    if k > 0:
        idx = rng.choice(len(train_x), size=k, replace=False)
        corrupted[idx] = rng.uniform(-outlier_range, outlier_range, size=(k, 2)).astype(np.float32)

    np.save(target_data / "train.npy", corrupted.astype(np.float32))
    np.save(target_data / "train_label.npy", train_y.astype(np.int64))
    np.save(target_data / "test.npy", test_x.astype(np.float32))
    np.save(target_data / "test_label.npy", test_y.astype(np.int64))
    np.save(target_data / "hidden_test.npy", hidden_x.astype(np.float32))
    np.save(target_data / "hidden_test_label.npy", hidden_y.astype(np.int64))
    with (target_data / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"outlier_ratio": ratio, "outlier_range": outlier_range, "seed": seed}, f, indent=2)
        f.write("\n")


def copy_clean_data(base_data: Path, target_data: Path) -> None:
    target_data.mkdir(parents=True, exist_ok=True)
    for name in (
        "train.npy",
        "train_label.npy",
        "test.npy",
        "test_label.npy",
        "hidden_test.npy",
        "hidden_test_label.npy",
        "metadata.json",
    ):
        shutil.copy2(base_data / name, target_data / name)


def link_main_checkpoints(main_out: Path, model_out: Path) -> None:
    src = main_out / "checkpoints"
    dst = model_out / "checkpoints"
    if not src.exists():
        raise FileNotFoundError(f"主实验 checkpoint 不存在: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("kde.pkl", "gmm.pkl", "diffusion.pt"):
        shutil.copy2(src / name, dst / name)


def append_summary_rows(rows: list[dict], metrics: dict, ratio: float) -> list[dict]:
    rows = [r for r in rows if r.get("outlier_ratio") != ratio]
    for model_name in ("kde", "gmm", "diffusion"):
        for class_name, vals in metrics[model_name].items():
            rows.append({"outlier_ratio": ratio, "model": model_name, "class": class_name, **vals})
    return rows


def cmd_robustness(args: argparse.Namespace) -> None:
    py = sys.executable
    output_root = args.output_root.resolve()
    main_out = (ROOT / args.main_output_dir).resolve()
    base_data = (ROOT / args.data_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "summary.json"
    detail_path = output_root / "evaluation_by_ratio.json"
    rows: list[dict] = []
    detail: dict[str, dict] = {}
    if summary_path.exists() and not args.force:
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if detail_path.exists() and not args.force:
        detail = json.loads(detail_path.read_text(encoding="utf-8"))

    tracker = ProgressTracker(len(args.ratios), desc="鲁棒性")
    for ratio in args.ratios:
        tag = f"ratio_{ratio:.3f}".replace(".", "p")
        run_dir = output_root / tag
        data_dir = run_dir / "data"
        model_out = run_dir
        eval_path = model_out / "metrics" / cfg.FILE_EVAL_TEST

        if not args.force and any(r.get("outlier_ratio") == ratio for r in rows) and eval_path.exists():
            tracker.update(extra=f"skip {ratio}")
            continue

        log(f"===== outlier ratio {ratio:.1%} =====")
        if ratio == 0.0:
            copy_clean_data(base_data, data_dir)
            model_out.mkdir(parents=True, exist_ok=True)
            link_main_checkpoints(main_out, model_out)
        else:
            build_corrupted_data(base_data, data_dir, ratio, args.seed, args.outlier_range)
            gmm_reg = 1e-3 if ratio <= 0.01 else (1e-2 if ratio <= 0.05 else 5e-2)
            run_command(
                [
                    py,
                    str(ROOT / "train.py"),
                    "baselines",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(model_out),
                    "--seed",
                    str(args.seed),
                    "--gmm-reg-covar",
                    str(gmm_reg),
                ],
                cwd=ROOT,
            )
            combo = (args.batch_size, args.lr, args.num_steps, args.ema_decay, 1, args.hidden_dim)
            run_command(diffusion_train_cmd(py, args, data_dir, model_out, combo), cwd=ROOT)

        run_command(
            [
                py,
                str(ROOT / "evaluate.py"),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(model_out),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--split",
                "test",
                "--no-figures",
            ],
            cwd=ROOT,
        )

        metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        detail[str(ratio)] = metrics
        rows = append_summary_rows(rows, metrics, ratio)
        summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        detail_path.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        tracker.update(extra=f"ratio={ratio}")

    macro_rows = [r for r in rows if r.get("class") == "macro_avg"]
    if macro_rows:
        with (output_root / "summary_macro.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(macro_rows[0].keys()))
            w.writeheader()
            w.writerows(macro_rows)
    if rows:
        with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        shutil.copy2(summary_path, output_root / cfg.ALIAS_ROBUSTNESS_SUMMARY)
    tracker.done(str(output_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MM26 extension experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p_c = sub.add_parser("conditional", help="Conditional generation eval")
    p_c.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    p_c.add_argument("--checkpoint", type=Path, default=cfg.CKPT_DIFFUSION)
    p_c.add_argument("--output-dir", type=Path, default=cfg.EXT_CONDITIONAL_DIR)
    p_c.add_argument("--samples-per-class", type=int, default=0)
    p_c.add_argument("--seed", type=int, default=42)
    p_c.add_argument("--device", type=str, default="cuda")
    p_c.add_argument("--force", action="store_true")

    p_s = sub.add_parser("sweep", help="Diffusion hyper-parameter sweep")
    p_s.add_argument("--grid-file", type=Path, default=cfg.SWEEP_GRID_FILE)
    p_s.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    p_s.add_argument("--sweep-dir", type=Path, default=cfg.EXT_SWEEP_DIR)
    p_s.add_argument("--seed", type=int, default=42)
    p_s.add_argument("--epochs", type=int, default=cfg.DIFFUSION_EPOCHS)
    p_s.add_argument("--gpus", type=int, default=cfg.DEFAULT_GPUS)
    p_s.add_argument("--device", type=str, default="cuda")
    p_s.add_argument("--precision", type=str, default=cfg.DIFFUSION_PRECISION)
    p_s.add_argument("--num-workers", type=int, default=cfg.DIFFUSION_NUM_WORKERS)
    p_s.add_argument("--warmup-epochs", type=int, default=cfg.DIFFUSION_WARMUP_EPOCHS)
    p_s.add_argument("--grad-clip", type=float, default=cfg.DIFFUSION_GRAD_CLIP)
    p_s.add_argument("--class-emb-dim", type=int, default=cfg.DIFFUSION_CLASS_EMB_DIM)
    p_s.add_argument("--compile", action="store_true")
    p_s.add_argument("--max-trials", type=int, default=0)
    p_s.add_argument("--random-order", action="store_true")
    p_s.add_argument("--random-seed", type=int, default=1234)
    p_s.add_argument("--min-steps-per-epoch", type=int, default=20)
    p_s.add_argument("--eval-only", action="store_true")
    p_s.add_argument("--force", action="store_true")

    p_r = sub.add_parser("robustness", help="Outlier robustness")
    p_r.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    p_r.add_argument("--main-output-dir", type=Path, default=cfg.OUTPUT_DIR)
    p_r.add_argument("--output-root", type=Path, default=cfg.EXT_ROBUSTNESS_DIR)
    p_r.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    p_r.add_argument("--seed", type=int, default=42)
    p_r.add_argument("--outlier-range", type=float, default=8.0)
    p_r.add_argument("--epochs", type=int, default=cfg.DIFFUSION_EPOCHS)
    p_r.add_argument("--batch-size", type=int, default=cfg.DIFFUSION_BATCH_SIZE)
    p_r.add_argument("--lr", type=float, default=cfg.DIFFUSION_LR)
    p_r.add_argument("--num-steps", type=int, default=cfg.DIFFUSION_NUM_STEPS)
    p_r.add_argument("--hidden-dim", type=int, default=cfg.DIFFUSION_HIDDEN_DIM)
    p_r.add_argument("--class-emb-dim", type=int, default=cfg.DIFFUSION_CLASS_EMB_DIM)
    p_r.add_argument("--ema-decay", type=float, default=cfg.DIFFUSION_EMA_DECAY)
    p_r.add_argument("--warmup-epochs", type=int, default=cfg.DIFFUSION_WARMUP_EPOCHS)
    p_r.add_argument("--grad-clip", type=float, default=cfg.DIFFUSION_GRAD_CLIP)
    p_r.add_argument("--gpus", type=int, default=cfg.DEFAULT_GPUS)
    p_r.add_argument("--device", type=str, default="cuda")
    p_r.add_argument("--precision", type=str, default=cfg.DIFFUSION_PRECISION)
    p_r.add_argument("--num-workers", type=int, default=cfg.DIFFUSION_NUM_WORKERS)
    p_r.add_argument("--compile", action="store_true")
    p_r.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "conditional":
        cmd_conditional(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    else:
        cmd_robustness(args)


if __name__ == "__main__":
    main()
