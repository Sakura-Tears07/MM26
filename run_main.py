#!/usr/bin/env python3
"""主实验：train.py baselines + diffusion → evaluate.py（test + hidden_test）。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import config as cfg
from progress import log


def run(cmd: list[str], cwd: Path) -> None:
    log(">> " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Main experiment pipeline.")
    p.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.OUTPUT_DIR,
        help=f"固定为 {cfg.OUTPUT_DIR}，勿在仓库根另建 outputs_*",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=cfg.DIFFUSION_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.DIFFUSION_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.DIFFUSION_LR)
    p.add_argument("--num-steps", type=int, default=cfg.DIFFUSION_NUM_STEPS)
    p.add_argument("--hidden-dim", type=int, default=cfg.DIFFUSION_HIDDEN_DIM)
    p.add_argument("--class-emb-dim", type=int, default=cfg.DIFFUSION_CLASS_EMB_DIM)
    p.add_argument("--gpus", type=int, default=cfg.DEFAULT_GPUS)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["test", "hidden_test"],
        choices=["test", "hidden_test"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable
    train_py = str(root / "train.py")

    skip_train = args.skip_train and not args.force
    eval_only = args.eval_only and not args.force
    if not eval_only and not skip_train:
        run(
            [py, train_py, "baselines", "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir), "--seed", str(args.seed)],
            cwd=root,
        )
        diff_tail = [
            train_py,
            "diffusion",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.output_dir),
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--num-steps",
            str(args.num_steps),
            "--hidden-dim",
            str(args.hidden_dim),
            "--class-emb-dim",
            str(args.class_emb_dim),
            "--device",
            args.device,
            "--precision",
            cfg.DIFFUSION_PRECISION,
            "--num-workers",
            str(cfg.DIFFUSION_NUM_WORKERS),
            "--warmup-epochs",
            str(cfg.DIFFUSION_WARMUP_EPOCHS),
            "--ema-decay",
            str(cfg.DIFFUSION_EMA_DECAY),
            "--grad-clip",
            str(cfg.DIFFUSION_GRAD_CLIP),
        ]
        if args.compile:
            diff_tail.append("--compile")
        if args.gpus > 1:
            diff_cmd = ["torchrun", "--standalone", "--nproc_per_node", str(args.gpus), *diff_tail]
        else:
            diff_cmd = [py, *diff_tail]
        run(diff_cmd, cwd=root)

    eval_cmd = [
        py,
        str(root / "evaluate.py"),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--split",
        *args.split,
    ]
    if args.no_figures:
        eval_cmd.append("--no-figures")
    run(eval_cmd, cwd=root)
    log(f"主实验完成 → {args.output_dir / 'metrics' / cfg.ALIAS_MAIN_EVAL}")


if __name__ == "__main__":
    main()
