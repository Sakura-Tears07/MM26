#!/usr/bin/env python3
"""主实验：训练 KDE / GMM / Diffusion → 采样 → 指标评估。"""
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
    p.add_argument("--output-dir", type=Path, default=cfg.OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=cfg.DIFFUSION_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.DIFFUSION_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.DIFFUSION_LR)
    p.add_argument("--num-steps", type=int, default=cfg.DIFFUSION_NUM_STEPS)
    p.add_argument("--hidden-dim", type=int, default=cfg.DIFFUSION_HIDDEN_DIM)
    p.add_argument("--class-emb-dim", type=int, default=cfg.DIFFUSION_CLASS_EMB_DIM)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable

    if not args.eval_only and not args.skip_train:
        run(
            [py, "train_baselines.py", "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir), "--seed", str(args.seed)],
            cwd=root,
        )
        diff_cmd = [
            py,
            "train_diffusion.py",
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
            "bf16",
            "--num-workers",
            "8",
            "--warmup-epochs",
            str(cfg.DIFFUSION_WARMUP_EPOCHS),
            "--ema-decay",
            str(cfg.DIFFUSION_EMA_DECAY),
            "--grad-clip",
            str(cfg.DIFFUSION_GRAD_CLIP),
        ]
        if args.compile:
            diff_cmd.append("--compile")
        if args.gpus > 1:
            diff_cmd = ["torchrun", "--standalone", "--nproc_per_node", str(args.gpus), *diff_cmd[1:]]
        run(diff_cmd, cwd=root)

    eval_cmd = [
        py,
        "eval_main.py",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.no_figures:
        eval_cmd.append("--no-figures")
    run(eval_cmd, cwd=root)
    log(f"主实验完成。指标: {args.output_dir / 'metrics' / 'evaluation.json'}")


if __name__ == "__main__":
    main()
