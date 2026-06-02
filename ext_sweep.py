#!/usr/bin/env python3
"""Diffusion 超参扫描：网格见 sweep_grid.json（默认 18 组）。"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

import config as cfg
from eval_utils import evaluate_diffusion_checkpoint
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep diffusion hyper-parameters from sweep_grid.json.")
    p.add_argument("--grid-file", type=Path, default=Path("sweep_grid.json"))
    p.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    p.add_argument("--sweep-dir", type=Path, default=Path("outputs/ext_sweep"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=cfg.DIFFUSION_EPOCHS)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--warmup-epochs", type=int, default=cfg.DIFFUSION_WARMUP_EPOCHS)
    p.add_argument("--grad-clip", type=float, default=cfg.DIFFUSION_GRAD_CLIP)
    p.add_argument("--class-emb-dim", type=int, default=cfg.DIFFUSION_CLASS_EMB_DIM)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--max-trials", type=int, default=0)
    p.add_argument("--random-order", action="store_true")
    p.add_argument("--random-seed", type=int, default=1234)
    p.add_argument("--min-steps-per-epoch", type=int, default=20)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--force", action="store_true", help="忽略已有 summary，重训/重评所有 trial")
    return p.parse_args()


def load_grid(path: Path) -> dict[str, list]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: raw[k] for k in GRID_KEYS}


def run_command(command: list[str], cwd: Path) -> int:
    log(">> " + " ".join(command))
    return int(subprocess.run(command, cwd=cwd).returncode)


def trial_suffix(bs: int, lr: float, ns: int, ema: float, acc: int, hd: int) -> str:
    return f"bs{bs}_lr{lr:.0e}_ns{ns}_ema{ema}_acc{acc}_hd{hd}"


def steps_per_epoch(n_train: int, batch_size: int, gpus: int, accum_steps: int) -> int:
    per_rank = max(1, math.ceil(n_train / max(1, gpus)))
    return max(1, math.ceil(per_rank / max(1, batch_size) / max(1, accum_steps)))


def main() -> None:
    args = parse_args()
    root = ROOT
    grid_path = args.grid_file if args.grid_file.is_absolute() else root / args.grid_file
    grid = load_grid(grid_path)
    sweep_root = args.sweep_dir.resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()

    train_x = np.load(data_dir / "train.npy")
    n_train = len(train_x)
    log(f"网格: {grid_path} | train={train_x.shape}")

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
    valid: list[tuple] = []
    for c in combos:
        bs, _, _, _, acc, _ = c
        if steps_per_epoch(n_train, bs, args.gpus, acc) >= args.min_steps_per_epoch:
            valid.append(c)

    if args.random_order:
        random.Random(args.random_seed).shuffle(valid)
    if args.max_trials > 0:
        valid = valid[: args.max_trials]

    summary_path = sweep_root / "summary.json"
    summary_rows: list[dict] = []
    if summary_path.exists() and not args.force:
        summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
    done = {r["trial"] for r in summary_rows}

    log(
        f"待跑 {len(valid)}/{len(combos)} 组 | lr×steps×hidden = "
        f"{len(grid['lrs'])}×{len(grid['num_steps_list'])}×{len(grid['hidden_dims'])}"
    )
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

        log(f"[{run_idx}/{len(valid)}] {suffix}")

        if not args.eval_only and (args.force or not ckpt.exists()):
            cmd = [
                "torchrun" if args.gpus > 1 else sys.executable,
                *(
                    ["--standalone", "--nproc_per_node", str(args.gpus), "train_diffusion.py"]
                    if args.gpus > 1
                    else ["train_diffusion.py"]
                ),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(trial_dir),
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
                cmd.append("--compile")
            if run_command(cmd, cwd=root) != 0:
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
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")

        train_summary = {}
        ts_path = trial_dir / "metrics" / "diffusion_train_summary.json"
        if ts_path.exists():
            train_summary = json.loads(ts_path.read_text(encoding="utf-8"))

        macro = result["macro_avg"]
        row = {
            "trial": tname,
            "batch_size": bs,
            "lr": lr,
            "num_steps": ns,
            "ema_decay": ema,
            "accum_steps": acc,
            "hidden_dim": hd,
            "steps_per_epoch": train_summary.get("steps_per_epoch"),
            "final_loss": train_summary.get("final_loss"),
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

    with (sweep_root / "best_trials.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_mmd": min(summary_rows, key=lambda r: r["mmd"]),
                "best_precision": max(summary_rows, key=lambda r: r["precision"]),
                "best_spiral_precision": max(
                    summary_rows, key=lambda r: r.get("spiral_precision") or -1.0
                ),
            },
            f,
            indent=2,
        )
        f.write("\n")
    tracker.done(str(summary_path))


if __name__ == "__main__":
    main()
