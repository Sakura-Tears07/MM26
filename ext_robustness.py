#!/usr/bin/env python3
"""拓展：训练集注入均匀异常点，比较三模型指标随污染比例的变化。"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import config as cfg
from progress import ProgressTracker, log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    p.add_argument("--main-output-dir", type=Path, default=cfg.OUTPUT_DIR, help="主实验 outputs，ratio=0 复用其 checkpoint")
    p.add_argument("--output-root", type=Path, default=Path("outputs/ext_robustness"))
    p.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outlier-range", type=float, default=8.0)
    p.add_argument("--epochs", type=int, default=cfg.DIFFUSION_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.DIFFUSION_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.DIFFUSION_LR)
    p.add_argument("--num-steps", type=int, default=cfg.DIFFUSION_NUM_STEPS)
    p.add_argument("--hidden-dim", type=int, default=cfg.DIFFUSION_HIDDEN_DIM)
    p.add_argument("--class-emb-dim", type=int, default=cfg.DIFFUSION_CLASS_EMB_DIM)
    p.add_argument("--ema-decay", type=float, default=cfg.DIFFUSION_EMA_DECAY)
    p.add_argument("--warmup-epochs", type=int, default=cfg.DIFFUSION_WARMUP_EPOCHS)
    p.add_argument("--grad-clip", type=float, default=cfg.DIFFUSION_GRAD_CLIP)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="bf16")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--force", action="store_true", help="忽略已有 summary，全部重跑")
    return p.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    log(">> " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


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
        raise FileNotFoundError(f"主实验 checkpoint 不存在: {src}，请先运行 run_main.py")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("kde.pkl", "gmm.pkl", "diffusion.pt"):
        shutil.copy2(src / name, dst / name)


def diffusion_train_cmd(args: argparse.Namespace, py: str, data_dir: Path, model_out: Path) -> list[str]:
    cmd = [
        py,
        "train_diffusion.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(model_out),
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
        args.precision,
        "--num-workers",
        str(args.num_workers),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--ema-decay",
        str(args.ema_decay),
        "--grad-clip",
        str(args.grad_clip),
    ]
    if args.compile:
        cmd.append("--compile")
    if args.gpus > 1:
        cmd = ["torchrun", "--standalone", "--nproc_per_node", str(args.gpus), *cmd[1:]]
    return cmd


def append_summary_rows(rows: list[dict], metrics: dict, ratio: float) -> list[dict]:
    rows = [r for r in rows if r.get("outlier_ratio") != ratio]
    for model_name in ("kde", "gmm", "diffusion"):
        payload = metrics[model_name]
        for class_name, vals in payload.items():
            rows.append({"outlier_ratio": ratio, "model": model_name, "class": class_name, **vals})
    return rows


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable
    output_root = args.output_root.resolve()
    main_out = (root / args.main_output_dir).resolve()
    base_data = (root / args.data_dir).resolve()
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
        model_out = run_dir / "outputs"
        eval_path = model_out / "metrics" / "evaluation.json"

        if not args.force and any(r.get("outlier_ratio") == ratio for r in rows) and eval_path.exists():
            log(f"跳过 ratio={ratio}")
            tracker.update(extra=f"skip {ratio}")
            continue

        log(f"===== outlier ratio {ratio:.1%} =====")
        if ratio == 0.0:
            copy_clean_data(base_data, data_dir)
            model_out.mkdir(parents=True, exist_ok=True)
            link_main_checkpoints(main_out, model_out)
            log(f"ratio=0：复用主实验 checkpoint（{main_out / 'checkpoints'}）")
        else:
            build_corrupted_data(base_data, data_dir, ratio, args.seed, args.outlier_range)
            gmm_reg = 1e-3 if ratio <= 0.01 else (1e-2 if ratio <= 0.05 else 5e-2)
            run_command(
                [
                    py,
                    "train_baselines.py",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(model_out),
                    "--seed",
                    str(args.seed),
                    "--gmm-reg-covar",
                    str(gmm_reg),
                ],
                cwd=root,
            )
            run_command(diffusion_train_cmd(args, py, data_dir, model_out), cwd=root)

        run_command(
            [
                py,
                "eval_main.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(model_out),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--no-figures",
            ],
            cwd=root,
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
            writer = csv.DictWriter(f, fieldnames=list(macro_rows[0].keys()))
            writer.writeheader()
            writer.writerows(macro_rows)
    if rows:
        with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    tracker.done(str(output_root))


if __name__ == "__main__":
    main()
