#!/usr/bin/env python3
"""训练入口：python train.py baselines | python train.py diffusion"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import config as cfg

ROOT = Path(__file__).resolve().parent


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=cfg.OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)


def parse_args(argv: list[str] | None = None) -> tuple[str, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Train KDE/GMM baselines or conditional diffusion.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("baselines", help="Train KDE and GMM per class")
    _add_common(p_base)
    p_base.add_argument("--kde-bandwidths", type=float, nargs="+", default=cfg.KDE_BANDWIDTHS)
    p_base.add_argument("--gmm-components", type=int, nargs="+", default=cfg.GMM_COMPONENTS)
    p_base.add_argument("--gmm-reg-covar", type=float, default=cfg.GMM_REG_COVAR)

    p_diff = sub.add_parser("diffusion", help="Train conditional diffusion (supports torchrun DDP)")
    _add_common(p_diff)
    p_diff.add_argument("--epochs", type=int, default=cfg.DIFFUSION_EPOCHS)
    p_diff.add_argument("--batch-size", type=int, default=cfg.DIFFUSION_BATCH_SIZE)
    p_diff.add_argument("--lr", type=float, default=cfg.DIFFUSION_LR)
    p_diff.add_argument("--num-steps", type=int, default=cfg.DIFFUSION_NUM_STEPS)
    p_diff.add_argument("--hidden-dim", type=int, default=cfg.DIFFUSION_HIDDEN_DIM)
    p_diff.add_argument("--time-emb-dim", type=int, default=cfg.DIFFUSION_TIME_EMB_DIM)
    p_diff.add_argument("--class-emb-dim", type=int, default=cfg.DIFFUSION_CLASS_EMB_DIM)
    p_diff.add_argument("--device", type=str, default="cuda")
    p_diff.add_argument("--precision", type=str, default=cfg.DIFFUSION_PRECISION, choices=["bf16", "fp16", "fp32"])
    p_diff.add_argument("--num-workers", type=int, default=cfg.DIFFUSION_NUM_WORKERS)
    p_diff.add_argument("--accum-steps", type=int, default=1)
    p_diff.add_argument("--warmup-epochs", type=int, default=cfg.DIFFUSION_WARMUP_EPOCHS)
    p_diff.add_argument("--ema-decay", type=float, default=cfg.DIFFUSION_EMA_DECAY)
    p_diff.add_argument("--grad-clip", type=float, default=cfg.DIFFUSION_GRAD_CLIP)
    p_diff.add_argument("--compile", action="store_true")
    p_diff.add_argument("--disable-tf32", action="store_true")
    p_diff.add_argument("--spiral-oversample", type=int, default=cfg.DIFFUSION_SPIRAL_OVERSAMPLE)
    p_diff.add_argument("--spiral-loss-weight", type=float, default=cfg.DIFFUSION_SPIRAL_LOSS_WEIGHT)
    p_diff.add_argument("--noise-schedule", type=str, default=cfg.DIFFUSION_NOISE_SCHEDULE, choices=["linear", "cosine"])
    p_diff.add_argument("--arch", type=str, default=cfg.DIFFUSION_ARCH, choices=["sinusoidal_v2", "resfourier_v3"])
    p_diff.add_argument("--fourier-freqs", type=int, default=cfg.DIFFUSION_FOURIER_FREQS)
    p_diff.add_argument("--res-blocks", type=int, default=cfg.DIFFUSION_RES_BLOCKS)
    p_diff.add_argument(
        "--no-posterior-var",
        action="store_true",
        help="采样时用 sqrt(beta) 而非 DDPM 后验方差",
    )

    ns = parser.parse_args(argv)
    return ns.command, ns


def train_baselines(args: argparse.Namespace) -> None:
    import numpy as np

    from data_utils import CLASS_NAMES, load_dataset, set_seed
    from gmm_model import GMMPerClass
    from kde_model import KDEPerClass
    from progress import log

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_dir)
    train_x, train_y = dataset["train_x"], dataset["train_y"]
    log(f"训练基线 | train={train_x.shape}")

    kde = KDEPerClass(bandwidth_candidates=list(args.kde_bandwidths))
    kde.fit(train_x, train_y)
    kde_path = ckpt_dir / "kde.pkl"
    kde.save(kde_path)
    log(f"KDE 完成，带宽={kde.best_bandwidth}")

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
        "class_names": CLASS_NAMES,
    }
    with (metrics_dir / "baseline_train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


def train_diffusion(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    from data_utils import load_dataset, set_seed
    from diffusion_model import (
        ConditionalDiffusion2D,
        DiffusionConfig,
        build_denoiser,
        compute_class_stats,
        make_beta_schedule,
    )
    from progress import ProgressTracker, log as prog_log

    def is_dist() -> bool:
        return "RANK" in os.environ and "WORLD_SIZE" in os.environ

    def setup_dist() -> tuple[int, int, int]:
        if not is_dist():
            return 0, 1, 0
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank

    def cleanup_dist() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def get_amp_dtype(precision: str) -> torch.dtype | None:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return None

    def log_rank(rank: int, msg: str) -> None:
        prog_log(msg, rank=rank)

    def update_ema(ema_state: dict, model: torch.nn.Module, decay: float) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ema_state[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    rank, world_size, local_rank = setup_dist()
    set_seed(args.seed + rank)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
    else:
        device = torch.device("cpu")

    if not args.disable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_dir)
    train_x_np = dataset["train_x"].astype(np.float32)
    train_y_np = dataset["train_y"].astype(np.int64)
    if args.spiral_oversample > 1:
        spiral_mask = train_y_np == cfg.SPIRAL_LABEL
        extra_x = np.repeat(train_x_np[spiral_mask], args.spiral_oversample - 1, axis=0)
        extra_y = np.repeat(train_y_np[spiral_mask], args.spiral_oversample - 1, axis=0)
        train_x_np = np.concatenate([train_x_np, extra_x], axis=0)
        train_y_np = np.concatenate([train_y_np, extra_y], axis=0)
        if rank == 0:
            log_rank(rank, f"SPIRAL 过采样 x{args.spiral_oversample}，训练集 -> {len(train_x_np)}")
    num_classes = int(train_y_np.max()) + 1

    config = DiffusionConfig(
        num_steps=args.num_steps,
        hidden_dim=args.hidden_dim,
        time_emb_dim=args.time_emb_dim,
        class_emb_dim=args.class_emb_dim,
        beta_start=cfg.DIFFUSION_BETA_START,
        beta_end=cfg.DIFFUSION_BETA_END,
        arch_version=args.arch,
        noise_schedule=args.noise_schedule,
        use_posterior_var=not args.no_posterior_var,
        fourier_freqs=args.fourier_freqs,
        num_res_blocks=args.res_blocks,
    )
    if rank == 0:
        log_rank(
            rank,
            f"arch={config.arch_version}, schedule={config.noise_schedule}, "
            f"posterior_var={config.use_posterior_var}, spiral_loss_w={args.spiral_loss_weight}",
        )

    train_x_cpu = torch.from_numpy(train_x_np)
    train_y_cpu = torch.from_numpy(train_y_np)
    class_mean, class_std = compute_class_stats(train_x_np, train_y_np, num_classes)
    class_mean = class_mean.to(device)
    class_std = class_std.to(device)
    if rank == 0:
        log_rank(rank, "按类归一化 (per-class normalization)")

    ds = TensorDataset(train_x_cpu, train_y_cpu)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    steps_per_epoch = len(loader)
    effective_batch_size = args.batch_size * world_size * max(1, args.accum_steps)
    if rank == 0:
        log_rank(
            rank,
            f"N={len(train_x_cpu)}, batch={args.batch_size}, world={world_size}, "
            f"eff_batch={effective_batch_size}, steps/epoch={steps_per_epoch}",
        )
    epoch_tracker = ProgressTracker(args.epochs, desc="Diffusion训练", rank=rank)

    denoiser = build_denoiser(config.arch_version, num_classes, config).to(device)
    if args.compile and hasattr(torch, "compile"):
        denoiser = torch.compile(denoiser)
    model_for_train = DDP(denoiser, device_ids=[local_rank]) if world_size > 1 else denoiser

    betas = make_beta_schedule(
        config.num_steps,
        config.noise_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    optimizer = torch.optim.AdamW(
        model_for_train.parameters(), lr=args.lr, weight_decay=cfg.DIFFUSION_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))

    amp_dtype = get_amp_dtype(args.precision) if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    losses: list[float] = []
    source_model = model_for_train.module if world_size > 1 else model_for_train
    if hasattr(source_model, "_orig_mod"):
        source_model = source_model._orig_mod
    ema_state = {
        name: param.detach().clone()
        for name, param in source_model.named_parameters()
        if param.requires_grad
    }

    model_for_train.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            mean_b = class_mean[y_batch]
            std_b = class_std[y_batch]
            x_batch = (x_batch - mean_b) / std_b
            bsz = x_batch.size(0)
            t = torch.randint(0, config.num_steps, (bsz,), device=device)
            alpha_bar_t = alpha_bars[t].unsqueeze(-1)
            noise = torch.randn_like(x_batch)
            x_noisy = torch.sqrt(alpha_bar_t) * x_batch + torch.sqrt(1.0 - alpha_bar_t) * noise

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_noise = model_for_train(x_noisy, t, y_batch)
                per_sample = ((pred_noise - noise) ** 2).mean(dim=1)
                weights = torch.ones_like(per_sample)
                if args.spiral_loss_weight != 1.0:
                    weights[y_batch == cfg.SPIRAL_LABEL] = args.spiral_loss_weight
                loss = (per_sample * weights).mean() / max(1, args.accum_steps)

            should_step = ((step_idx + 1) % max(1, args.accum_steps) == 0) or ((step_idx + 1) == len(loader))
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if should_step:
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_for_train.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema_state, source_model, args.ema_decay)
            else:
                loss.backward()
                if should_step:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_for_train.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema_state, source_model, args.ema_decay)

            running_loss += float(loss.item()) * bsz * max(1, args.accum_steps)
            n_seen += bsz

        if epoch < args.warmup_epochs:
            warmup_scale = float(epoch + 1) / float(max(1, args.warmup_epochs))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup_scale
        else:
            scheduler.step()
        local_epoch_loss = torch.tensor([running_loss, float(n_seen)], dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(local_epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss = float(local_epoch_loss[0].item() / max(local_epoch_loss[1].item(), 1.0))
        losses.append(epoch_loss)
        epoch_tracker.update(extra=f"loss={epoch_loss:.4f}")

    epoch_tracker.done("保存 checkpoint")

    if rank == 0:
        ckpt_path = ckpt_dir / "diffusion.pt"
        runtime_model = ConditionalDiffusion2D(config=config, num_classes=num_classes, device="cpu")
        trained_state_dict = source_model.state_dict()
        for name, param in runtime_model.model.named_parameters():
            if name in ema_state:
                trained_state_dict[name] = ema_state[name].cpu()
        runtime_model.model.load_state_dict(trained_state_dict)
        runtime_model.set_class_stats(class_mean.cpu(), class_std.cpu())
        runtime_model.save(ckpt_path)
        runtime_model.save_loss_curve(losses, metrics_dir / "diffusion_loss.json")

        summary = {
            "diffusion_checkpoint": str(ckpt_path),
            "epochs": args.epochs,
            "batch_size_per_gpu": args.batch_size,
            "world_size": world_size,
            "effective_batch_size": effective_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "accum_steps": args.accum_steps,
            "lr": args.lr,
            "num_steps": args.num_steps,
            "precision": args.precision if device.type == "cuda" else "fp32",
            "ema_decay": args.ema_decay,
            "warmup_epochs": args.warmup_epochs,
            "norm_mode": "per_class",
            "hidden_dim": args.hidden_dim,
            "time_emb_dim": args.time_emb_dim,
            "class_emb_dim": args.class_emb_dim,
            "beta_start": config.beta_start,
            "beta_end": config.beta_end,
            "noise_schedule": cfg.DIFFUSION_NOISE_SCHEDULE,
            "prediction_target": cfg.DIFFUSION_PREDICTION_TARGET,
            "weight_decay": cfg.DIFFUSION_WEIGHT_DECAY,
            "spiral_oversample": args.spiral_oversample,
            "spiral_loss_weight": args.spiral_loss_weight,
            "arch_version": config.arch_version,
            "fourier_freqs": config.fourier_freqs,
            "num_res_blocks": config.num_res_blocks,
            "use_posterior_var": config.use_posterior_var,
            "final_loss": losses[-1],
        }
        with (metrics_dir / "diffusion_train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(json.dumps(summary, indent=2))

    cleanup_dist()


def main(argv: list[str] | None = None) -> None:
    command, args = parse_args(argv)
    if command == "baselines":
        train_baselines(args)
    else:
        train_diffusion(args)


if __name__ == "__main__":
    main()
