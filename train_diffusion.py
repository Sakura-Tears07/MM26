#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from data_utils import load_dataset, set_seed
from diffusion_model import (
    ConditionalDiffusion2D,
    DenoiseMLP,
    DiffusionConfig,
    compute_class_stats,
)
from progress import ProgressTracker, log as prog_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional diffusion model on 2D distributions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument("--class-emb-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for denoiser network")
    parser.add_argument("--disable-tf32", action="store_true", help="Disable TF32 matmul/cudnn acceleration")
    parser.add_argument(
        "--spiral-oversample",
        type=int,
        default=1,
        help="训练时复制 SPIRAL 样本的次数（>1 可加强螺旋类学习）",
    )
    return parser.parse_args()


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


def log(rank: int, msg: str) -> None:
    prog_log(msg, rank=rank)


def update_ema(ema_state: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ema_state[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_dist()

    seed = args.seed + rank
    set_seed(seed)

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
    if args.spiral_oversample > 1 and rank == 0:
        spiral_mask = train_y_np == 3
        extra_x = np.repeat(train_x_np[spiral_mask], args.spiral_oversample - 1, axis=0)
        extra_y = np.repeat(train_y_np[spiral_mask], args.spiral_oversample - 1, axis=0)
        train_x_np = np.concatenate([train_x_np, extra_x], axis=0)
        train_y_np = np.concatenate([train_y_np, extra_y], axis=0)
        log(rank, f"SPIRAL 过采样 x{args.spiral_oversample}，训练集 -> {len(train_x_np)}")
    num_classes = int(train_y_np.max()) + 1

    config = DiffusionConfig(
        num_steps=args.num_steps,
        hidden_dim=args.hidden_dim,
        time_emb_dim=args.time_emb_dim,
        class_emb_dim=args.class_emb_dim,
    )

    train_x_cpu = torch.from_numpy(train_x_np)
    train_y_cpu = torch.from_numpy(train_y_np)
    class_mean, class_std = compute_class_stats(train_x_np, train_y_np, num_classes)
    class_mean = class_mean.to(device)
    class_std = class_std.to(device)
    log(rank, "使用按类归一化 (per-class normalization)")

    ds = TensorDataset(train_x_cpu, train_y_cpu)
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
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
    n_train = len(train_x_cpu)
    log(rank, f"训练集样本数={n_train}, 每卡batch={args.batch_size}, world_size={world_size}, accum={args.accum_steps}")
    log(rank, f"有效batch={effective_batch_size}, 每epoch步数={steps_per_epoch}, 总优化步数≈{steps_per_epoch * args.epochs}")
    if rank == 0 and steps_per_epoch < 20:
        suggested_bs = max(16, n_train // max(40 * world_size, 1))
        log(
            rank,
            (
                f"[Warn] 每个 epoch 只有 {steps_per_epoch} 个 step，effective batch={effective_batch_size}。"
                f"建议: --batch-size {suggested_bs}（目标约 40 step/epoch）。"
            ),
        )
    elif rank == 0:
        log(rank, f"[OK] 每 epoch {steps_per_epoch} step，总优化步数约 {steps_per_epoch * args.epochs}")
    epoch_tracker = ProgressTracker(args.epochs, desc="Diffusion训练", rank=rank)

    denoiser = DenoiseMLP(
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        time_emb_dim=config.time_emb_dim,
        class_emb_dim=config.class_emb_dim,
        max_steps=config.num_steps,
    ).to(device)
    if args.compile and hasattr(torch, "compile"):
        denoiser = torch.compile(denoiser)
    model_for_train = DDP(denoiser, device_ids=[local_rank]) if world_size > 1 else denoiser

    betas = torch.linspace(config.beta_start, config.beta_end, config.num_steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    optimizer = torch.optim.AdamW(model_for_train.parameters(), lr=args.lr, weight_decay=1e-4)
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
                loss = F.mse_loss(pred_noise, noise) / max(1, args.accum_steps)

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
        local_epoch_loss = torch.tensor(
            [running_loss, float(n_seen)],
            dtype=torch.float64,
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(local_epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss = float(local_epoch_loss[0].item() / max(local_epoch_loss[1].item(), 1.0))
        losses.append(epoch_loss)

        epoch_tracker.update(extra=f"loss={epoch_loss:.4f}")
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log(rank, f"[Epoch {epoch+1}/{args.epochs}] loss={epoch_loss:.6f}, lr={current_lr:.6e}, steps/epoch={steps_per_epoch}")
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
            "class_emb_dim": args.class_emb_dim,
            "final_loss": losses[-1],
        }
        with (metrics_dir / "diffusion_train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

        print("Diffusion model trained.")
        print(json.dumps(summary, indent=2))

    cleanup_dist()


if __name__ == "__main__":
    main()
