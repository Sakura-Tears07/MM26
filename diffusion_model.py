from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


from data_utils import compute_class_stats as _compute_class_stats_np


def compute_class_stats(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """每类独立的 mean/std，形状均为 (num_classes, 2)。"""
    mean_np, std_np = _compute_class_stats_np(x, y, num_classes)
    return torch.from_numpy(mean_np), torch.from_numpy(std_np)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_steps: int = 1000) -> None:
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long
        ang = t.float().unsqueeze(1) * self.freq.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class DenoiseMLP(nn.Module):
    """当前默认结构：正弦时间嵌入 + 按类归一化训练。"""

    arch_version = "sinusoidal_v2"

    def __init__(
        self,
        num_classes: int = 4,
        hidden_dim: int = 256,
        time_emb_dim: int = 64,
        class_emb_dim: int = 32,
        max_steps: int = 200,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim, max_steps=max_steps)
        self.time_proj = nn.Linear(time_emb_dim, time_emb_dim)
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        in_dim = 2 + time_emb_dim + class_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_proj(self.time_embed(t))
        y_emb = self.class_embed(y)
        h = torch.cat([x_t, t_emb, y_emb], dim=-1)
        return self.net(h)


class DenoiseMLPLegacy(nn.Module):
    """兼容 sweep 早期保存的 checkpoint（nn.Embedding 时间步）。"""

    arch_version = "embedding_v1"

    def __init__(
        self,
        num_classes: int = 4,
        hidden_dim: int = 128,
        time_emb_dim: int = 64,
        class_emb_dim: int = 16,
        max_steps: int = 200,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Embedding(max_steps, time_emb_dim)
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        in_dim = 2 + time_emb_dim + class_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        y_emb = self.class_embed(y)
        h = torch.cat([x_t, t_emb, y_emb], dim=-1)
        return self.net(h)


def detect_arch_version(state_dict: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("blocks.0.cond.") for k in state_dict):
        return "resfourier_v4"
    if "in_proj.weight" in state_dict:
        return "resfourier_v3"
    if "time_proj.weight" in state_dict:
        return "sinusoidal_v2"
    if "time_embed.weight" in state_dict:
        return "embedding_v1"
    raise ValueError("无法识别 diffusion checkpoint 的网络结构")


def build_denoiser(
    arch_version: str,
    num_classes: int,
    config: DiffusionConfig,
) -> nn.Module:
    if arch_version == "sinusoidal_v2":
        return DenoiseMLP(
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            time_emb_dim=config.time_emb_dim,
            class_emb_dim=config.class_emb_dim,
            max_steps=config.num_steps,
        )
    if arch_version == "embedding_v1":
        return DenoiseMLPLegacy(
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            time_emb_dim=config.time_emb_dim,
            class_emb_dim=config.class_emb_dim,
            max_steps=config.num_steps,
        )
    if arch_version == "resfourier_v3":
        return DenoiseMLPResFourier(
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            time_emb_dim=config.time_emb_dim,
            class_emb_dim=config.class_emb_dim,
            max_steps=config.num_steps,
            fourier_freqs=config.fourier_freqs,
            num_res_blocks=config.num_res_blocks,
        )
    if arch_version == "resfourier_v4":
        return DenoiseResFourierV4(
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            time_emb_dim=config.time_emb_dim,
            class_emb_dim=config.class_emb_dim,
            max_steps=config.num_steps,
            fourier_freqs=config.fourier_freqs,
            num_res_blocks=config.num_res_blocks,
        )
    raise ValueError(f"未知 arch_version: {arch_version}")


SPIRAL_LABEL = 3


def make_beta_schedule(
    num_steps: int,
    schedule: str,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """训练与采样共用的 beta 序列。"""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)

    if schedule == "cosine":
        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float32, device=device)
        f = torch.cos(((steps / num_steps) + s) / (1.0 + s) * math.pi * 2.0) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-5, 0.999)

    raise ValueError(f"未知 noise schedule: {schedule!r}")


def fourier_features(x: torch.Tensor, num_freqs: int = 4) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    xb = x[..., None] * freqs
    xb = xb.reshape(x.shape[0], -1)
    return torch.cat([torch.sin(math.pi * xb), torch.cos(math.pi * xb)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FiLMResBlock(nn.Module):
    """FiLM 条件残差块：time/class 嵌入调制 LayerNorm + 两层 MLP。"""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.cond = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class DenoiseResFourierV4(nn.Module):
    """FiLM 条件 ResNet + 坐标 Fourier 特征（默认训练结构）。"""

    arch_version = "resfourier_v4"

    def __init__(
        self,
        num_classes: int = 4,
        hidden_dim: int = 384,
        time_emb_dim: int = 64,
        class_emb_dim: int = 64,
        max_steps: int = 500,
        fourier_freqs: int = 6,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.fourier_freqs = fourier_freqs
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim, max_steps=max_steps)
        self.time_proj = nn.Linear(time_emb_dim, time_emb_dim)
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        cond_dim = time_emb_dim + class_emb_dim
        x_in_dim = 2 + 2 * 2 * fourier_freqs
        in_dim = x_in_dim + cond_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMResBlock(hidden_dim, cond_dim) for _ in range(num_res_blocks)]
        )
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_feat = torch.cat([x_t, fourier_features(x_t, self.fourier_freqs)], dim=-1)
        t_emb = self.time_proj(self.time_embed(t))
        y_emb = self.class_embed(y)
        cond = torch.cat([t_emb, y_emb], dim=-1)
        h = self.in_proj(torch.cat([x_feat, cond], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


class DenoiseMLPResFourier(nn.Module):
    """残差 MLP + 坐标 Fourier 特征（兼容旧 checkpoint）。"""

    arch_version = "resfourier_v3"

    def __init__(
        self,
        num_classes: int = 4,
        hidden_dim: int = 256,
        time_emb_dim: int = 64,
        class_emb_dim: int = 32,
        max_steps: int = 200,
        fourier_freqs: int = 4,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.fourier_freqs = fourier_freqs
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim, max_steps=max_steps)
        self.time_proj = nn.Linear(time_emb_dim, time_emb_dim)
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        x_in_dim = 2 + 2 * 2 * fourier_freqs
        in_dim = x_in_dim + time_emb_dim + class_emb_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_feat = torch.cat([x_t, fourier_features(x_t, self.fourier_freqs)], dim=-1)
        t_emb = self.time_proj(self.time_embed(t))
        y_emb = self.class_embed(y)
        h = self.in_proj(torch.cat([x_feat, t_emb, y_emb], dim=-1))
        h = self.blocks(h)
        return self.out(h)


@dataclass
class DiffusionConfig:
    num_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_dim: int = 256
    time_emb_dim: int = 64
    class_emb_dim: int = 32
    arch_version: str = "resfourier_v3"
    noise_schedule: str = "cosine"
    use_posterior_var: bool = True
    fourier_freqs: int = 6
    num_res_blocks: int = 6
    spiral_sample_step_mult: float = 3.0
    max_sample_steps: int = 1000


class ConditionalDiffusion2D:
    def __init__(self, config: DiffusionConfig, num_classes: int = 4, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.norm_mode = "per_class"
        self.arch_version = config.arch_version
        self.model = build_denoiser(self.arch_version, num_classes, config).to(self.device)

        betas = make_beta_schedule(
            config.num_steps,
            getattr(config, "noise_schedule", "linear"),
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=self.device,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        # 兼容旧 checkpoint 的全局统计量
        self.x_mean = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.x_std = torch.ones(2, dtype=torch.float32, device=self.device)
        self.class_mean = torch.zeros(num_classes, 2, dtype=torch.float32, device=self.device)
        self.class_std = torch.ones(num_classes, 2, dtype=torch.float32, device=self.device)

    def set_class_stats(self, class_mean: torch.Tensor, class_std: torch.Tensor) -> None:
        self.norm_mode = "per_class"
        self.class_mean = class_mean.to(self.device)
        self.class_std = class_std.to(self.device)

    def _norm_params(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm_mode == "per_class":
            return self.class_mean[y], self.class_std[y]
        mean = self.x_mean.view(1, 2).expand(y.shape[0], -1)
        std = self.x_std.view(1, 2).expand(y.shape[0], -1)
        return mean, std

    def normalize(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, std = self._norm_params(y)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, std = self._norm_params(y)
        return x * std + mean

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 250,
        batch_size: int = 512,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> list[float]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        class_mean, class_std = compute_class_stats(x, y, self.num_classes)
        self.set_class_stats(class_mean, class_std)

        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y.astype(np.int64)).to(self.device)
        x_norm = self.normalize(x_t, y_t)

        dataset = TensorDataset(x_norm, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        losses: list[float] = []
        self.model.train()

        for _ in range(epochs):
            epoch_loss = 0.0
            n_seen = 0
            for x_batch, y_batch in loader:
                bsz = x_batch.size(0)
                t = torch.randint(0, self.config.num_steps, (bsz,), device=self.device)

                alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
                noise = torch.randn_like(x_batch)
                x_noisy = torch.sqrt(alpha_bar_t) * x_batch + torch.sqrt(1.0 - alpha_bar_t) * noise

                pred_noise = self.model(x_noisy, t, y_batch)
                loss = F.mse_loss(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item()) * bsz
                n_seen += bsz
            losses.append(epoch_loss / max(1, n_seen))
        return losses

    def _schedule_for_steps(self, num_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = make_beta_schedule(
            num_steps,
            getattr(self.config, "noise_schedule", "linear"),
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            device=self.device,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    def _sample_steps_for_label(self, label: int) -> int:
        base = self.config.num_steps
        cap = int(getattr(self.config, "max_sample_steps", 1000))
        if int(label) == SPIRAL_LABEL:
            return int(min(cap, round(base * self.config.spiral_sample_step_mult)))
        return base

    @torch.no_grad()
    def sample_by_label(self, label: int, n_samples: int, seed: int = 42) -> np.ndarray:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        self.model.eval()
        x = torch.randn((n_samples, 2), generator=g, device=self.device)
        y = torch.full((n_samples,), int(label), dtype=torch.long, device=self.device)

        n_steps = self._sample_steps_for_label(label)
        betas, alphas, alpha_bars = self._schedule_for_steps(n_steps)
        train_steps = self.config.num_steps

        for t_idx in range(n_steps - 1, -1, -1):
            # 将稠密采样时间步映射到训练时见过的步索引
            t_train = int(round(t_idx * (train_steps - 1) / max(n_steps - 1, 1)))
            t_train = min(t_train, train_steps - 1)
            t = torch.full((n_samples,), t_train, dtype=torch.long, device=self.device)
            pred_noise = self.model(x, t, y)

            alpha_t = alphas[t_idx]
            alpha_bar_t = alpha_bars[t_idx]
            beta_t = betas[t_idx]

            mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)

            if t_idx > 0:
                z = torch.randn(x.shape, generator=g, device=self.device)
                if getattr(self.config, "use_posterior_var", True):
                    alpha_bar_prev = alpha_bars[t_idx - 1]
                    posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    posterior_var = posterior_var.clamp(min=1e-20)
                    x = mean + torch.sqrt(posterior_var) * z
                else:
                    x = mean + torch.sqrt(beta_t) * z
            else:
                x = mean

        x = self.denormalize(x, y)
        return x.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def sample_from_counts(self, counts: dict[int, int], seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        rng = np.random.default_rng(seed)
        for label in sorted(counts.keys()):
            s = int(rng.integers(1, 2**31 - 1))
            x_part = self.sample_by_label(label=label, n_samples=counts[label], seed=s)
            y_part = np.full(counts[label], label, dtype=np.int64)
            xs.append(x_part)
            ys.append(y_part)
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        order = rng.permutation(len(y))
        return x[order], y[order]

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(self.config),
            "num_classes": self.num_classes,
            "arch_version": getattr(self.model, "arch_version", self.config.arch_version),
            "norm_mode": self.norm_mode,
            "state_dict": self.model.state_dict(),
            "class_mean": self.class_mean.detach().cpu().numpy().tolist(),
            "class_std": self.class_std.detach().cpu().numpy().tolist(),
            # 兼容旧代码读取
            "x_mean": self.class_mean.mean(dim=0).detach().cpu().numpy().tolist(),
            "x_std": self.class_std.mean(dim=0).detach().cpu().numpy().tolist(),
        }
        torch.save(payload, checkpoint_path)

    @staticmethod
    def load(checkpoint_path: str | Path, device: str = "cpu") -> "ConditionalDiffusion2D":
        payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        raw_cfg = dict(payload["config"])
        raw_cfg.setdefault("noise_schedule", "linear")
        raw_cfg.setdefault("use_posterior_var", True)
        raw_cfg.setdefault("fourier_freqs", 6)
        raw_cfg.setdefault("num_res_blocks", 6)
        raw_cfg.setdefault("spiral_sample_step_mult", 3.0)
        raw_cfg.setdefault("max_sample_steps", 1000)
        arch_version = payload.get("arch_version") or raw_cfg.get("arch_version") or detect_arch_version(
            payload["state_dict"]
        )
        raw_cfg["arch_version"] = arch_version
        config = DiffusionConfig(**{k: v for k, v in raw_cfg.items() if k in DiffusionConfig.__dataclass_fields__})
        model = ConditionalDiffusion2D(
            config=config,
            num_classes=payload["num_classes"],
            device=device,
        )
        model.arch_version = arch_version
        model.model = build_denoiser(arch_version, payload["num_classes"], config).to(model.device)
        model.model.load_state_dict(payload["state_dict"])
        if "class_mean" in payload and payload.get("norm_mode") == "per_class":
            model.class_mean = torch.tensor(payload["class_mean"], dtype=torch.float32, device=model.device)
            model.class_std = torch.tensor(payload["class_std"], dtype=torch.float32, device=model.device)
            model.norm_mode = "per_class"
        else:
            model.norm_mode = "global"
            model.x_mean = torch.tensor(payload["x_mean"], dtype=torch.float32, device=model.device)
            model.x_std = torch.tensor(payload["x_std"], dtype=torch.float32, device=model.device)
        return model

    def save_loss_curve(self, losses: list[float], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"losses": losses}, f, indent=2)
            f.write("\n")
