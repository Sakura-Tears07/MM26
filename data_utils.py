from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config as cfg


CLASS_NAMES = ["gaussian_mixture", "ring", "two_moons", "spiral"]


@dataclass
class ClassNormStats:
    """每类独立的 mean/std，形状均为 (num_classes, 2)。"""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, num_classes: int | None = None) -> ClassNormStats:
        if num_classes is None:
            num_classes = int(y.max()) + 1
        mean = np.zeros((num_classes, 2), dtype=np.float32)
        std = np.ones((num_classes, 2), dtype=np.float32)
        for c in range(num_classes):
            pts = x[y == c]
            if len(pts) == 0:
                continue
            mean[c] = pts.mean(axis=0)
            std[c] = np.clip(pts.std(axis=0), 1e-6, None)
        return cls(mean=mean, std=std)

    def normalize(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = x.astype(np.float32, copy=True)
        for c in range(len(self.mean)):
            mask = y == c
            if not np.any(mask):
                continue
            out[mask] = (out[mask] - self.mean[c]) / self.std[c]
        return out

    def denormalize(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = x.astype(np.float32, copy=True)
        for c in range(len(self.mean)):
            mask = y == c
            if not np.any(mask):
                continue
            out[mask] = out[mask] * self.std[c] + self.mean[c]
        return out

    def denormalize_by_label(self, x: np.ndarray, label: int) -> np.ndarray:
        y = np.full(len(x), label, dtype=np.int64)
        return self.denormalize(x, y)


def compute_class_stats(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy 版统计量，供 Diffusion 训练脚本使用。"""
    stats = ClassNormStats.from_arrays(x, y, num_classes)
    return stats.mean, stats.std


def prepare_training_data(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    spiral_oversample: int = 1,
    spiral_label: int = cfg.SPIRAL_LABEL,
    use_per_class_norm: bool = True,
) -> tuple[np.ndarray, np.ndarray, ClassNormStats | None]:
    """统一训练数据：spiral 过采样 + 返回按类归一化统计量（不在此函数内归一化 x）。"""
    x = train_x.astype(np.float32, copy=True)
    y = train_y.astype(np.int64, copy=True)

    if spiral_oversample > 1:
        spiral_mask = y == spiral_label
        extra_x = np.repeat(x[spiral_mask], spiral_oversample - 1, axis=0)
        extra_y = np.repeat(y[spiral_mask], spiral_oversample - 1, axis=0)
        x = np.concatenate([x, extra_x], axis=0)
        y = np.concatenate([y, extra_y], axis=0)

    norm_stats: ClassNormStats | None = None
    if use_per_class_norm:
        num_classes = int(y.max()) + 1
        # 统计量基于过采样后的原始坐标；归一化由各模型训练时各自施加一次
        norm_stats = ClassNormStats.from_arrays(x, y, num_classes)

    return x, y, norm_stats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(data_dir / f"{split}.npy").astype(np.float32)
    y = np.load(data_dir / f"{split}_label.npy").astype(np.int64)
    return x, y


def load_dataset(data_dir: str | Path, *, include_hidden: bool = True) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir)
    train_x, train_y = load_split(data_dir, "train")
    test_x, test_y = load_split(data_dir, "test")

    payload = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }

    if include_hidden and (data_dir / "hidden_test.npy").exists():
        hx, hy = load_split(data_dir, "hidden_test")
        payload["hidden_test_x"] = hx
        payload["hidden_test_y"] = hy

    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            payload["metadata"] = json.load(f)
    else:
        payload["metadata"] = {"class_names": CLASS_NAMES}

    return payload


def get_eval_split(dataset: dict[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray]:
    if split == "test":
        return dataset["test_x"], dataset["test_y"]
    if split == "hidden_test":
        if "hidden_test_x" not in dataset:
            raise FileNotFoundError("hidden_test 不存在，请先运行 generate_data.py")
        return dataset["hidden_test_x"], dataset["hidden_test_y"]
    raise ValueError(f"未知 split: {split!r}，可选 test / hidden_test")


def class_counts(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(labels, counts)}


def split_by_class(x: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    return {label: x[y == label] for label in np.unique(y)}
