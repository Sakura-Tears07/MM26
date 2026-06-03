from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


CLASS_NAMES = ["gaussian_mixture", "ring", "two_moons", "spiral"]


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
