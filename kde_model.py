from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.neighbors import KernelDensity

from data_utils import ClassNormStats


SPIRAL_LABEL = 3


@dataclass
class KDEPerClass:
    bandwidth_candidates: list[float] = field(
        default_factory=lambda: [0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
    )
    class_bandwidth_candidates: dict[int, list[float]] = field(
        default_factory=lambda: {
            SPIRAL_LABEL: [0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12],
        }
    )
    models: dict[int, KernelDensity] = field(default_factory=dict)
    best_bandwidth: dict[int, float] = field(default_factory=dict)
    norm_stats: ClassNormStats | None = None

    def _bandwidths_for_label(self, label: int | None) -> list[float]:
        if label is not None and label in self.class_bandwidth_candidates:
            return list(self.class_bandwidth_candidates[label])
        return list(self.bandwidth_candidates)

    def _fit_one_class(self, x: np.ndarray, label: int | None = None) -> tuple[KernelDensity, float]:
        candidates = self._bandwidths_for_label(label)
        best_model = None
        best_bw = None
        best_ll = -np.inf
        for bw in candidates:
            model = KernelDensity(kernel="gaussian", bandwidth=bw)
            model.fit(x)
            ll = float(model.score(x) / len(x))
            if ll > best_ll:
                best_ll = ll
                best_bw = bw
                best_model = model
        assert best_model is not None and best_bw is not None
        return best_model, best_bw

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        norm_stats: ClassNormStats | None = None,
    ) -> None:
        """在（可选）按类归一化后的坐标上拟合 KDE。"""
        self.models.clear()
        self.best_bandwidth.clear()
        self.norm_stats = norm_stats
        x_fit = x
        if norm_stats is not None:
            x_fit = norm_stats.normalize(x, y)
        labels = np.unique(y)
        for label in labels:
            class_x = x_fit[y == label]
            model, bw = self._fit_one_class(class_x, label=int(label))
            self.models[int(label)] = model
            self.best_bandwidth[int(label)] = float(bw)

    def sample_by_label(self, label: int, n_samples: int, seed: int = 0) -> np.ndarray:
        if label not in self.models:
            raise KeyError(f"Label {label} not found in KDE models")
        samples = self.models[label].sample(n_samples=n_samples, random_state=seed).astype(np.float32)
        if self.norm_stats is not None:
            samples = self.norm_stats.denormalize_by_label(samples, label)
        return samples

    def sample_from_counts(self, counts: dict[int, int], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        rng = np.random.default_rng(seed)
        for label in sorted(counts.keys()):
            sample_seed = int(rng.integers(1, 2**31 - 1))
            x_part = self.sample_by_label(label, counts[label], seed=sample_seed)
            y_part = np.full(counts[label], label, dtype=np.int64)
            xs.append(x_part)
            ys.append(y_part)
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        order = rng.permutation(len(y))
        return x[order], y[order]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def negative_log_likelihood(self, x: np.ndarray, y: np.ndarray) -> float:
        x_eval = x
        if self.norm_stats is not None:
            x_eval = self.norm_stats.normalize(x, y)
        nlls: list[float] = []
        for label in np.unique(y):
            mask = y == label
            pts = np.asarray(x_eval[mask], dtype=np.float64)
            if len(pts) == 0:
                continue
            lp = self.models[int(label)].score_samples(pts)
            nlls.append(float(-np.mean(lp)))
        return float(np.mean(nlls)) if nlls else float("inf")

    @staticmethod
    def load(path: str | Path) -> "KDEPerClass":
        with Path(path).open("rb") as f:
            return pickle.load(f)
