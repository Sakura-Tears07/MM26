from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


SPIRAL_LABEL = 3


@dataclass
class GMMPerClass:
    component_candidates: list[int] = field(default_factory=lambda: [1, 2, 4, 6, 8, 10, 12])
    # 螺旋为 1D 流形，需要更多椭球分量才能沿曲线铺展
    class_component_candidates: dict[int, list[int]] = field(
        default_factory=lambda: {
            SPIRAL_LABEL: [12, 16, 20, 24, 28, 32, 36, 40],
        }
    )
    class_reg_covar: dict[int, float] = field(
        default_factory=lambda: {SPIRAL_LABEL: 5e-4}
    )
    covariance_type: str = "full"
    reg_covar: float = 1e-3
    random_state: int = 42
    models: dict[int, GaussianMixture] = field(default_factory=dict)
    best_components: dict[int, int] = field(default_factory=dict)

    def _candidates_for_label(self, label: int | None) -> list[int]:
        if label is not None and label in self.class_component_candidates:
            return list(self.class_component_candidates[label])
        return list(self.component_candidates)

    def _reg_for_label(self, label: int | None) -> float:
        if label is not None and label in self.class_reg_covar:
            return float(self.class_reg_covar[label])
        return self.reg_covar

    def _fit_one_class(self, x: np.ndarray, label: int | None = None) -> tuple[GaussianMixture, int]:
        x_fit = np.asarray(x, dtype=np.float64)
        candidates = self._candidates_for_label(label)
        reg = self._reg_for_label(label)
        best_model = None
        best_k = None
        best_bic = np.inf
        for k in candidates:
            if k > len(x_fit):
                continue
            model = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                reg_covar=reg,
                random_state=self.random_state,
                max_iter=500,
            )
            try:
                model.fit(x_fit)
                bic = float(model.bic(x_fit))
            except ValueError:
                continue
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_model = model
        if best_model is None:
            # 最后兜底：单分量 + 更强正则
            model = GaussianMixture(
                n_components=1,
                covariance_type=self.covariance_type,
                reg_covar=max(reg, 1e-2),
                random_state=self.random_state,
            )
            model.fit(x_fit)
            return model, 1
        return best_model, int(best_k)

    def fit(self, x: np.ndarray, y: np.ndarray, verbose: bool = False) -> None:
        self.models.clear()
        self.best_components.clear()
        labels = np.unique(y)
        for label in labels:
            class_x = x[y == label]
            if verbose:
                print(f"[GMM] 拟合类别 {label}, n={len(class_x)}", flush=True)
            model, k = self._fit_one_class(class_x, label=int(label))
            self.models[int(label)] = model
            self.best_components[int(label)] = int(k)
            if verbose:
                print(f"[GMM] 类别 {label} 最优分量数 k={k}", flush=True)

    def sample_by_label(self, label: int, n_samples: int) -> np.ndarray:
        if label not in self.models:
            raise KeyError(f"Label {label} not found in GMM models")
        x, _ = self.models[label].sample(n_samples=n_samples)
        return x.astype(np.float32)

    def sample_from_counts(self, counts: dict[int, int], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        rng = np.random.default_rng(seed)
        for label in sorted(counts.keys()):
            x_part = self.sample_by_label(label, counts[label])
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
        nlls: list[float] = []
        for label in np.unique(y):
            mask = y == label
            pts = np.asarray(x[mask], dtype=np.float64)
            if len(pts) == 0:
                continue
            lp = self.models[int(label)].score_samples(pts)
            nlls.append(float(-np.mean(lp)))
        return float(np.mean(nlls)) if nlls else float("inf")

    @staticmethod
    def load(path: str | Path) -> "GMMPerClass":
        with Path(path).open("rb") as f:
            return pickle.load(f)
