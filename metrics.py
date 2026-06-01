from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity


def _median_heuristic_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    merged = np.concatenate([x, y], axis=0)
    pairwise = cdist(merged, merged, metric="euclidean")
    median = np.median(pairwise[pairwise > 0.0])
    return float(max(median, 1e-3))


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    if sigma is None:
        sigma = _median_heuristic_bandwidth(x, y)
    gamma = 1.0 / (2.0 * sigma * sigma)

    k_xx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))

    m = x.shape[0]
    n = y.shape[0]
    return float(k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2.0 * k_xy.sum() / (m * n))


def wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 512,
    seed: int = 42,
) -> float:
    """2D 经验 Wasserstein-1（等质量子采样 + 最优匹配）。"""
    n = min(len(x), len(y), max_samples)
    rng = np.random.default_rng(seed)
    idx_x = rng.choice(len(x), n, replace=len(x) < n)
    idx_y = rng.choice(len(y), n, replace=len(y) < n)
    xs = x[idx_x]
    ys = y[idx_y]
    cost = cdist(xs, ys, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def sliced_wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 128,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    projections = rng.normal(size=(n_projections, dim))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    distances = []
    for vec in projections:
        px = np.sort(x @ vec)
        py = np.sort(y @ vec)
        distances.append(np.mean(np.abs(px - py)))
    return float(np.mean(distances))


def _real_manifold_radius(real_x: np.ndarray, quantile: float = 0.95) -> float:
    real_pair = cdist(real_x, real_x, metric="euclidean")
    np.fill_diagonal(real_pair, np.inf)
    return float(np.quantile(np.min(real_pair, axis=1), quantile))


def coverage_score(real_x: np.ndarray, gen_x: np.ndarray, quantile: float = 0.95) -> float:
    radius = _real_manifold_radius(real_x, quantile)
    cross = cdist(real_x, gen_x, metric="euclidean")
    covered = np.min(cross, axis=1) <= radius
    return float(np.mean(covered))


def precision_score(real_x: np.ndarray, gen_x: np.ndarray, quantile: float = 0.95) -> float:
    """Precision：生成样本落在真实流形邻域内的比例。"""
    radius = _real_manifold_radius(real_x, quantile)
    cross = cdist(gen_x, real_x, metric="euclidean")
    precise = np.min(cross, axis=1) <= radius
    return float(np.mean(precise))


def mode_coverage_gmm(real_x: np.ndarray, gen_x: np.ndarray, n_modes: int = 8) -> float:
    real_theta = np.mod(np.arctan2(real_x[:, 1], real_x[:, 0]), 2 * math.pi)
    gen_theta = np.mod(np.arctan2(gen_x[:, 1], gen_x[:, 0]), 2 * math.pi)
    bins = np.linspace(0.0, 2.0 * math.pi, n_modes + 1)

    real_hist, _ = np.histogram(real_theta, bins=bins)
    gen_hist, _ = np.histogram(gen_theta, bins=bins)

    active_real = real_hist > max(3, int(0.02 * len(real_x)))
    active_gen = gen_hist > max(3, int(0.01 * len(gen_x)))

    denom = np.sum(active_real)
    if denom == 0:
        return 0.0
    return float(np.sum(active_real & active_gen) / denom)


def negative_log_likelihood_kde(x: np.ndarray, kde: KernelDensity) -> float:
    return float(-np.mean(kde.score_samples(np.asarray(x, dtype=np.float64))))


def negative_log_likelihood_kde_on_generated(
    test_x: np.ndarray,
    gen_x: np.ndarray,
    bandwidth: float = 0.12,
) -> float:
    """对无显式密度的模型：在生成样本上拟合 KDE，在测试集上估计 NLL。"""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(np.asarray(gen_x, dtype=np.float64))
    return negative_log_likelihood_kde(test_x, kde)


def compute_all_metrics(
    real_x: np.ndarray,
    gen_x: np.ndarray,
    *,
    nll: float | None = None,
    include_mode_coverage: bool = False,
    seed: int = 42,
) -> dict[str, float]:
    out = {
        "mmd": mmd_rbf(real_x, gen_x),
        "wasserstein": wasserstein_distance(real_x, gen_x, seed=seed),
        "swd": sliced_wasserstein_distance(real_x, gen_x, seed=seed),
        "coverage": coverage_score(real_x, gen_x),
        "precision": precision_score(real_x, gen_x),
    }
    if nll is not None:
        out["nll"] = float(nll)
    if include_mode_coverage:
        out["mode_coverage"] = mode_coverage_gmm(real_x, gen_x, n_modes=8)
    return out


def macro_average(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = set()
    for v in per_class.values():
        keys.update(v.keys())
    macro: dict[str, float] = {}
    for k in sorted(keys):
        vals = [v[k] for v in per_class.values() if k in v]
        macro[k] = float(np.mean(vals))
    return macro
