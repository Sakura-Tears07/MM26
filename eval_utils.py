from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_utils import CLASS_NAMES, class_counts, load_dataset
from diffusion_model import ConditionalDiffusion2D
from metrics import compute_all_metrics, macro_average, negative_log_likelihood_kde_on_generated


def evaluate_by_class(
    gen_x: np.ndarray,
    gen_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    nll_per_label: dict[int, float] | None = None,
    nll_from_generated_kde: bool = False,
    seed: int = 42,
) -> dict[str, dict]:
    per_class: dict[str, dict[str, float]] = {}
    for label, class_name in enumerate(CLASS_NAMES):
        real = test_x[test_y == label]
        fake = gen_x[gen_y == label]
        nll = None
        if nll_per_label is not None and label in nll_per_label:
            nll = nll_per_label[label]
        elif nll_from_generated_kde:
            nll = negative_log_likelihood_kde_on_generated(real, fake, bandwidth=0.1 if label == 3 else 0.12)
        per_class[class_name] = compute_all_metrics(
            real,
            fake,
            nll=nll,
            include_mode_coverage=(label == 0),
            seed=seed + label,
        )
    out = dict(per_class)
    out["macro_avg"] = macro_average(per_class)
    return out


def evaluate_diffusion_checkpoint(
    checkpoint: str | Path,
    data_dir: str | Path,
    *,
    device: str = "cuda",
    seed: int = 42,
    save_figure: Path | None = None,
    save_samples_dir: Path | None = None,
) -> dict[str, dict]:
    """加载 Diffusion checkpoint，在 test 上采样并返回按类 + macro 指标。"""
    dataset = load_dataset(Path(data_dir))
    test_x, test_y = dataset["test_x"], dataset["test_y"]
    counts = class_counts(test_y)

    model = ConditionalDiffusion2D.load(checkpoint, device=device)
    gen_x, gen_y = model.sample_from_counts(counts, seed=seed + 1)
    result = evaluate_by_class(gen_x, gen_y, test_x, test_y, nll_from_generated_kde=True, seed=seed)

    if save_samples_dir is not None:
        save_samples_dir = Path(save_samples_dir)
        save_samples_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_samples_dir / "diffusion_samples.npy", gen_x)
        np.save(save_samples_dir / "diffusion_labels.npy", gen_y)

    if save_figure is not None:
        plot_gen_vs_real(save_figure, test_x, test_y, gen_x, gen_y, title="Diffusion")

    return result


def plot_gen_vs_real(
    save_path: Path,
    real_x: np.ndarray,
    real_y: np.ndarray,
    gen_x: np.ndarray,
    gen_y: np.ndarray,
    title: str = "",
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for label, class_name in enumerate(CLASS_NAMES):
        r = real_x[real_y == label]
        f = gen_x[gen_y == label]
        axes[0, label].scatter(r[:, 0], r[:, 1], s=4, alpha=0.55, linewidths=0)
        axes[1, label].scatter(f[:, 0], f[:, 1], s=4, alpha=0.55, linewidths=0)
        axes[0, label].set_title(f"Real: {class_name}")
        axes[1, label].set_title(f"Gen: {class_name}")
        for ax in (axes[0, label], axes[1, label]):
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
