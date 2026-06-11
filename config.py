"""项目唯一默认超参与固定产物路径（脚本 argparse 仅作兜底）。"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
TRAIN_PER_CLASS = 4000
TEST_PER_CLASS = 2000
HIDDEN_PER_CLASS = 2000
SPIRAL_LABEL = 3

# ---------------------------------------------------------------------------
# 产物根目录（仅此一棵，勿在仓库根新建 outputs_*）
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")

# 主实验
CKPT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
SAMPLES_DIR = OUTPUT_DIR / "samples"
FIGURES_DIR = OUTPUT_DIR / "figures"

CKPT_KDE = CKPT_DIR / "kde.pkl"
CKPT_GMM = CKPT_DIR / "gmm.pkl"
CKPT_DIFFUSION = CKPT_DIR / "diffusion.pt"

# 拓展实验（均在 outputs/ 下）
EXT_CONDITIONAL_DIR = OUTPUT_DIR / "conditional"
EXT_SWEEP_DIR = OUTPUT_DIR / "sweep"
EXT_ROBUSTNESS_DIR = OUTPUT_DIR / "robustness"
SWEEP_GRID_FILE = Path("sweep_grid.json")

# 指标文件名（主实验 metrics/ 内）
FILE_EVAL_TEST = "evaluation.json"
FILE_EVAL_TEST_CSV = "evaluation.csv"
FILE_EVAL_HIDDEN = "evaluation_hidden_test.json"
FILE_EVAL_HIDDEN_CSV = "evaluation_hidden_test.csv"
ALIAS_MAIN_EVAL = "main_evaluation.json"
ALIAS_HIDDEN_EVAL = "hidden_test_evaluation.json"
# --diffusion-only 专用，勿覆盖主实验三模型别名
FILE_DIFFUSION_ONLY_TEST = "diffusion_only_evaluation.json"
FILE_DIFFUSION_ONLY_HIDDEN = "diffusion_only_hidden_test_evaluation.json"

# 拓展汇总别名
ALIAS_CONDITIONAL_EVAL = "conditional_evaluation.json"
ALIAS_SWEEP_SUMMARY = "sweep_summary.json"
ALIAS_ROBUSTNESS_SUMMARY = "robustness_summary.json"

# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
KDE_NLL_BW_SPIRAL = 0.10
KDE_NLL_BW_DEFAULT = 0.12
MAX_METRIC_SAMPLES = 2000
WASSERSTEIN_MAX_SAMPLES = 512

# ---------------------------------------------------------------------------
# 统一训练数据（三模型共用）
# ---------------------------------------------------------------------------
TRAIN_SPIRAL_OVERSAMPLE = 4
TRAIN_USE_PER_CLASS_NORM = True

# ---------------------------------------------------------------------------
# 分布式训练默认 GPU 数（torchrun --nproc_per_node）
# ---------------------------------------------------------------------------
DEFAULT_GPUS = 4

# ---------------------------------------------------------------------------
# Diffusion 训练默认
# ---------------------------------------------------------------------------
DIFFUSION_EPOCHS = 600
DIFFUSION_BATCH_SIZE = 32
DIFFUSION_LR = 1e-3
DIFFUSION_NUM_STEPS = 500
DIFFUSION_HIDDEN_DIM = 384
DIFFUSION_TIME_EMB_DIM = 64
DIFFUSION_CLASS_EMB_DIM = 64
DIFFUSION_WARMUP_EPOCHS = 40
DIFFUSION_EMA_DECAY = 0.999
DIFFUSION_GRAD_CLIP = 1.0
DIFFUSION_WEIGHT_DECAY = 1e-4
DIFFUSION_BETA_START = 1e-4
DIFFUSION_BETA_END = 2e-2
DIFFUSION_NOISE_SCHEDULE = "cosine"
DIFFUSION_PREDICTION_TARGET = "epsilon"
DIFFUSION_PRECISION = "bf16"
DIFFUSION_NUM_WORKERS = 8
DIFFUSION_SPIRAL_OVERSAMPLE = TRAIN_SPIRAL_OVERSAMPLE
DIFFUSION_SPIRAL_LOSS_WEIGHT = 3.0
DIFFUSION_ARCH = "resfourier_v4"
DIFFUSION_FOURIER_FREQS = 6
DIFFUSION_RES_BLOCKS = 6
DIFFUSION_USE_POSTERIOR_VAR = True
DIFFUSION_SPIRAL_SAMPLE_STEP_MULT = 3.0
DIFFUSION_MAX_SAMPLE_STEPS = 1000

# 基线
KDE_BANDWIDTHS = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
GMM_COMPONENTS = [1, 2, 4, 6, 8, 10, 12]
GMM_REG_COVAR = 1e-3


def ensure_output_dirs(root: Path = OUTPUT_DIR) -> None:
    """创建主实验常用子目录。"""
    for d in (root / "checkpoints", root / "metrics", root / "samples", root / "figures"):
        d.mkdir(parents=True, exist_ok=True)
