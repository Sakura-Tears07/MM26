"""主实验与拓展实验共用的默认超参。"""
from __future__ import annotations

from pathlib import Path

# 数据（与 generate_data.py 默认一致）
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
TRAIN_PER_CLASS = 4000
TEST_PER_CLASS = 2000
HIDDEN_PER_CLASS = 2000

# Diffusion 训练（与 run_main.py 一致）
DIFFUSION_EPOCHS = 400
DIFFUSION_BATCH_SIZE = 32
DIFFUSION_LR = 1e-3
DIFFUSION_NUM_STEPS = 300
DIFFUSION_HIDDEN_DIM = 256
DIFFUSION_CLASS_EMB_DIM = 32
DIFFUSION_WARMUP_EPOCHS = 30
DIFFUSION_EMA_DECAY = 0.999
DIFFUSION_GRAD_CLIP = 1.0
