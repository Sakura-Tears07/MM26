# 鲁棒性实验说明

`ratio_0p000/` 直接复用主实验 `outputs/checkpoints/`（**改进版 Diffusion**）。

`ratio > 0` 时在污染训练集上重训 KDE/GMM/Diffusion；Diffusion 使用 `config.py` 改进默认（cosine、`resfourier_v3`、spiral 过采样等）。

若目录内 Diffusion spiral precision 仍约 **0.62**，说明未用改进 checkpoint 或未 `--force` 重跑。
