# 超参扫描说明

本目录为 **原始 Diffusion** 在固定网格上的扫描结果（`sweep_grid.json`），训练时未使用改进版默认：

- 未固定 `resfourier_v3` / cosine schedule / posterior variance
- 未使用 spiral 过采样与加权损失

因此 `best_trials.json` 中 MMD 最优 trial 的 spiral precision 仍约 **0.61**，不能与 `outputs/checkpoints/diffusion.pt`（改进版，spiral precision ≈ **0.84**）直接对比。

报告用途：说明「增大 `num_steps` 等对原始模型的影响」，改进版结论以主实验 `outputs/metrics/main_evaluation.json` 为准。
