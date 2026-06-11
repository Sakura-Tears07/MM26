# 超参扫描说明

本目录由 `extensions.py sweep` 生成，网格定义见仓库根目录 `sweep_grid.json`。

**当前默认**与主实验一致：`resfourier_v4`、cosine schedule、per-class norm、spiral 过采样与加权损失（见 `config.py`）。

扫描变量：`batch_size` × `lr` × `num_steps` × `hidden_dim`（共 18 trials）。

报告用途：

- 对比 `num_steps`、`lr`、`hidden_dim` 对 macro MMD / spiral precision 的影响；
- 主实验结论以 `outputs/checkpoints/diffusion.pt` 与 `outputs/metrics/main_evaluation.json` 为准；
- 若目录内为**旧版 MLP / v3** 结果，请 `extensions.py sweep --force` 重跑后再写入报告。
