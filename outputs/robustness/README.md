# 鲁棒性实验说明

向训练集注入比例 `ratio ∈ {0, 1%, 5%, 10%}` 的均匀异常点（默认范围 `[-8,8]²`），分别重训 KDE / GMM / Diffusion 并评估。

- `ratio_0p000/`：复用主实验 checkpoint（`ratio=0` 时不重训）。
- `ratio > 0`：在污染数据上重训三模型；Diffusion 使用 `config.py` 当前默认（v4 + 统一数据管线）。

汇总：`summary.json`、`summary_macro.csv`。

**须在主实验 `outputs/checkpoints/diffusion.pt` 更新后** 执行：

```bash
python extensions.py robustness --force --ratios 0 0.01 0.05 0.1 --gpus 4 --device cuda --compile
```
