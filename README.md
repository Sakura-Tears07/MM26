# MM26 · 二维分布生成建模

KDE / GMM / 条件 Diffusion 在四类 2D 分布上的采样与密度拟合对比。

| 标签 | 分布 |
|------|------|
| 0 | gaussian_mixture |
| 1 | ring |
| 2 | two_moons |
| 3 | spiral |

---

## 环境

```bash
conda activate MM26
pip install -r requirements.txt
python generate_data.py --force
python generate_data.py --force --plot   # 可选 preview
```

---

## 命令

| 脚本 | 作用 |
|------|------|
| `generate_data.py` | 生成 `data/` |
| `train.py` | `baselines` / `diffusion` |
| `evaluate.py` | 三模型评估；`--diffusion-only` 写入独立 JSON，不覆盖主结果 |
| `run_main.py` | 主实验：三模型训练 + test/hidden 评估 |
| `extensions.py` | `conditional` / `sweep` / `robustness` |

```bash
# 主实验（产物一律写入 outputs/，见下方目录表）
python run_main.py --force --gpus 4 --device cuda --compile

python evaluate.py --split test hidden_test --device cuda

# 拓展
python extensions.py conditional --force --device cuda
python extensions.py sweep --force --gpus 4 --device cuda --compile
python extensions.py robustness --force --ratios 0 0.01 0.05 0.1 --gpus 4 --device cuda --compile
```

仅重训 Diffusion（仍写入 `outputs/checkpoints/diffusion.pt`）：

```bash
torchrun --standalone --nproc_per_node 4 train.py diffusion \
  --data-dir data --output-dir outputs \
  --epochs 400 --compile --spiral-oversample 2
python evaluate.py --diffusion-only --split test hidden_test --device cuda
# 产物：metrics/diffusion_only_evaluation.json（不覆盖 main_evaluation.json）
```

超参与路径默认值见 `config.py`。

### 原始 Diffusion vs 改进版（报告务必区分）

| 版本 | 典型 spiral precision (test) | 说明 |
|------|------------------------------|------|
| 原始 | ~0.618 | 默认 MLP + linear schedule；spiral 易呈圆盘状 |
| 改进版 | ~0.841 | `resfourier_v3`、`cosine`、后验方差、`spiral_oversample=2`、`spiral_loss_weight=2.0`、`num_steps=500` |

改进版主实验 macro precision ≈ **0.930**，macro MMD ≈ **0.000328**；coverage ≈ **0.918** 仍低于 KDE/GMM，生成分布偏保守。

`sweep/` 为**原始 Diffusion 超参扫描**（旧网络与训练默认），勿与改进版主 checkpoint 混比。`conditional/`、`robustness/ratio_0` 须用 `outputs/checkpoints/diffusion.pt`（改进版）重跑后写入报告。

---

## 产物目录（固定）

所有实验结果只在 **`outputs/`** 下，不要在仓库根目录创建 `outputs_v2`、`outputs_spiral` 等。

```
outputs/
├── checkpoints/          # kde.pkl, gmm.pkl, diffusion.pt
├── metrics/              # 主实验指标
├── samples/
├── figures/
├── conditional/          # 条件生成拓展
├── sweep/                # 超参扫描（trial_*/、summary.json）
└── robustness/           # 鲁棒性（ratio_0p000/ …）
    ├── summary.json
    └── ratio_0p010/
        ├── data/         # 污染后的训练副本
        ├── checkpoints/
        └── metrics/
```

| 文件 | 说明 |
|------|------|
| `metrics/evaluation.json` | test 三模型指标 |
| `metrics/main_evaluation.json` | test 三模型（kde + gmm + diffusion） |
| `metrics/diffusion_only_evaluation.json` | 仅 Diffusion 的快速评估 |
| `metrics/evaluation_hidden_test.json` | hidden_test 三模型 |
| `metrics/diffusion_only_hidden_test_evaluation.json` | hidden_test 仅 Diffusion |
| `metrics/diffusion_train_summary.json` | Diffusion 训练配置摘要 |
| `conditional/conditional_evaluation.json` | 条件生成指标 |
| `sweep/sweep_summary.json` | 扫描汇总别名 |
| `robustness/robustness_summary.json` | 鲁棒性汇总别名 |

`data/`：`train.npy`、`test.npy`、`hidden_test.npy`、`metadata.json`（均由 `generate_data.py` 生成）。

---

## 代码

`config.py` · `data_utils.py` · `metrics.py` · `eval_utils.py` · `progress.py`  
`kde_model.py` `gmm_model.py` `diffusion_model.py`  
`generate_data.py` · `train.py` · `evaluate.py` · `run_main.py` · `extensions.py` · `sweep_grid.json`
