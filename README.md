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
python generate_data.py --force      # 覆盖 data/*.npy，默认 train 16000 / test 8000
python generate_data.py --force --plot
```

等价于：`python generate_data.py --train-per-class 4000 --test-per-class 2000`

---

## 命令

```bash
# 主实验（--force：重训并覆盖 outputs/ 下 checkpoint、samples、metrics、figures）
python run_main.py --force --data-dir data --output-dir outputs \
  --epochs 400 --batch-size 32 --gpus 8 --device cuda --compile

# 拓展（均带 --force，忽略已有结果并覆盖输出目录）
python ext_conditional.py --force --checkpoint outputs/checkpoints/diffusion.pt --device cuda
python ext_sweep.py --force --gpus 8 --device cuda --compile    # sweep_grid.json，18 组
python ext_robustness.py --force --main-output-dir outputs --ratios 0 0.01 0.05 0.1 --gpus 8 --device cuda --compile
```

修改扫描维：编辑 `sweep_grid.json` 后同样加 `--force` 重跑；旧 `outputs/ext_sweep/trial_*` 会被各 trial 新结果覆盖（目录名不变时建议先删再扫，避免残留）。

---

## 数据与产物路径

### `data/`（`generate_data.py`）

| 文件 | 形状/说明 |
|------|-----------|
| `train.npy` | `(16000, 2)`，每类 4000 |
| `train_label.npy` | `(16000,)` |
| `test.npy` | `(8000, 2)`，每类 2000 |
| `test_label.npy` | `(8000,)` |
| `hidden_test.npy` / `hidden_test_label.npy` | 默认每类 2000 |
| `metadata.json` | seed、类别映射、形状 |
| `preview.png`、`figures/preview_*.png` | `--plot` 时生成 |

### `outputs/`（主实验）

| 路径 | 说明 |
|------|------|
| `checkpoints/kde.pkl` `gmm.pkl` `diffusion.pt` | 三模型权重 |
| `samples/{kde,gmm,diffusion}_{samples,labels}.npy` | 生成样本 |
| `metrics/evaluation.json` `.csv` | 指标 |
| `metrics/baseline_train_summary.json` | KDE/GMM 训练摘要 |
| `metrics/diffusion_train_summary.json` `diffusion_loss.json` | Diffusion 训练 |
| `figures/{kde,gmm,diffusion}_vs_real.png` | 对比图 |

### `outputs/ext_conditional/`

`evaluation.json`、`evaluation.csv`、`conditional_vs_real.png`、`samples_class_*.npy`、`labels_class_*.npy`、`samples_all.npy`

### `outputs/ext_sweep/`

| 路径 | 说明 |
|------|------|
| 根目录 `sweep_grid.json` | 扫描网格定义 |
| `summary.json` `summary.csv` `best_trials.json` | 汇总 |
| `trial_*/checkpoints/diffusion.pt` | 各组权重 |
| `trial_*/metrics/diffusion_eval.json` | 各组指标 |
| `trial_*/figures/diffusion_vs_real.png` | 各组对比图 |

当前网格（固定 `bs=32, ema=0.999, accum=1`）：`lr` × `num_steps` × `hidden_dim` = 3×3×2 = **18** 组。

### `outputs/ext_robustness/`

`summary.json`、`summary.csv`、`summary_macro.csv`、`evaluation_by_ratio.json`；`ratio_0p000/outputs/` 复用主实验 checkpoint；`ratio_0p010/`、`ratio_0p050/`、`ratio_0p100/` 为污染数据与重训结果。

---

## 代码

`generate_data.py` · `config.py` · `data_utils.py` · `metrics.py` · `eval_utils.py` · `progress.py`  
`kde_model.py` `gmm_model.py` `diffusion_model.py`  
`train_baselines.py` `train_diffusion.py` · `eval_main.py` `run_main.py`  
`ext_conditional.py` `ext_sweep.py` `ext_robustness.py` · `sweep_grid.json`

Diffusion 可选：`train_diffusion.py --spiral-oversample 2`；螺旋推理步数 ×2（代码内默认）。
