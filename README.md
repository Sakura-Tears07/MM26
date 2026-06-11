# MM26 · 二维分布生成建模

**课程期末大作业** — 选题「二维分布生成建模」（`final_project.pdf` 第 19–23 页）

在四类合成二维分布上实现并对比 **KDE**、**GMM**、**条件 Diffusion（resfourier_v4）**，含主实验与三项拓展（条件生成 / 超参扫描 / 鲁棒性）。

| 标签 | 分布 | 几何特点 |
|------|------|----------|
| 0 | `gaussian_mixture` | 8 峰环形多模态 |
| 1 | `ring` | 单环 + 径向/切向噪声 |
| 2 | `two_moons` | 双月牙非凸 |
| 3 | `spiral` | 双臂螺旋，建模难度最高 |

---

## 环境

```bash
conda activate work          # 或 MM26；Python 3.10+，torch>=2.0
pip install -r requirements.txt
```

依赖：`numpy`、`scipy`、`scikit-learn`、`matplotlib`、`torch`。Diffusion / sweep / robustness 推荐 **4× GPU**；KDE/GMM 可在 CPU 完成。

---

## 全量重跑（按顺序复制执行）

```bash
cd ~/my_stuff/MM26/MM26
conda activate work

# 0. 数据
python generate_data.py --force

# 1. 主实验（含 test + hidden_test 评估）
python run_main.py --force --gpus 4 --device cuda --compile

# 2. 拓展（依赖 outputs/checkpoints/）
python extensions.py conditional --force --device cuda
python extensions.py sweep --force --gpus 4 --device cuda --compile
python extensions.py robustness --force --ratios 0 0.01 0.05 0.1 --gpus 4 --device cuda --compile
```

`sweep` 为 18 trials × 600 epoch，耗时最长；时间紧可先跳过第 4 行。

**tmux 后台示例**（自行把上面命令贴进 session）：

```bash
tmux new -s mm26
# 在 tmux 内粘贴上述命令
# 脱离：Ctrl+B  D
tmux attach -t mm26
```

**停止正在跑的训练**：

```bash
# 若在 tmux 里：Ctrl+C，或
tmux kill-session -t mm26

# 若 torchrun 僵死，按路径杀 MM26 进程（勿误杀其他项目）：
pkill -f "/home/zy/my_stuff/MM26/MM26.*train.py"
pkill -f "/home/zy/my_stuff/MM26/MM26.*run_main.py"
pkill -f "/home/zy/my_stuff/MM26/MM26.*extensions.py"
```

---

## 分步命令

### 0. 数据（任务一）

```bash
python generate_data.py --force
python generate_data.py --force --plot   # 可选：data/preview.png
```

每类 4000 train / 2000 test / 2000 hidden_test → `data/*.npy`。

### 1. 主实验（任务二～四）

训练三模型并在 **test + hidden_test** 上评估（`run_main.py` 已含 evaluate，无需再单独跑）：

```bash
python run_main.py --force --gpus 4 --device cuda --compile
```

产物：

- `outputs/checkpoints/{kde.pkl,gmm.pkl,diffusion.pt}`
- `outputs/metrics/main_evaluation.json`
- `outputs/figures/*_vs_real.png`

### 2. 拓展实验（须在主实验 checkpoint 生成**之后**）

```bash
# 条件生成 p(x|c)
python extensions.py conditional --force --device cuda

# 超参扫描（resfourier_v4，网格见 sweep_grid.json，18 trials）
python extensions.py sweep --force --gpus 4 --device cuda --compile

# 鲁棒性：训练集异常点 0 / 1% / 5% / 10%
python extensions.py robustness --force --ratios 0 0.01 0.05 0.1 --gpus 4 --device cuda --compile
```

### 3. 仅重训 Diffusion

```bash
torchrun --standalone --nproc_per_node 4 train.py diffusion \
  --data-dir data --output-dir outputs --compile

python evaluate.py --split test hidden_test --device cuda
# 仅 Diffusion：evaluate.py --diffusion-only ...
```

---

## 统一训练设定

三模型通过 `data_utils.prepare_training_data()` 对齐：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TRAIN_SPIRAL_OVERSAMPLE` | 4 | spiral 4× 过采样（16000→20000） |
| `TRAIN_USE_PER_CLASS_NORM` | true | 每类 z-score；**统计量在原始坐标上计算，归一化只施加一次** |
| 评估 | 原始坐标 | 采样后反归一化，与 test 直接对比 |

超参默认值见 `config.py`（Diffusion：600 epoch、`resfourier_v4`、hidden=384、spiral loss×3、推理步×3）。

---

## 模型概览

| 模型 | 类型 | 要点 |
|------|------|------|
| **KDE** | 非参数 | 按类带宽网格搜索；归一化空间拟合 |
| **GMM** | 参数混合 | 按类 BIC 选 k；spiral 最多 48 成分 |
| **Diffusion** | DDPM | **FiLM-ResNet v4** + cosine schedule + EMA |

**Diffusion v4**：6× FiLMResBlock、Fourier 特征×6、class_emb=64、DDPM 后验方差采样。

---

## 评价指标

| 指标 | 含义 | 注意 |
|------|------|------|
| **MMD** | RBF 核两样本距离，↓越好 | median heuristic 带宽 |
| **Wasserstein / SWD** | 最优传输 / 切片 Wasserstein | 子采样近似 |
| **Coverage** | 真实点被生成覆盖的比例，↑越好 | 95% k-NN 半径 |
| **Precision** | 生成点落在真实邻域的比例，↑越好 | 与 Coverage 不对称 |
| **NLL** | KDE/GMM 用模型密度；Diffusion 在生成样本上拟合 KDE 估计 | **量级不可跨模型直接比** |
| **Mode Coverage** | 仅 gaussian_mixture | 8 扇区模态覆盖 |

---

## 当前主实验结果（test，seed=42）

> 重跑后请查看 `outputs/metrics/main_evaluation.json` 更新下表。

| 模型 | MMD ↓ | Coverage ↑ | Precision ↑ |
|------|-------|------------|-------------|
| KDE | 0.000436 | **0.952** | 0.916 |
| GMM | 0.000389 | 0.946 | 0.930 |
| **Diffusion v4** | **0.000289** | 0.914 | **0.959** |

**Spiral（原难点）**

| 模型 | Precision | Coverage | MMD |
|------|-----------|----------|-----|
| GMM | **0.935** | **0.932** | **0.000338** |
| Diffusion | 0.926 | 0.922 | 0.000497 |
| KDE | 0.923 | 0.926 | 0.000419 |

要点：

- Diffusion **macro MMD 最低、Precision 最高**；Coverage 略低于 KDE/GMM（生成分布偏保守）。
- v4 将 spiral precision 从旧版 ~0.785 提升至 **~0.926**。
- 可视化：`outputs/figures/kde_vs_real.png` 等。

Hidden test 泛化正常（Diffusion macro precision test 0.959 → hidden 0.954）。

---

## 产物目录

所有结果仅在 **`outputs/`**（勿新建 `outputs_*`）。

```
outputs/
├── checkpoints/          # kde.pkl, gmm.pkl, diffusion.pt
├── metrics/              # main_evaluation.json, diffusion_train_summary.json, ...
├── figures/              # *_vs_real.png
├── samples/
├── conditional/          # conditional_evaluation.json
├── sweep/                # trial_*/ , summary.json, best_trials.json
└── robustness/           # ratio_0p010/ ... , summary.json
```

| 文件 | 说明 |
|------|------|
| `metrics/main_evaluation.json` | test 三模型 |
| `metrics/evaluation_hidden_test.json` | hidden_test 三模型 |
| `metrics/diffusion_train_summary.json` | Diffusion 训练摘要 |
| `conditional/conditional_evaluation.json` | 条件生成 |
| `sweep/summary.json` | 超参扫描汇总 |
| `robustness/summary.json` | 鲁棒性汇总 |

---

## 仓库结构

```
MM26/
├── generate_data.py
├── config.py
├── data_utils.py         # prepare_training_data / ClassNormStats
├── kde_model.py  gmm_model.py  diffusion_model.py
├── train.py  evaluate.py  run_main.py  extensions.py
├── sweep_grid.json
└── data/  outputs/
```

---

## 与作业要求对照

| 要求 | 对应 |
|------|------|
| 任务一 数据分析 | `generate_data.py --plot` + 报告 |
| 任务二 ≥2 种生成模型 | KDE + GMM + Diffusion |
| 任务三 样本生成与对比 | `evaluate.py` + `outputs/figures/` |
| 任务四 ≥2 种指标 | MMD、W1、SWD、Coverage、Precision、NLL、ModeCov |
| 拓展一 超参 | `extensions.py sweep` |
| 拓展二 条件生成 | `extensions.py conditional` |
| 拓展三 鲁棒性 | `extensions.py robustness` |

**提交**：数学建模竞赛格式报告 + 队员分工；Deadline **2026-06-14**，汇报 **2026-06-17**。

---

## 命令速查

| 命令 | 作用 |
|------|------|
| `python run_main.py --force --gpus 4 --device cuda --compile` | 主实验 |
| `python train.py baselines` | 仅 KDE/GMM |
| `python extensions.py conditional \| sweep \| robustness` | 单项拓展 |
