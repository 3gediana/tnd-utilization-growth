# REPRODUCE：安装 → 冒烟 → 完整三 seed

所有命令都在 `repo/` 目录下执行（Windows / Linux / macOS 均可，纯 CPU）。

## 1. 环境

| 项 | 实测值 | 说明 |
|---|---|---|
| Python | 3.11.9 | 3.9+ 应该都行（用了 `from __future__ import annotations`） |
| PyTorch | 2.10.0（CPU） | 只用 `torch` + `numpy`；**不需要 GPU / CUDA** |
| 线程 | `torch.set_num_threads(4)`（默认） | 线程数会影响浮点求和顺序，复现存档数字请用 4 |
| 内存 | < 2 GB | 数据张量约 73 MB，模型 361,224 参数 |
| 系统 | Windows 11 | 评测是纯 Python 逻辑，无 pygame / 显示依赖 |

## 2. 安装

```bash
cd repo
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

仓库自带 `data/c2.txt`（19999 行 Pong 行为克隆数据，来自上游 `brcai/tnd_pong` 的 `dt/c2.txt`），
不需要额外下载。

## 3. 冒烟（必跑，约 1 分钟）

```bash
cd code
python -m tnd_growth --smoke
```

预期：`[smoke checks] ALL PASS (13/13)`，退出码 0，全流程 **~58 秒**（Windows 11 / 4 线程实测）。
它跑通：数据切批 → 探针 → 四臂重连 → 各 60 步训练 → 3 局评测 → 时延测量，
并断言边数守恒、参数量恒定、谱归一化、探针决定性。

**注意**：冒烟不加载 checkpoint（只热身 30 步），catch rate 没有科学含义；
真实数字见 §5。

## 4. 端口校验（可选，约 8 分钟）

如果你手上有上游 checkpoint，可以验证「本仓代码 == 原始实验」：

```bash
python code/validate_port.py --checkpoint /path/to/snn_dt_model_0.8_600_0.4_25_40.pth
```

期望（与存档逐位对比）：

```
PROBE hidden_dissociation_ratio=0.20418848167539266  count=117  prune=803  growth=10
B0    mean_catch_rate=0.7036764705882353（存档 0.7036764705882352，差 1 ULP）
      per_episode=[0.75, 0.8125, 0.5882352941176471, 0.75, 0.8125,
                   0.5294117647058824, 0.6875, 0.8125, 0.5882352941176471, 0.7058823529411765]
```

实测输出见 `RESULTS.md` §B。**这一步通过 = 抽出来的代码与原始实验等价**。

## 5. 完整三 seed 复现（约 25 分钟 CPU）

### 5.1 前置：checkpoint

本仓**不附带**预训练权重（第三方权重，许可证未声明，不随仓分发）。二选一：

- **A. 用上游权重**：从 `brcai/tnd_pong` 取 `save/snn_dt_model_0.8_600_0.4_25_40.pth`
  （600 神经元 / p=0.4 / dt=0.8 / batch=25 / L=40）。它在本机的实测表现为 10 局 catch 0.9632。
- **B. 自己训**：跑上游 `train_tnd.py`（`--neuron 600 --p 0.4 --dt 0.8 --l 40 --bt 25`，
  800 epoch 上限，`running_loss < 0.13` 早停），再拿来用。

没有 checkpoint 也能跑 `--full`，但那是「从随机初始化开始」的另一个工作点，**不可与本文数字比较**。

### 5.2 命令（每个 seed 一条）

```bash
cd code
python -m tnd_growth --full --seed 45 --checkpoint /path/to/snn_dt_model_0.8_600_0.4_25_40.pth
python -m tnd_growth --full --seed 46 --checkpoint /path/to/snn_dt_model_0.8_600_0.4_25_40.pth
python -m tnd_growth --full --seed 47 --checkpoint /path/to/snn_dt_model_0.8_600_0.4_25_40.pth
```

`--full` 的默认协议（与原始实验一致）：

- **Stage A**：从 checkpoint 再训练 **2000 步**（batch 序 = `RandomState(seed)` 固定置换，逐 epoch 同序）
  → 终点探针（250 条 calibration 序列）→ 剪/长候选；
- **Stage B**：四臂 **各 190 步**（B0/B1/B2/B3，同起点同数据序），
  评测 **10 局 × 2000 步**（episode seeds 2026–2035），时延 3 轮 × 300 次；
- k=10，剪枝预算 5%，统一谱归一化 0.95。

每 seed 实测耗时（本机 CPU / 4 线程）：Stage A 约 **5.5 分钟**（341s），Stage B 约 **2.5 分钟**。
三 seed 合计约 **25 分钟**，不需要 GPU。

### 5.3 期望读数（逐 seed）

| seed | 探针解离占比 | 解离数 | 剪枝候选 | 生长候选 | B0 catch | B2 catch | B2−B0 |
|---|---|---|---|---|---|---|---|
| 45 | 0.20418848167539266 | 117 | 803 | 10 | 0.703676 | 0.918750 | +21.51pp |
| 46 | 0.19895287958115182 | 114 | 978 | 10 | 0.702941 | 0.907721 | +20.48pp |
| 47 | 0.19197207678883071 | 110 | 1052 | 10 | 0.416013 | 0.477859 | +6.18pp |
| 合并 | — | — | — | — | 0.607544 | 0.768110 | **+16.06pp**（26/30 对为正） |

逐位复现的前提：同一 checkpoint、`torch.set_num_threads(4)`、同一 torch 版本。
换了 CPU / 线程数 / torch 版本，浮点求和顺序会变，catch 可能有 ±1 局的抖动。

## 6. 哪些东西不可复现 / 不可比

1. **时延绝对值**：0.13–0.19 ms 量级，强依赖机器、负载、线程调度；本仓只用于比值，
   而比值本身噪声极大（同臂同边数两次测量可差 30–60 个百分点，见 `notes/NEGATIVE-CORNER.md` §3）。
   **不要把时延数字当机制结论。**
2. **上游 checkpoint 不随仓分发**：没有它就无法复现「本文的工作点」。用 `--warmup` 从
   随机初始化重训是另一个工作点。
3. **seed 41/43/44（S4）**：需要各自的全链重探针候选，本仓能跑（同一条 `--full --seed 41`），
   但存档的 S4 产物依赖当时的候选文件与原始脚本；本仓未逐位校验那三个 seed。
4. **评测局表**：只覆盖 2026–2035。换局表（如 2101–2110）会改变绝对 catch（实测摆幅约 9.3pp），
   本仓不声称跨局表稳健。
5. **wall-clock**：`train_seconds` / `elapsed_seconds` 每次都不同，不参与任何比较。
6. **GPU**：原始实验与本文全部在 CPU 上跑；GPU 上的数值可能与 CPU 有微小差异。