# TND-Grow：按利用率生长神经元拓扑，Pong 接球率 **+16.06pp**

**一句话问题**：TND（arXiv:2606.21295，AAAI 2027）把序列建模画成一张神经元图，但图在初始化后
就固定不动——如果让这张图在训练中按「谁在被使用」自己长，同一套 Pong 行为克隆协议下能不能多接球？

**增益（本仓最重要的一行）**：只生长臂 **B2** 相对固定拓扑 **B0**，跨 **3 个训练流（seed 45/46/47）**
配对平均 **+16.06pp** catch rate（+21.51 / +20.48 / +6.18pp），**3/3 个 seed 为正**，
**30 对 episode 中 26 对为正**（10 局 × 3 seed）。活跃边数 137063 → 137063、参数量 361,224 不变
（「长 k 换 k」守恒），增益不是靠加参数换来的。

**一条命令跑通（CPU，实测 57.9 秒）**：

```bash
cd code && python -m tnd_growth --smoke
```

13 项硬不变量全部 PASS（边数守恒 / 参数量恒定 / 拓扑确实被改 / 探针可复现），
真实终端输出见 [`RESULTS.md` §A](RESULTS.md)。

## 结果（10 局/seed × 3 seed，配对）

| seed | B0 catch | B2 catch | Δ | B2 时延变化 | 10 对中 B2 胜 |
|---|---|---|---|---|---|
| 45 | 0.7037 | 0.9188 | **+21.51pp** | −8.33% | 10 |
| 46 | 0.7029 | 0.9077 | **+20.48pp** | +4.62% | 9 |
| 47 | 0.4160 | 0.4779 | **+6.18pp** | **+46.14%** | 7 |
| **合并** | 0.6075 | 0.7681 | **+16.06pp** | 中位 +4.62% | **26/30** |

原始证据：[`results/S6-confirm-summary.json`](results/S6-confirm-summary.json)（课题仓原件拷贝）、
[`results/per-seed.csv`](results/per-seed.csv)。代码与原始实验的等价性经过逐位校验：
探针锚点与 B0 臂十局逐局 catch **完全相同**（[`RESULTS.md` §B](RESULTS.md)）。

## 做法（30 秒版）

一次探针（固定 250 条 calibration 序列，前向+反向）算出三个量：单元平均激活 `A_i`、
梯度幅值 `G_i`、边贡献 `S_ij = mean |o_j|·|∂L/∂z_i|`。然后：

- **剪枝**：剪掉「源/目标至少一端是低激活·低梯度」且 `S_ij` 落在最低 10% 的边，
  预算 5%、保留最小入/出度 1；
- **生长**：在高误差样本上找「梯度高于中位数」的瓶颈单元，从未连边里挑与残差相关性最高、
  且 discovery/calibration 两段相关**同号**的源，长 k=10 条；
- 新边幅值 = 被剪边 |w| 均值，符号 = sign(corr)；长 k 条就再关掉 k 条 |w| 最低的边（边数守恒）；
- 所有臂（含 B0）统一做谱归一化 `max|eig| ≤ 0.95`，从同一缩放后起点出发。

完整定义（含阈值、退化情形、四臂对照）见 [`METHOD.md`](METHOD.md)。

## 失效边界（必读，负结果只写在这里和 notes/）

- **seed47 时延 +46.14% 破 20% 预算** → 按预注册判据，正式 verdict 是 **inconclusive，不是 supported**。
  唯一卡点就是这条时延门（三个 seed 的增益方向都是正的）。
- **时延门本身不硬**：同一 seed、同一 k=10 臂、边数同为 137063，两次独立测量可以差几十个百分点
  （−8.33% vs +12.19%；+46.14% vs −20.09%）。所以「B2 慢 46%」不是机制结论，是测量问题；
  但本仓不据此翻案，照实记破门。
- **只剪枝 B1 单独有害**：S4 seed41 **−18.97pp**、合并 **−2.40pp**（S6 描述性读数又翻正，
  两批 seed 方向矛盾，从未被证明有稳定增益）。
- **剪+长 B3 不如只长**：S4 合并 **−2.70pp**（H_TND 判 inconclusive）；
  S6 描述性 +11.45pp，低于 B2 的 +16.06pp，3 个 seed 里 2 个低于 B2。
- **pilot 的 +17.5pp 是 3 局噪声**：换 10 局口径只剩 +1.5pp，换训练序直接翻负（−15.1/−17.5pp）。

完整数字与读法见 [`notes/NEGATIVE-CORNER.md`](notes/NEGATIVE-CORNER.md)。

## 目录

```
repo/
├── README.md              本页
├── METHOD.md              生长规则定义、四臂对照、退化情形
├── REPRODUCE.md           安装 → 冒烟 → 端口校验 → 完整三 seed
├── RESULTS.md             真实终端输出（smoke + 端口校验）+ 主结果表
├── requirements.txt       只有 torch + numpy
├── code/
│   ├── tnd_growth/        模型 / 数据 / 探针 / 重连 / 评测 / 执行器（可直接 import）
│   └── validate_port.py   与原始实验的逐位校验（需 checkpoint）
├── data/c2.txt            Pong 行为克隆数据（19999 行，来自上游 tnd_pong）
├── results/               主表 + 原始汇总 JSON + 冒烟输出
└── notes/NEGATIVE-CORNER.md   负结果与失效边界
```

## 依赖

`torch>=2.0` + `numpy>=1.24`，**纯 CPU**，不需要 GPU、不需要 pygame、不需要下载数据集。
实测环境 Python 3.11.9 / torch 2.10.0 / Windows 11 / 4 线程。

## 复现三个 seed

需要上游预训练 checkpoint（本仓不附带，见 [`REPRODUCE.md` §5](REPRODUCE.md)），
每个 seed 约 8 分钟 CPU，三 seed 约 25 分钟：

```bash
cd code && python -m tnd_growth --full --seed 45 --checkpoint /path/to/snn_dt_model.pth
```

## 溯源与许可

- 模型、数据与 Pong 协议来自 **TND** 作者仓库 [`brcai/tnd_pong`](https://github.com/brcai/tnd_pong)
  （论文 arXiv:2606.21295）。`data/c2.txt` 与 `code/tnd_growth/model.py`、`pong.py` 的语义
  均源自该仓库；本仓只做抽取、去 pygame、去绝对路径，并保持逐位等价（见 `RESULTS.md` §B）。
- 上游仓库未声明开源许可证；**推公开仓前请先确认上游许可**，或把 `data/` 换成自行生成的数据。
- 本仓新增部分：利用率探针的候选规则、长 k 换 k 重连、四臂执行器、纯 Python 评测环境。