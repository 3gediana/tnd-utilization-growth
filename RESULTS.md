# RESULTS：真实终端输出与主结果表

本文件只放**跑出来的东西**：`--smoke` 的原始输出、端口校验的原始输出、主结果表。
解读在 README / METHOD，负结果在 `notes/NEGATIVE-CORNER.md`。

---

## A. `--smoke` 原始输出（CPU，实测 57.9 秒）

命令（在 `repo/` 下）：

```
cd code && python -m tnd_growth --smoke
```

终端输出（原样，未删改；`[smoke checks]` 是本仓的硬不变量自检）：

```
====================================================================
TND growth experiment [smoke] seed=45 steps=60 episodes=3x800
====================================================================
[data] batches=19 x=(19, 1000, 40, 24) y=(19, 1000, 3)
[model] no checkpoint: init from seed + 30 warmup steps
[warmup] steps=30 loss 0.579299 -> 0.179298 (4.38s)
[model] edges=137698 params=361,224
[probe] calib_loss=0.191177 dissoc_hidden=0.1710(98/573) prune_candidates=1838 growth_candidates=10
[gate] stage_b_trigger=True probe_deterministic=True

----- arm B0_fixed (k=0) -----
[graph] edges 137698->137698 (pruned=0 grown=0 compensated=0; added=0 removed=0) spec_radius 1.5965->0.9500
[train] 60 steps loss 0.192713 -> 0.241319 (8.66s)
[eval] mean_catch=0.3016 per_ep=[0.3333, 0.2857, 0.2857]
[latency] median=0.129ms p95=0.175ms (3 rounds x 100 calls)

----- arm B1_prune_only (k=10) -----
[graph] edges 137698->137688 (pruned=10 grown=0 compensated=0; added=0 removed=10) spec_radius 1.5963->0.9500
[train] 60 steps loss 0.193063 -> 0.239962 (8.35s)
[eval] mean_catch=0.3016 per_ep=[0.3333, 0.2857, 0.2857]
[latency] median=0.148ms p95=0.163ms (3 rounds x 100 calls)

----- arm B2_grow_only (k=10) -----
[graph] edges 137698->137698 (pruned=0 grown=10 compensated=10; added=10 removed=10) spec_radius 1.5967->0.9500
[train] 60 steps loss 0.192852 -> 0.233681 (8.83s)
[eval] mean_catch=0.2460 per_ep=[0.1667, 0.2857, 0.2857]
[latency] median=0.168ms p95=0.188ms (3 rounds x 100 calls)

----- arm B3_prune_grow (k=10) -----
[graph] edges 137698->137698 (pruned=10 grown=10 compensated=0; added=10 removed=10) spec_radius 1.5970->0.9500
[train] 60 steps loss 0.193573 -> 0.242441 (8.60s)
[eval] mean_catch=0.3016 per_ep=[0.3333, 0.2857, 0.2857]
[latency] median=0.245ms p95=0.268ms (3 rounds x 100 calls)

[paired vs B0]
  B1_prune_only: +0.00pp (0/3 episodes positive), latency +14.69%
  B2_grow_only: -5.56pp (0/3 episodes positive), latency +30.58%
  B3_prune_grow: +0.00pp (0/3 episodes positive), latency +90.04%

[output] wrote .../repo/results/smoke_run.json
[done] 55.3s

[smoke checks]
  PASS  probe deterministic: 两次探针候选完全一致
  PASS  stage-B gate: 解离 0.1710 / prune 1838 / grow 10
  PASS  B0 unchanged: 137698 -> 137698
  PASS  B1 prunes k: 137698 -> 137688 (k=10)
  PASS  B2 conserves edges: 137698 -> 137698
  PASS  B3 conserves edges: 137698 -> 137698
  PASS  B2 grows k edges: added=10 removed=10
  PASS  B3 grows+prunes k: added=10 removed=10
  PASS  B1 topology only removes: added=0 removed=10
  PASS  new weights nonzero: range=[-0.00594977056607604, 0.00594977056607604]
  PASS  params constant: 361,224
  PASS  spectral radius <= 0.95: B0_fixed=0.949998, B1_prune_only=0.949999, B2_grow_only=0.949999, B3_prune_grow=0.950000
  PASS  eval ran: 3 episodes/arm
[smoke checks] ALL PASS (13/13)
[smoke note] catch rate 在无 checkpoint 的冒烟里不具备科学含义：模型只热了几十步，策略退化为常动作，四臂数字会相同。

real    0m57.897s
```

### 这段输出说明了什么 / 不说明什么

- **说明**：数据管线、探针、剪枝/生长规则、预算守恒（B2/B3 边数 137698→137698）、
  参数量恒定（361,224）、谱归一化（全部落在 0.95）、两次探针完全一致、四臂评测都能跑。
- **不说明**：catch rate。冒烟没有 checkpoint，只热身 30 步（loss 0.18），策略退化为常动作，
  所以四臂 catch 相同/接近——这不是结果，是管线读数。
- 顺带一个观察：同一条命令连跑两次，B2 的时延读数从 **+13.50%** 变成 **+30.58%**（B3 从 +33.75% 到 +90.04%），
  活跃边数完全相同。这就是 `notes/NEGATIVE-CORNER.md` §3 说的「时延比值门测的是噪声」。

---

## B. 端口校验：抽出来的代码与原始实验逐位一致

用本仓代码 + 上游 checkpoint，重跑 S6 seed45 的两个锚点（命令：

```
python code/validate_port.py --checkpoint /path/to/snn_dt_model_0.8_600_0.4_25_40.pth
```

输出（`results/port_validation.json`）：

```
PROBE {"elapsed_s": 341.1, "train_loss": [0.01984548009932041, 0.00908588245511055],
 "hidden_dissociation_ratio": 0.20418848167539266, "hidden_dissociation_count": 117,
 "prune_candidates_strict": 803, "growth_candidate_edges": 10, "growth_bottleneck_targets": 287,
 "expected": {"ratio": 0.20418848167539266, "count": 117, "prune": 803, "growth": 10}}
B0 {"elapsed_s": 39.6, "pre_norm_spectral_radius": 2.799077272415161,
 "mean_catch_rate": 0.7036764705882353,
 "per_episode": [0.75, 0.8125, 0.5882352941176471, 0.75, 0.8125, 0.5294117647058824,
                 0.6875, 0.8125, 0.5882352941176471, 0.7058823529411765],
 "expected_mean": 0.7036764705882352}
```

对比存档：

| 锚点 | 存档值 | 本仓重跑 | 结论 |
|---|---|---|---|
| 探针解离占比 | 0.20418848167539266 | 0.20418848167539266 | **逐位相同** |
| 解离计数 | 117 | 117 | 相同 |
| 剪枝候选 | 803 | 803 | 相同 |
| 生长候选 | 10 | 10 | 相同 |
| B0 mean catch | 0.7036764705882352 | 0.7036764705882353 | 差 1.1e-16（浮点求和顺序） |
| B0 十局逐局 catch | `[0.75, 0.8125, 0.5882…, 0.7059]` | 完全相同 | **逐局相同** |

结论：本仓的模型前向、探针公式、切批顺序、训练循环、评测环境与原始实验**等价**，
不是「重新实现了一个像的东西」。

---

## C. 主结果表（3 seed，10 局/seed，来自 S6）

| seed | B0 catch | B2 catch | Δ（pp） | B0 时延(ms) | B2 时延(ms) | 时延变化 | 10 对中 B2 胜 |
|---|---|---|---|---|---|---|---|
| 45 | 0.7037 | 0.9188 | **+21.51** | 0.1549 | 0.1420 | −8.33% | 10 |
| 46 | 0.7029 | 0.9077 | **+20.48** | 0.1569 | 0.1641 | +4.62% | 9 |
| 47 | 0.4160 | 0.4779 | **+6.18** | 0.1294 | 0.1891 | **+46.14%** | 7 |
| **合并** | 0.6075 | 0.7681 | **+16.06** | 0.1471 | 0.1651 | 中位 +4.62% | **26/30** |

- 「合并 Δ」= 三个 seed 配对差的平均（16.06pp），不是「合并 catch 之差」的另一种算法；
  两者在此处数值一致（0.7681−0.6075=16.06pp）。
- 预注册判据要求「B2 时延 ≤ +20%」，seed47 破门 → **正式 verdict = inconclusive**，
  不是 supported。详见 `notes/NEGATIVE-CORNER.md`。

原始证据：`results/S6-confirm-summary.json`（课题仓原件拷贝）、`results/per-seed.csv`。

## D. 复现这三个 seed 需要什么

需要上游预训练 checkpoint + 每 seed 约 6–8 分钟 CPU（详见 `REPRODUCE.md`）。
本仓不附带 `.pth`，也不改存档数字。