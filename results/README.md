# results/：主结果证据

| 文件 | 内容 |
|---|---|
| `per-seed.csv` | 主表：三行每 seed（B0/B2 的 mean catch 与中位时延）+ 汇总行 |
| `S6-confirm-summary.json` | 原始汇总证据（直接拷贝自课题仓 `results/`，未改一字） |
| `smoke_run.json` | 本仓 `--smoke` 的真实输出（管线自检，不是论文数字） |
| `port_validation.json` | 端口逐位校验（需要上游 checkpoint；见 REPRODUCE §4） |

## per-seed.csv 读法

- `B0_mean_catch` / `B2_mean_catch`：10 局 × 2000 步的 catch rate 均值。
- `delta_pp`：同 seed 内 10 对 episode 的配对差均值（B2−B0）× 100。
- `B0_latency_ms` / `B2_latency_ms`：单步推理时延，3 轮 × 300 次取中位的中位。
- `episodes_positive_of_10`：该 seed 内 10 对 episode 中有几对 B2 > B0。

**汇总行口径**（容易被误读，写清楚）：

- `delta_pp` = 三个 seed 配对差的平均 = **+16.06pp**（不是「平均 catch 之差」再取整；
  两者在此处恰好接近：0.768110 − 0.607544 = 16.06pp）。
- `latency_increase_pct` = 三个 seed 增幅的**中位数** = **+4.62%**，
  不是「平均时延之比」（那会得到 +12.25%，把 seed47 的破门稀释掉）。
- `n_episodes_positive` = 26/30 对 episode 为正。
- seed47 的时延 +46.14% 超 20% 预算 → 预注册 verdict 为 **inconclusive**，详见
  `../notes/NEGATIVE-CORNER.md` §3。

## 复算

```bash
python - <<'EOF'
import json
for s in (45, 46, 47):
    d = json.load(open(f"../tnd-.../results/S6-confirm-seed{s}.json"))  # 课题仓原件
    b0, b2 = d["arms"]["B0_fixed"], d["arms"]["B2_grow_only"]
    print(s, b0["eval"]["mean_catch_rate"], b2["eval"]["mean_catch_rate"],
          b0["latency"]["median_ms"], b2["latency"]["median_ms"])
EOF
```

本仓库不重算这三个 seed（需要上游 checkpoint + 每 seed 约 5 分钟探针 + 2.5 分钟四臂），
只搬运已落盘证据；重跑方法见 `../REPRODUCE.md`。