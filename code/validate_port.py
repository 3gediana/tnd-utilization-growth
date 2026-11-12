"""端口校验：用本仓代码重跑原始实验的两处锚点，比对存档数字。

锚点 1（探针，S6 seed45 Stage A）：2000 步训练 + 终点探针
  期望 hidden_dissociation_ratio=0.20418848167539266 / count=117 / prune=803 / growth=10
锚点 2（B0 臂，S6 seed45）：checkpoint + 统一谱归一化 + 190 步 + 10 局评测
  期望 mean_catch_rate=0.7036764705882352

用法：
  python code/validate_port.py --checkpoint /path/to/snn_dt_model_0.8_600_0.4_25_40.pth
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tnd_growth.data import prepare_data
from tnd_growth.experiment import build_model, train_model
from tnd_growth.probe import probe
from tnd_growth.rewire import apply_spectral_normalization
from tnd_growth import pong

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="上游 TND 预训练 checkpoint")
    ap.add_argument("--data", default=str(REPO / "data" / "c2.txt"))
    ap.add_argument("--out", default=str(REPO / "results" / "port_validation.json"))
    args = ap.parse_args()

    torch.set_num_threads(4)
    x, y, n = prepare_data(args.data)
    calib_x = x[:5, :50].reshape(-1, 40, 24).contiguous()
    calib_y = y[:5, :50].reshape(-1, 3).contiguous()
    out = {}

    t0 = time.time()
    m = build_model(seed=45)
    m.load_checkpoint(args.checkpoint)
    info = train_model(m, x, y, n, 2000, order_seed=45)
    m.eval()
    s = probe(m, calib_x, calib_y)
    out["probe"] = {
        "elapsed_s": round(time.time() - t0, 1),
        "train_loss": [info["first_loss"], info["last_loss"]],
        "hidden_dissociation_ratio": s["hidden_dissociation_ratio"],
        "hidden_dissociation_count": s["hidden_dissociation_count"],
        "prune_candidates_strict": s["prune_candidates_strict"],
        "growth_candidate_edges": s["growth_candidate_edges"],
        "growth_bottleneck_targets": s["growth_bottleneck_targets"],
        "expected": {"ratio": 0.20418848167539266, "count": 117, "prune": 803, "growth": 10},
    }
    print("PROBE", json.dumps(out["probe"], ensure_ascii=False), flush=True)

    t0 = time.time()
    m2 = build_model(seed=45)
    m2.load_checkpoint(args.checkpoint)
    pre = apply_spectral_normalization(m2)
    train_model(m2, x, y, n, 190, order_seed=45)
    ev = pong.evaluate(m2, list(range(2026, 2036)))
    out["B0"] = {
        "elapsed_s": round(time.time() - t0, 1),
        "pre_norm_spectral_radius": pre,
        "mean_catch_rate": ev["mean_catch_rate"],
        "per_episode": [e["catch_rate"] for e in ev["episodes"]],
        "expected_mean": 0.7036764705882352,
    }
    print("B0", json.dumps(out["B0"], ensure_ascii=False), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()