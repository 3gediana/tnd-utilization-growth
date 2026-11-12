"""重连算子：把探针给出的候选边应用到拓扑上。

三条规则（预注册，原样搬运）：
  1. 剪枝：逐条关闭候选边（候选集已含预算与最小度约束）。
  2. 预算守恒（长 k 换 k）：只长臂/剪+长臂生长 k 条边时，从现存边里关掉 k 条 |w| 最低的边
     （规则剪枝已腾出的槽位先抵扣），保证活跃边数不变、参数量不变——"增益不是靠加参数换来的"。
  3. 新边赋权：幅值 = 被剪边 |w| 的均值（无剪枝则 0.01），符号 = sign(corr_disc)。

重连后统一做谱归一化（max|eig| <= 0.95），与基线初始化同一稳定化流程。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model import TND


def apply_spectral_normalization(model: TND, target_radius: float = 0.95) -> float:
    """把 max|eig(W⊙A)| 缩到 target_radius，返回缩放前的谱半径。"""
    with torch.no_grad():
        eff = model.weights * model.adj_mask
        max_eig = float(torch.max(torch.abs(torch.linalg.eigvals(eff.cpu()))).item())
        if max_eig > 0:
            model.weights.data = (model.weights.data / max_eig) * target_radius
    return max_eig


def rewire(model: TND, prune_edges: list[dict[str, Any]], growth_edges: list[dict[str, Any]],
           conserve_budget: bool) -> dict[str, Any]:
    """按候选集改拓扑；返回实际生效的剪/长数量与守恒信息。"""
    adj = model.adj_mask.clone()
    W = model.weights
    k_growth = len(growth_edges)
    pruned_w_vals: list[float] = []
    applied_prune: list[dict[str, Any]] = []

    with torch.no_grad():
        for e in prune_edges:
            i, j = int(e["target_i"]), int(e["source_j"])
            if adj[i, j] == 1:
                pruned_w_vals.append(float(abs(W[i, j].item())))
                adj[i, j] = 0
                applied_prune.append(e)

        extra_needed = 0
        if conserve_budget and k_growth > 0:
            # 规则剪枝已腾出的槽位先抵扣，剩余缺口用最低 |w| 边补偿
            extra_needed = max(0, k_growth - len(applied_prune))
            eff = (W * adj).abs()
            flat = []
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i, j] == 1:
                        flat.append((float(eff[i, j].item()), int(i), int(j)))
            flat.sort(key=lambda x: x[0])
            grown_pos = {(int(g["target_i"]), int(g["source_j"])) for g in growth_edges}
            removed = 0
            in_deg = adj.sum(axis=1).clone()
            out_deg = adj.sum(axis=0).clone()
            for _, i, j in flat:
                if removed >= extra_needed:
                    break
                if (i, j) in grown_pos:
                    continue
                if int(in_deg[i]) - 1 < 1 or int(out_deg[j]) - 1 < 1:
                    continue
                adj[i, j] = 0
                in_deg[i] -= 1
                out_deg[j] -= 1
                removed += 1
            extra_needed = removed

        scale = float(np.mean(pruned_w_vals)) if pruned_w_vals else 0.01
        applied_growth = []
        for g in growth_edges:
            i, j = int(g["target_i"]), int(g["source_j"])
            if adj[i, j] == 0:
                sign = 1.0 if float(g.get("corr_disc", 0.0)) >= 0 else -1.0
                adj[i, j] = 1
                W[i, j] = sign * scale
                applied_growth.append(g)

    model.adj_mask = adj.to(model.adj_mask.device)
    return {
        "pruned": len(applied_prune),
        "grown": len(applied_growth),
        "compensated_edges": int(extra_needed),
        "new_edge_scale": scale,
        "net_edge_change": len(applied_growth) - len(applied_prune) - int(extra_needed),
    }