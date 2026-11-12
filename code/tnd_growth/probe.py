"""利用率探针：把"谁在被用、谁在拖后腿"变成可排序的统计量，并据此产出剪/长候选。

定义（与 notes/design-intervention.md §2/§3 一致）：
  A_i  = 单元 i 的时序平均激活幅值      mean_{b,t} |o_i|
  G_i  = 单元 i 的梯度幅值              mean_{b,t} |dL/dz_i|
  S_ij = 边贡献                         mean_{b,t} |o_j| * |dL/dz_i|   （i 目标，j 源）

剪枝：源/目标同时落在 A 与 G 的 bottom-25%（"真低利用"，排除"激活低但梯度高"的解离型），
      且 S_ij 落在现存边最低 10%，取最低的 5% 边预算，保留最小入/出度 1。
生长：高误差样本（残差 top-25%）下 |dL/dz| 高于中位数的隐单元=瓶颈目标；
      候选源 = 与残差相关性最高的未连边单元，前 60% 序列做 discovery、后 40% 做 calibration，
      两段相关系数同号才接受（防过拟合噪声）。新边权重：幅值=被剪边 |w| 均值，符号=sign(corr)。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .model import TND

K_GROWTH = 10          # D9 冻结：生长条数 k=10（S5 做过剂量扫描，主结果仍取 k=10）
PRUNE_CAP_FRAC = 0.05  # 剪枝预算 = 现存边的 5%
PRUNE_S_PCT = 10.0     # S 的低 10% 分位
UNIT_PCT = 25.0        # A/G 的四分位阈值
HIGH_ERR_Q = 0.75      # 高误差样本分位
CORR_TOP = 5           # 每个目标最多考察 5 个候选源


class ProbeTND(TND):
    """TND + 可反传的探针前向（保留每个时间步的 o 与 z 的梯度）。"""

    def forward_probe(self, input_seq, states=None, last_neuron_outputs=None):
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device

        if states is None:
            states = torch.zeros(batch_size, self.num_neurons, device=device, requires_grad=True)
        if last_neuron_outputs is None:
            last_neuron_outputs = torch.zeros(batch_size, self.num_neurons, device=device,
                                              requires_grad=True)

        # 关键：整段递归必须在 enable_grad 内，否则 retain_grad 会抛 requires_grad=False
        with torch.enable_grad():
            effective_weights = self.weights * self.adj_mask
            full_input_stream = torch.zeros(batch_size, seq_len, self.num_neurons, device=device)
            full_input_stream[:, :, self.input_indices] = input_seq * self.input_weight
            w_t = effective_weights.t()

            all_o, all_z = [], []
            for t in range(seq_len):
                weighted_sum = torch.matmul(last_neuron_outputs, w_t)
                total_input = full_input_stream[:, t, :] + weighted_sum + self.bias + self.alpha * states
                if total_input.requires_grad:
                    total_input.retain_grad()
                derivative = torch.tanh(total_input) - states
                states = states + self.dt * derivative
                last_neuron_outputs = torch.tanh(total_input)
                if last_neuron_outputs.requires_grad:
                    last_neuron_outputs.retain_grad()
                all_o.append(last_neuron_outputs)
                all_z.append(total_input)

        outputs = last_neuron_outputs[:, self.output_indices]
        return outputs, states, last_neuron_outputs, all_o, all_z


def collect_grads(model: ProbeTND, calib_x, calib_y):
    """一次前向+反向，返回 loss 与逐步的 o / grad_o / grad_z。"""
    model.zero_grad()
    outputs, states, final_o, all_o, all_z = model.forward_probe(calib_x.clone())
    loss = nn.MSELoss()(outputs, calib_y)
    loss.backward()
    grad_o = torch.stack([t.grad for t in all_o], dim=1)      # (B, T, N)
    grad_z = torch.stack([t.grad for t in all_z], dim=1)      # (B, T, N)
    return loss, all_o, grad_o, grad_z


def quadrant_stats(A: np.ndarray, G: np.ndarray, mask: np.ndarray | None = None,
                   label: str = "overall") -> dict[str, Any]:
    if mask is not None:
        A, G = A[mask], G[mask]
    n = len(A)
    a_thresh = float(np.percentile(A, UNIT_PCT))
    g_thresh = float(np.percentile(G, UNIT_PCT))
    a_bottom = A <= a_thresh
    g_bottom = G <= g_thresh
    return {
        f"{label}_n": int(n),
        f"{label}_A_bottom25_threshold": a_thresh,
        f"{label}_G_bottom25_threshold": g_thresh,
        f"{label}_q_A_bot_G_bot": float(np.sum(a_bottom & g_bottom) / n),
        f"{label}_q_A_bot_G_top": float(np.sum(a_bottom & ~g_bottom) / n),
        f"{label}_q_A_top_G_bot": float(np.sum(~a_bottom & g_bottom) / n),
        f"{label}_q_A_top_G_top": float(np.sum(~a_bottom & ~g_bottom) / n),
        f"{label}_dissociation_ratio": float(np.sum(a_bottom & ~g_bottom) / n),
        f"{label}_dissociation_count": int(np.sum(a_bottom & ~g_bottom)),
    }


def edge_stats_and_candidates(model: ProbeTND, stacked_o, grad_o, grad_z, per_sample_loss,
                              growth_cap: int = K_GROWTH) -> dict[str, Any]:
    """由一次探针的张量算出统计量 + 剪/长候选（规则本体，别在这里调参）。"""
    with torch.no_grad():
        A = stacked_o.abs().mean(dim=(0, 1)).cpu().numpy()       # (N,)
        G = grad_o.abs().mean(dim=(0, 1)).cpu().numpy()          # (N,)
        adj = model.adj_mask.cpu().numpy()
        N = model.num_neurons
        hidden = np.arange(model.num_inputs + model.num_outputs, N)

        # 边贡献 S_ij（只对现存边有意义，乘 adj 后为 0 的边不参与统计）
        S_raw = torch.einsum("bti,btj->ij", stacked_o.abs(), grad_z.abs()).cpu().numpy()
        S_raw /= stacked_o.shape[0] * stacked_o.shape[1]
        S = S_raw * adj
        S_vals = S[S > 0]
        mean_S = float(S_vals.mean()) if len(S_vals) else 0.0
        median_S = float(np.median(S_vals)) if len(S_vals) else 0.0

        quad_all = quadrant_stats(A, G)
        quad_hidden = quadrant_stats(A, G, mask=hidden, label="hidden")

        # ---- 剪枝：真低利用单元 ∩ 最低 S 边 ∩ 5% 预算 ∩ 最小度 1 ----
        a25 = float(np.percentile(A[hidden], UNIT_PCT))
        g25 = float(np.percentile(G[hidden], UNIT_PCT))
        low_unit = (A <= a25) & (G <= g25)
        s10 = float(np.percentile(S_vals, PRUNE_S_PCT)) if len(S_vals) else 0.0

        E = int(adj.sum())
        cap = max(1, int(np.floor(PRUNE_CAP_FRAC * E)))
        prune_pairs = []
        for i in range(N):
            for j in range(N):
                if adj[i, j] != 1:
                    continue
                if not (low_unit[j] or low_unit[i]):
                    continue
                if S[i, j] <= 0 or S[i, j] > s10:
                    continue
                prune_pairs.append((int(i), int(j), float(S[i, j])))
        prune_pairs.sort(key=lambda p: p[2])

        in_deg = adj.sum(axis=1).copy()
        out_deg = adj.sum(axis=0).copy()
        kept = []
        for i, j, s in prune_pairs:
            if len(kept) >= cap:
                break
            if in_deg[i] - 1 < 1 or out_deg[j] - 1 < 1:
                continue
            in_deg[i] -= 1
            out_deg[j] -= 1
            kept.append({"target_i": i, "source_j": j, "S_ij": s})

        # ---- 生长：高误差瓶颈目标 + 相关源 + discovery/calibration 同号验证 ----
        residual = per_sample_loss.detach().cpu().numpy()
        high_err_thresh = float(np.quantile(residual, HIGH_ERR_Q))
        high_err = per_sample_loss >= high_err_thresh
        b_score = grad_z.abs()[high_err].mean(dim=(0, 1)).cpu().numpy()
        b_median = float(np.median(b_score[hidden]))
        targets = [int(i) for i in hidden if b_score[i] >= b_median]

        o_mean_t = stacked_o.detach().mean(dim=1).cpu().numpy()     # (B, N)
        r = residual.reshape(-1, 1)
        rm = r - r.mean()
        om = o_mean_t - o_mean_t.mean(axis=0, keepdims=True)
        num = (rm * om).sum(axis=0)
        den = np.sqrt((rm ** 2).sum() * (om ** 2).sum(axis=0)) + 1e-12
        corrs = num / den

        B_total = stacked_o.shape[0]
        n_disc = int(B_total * 0.6)
        num_c = (rm[n_disc:] * om[n_disc:]).sum(axis=0)
        den_c = np.sqrt((rm[n_disc:] ** 2).sum() * (om[n_disc:] ** 2).sum(axis=0)) + 1e-12
        corrs_c = num_c / den_c

        growth_edges = []
        n_src_pool_empty = 0
        n_calib_fail = 0
        for i in targets[:growth_cap]:
            src_pool = [j for j in hidden if j != i and adj[i, j] == 0]
            if len(src_pool) == 0:
                n_src_pool_empty += 1
                continue
            pool_idx = np.array(src_pool)
            order = np.argsort(-np.abs(corrs[pool_idx]))
            placed = False
            for k in order[:CORR_TOP]:
                j = int(pool_idx[k])
                c_disc = float(corrs[j])
                c_cal = float(corrs_c[j])
                if np.sign(c_disc) == np.sign(c_cal) and abs(c_cal) > 0.0:
                    growth_edges.append({"target_i": i, "source_j": j,
                                         "corr_disc": c_disc, "corr_calib": c_cal})
                    placed = True
                    break
            if not placed:
                n_calib_fail += 1

        return {
            "mean_activation": float(np.mean(A)),
            "mean_gradient": float(np.mean(G)),
            "mean_edge_contribution_S": mean_S,
            "median_edge_contribution_S": median_S,
            "num_existing_edges": E,
            **quad_all,
            **quad_hidden,
            "prune_candidates_strict": len(kept),
            "prune_edge_list": kept,
            "prune_cap_5pct": cap,
            "S_bottom10_threshold": s10,
            "growth_bottleneck_targets": len(targets),
            "growth_candidate_edges": len(growth_edges),
            "growth_edge_list": growth_edges,
            "growth_src_pool_empty": n_src_pool_empty,
            "growth_calib_reject": n_calib_fail,
            "growth_cap": growth_cap,
        }


def probe(model: ProbeTND, calib_x, calib_y, growth_cap: int = K_GROWTH) -> dict[str, Any]:
    """跑一次探针：返回统计量 + 候选（可直接喂给 rewire）。"""
    loss_calib, all_o, grad_o, grad_z = collect_grads(model, calib_x, calib_y)
    stacked_o = torch.stack(all_o, dim=1)
    per_sample_loss = nn.MSELoss(reduction="none")(
        all_o[-1][:, model.output_indices], calib_y).mean(dim=1)
    stats = edge_stats_and_candidates(model, stacked_o, grad_o, grad_z, per_sample_loss,
                                      growth_cap=growth_cap)
    stats["calib_loss"] = float(loss_calib.item())
    return stats