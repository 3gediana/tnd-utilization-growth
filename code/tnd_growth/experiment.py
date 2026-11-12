"""实验臂的执行器：训练 → 探针 → 重连 → 再训练 → 配对评测 → 时延。

四臂（同起点、同数据序、同步数、同评测局表）：
  B0 固定拓扑（对照）
  B1 只剪枝（剪掉最低贡献的 k 条边，不补）
  B2 只生长（长 k 条边，长 k 换 k，边数守恒）
  B3 剪+长（先剪 k 条再长 k 条，边数守恒）
"""
from __future__ import annotations

import copy
import json
import time
from statistics import median
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import pong
from .data import batch_order, prepare_data
from .model import TND
from .probe import K_GROWTH, ProbeTND, probe
from .rewire import apply_spectral_normalization, rewire

OPTIMIZER = dict(lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
TRAIN_SEQS = 200
NUM_INPUTS, NUM_OUTPUTS = 24, 3
CALIB_BATCHES, CALIB_SEQS = 5, 50


def set_seed(seed: int):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    import random
    random.seed(int(seed))


def build_model(seed: int = 0) -> ProbeTND:
    m = ProbeTND(num_neurons=600, num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS,
                 dt=0.8, connection_prob=0.4, seed=seed)
    return m


def train_model(model, x_all, y_all, num_batches: int, n_steps: int,
                order_seed: int | None = None, train_seqs: int = TRAIN_SEQS,
                record_history: bool = False) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
    criterion = nn.MSELoss()
    model.train()
    step = 0
    first_loss = last_loss = None
    losses: list[float] = []
    t0 = time.time()
    order = batch_order(num_batches, order_seed)
    while step < n_steps:
        for idx in order:
            if step >= n_steps:
                break
            batch_x = x_all[idx, :train_seqs]
            batch_y = y_all[idx, :train_seqs]
            optimizer.zero_grad()
            outputs, states, neuron_outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if first_loss is None:
                first_loss = float(loss.item())
            last_loss = float(loss.item())
            if record_history:
                losses.append(last_loss)
            states = states.detach()
            neuron_outputs = neuron_outputs.detach()
    info = {
        "steps": step,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "train_seconds": round(time.time() - t0, 2),
        "batch_order_mode": ("original" if order_seed is None
                             else f"RandomState({order_seed}) fixed permutation, same each epoch"),
    }
    if record_history:
        info["loss_history"] = losses
    return info


def measure_latency(model, calls: int = 300, rounds: int = 1, device: str = "cpu") -> dict[str, Any]:
    """单步推理时延（ms）：固定输入、固定线程；rounds>1 取各轮中位数的中位。"""
    model.eval()
    tmp = []
    tmp.extend(pong.int_to_binary_list(50))
    tmp.extend(pong.int_to_binary_list(60))
    tmp.extend(pong.int_to_binary_list(75))
    x_input = torch.tensor(tmp, dtype=torch.float32).reshape(1, -1, 24)
    states = torch.zeros(1, model.num_neurons)
    neuron_outputs = torch.zeros(1, model.num_neurons)

    per_round = []
    with torch.no_grad():
        for _ in range(rounds):
            for _ in range(20):
                model.pred(x_input, states, neuron_outputs)
            ts = []
            for _ in range(calls):
                t0 = time.perf_counter()
                model.pred(x_input, states, neuron_outputs)
                ts.append((time.perf_counter() - t0) * 1000.0)
            ts.sort()
            per_round.append({
                "median_ms": float(median(ts)),
                "p95_ms": float(ts[min(int(0.95 * len(ts)), len(ts) - 1)]),
            })
    return {
        "median_ms": float(median([r["median_ms"] for r in per_round])),
        "p95_ms": float(median([r["p95_ms"] for r in per_round])),
        "rounds_used": rounds,
        "calls_per_round": calls,
        "rounds": per_round,
    }


def arm_specs(prune_edges, growth_edges, k: int = K_GROWTH) -> dict[str, dict[str, Any]]:
    kk = min(int(k), len(prune_edges), len(growth_edges))
    return {
        "B0_fixed": {"prune": [], "grow": [], "conserve": False, "k": 0},
        "B1_prune_only": {"prune": prune_edges[:kk], "grow": [], "conserve": False, "k": kk},
        "B2_grow_only": {"prune": [], "grow": growth_edges[:kk], "conserve": True, "k": kk},
        "B3_prune_grow": {"prune": prune_edges[:kk], "grow": growth_edges[:kk],
                          "conserve": True, "k": kk},
    }


def run(seed: int = 45, checkpoint: str | None = None, warmup_steps: int = 0,
        steps: int = 190, episode_seeds: list[int] | None = None,
        max_steps_per_episode: int = pong.MAX_STEPS, k: int = K_GROWTH,
        latency_rounds: int = 3, latency_calls: int = 300, threads: int = 4,
        data_path: str | None = None, arms: list[str] | None = None,
        out_path: str | None = None, label: str = "run") -> dict[str, Any]:
    torch.set_num_threads(int(threads))
    set_seed(seed)
    t_start = time.time()
    episode_seeds = list(episode_seeds) if episode_seeds else [2026, 2027, 2028]

    print("=" * 68)
    print(f"TND growth experiment [{label}] seed={seed} steps={steps} "
          f"episodes={len(episode_seeds)}x{max_steps_per_episode}")
    print("=" * 68)

    x_all, y_all, num_batches = prepare_data(data_path)
    print(f"[data] batches={num_batches} x={tuple(x_all.shape)} y={tuple(y_all.shape)}")

    model = build_model(seed=seed)
    if checkpoint:
        model.load_checkpoint(checkpoint)
        print(f"[model] loaded checkpoint: {checkpoint}")
    else:
        print("[model] no checkpoint: init from seed" + (f" + {warmup_steps} warmup steps" if warmup_steps else ""))
        if warmup_steps > 0:
            info = train_model(model, x_all, y_all, num_batches, warmup_steps, order_seed=seed)
            print(f"[warmup] steps={info['steps']} loss {info['first_loss']:.6f} -> "
                  f"{info['last_loss']:.6f} ({info['train_seconds']}s)")

    print(f"[model] edges={model.edge_count()} params={sum(p.numel() for p in model.parameters()):,}")

    calib_x = x_all[:CALIB_BATCHES, :CALIB_SEQS].reshape(-1, 40, NUM_INPUTS).contiguous()
    calib_y = y_all[:CALIB_BATCHES, :CALIB_SEQS].reshape(-1, NUM_OUTPUTS).contiguous()
    model.eval()
    stats = probe(model, calib_x, calib_y, growth_cap=max(k, K_GROWTH))
    stats_repeat = probe(model, calib_x, calib_y, growth_cap=max(k, K_GROWTH))
    probe_deterministic = (
        json.dumps(stats["prune_edge_list"], sort_keys=True)
        == json.dumps(stats_repeat["prune_edge_list"], sort_keys=True)
        and json.dumps(stats["growth_edge_list"], sort_keys=True)
        == json.dumps(stats_repeat["growth_edge_list"], sort_keys=True))
    print(f"[probe] calib_loss={stats['calib_loss']:.6f} "
          f"dissoc_hidden={stats['hidden_dissociation_ratio']:.4f}"
          f"({stats['hidden_dissociation_count']}/{stats['hidden_n']}) "
          f"prune_candidates={stats['prune_candidates_strict']} "
          f"growth_candidates={stats['growth_candidate_edges']}")
    gate = {
        "dissociation_ratio_hidden": stats["hidden_dissociation_ratio"],
        "dissociation_count_hidden": stats["hidden_dissociation_count"],
        "prune_candidates": stats["prune_candidates_strict"],
        "growth_candidates": stats["growth_candidate_edges"],
        "stage_b_trigger": bool(stats["hidden_dissociation_ratio"] >= 0.01
                                and (stats["prune_candidates_strict"] > 0
                                     or stats["growth_candidate_edges"] > 0)),
    }
    print(f"[gate] stage_b_trigger={gate['stage_b_trigger']} "
          f"probe_deterministic={probe_deterministic}")

    specs = arm_specs(stats["prune_edge_list"], stats["growth_edge_list"], k=k)
    if arms:
        specs = {name: specs[name] for name in specs if name in arms}
    if specs["B3_prune_grow"]["k"] < k:
        print(f"[warn] k_used={specs['B3_prune_grow']['k']} < requested k={k} "
              f"(候选不足：prune={len(stats['prune_edge_list'])}, growth={len(stats['growth_edge_list'])})")

    results: dict[str, Any] = {}
    base_adj = model.adj_mask.clone()
    for name, spec in specs.items():
        print(f"\n----- arm {name} (k={spec['k']}) -----", flush=True)
        m = copy.deepcopy(model)
        E0 = m.edge_count()
        wnorm0 = float(m.weights.norm().item())
        sr0 = m.spectral_radius()

        rew = rewire(m, spec["prune"], spec["grow"], spec["conserve"])
        pre_norm_sr = m.spectral_radius()
        apply_spectral_normalization(m)
        post_norm_sr = m.spectral_radius()   # 归一化后立刻读；训练后权重会再变
        E1 = m.edge_count()
        added = int(((m.adj_mask == 1) & (base_adj == 0)).sum().item())
        removed = int(((m.adj_mask == 0) & (base_adj == 1)).sum().item())
        new_w = [float(m.weights[i, j].item()) for i, j in
                 zip(((m.adj_mask == 1) & (base_adj == 0)).nonzero(as_tuple=True)[0].tolist(),
                     ((m.adj_mask == 1) & (base_adj == 0)).nonzero(as_tuple=True)[1].tolist())]
        print(f"[graph] edges {E0}->{E1} (pruned={rew['pruned']} grown={rew['grown']} "
              f"compensated={rew['compensated_edges']}; added={added} removed={removed}) "
              f"spec_radius {pre_norm_sr:.4f}->{m.spectral_radius():.4f}")

        train_info = train_model(m, x_all, y_all, num_batches, steps, order_seed=seed,
                                 record_history=True)
        print(f"[train] {train_info['steps']} steps loss {train_info['first_loss']:.6f} -> "
              f"{train_info['last_loss']:.6f} ({train_info['train_seconds']}s)")

        ev = pong.evaluate(m, episode_seeds, max_steps=max_steps_per_episode)
        print(f"[eval] mean_catch={ev['mean_catch_rate']:.4f} "
              f"per_ep={[round(e['catch_rate'], 4) for e in ev['episodes']]}")

        lat = measure_latency(m, calls=latency_calls, rounds=latency_rounds)
        print(f"[latency] median={lat['median_ms']:.3f}ms p95={lat['p95_ms']:.3f}ms "
              f"({latency_rounds} rounds x {latency_calls} calls)")

        results[name] = {
            "arm": name,
            "k": spec["k"],
            "spec": {"prune_edges": len(spec["prune"]), "grow_edges": len(spec["grow"]),
                     "conserve_budget": spec["conserve"]},
            "edges_before": E0, "edges_after": E1,
            "topology": {
                "added_edges": added,
                "removed_edges": removed,
                "new_edge_weight_min": (min(new_w) if new_w else None),
                "new_edge_weight_max": (max(new_w) if new_w else None),
            },
            "params": int(sum(p.numel() for p in m.parameters())),
            "rewire": {**rew, "spectral_radius_pre_norm": pre_norm_sr,
                       "spectral_radius_post_norm": post_norm_sr},
            "weight_norm_before": wnorm0, "spectral_radius_before": sr0,
            "train": train_info,
            "eval": ev,
            "latency": lat,
        }

    b0 = results.get("B0_fixed")
    deltas = {}
    if b0:
        for name, r in results.items():
            if name == "B0_fixed":
                continue
            diffs = [e[1]["catch_rate"] - e[0]["catch_rate"]
                     for e in zip(b0["eval"]["episodes"], r["eval"]["episodes"])]
            deltas[name] = {
                "mean_diff_pp": round(float(np.mean(diffs)) * 100, 2),
                "n_episodes_positive": int(sum(1 for d in diffs if d > 0)),
                "n_pairs": len(diffs),
                "latency_increase_pct": round(
                    (r["latency"]["median_ms"] - b0["latency"]["median_ms"])
                    / b0["latency"]["median_ms"] * 100, 2) if b0["latency"]["median_ms"] else None,
                "per_episode": [
                    {"episode_seed": a["episode_seed"],
                     "B0": a["catch_rate"], name: b["catch_rate"], "diff": b["catch_rate"] - a["catch_rate"]}
                    for a, b in zip(b0["eval"]["episodes"], r["eval"]["episodes"])
                ],
            }
        print("\n[paired vs B0]")
        for name, d in deltas.items():
            print(f"  {name}: {d['mean_diff_pp']:+.2f}pp "
                  f"({d['n_episodes_positive']}/{d['n_pairs']} episodes positive), "
                  f"latency {d['latency_increase_pct']:+.2f}%")

    payload = {
        "label": label,
        "seed": seed,
        "checkpoint": checkpoint,
        "config": {
            "neurons": 600, "inputs": NUM_INPUTS, "outputs": NUM_OUTPUTS, "dt": 0.8,
            "p_connect": 0.4, "steps_per_arm": steps, "train_seqs": TRAIN_SEQS,
            "k": k, "episode_seeds": episode_seeds,
            "max_steps_per_episode": max_steps_per_episode,
            "latency_rounds": latency_rounds, "latency_calls": latency_calls,
            "threads": threads, "torch": torch.__version__,
        },
        "probe": {kk: vv for kk, vv in stats.items() if kk != "prune_edge_list"
                  and kk != "growth_edge_list"},
        "gate": gate,
        "probe_deterministic": bool(probe_deterministic),
        "arms": results,
        "paired_vs_B0": deltas,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[output] wrote {out_path}")
    print(f"[done] {payload['elapsed_seconds']}s")
    return payload