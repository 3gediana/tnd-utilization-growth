"""TND 按利用率生长的最小实现（可直接推 GitHub 的实验代码）。

入口：
    python -m tnd_growth --smoke     # CPU 冒烟：1 条命令，几分钟内跑完四臂
    python -m tnd_growth --full --seed 45 --checkpoint /path/to/snn_dt_model.pth

设计边界见 README.md / METHOD.md；结果的失效边界见 notes/NEGATIVE-CORNER.md。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiment import run
from .probe import K_GROWTH


def parse_episode_seeds(spec: str) -> list[int]:
    spec = str(spec).strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tnd_growth",
        description="TND 固定随机拓扑 vs 利用率驱动生长：最小可跑实现")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="CPU 冒烟：小步数四臂，验证管线（不是论文数字）")
    mode.add_argument("--full", action="store_true",
                      help="完整单 seed 协议（190 步/臂 + 10 局评测）")

    p.add_argument("--seed", type=int, default=45,
                   help="训练流 seed（batch 序=RandomState(seed) 固定置换）")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="可选：TND 预训练 checkpoint（.pth）；不给则从 seed 初始化")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="无 checkpoint 时先训练的步数（默认 smoke=30；full 协议固定 2000=Stage A）")
    p.add_argument("--steps", type=int, default=None, help="每臂训练步数（默认 smoke=60, full=190）")
    p.add_argument("--episode-seeds", type=str, default=None,
                   help="评测局表：'2026-2035' 或 '2026,2027'（默认 smoke=3 局, full=10 局）")
    p.add_argument("--max-steps-per-episode", type=int, default=None,
                   help="每局步数（默认 smoke=800, full=2000）")
    p.add_argument("--k", type=int, default=K_GROWTH, help="剪/长条数 k（默认 10）")
    p.add_argument("--latency-rounds", type=int, default=3)
    p.add_argument("--latency-calls", type=int, default=None,
                   help="每轮时延测量次数（默认 smoke=100, full=300）")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data", type=str, default=None, help="c2.txt 路径（默认 repo/data/c2.txt）")
    p.add_argument("--out", type=str, default=None, help="结果 JSON 路径")
    args = p.parse_args(argv)

    if not args.smoke and not args.full:
        p.print_help()
        return 2

    if args.smoke:
        cfg = dict(label="smoke", steps=60, warmup_steps=30, episode_seeds=[2026, 2027, 2028],
                   max_steps_per_episode=800, latency_calls=100)
        default_out = repo_root() / "results" / "smoke_run.json"
    else:
        # 完整协议：Stage A = 从 checkpoint 再训练 2000 步 → 探针；Stage B = 四臂各 190 步
        cfg = dict(label="full", steps=190, warmup_steps=2000,
                   episode_seeds=parse_episode_seeds("2026-2035"),
                   max_steps_per_episode=2000, latency_calls=300)
        default_out = repo_root() / "results" / f"full_seed{args.seed}.json"

    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.warmup_steps is not None:
        cfg["warmup_steps"] = args.warmup_steps
    if args.episode_seeds:
        cfg["episode_seeds"] = parse_episode_seeds(args.episode_seeds)
    if args.max_steps_per_episode is not None:
        cfg["max_steps_per_episode"] = args.max_steps_per_episode
    if args.latency_calls is not None:
        cfg["latency_calls"] = args.latency_calls
    if not args.checkpoint and cfg["warmup_steps"] == 0:
        # 没有 checkpoint 又不热身 → 等于评一个随机初始化的网络；冒烟可以，完整跑要提醒
        print("[warn] no --checkpoint and --warmup-steps=0: model is random-init", file=sys.stderr)

    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = run(seed=args.seed, checkpoint=args.checkpoint, steps=cfg["steps"],
                  warmup_steps=cfg["warmup_steps"], episode_seeds=cfg["episode_seeds"],
                  max_steps_per_episode=cfg["max_steps_per_episode"], k=args.k,
                  latency_rounds=args.latency_rounds, latency_calls=cfg["latency_calls"],
                  threads=args.threads, data_path=args.data, out_path=str(out), label=cfg["label"])

    if not args.smoke:
        return 0

    # ---- 冒烟硬不变量：机制必须成立，catch rate 只是附带的管线读数 ----
    arms = payload["arms"]
    b0, b1, b2, b3 = arms["B0_fixed"], arms["B1_prune_only"], arms["B2_grow_only"], arms["B3_prune_grow"]
    k = b2["k"]
    checks = [
        ("probe deterministic", payload["probe_deterministic"], "两次探针候选完全一致"),
        ("stage-B gate", payload["gate"]["stage_b_trigger"],
         f"解离 {payload['gate']['dissociation_ratio_hidden']:.4f} / "
         f"prune {payload['gate']['prune_candidates']} / grow {payload['gate']['growth_candidates']}"),
        ("B0 unchanged", b0["edges_after"] == b0["edges_before"],
         f"{b0['edges_before']} -> {b0['edges_after']}"),
        ("B1 prunes k", b1["edges_after"] == b1["edges_before"] - k,
         f"{b1['edges_before']} -> {b1['edges_after']} (k={k})"),
        ("B2 conserves edges", b2["edges_after"] == b2["edges_before"],
         f"{b2['edges_before']} -> {b2['edges_after']}"),
        ("B3 conserves edges", b3["edges_after"] == b3["edges_before"],
         f"{b3['edges_before']} -> {b3['edges_after']}"),
        ("B2 grows k edges", b2["topology"]["added_edges"] == k
         and b2["topology"]["removed_edges"] == k,
         f"added={b2['topology']['added_edges']} removed={b2['topology']['removed_edges']}"),
        ("B3 grows+prunes k", b3["topology"]["added_edges"] == k
         and b3["topology"]["removed_edges"] == k,
         f"added={b3['topology']['added_edges']} removed={b3['topology']['removed_edges']}"),
        ("B1 topology only removes", b1["topology"]["added_edges"] == 0
         and b1["topology"]["removed_edges"] == k,
         f"added={b1['topology']['added_edges']} removed={b1['topology']['removed_edges']}"),
        ("new weights nonzero", all(abs(w) > 0 for w in
                                   [b2["topology"]["new_edge_weight_min"], b2["topology"]["new_edge_weight_max"]]
                                   if w is not None),
         f"range=[{b2['topology']['new_edge_weight_min']}, {b2['topology']['new_edge_weight_max']}]"),
        ("params constant", len({a["params"] for a in arms.values()}) == 1,
         f"{b0['params']:,}"),
        ("spectral radius <= 0.95", all(a["rewire"]["spectral_radius_post_norm"] <= 0.950001
                                       for a in arms.values()),
         ", ".join(f"{n}={a['rewire']['spectral_radius_post_norm']:.6f}" for n, a in arms.items())),
        ("eval ran", all(a["eval"]["num_episodes"] == len(cfg["episode_seeds"]) for a in arms.values()),
         f"{len(cfg['episode_seeds'])} episodes/arm"),
    ]
    print("\n[smoke checks]")
    ok = True
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
        ok = ok and bool(passed)
    print(f"[smoke checks] {'ALL PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(1 for _, p, _ in checks if p)}/{len(checks)})")
    print("[smoke note] catch rate 在无 checkpoint 的冒烟里不具备科学含义："
          "模型只热了几十步，策略退化为常动作，四臂数字会相同。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())