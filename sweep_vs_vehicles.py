#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sweep vehicles and compare models (GAT vs SAGE):
- For vehicle counts [20,40] (可通过 --veh-list 调整)
- Train (short) then evaluate average V2I/V2V
- Benchmark decision time under "complete" vs "incomplete" graph
  * kind=full: 端到端（逐节点 DQN 决策）
  * kind=gnn: 仅 GNN 批量前向推理（embedding），更客观反映 GNN 本体速度
Outputs:
- runs/sweep/sweep_results.csv
- runs/sweep/decision_time.csv
"""

import argparse
import csv
import time
from pathlib import Path
import random

import numpy as np
import tensorflow as tf

from highway_environment import HighwayTopoEnv
from agent import Agent
from gnn_factory import build_gnn


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def build_env(n_up, n_down, base_y, height, topo, bs_layout, bs_spacing):
    env = HighwayTopoEnv(
        n_up=n_up, n_down=n_down, lanes_per_dir=4, spacing=20.0,
        base_y=base_y, height=height, topology_type=topo,
        v2i_mode="rsu", bs_layout=bs_layout, bs_spacing=bs_spacing
    )
    env.new_random_game()
    return env


def parse_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def _safe_v2v_success_from_record(rec) -> float:
    """
    兼容不同版本 Agent.test_history 的第三项定义：
    - 若在 [0,1] 内，视为已经是 success rate
    - 若在 (-inf, +inf) 且不在 [0,1]，视为 fail ratio，返回 1 - x
    - 最终做 [0,1] 截断
    """
    try:
        x = float(rec[2])
    except Exception:
        if isinstance(rec, dict):
            x = None
            for k in ("v2v_success", "v2v", "v2v_mean", "success"):
                if k in rec:
                    x = float(rec[k])
                    break
            if x is None:
                for k in ("fail", "v2v_fail", "fail_ratio"):
                    if k in rec:
                        x = 1.0 - float(rec[k])
                        break
            if x is None:
                return 0.0
        else:
            return 0.0
    v = x if 0.0 <= x <= 1.0 else 1.0 - x
    return max(0.0, min(1.0, v))


def main():
    ap = argparse.ArgumentParser(description="Sweep vehicles and compare GAT vs GraphSAGE.")
    ap.add_argument("--veh-list", default="20,40", help="Comma-separated vehicle counts, e.g., 20,40")
    ap.add_argument("--topo", default="star", choices=["star", "tree"])
    ap.add_argument("--base-y", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1000.0)
    ap.add_argument("--bs-layout", default="median", choices=["median", "dual-roadside"])
    ap.add_argument("--bs-spacing", type=float, default=250.0)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--test-sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2025, help="Base seed for training randomness")
    ap.add_argument("--eval-seed", type=int, default=4242, help="Base seed for test sampling; enforce same test set across models")
    ap.add_argument("--out-dir", default="runs/sweep")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    veh_list = parse_list(args.veh_list)

    sweep_rows = []
    dec_rows = []

    for total in veh_list:
        # 为该车辆规模固定一组评测随机种子，两模型复用
        seed_train_base = int(args.seed + total * 1000)
        seed_eval_base = int(args.eval_seed + total * 1000)

        for model in ["gat", "sage"]:
            # 训练与评测前分别设定随机种子
            set_all_seeds(seed_train_base)
            set_all_seeds(seed_eval_base)

            n_up = total // 2
            n_down = total - n_up
            env = build_env(n_up, n_down, args.base_y, args.height, args.topo, args.bs_layout, args.bs_spacing)

            # 初始化 Agent（speed_mode=True 走快速模式）
            agent = Agent([], env, gnn_type=model, warmup_steps=500, epsilon_decay_steps=8000, speed_mode=True)

            # 训练 + 中期一次评测（test_every_steps）
            agent.train(max_steps=args.train_steps, test_every_steps=max(1, args.train_steps // 2), test_sample=args.test_sample)

            # 读取最后一次测试结果
            if getattr(agent, "test_history", None):
                rec = agent.test_history[-1]
                try:
                    v2i_mean = float(rec[1])
                except Exception:
                    if isinstance(rec, dict):
                        v2i_mean = float(rec.get("v2i_mean", 0.0))
                    else:
                        v2i_mean = 0.0
                v2v_succ = _safe_v2v_success_from_record(rec)
            else:
                v2i_mean, v2v_succ = 0.0, 0.0

            sweep_rows.append(dict(
                model=model, n_veh=total, v2i_mean=v2i_mean, v2v_mean=v2v_succ,
                seed_train=seed_train_base, seed_eval=seed_eval_base
            ))

            # 决策时间对比：构造两种图连边密度
            for mode, thr in [("incomplete", 150.0), ("complete", 1e9)]:
                # 重新构建 GNN 并初始化图（不改 DQN）
                agent.G = build_gnn(env, gnn_type=model, distance_threshold=thr, lr=5e-4, gat_train_interval=20)
                _ = agent.initial_better_state(0, True)

                # 更新整图节点特征（一次性）
                for i in range(len(env.vehicles)):
                    for j in range(3):
                        s_old = agent.get_state([i, j])
                        agent.G.features[3 * i + j, :] = s_old[:60]

                # 仅 GNN 批量前向时间（兼容无 _forward_all 的 SAGE）
                t0 = time.perf_counter()
                if hasattr(agent.G, "_forward_all"):
                    _ = agent.G._forward_all(training=False).numpy()
                else:
                    idx_all = list(range(3 * len(env.vehicles)))
                    _ = agent.G.use_GraphSAGE(agent.channel_reward, 0, idx_all, False)
                t1 = time.perf_counter()
                per_node_gnn = (t1 - t0) / max(1, len(env.vehicles) * 3)
                dec_rows.append(dict(
                    model=model, n_veh=total, graph=mode, kind="gnn", decision_time_s=per_node_gnn,
                    seed_train=seed_train_base, seed_eval=seed_eval_base
                ))

                # 端到端逐节点（Full decision）
                t0 = time.perf_counter()
                for i in range(len(env.vehicles)):
                    for j in range(3):
                        idx = [3 * i + j]
                        emb = agent.G.use_GraphSAGE(agent.channel_reward, 0, idx, False).squeeze()
                        emb = emb / (np.max(np.abs(emb)) + 1e-4)
                        s_old = agent.get_state([i, j])
                        _ = agent.predict(np.concatenate((emb, s_old), axis=0), 0, test_ep=True)
                t1 = time.perf_counter()
                per_decision_full = (t1 - t0) / max(1, len(env.vehicles) * 3)
                dec_rows.append(dict(
                    model=model, n_veh=total, graph=mode, kind="full", decision_time_s=per_decision_full,
                    seed_train=seed_train_base, seed_eval=seed_eval_base
                ))

    # 保存
    sweep_csv = out_dir / "sweep_results.csv"
    with sweep_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_veh", "v2i_mean", "v2v_mean", "seed_train", "seed_eval"])
        w.writeheader()
        w.writerows(sweep_rows)

    dec_csv = out_dir / "decision_time.csv"
    with dec_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_veh", "graph", "kind", "decision_time_s", "seed_train", "seed_eval"])
        w.writeheader()
        w.writerows(dec_rows)

    print("[OK] Saved:", sweep_csv, dec_csv)


if __name__ == "__main__":
    main()