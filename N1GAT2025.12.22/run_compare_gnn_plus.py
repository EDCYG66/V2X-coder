#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare GATv2 vs GraphSAGE with flat output (single folder).

变更要点（相对你当前版本）：
- 不再创建 curve_{tag} 子目录；所有产出写入同一个 out_dir。
- 以文件名前缀区分模型：gat_* 与 graphsage_*。
- 对比图沿用 compare_* 命名，放在同一 out_dir。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from highway_environment import HighwayTopoEnv
from agent import Agent
from gnn_factory import build_gnn


# ================= 工具函数 =================

def compute_entropy(prob: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    s = prob.sum()
    if s <= 0:
        return 0.0
    p = prob / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_gini(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return 0.0
    s = counts.sum()
    if s <= 0:
        return 0.0
    diffs = np.abs(counts[:, None] - counts[None, :])
    return float(diffs.sum() / (2 * counts.size * s))


def compute_convergence_step(steps: List[int], values: List[float], ratio: float = 0.9) -> int:
    if not steps or not values:
        return -1
    final_v = values[-1]
    thr = ratio * final_v
    for s, v in zip(steps, values):
        if v >= thr:
            return int(s)
    return int(steps[-1])


def ensure_env_difficulty(env: HighwayTopoEnv, demand_amount: float, v2v_limit: float):
    env.demand_amount = float(demand_amount)
    env.demand = env.demand_amount * np.ones((env.n_Veh, 3))
    env.V2V_limit = float(v2v_limit)
    env.individual_time_limit = env.V2V_limit * np.ones((env.n_Veh, 3))
    if hasattr(env, "activate_links"):
        env.activate_links[:] = False


def parse_list_int(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_int_strict(s: str) -> List[int]:
    try:
        return parse_list_int(s)
    except Exception:
        return []


# ================ 带统计的 Agent（复用原 Agent 能力） ================

class InstrumentedAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rb_usage_counts = np.zeros(self.RB_number, dtype=np.int64)
        self.full_decision_time_acc = []
        self.gnn_only_time_acc = []
        self.v2i_curve_steps = []
        self.v2i_curve_vals = []
        self.v2v_curve_vals = []
        # 保存一次详细测试时序（来自 test_environment(detailed=True)）
        self.last_timeseries = None

    def _collect_power_and_rb(self, i: int, j: int, s_old_raw: np.ndarray, action: int):
        rb = action % self.RB_number
        pw = action // self.RB_number
        self.action_all_with_power_training[i, j, 0] = rb
        self.action_all_with_power_training[i, j, 1] = pw
        if self.power_log_stride and (self.step % self.power_log_stride == 0):
            time_left_s = float(s_old_raw[-1]) * float(self.env.V2V_limit)
            self.power_log.append((time_left_s, pw))
            if len(self.power_log) > self.power_log_max:
                self.power_log = self.power_log[-self.power_log_max:]
        if 0 <= rb < self.RB_number:
            self.rb_usage_counts[rb] += 1

    def train(self, max_steps=1000, test_every_steps=200, test_sample=80):
        self.dqn.update_target_network()
        self.warmup()

        self.env.new_random_game()
        _ = self.initial_better_state(0, True)

        for self.step in range(1, max_steps + 1):
            is_test = (test_every_steps > 0 and self.step % test_every_steps == 0)

            if is_test:
                self.training = False
                mean_v2i, fail = self.test_environment(test_sample=test_sample, detailed=False)
                self.test_history.append((self.step, float(mean_v2i), float(fail)))
                self.v2i_curve_steps.append(self.step)
                self.v2i_curve_vals.append(float(mean_v2i))
                self.v2v_curve_vals.append(float(1.0 - fail))
                self.training = True

            # 刷新节点特征
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    s = self.get_state([i, j])
                    self.G.features[3 * i + j, :] = s[:60]

            # GNN 训练（间隔）
            interval = getattr(self.G, "gat_train_interval", 20)
            if self.GraphSAGE and (self.step % interval == 0) and (not is_test):
                idx_all = list(range(3 * len(self.env.vehicles)))
                _ = self.G.use_GraphSAGE(self.channel_reward, self.step, idx_all, True)

            # 整图前向计时
            N = 3 * len(self.env.vehicles)
            t0 = time.perf_counter()
            if hasattr(self.G, "_forward_all"):
                emb_all_t = self.G._forward_all(training=False)
                emb_all = emb_all_t.numpy() if hasattr(emb_all_t, "numpy") else np.asarray(emb_all_t)
            else:
                idx_all = list(range(N))
                emb_all = self.G.use_GraphSAGE(self.channel_reward, self.step, idx_all, False)
            t1 = time.perf_counter()
            emb_all = emb_all / (np.max(np.abs(emb_all)) + 1e-4)
            self.gnn_only_time_acc.append((t1 - t0) / max(1, N))

            # 动作选择与记录
            t2 = time.perf_counter()
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    s_old = self.get_state([i, j])
                    emb_old = emb_all[3 * i + j]
                    better_old = np.concatenate((emb_old, s_old), axis=0)
                    action = self.predict(better_old, self.step)
                    self._collect_power_and_rb(i, j, s_old, action)
            t3 = time.perf_counter()
            self.full_decision_time_acc.append((t3 - t2) / max(1, N))

            # 环境推进（逐链路）
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    _ = self.env.act_for_training(self.action_all_with_power_training, [i, j])

            # DQN 更新
            if self.step >= self.warmup_steps and self.step % self.train_every_n_steps == 0:
                self.q_learning_mini_batch()
            if self.step % self.target_q_update_step == 0:
                self.dqn.update_target_network()

            used_blocks = np.unique(self.action_all_with_power_training[:, :, 0])
            self.used_blocks_history.append((self.step, len(used_blocks)))

        # 最后一次详细测试，获取时序
        mean_v2i, fail = self.test_environment(test_sample=test_sample, detailed=True)
        self.test_history.append((self.step, float(mean_v2i), float(fail)))
        self.v2i_curve_steps.append(self.step)
        self.v2i_curve_vals.append(float(mean_v2i))
        self.v2v_curve_vals.append(float(1.0 - fail))
        # 从父类里取 _last_test_detailed（t, v2i, v2v_succ, nveh）
        if hasattr(self, "_last_test_detailed") and self._last_test_detailed:
            self.last_timeseries = self._last_test_detailed.copy()

        self._export_results(mean_v2i, fail)


# ================= 单次运行（训练+评测+指标） =================

def run_one(model_type: str, args: argparse.Namespace, out_dir: Path) -> Dict:
    tag = model_type.lower()
    prefix = f"{tag}_"  # 关键：统一目录下用前缀区分

    env = HighwayTopoEnv(
        n_up=args.n_up,
        n_down=args.n_down,
        lanes_per_dir=args.lanes,
        spacing=args.spacing,
        base_y=args.base_y,
        height=args.height,
        topology_type=args.topo,
        leader_at_front=bool(args.leader_front),
        leader_lane_up=args.leader_lane_up,
        leader_lane_down=args.leader_lane_down,
        leader_dynamic=bool(args.leader_dynamic),
        move_speed=float(args.use_move_speed),
        v2i_mode=args.v2i_mode,
        bs_layout=args.bs_layout,
        bs_spacing=args.bs_spacing,
        bs_min_stay_steps=args.bs_min_stay,
        bs_handover_hyst_m=args.bs_hyst,
        seed=args.seed
    )
    env.new_random_game()
    ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

    # 注入奖励相关超参到 env（batch_reward_all 会通过 getattr 读取）
    env.beta_urgency_pos = float(args.beta_urgency_pos)
    env.beta_urgency_neg = float(args.beta_urgency_neg)
    env.urgency_threshold = float(args.urgency_threshold)
    env.rb_anti_conc_alpha = float(args.rb_anti_conc_alpha)
    env.rb_hot_threshold = float(args.rb_hot_threshold)
    env.rb_softmask_alpha = float(args.rb_softmask_alpha)
    env.rb_softmask_window = int(args.rb_softmask_window)

    # 构建 GNN：通过 CLI 透传 GraphGAT 的重剪枝参数
    G = build_gnn(
        env,
        gnn_type=tag,
        distance_threshold=150.0,
        lr=5e-4,
        gat_train_interval=20,
        grad_clip=5.0,
        reprune_every=args.reprune_every,
        hysteresis_keep=args.hysteresis_keep,
        reprune_start_step=args.reprune_start_step,
        reg_attn_w=args.reg_attn_w,
    )

    # Agent
    agent = InstrumentedAgent(
        [],
        env,
        gnn_type=tag,
        warmup_steps=args.warmup_steps,
        epsilon_decay_steps=2000,
        speed_mode=False,
        plot_dpi=args.dpi,
        beta_urgency_pos=args.beta_urgency_pos,
        beta_urgency_neg=args.beta_urgency_neg,
        urgency_threshold=args.urgency_threshold,
        rb_anti_conc_alpha=args.rb_anti_conc_alpha,
        rb_hot_threshold=args.rb_hot_threshold,
        rb_softmask_alpha=args.rb_softmask_alpha,
        rb_softmask_window=args.rb_softmask_window,
    )
    agent.G = G

    agent.train(max_steps=args.train_steps, test_every_steps=args.test_every, test_sample=args.test_sample)

    # 扩展指标
    power_counts = np.zeros(3, dtype=np.int64)
    for _, pw in agent.power_log:
        if 0 <= pw < 3:
            power_counts[pw] += 1
    power_entropy = compute_entropy(power_counts)
    rb_gini = compute_gini(agent.rb_usage_counts)
    unique_rb_ratio = float(np.mean([c / agent.RB_number for _, c in agent.used_blocks_history]))
    conv_step_v2i = compute_convergence_step(agent.v2i_curve_steps, agent.v2i_curve_vals, 0.9)
    full_mean = float(np.mean(agent.full_decision_time_acc)) if agent.full_decision_time_acc else 0.0
    gnn_only_mean = float(np.mean(agent.gnn_only_time_acc)) if agent.gnn_only_time_acc else 0.0

    summary = dict(
        model=tag,
        v2i_final=float(agent.v2i_curve_vals[-1]),
        v2v_final=float(agent.v2v_curve_vals[-1]),
        v2i_conv_step=int(conv_step_v2i),
        power_entropy=float(power_entropy),
        power_counts=power_counts.tolist(),
        rb_unique_ratio=float(unique_rb_ratio),
        rb_gini=float(rb_gini),
        decision_time_full_s=full_mean,
        decision_time_gnn_only_s=gnn_only_mean,
        steps=len(agent.v2i_curve_steps),
        v2i_curve=list(zip(agent.v2i_curve_steps, agent.v2i_curve_vals)),
        v2v_curve=list(zip(agent.v2i_curve_steps, agent.v2v_curve_vals)),
        rb_usage_counts=agent.rb_usage_counts.tolist(),
        last_timeseries=agent.last_timeseries if agent.last_timeseries else None,
        agent_ref=agent,  # 供后续“完全图 vs 非完全图”耗时对比使用
    )

    # 不再创建子目录；按前缀写入 out_dir
    (out_dir).mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{prefix}summary.json").open("w", encoding="utf-8") as f:
        s_copy = summary.copy()
        s_copy.pop("agent_ref", None)
        json.dump(s_copy, f, indent=2, ensure_ascii=False)
    with (out_dir / f"{prefix}v2i_curve.csv").open("w", encoding="utf-8") as f:
        f.write("step,v2i_mean\n")
        for s, v in summary["v2i_curve"]:
            f.write(f"{s},{v}\n")
    with (out_dir / f"{prefix}v2v_curve.csv").open("w", encoding="utf-8") as f:
        f.write("step,v2v_success\n")
        for s, v in summary["v2v_curve"]:
            f.write(f"{s},{v}\n")
    return summary


# ================= 画图：基本对比 =================

def plot_basic_comparison(gat_sum: Dict, sage_sum: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    width = 0.35

    # V2I 曲线
    plt.figure(figsize=(8, 5))
    gs, gv = zip(*gat_sum["v2i_curve"])
    ss, sv = zip(*sage_sum["v2i_curve"])
    plt.plot(gs, gv, label="GAT V2I")
    plt.plot(ss, sv, label="GraphSAGE V2I")
    plt.axvline(gat_sum["v2i_conv_step"], color="#1976d2", ls="--", alpha=0.5,
                label=f"GAT conv@{gat_sum['v2i_conv_step']}")
    plt.axvline(sage_sum["v2i_conv_step"], color="#d32f2f", ls="--", alpha=0.5,
                label=f"SAGE conv@{sage_sum['v2i_conv_step']}")
    plt.xlabel("Step (test points)")
    plt.ylabel("V2I Mean")
    plt.title("V2I Mean Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_v2i_curve.png", dpi=160)
    plt.close()

    # V2V 曲线
    plt.figure(figsize=(8, 5))
    gs2, gv2 = zip(*gat_sum["v2v_curve"])
    ss2, sv2 = zip(*sage_sum["v2v_curve"])
    plt.plot(gs2, gv2, label="GAT V2V")
    plt.plot(ss2, sv2, label="GraphSAGE V2V")
    plt.xlabel("Step (test points)")
    plt.ylabel("V2V Success")
    plt.title("V2V Success Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_v2v_curve.png", dpi=160)
    plt.close()

    # 功率分布 & 熵
    labels = ["23dB(P0)", "10dB(P1)", "5dB(P2)"]
    x = np.arange(3)
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, gat_sum["power_counts"], width, label=f"GAT H={gat_sum['power_entropy']:.3f}")
    plt.bar(x + width/2, sage_sum["power_counts"], width, label=f"SAGE H={sage_sum['power_entropy']:.3f}")
    plt.xticks(x, labels)
    plt.ylabel("Counts")
    plt.title("Power Selection Distribution & Entropy")
    plt.grid(alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_power_distribution.png", dpi=160)
    plt.close()

    # RB 使用均匀度
    plt.figure(figsize=(8, 5))
    metrics = ["unique_ratio", "gini"]
    gat_vals = [gat_sum["rb_unique_ratio"], gat_sum["rb_gini"]]
    sage_vals = [sage_sum["rb_unique_ratio"], sage_sum["rb_gini"]]
    xi = np.arange(len(metrics))
    plt.bar(xi - width/2, gat_vals, width, label="GAT")
    plt.bar(xi + width/2, sage_vals, width, label="GraphSAGE")
    plt.xticks(xi, metrics)
    plt.ylabel("Value")
    plt.title("RB Usage Uniformity Comparison")
    plt.grid(alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_rb_uniformity.png", dpi=160)
    plt.close()

    # 决策时间（平均）
    plt.figure(figsize=(8, 5))
    labels_t = ["Full decision", "GNN only"]
    gat_t = [gat_sum["decision_time_full_s"], gat_sum["decision_time_gnn_only_s"]]
    sage_t = [sage_sum["decision_time_full_s"], sage_sum["decision_time_gnn_only_s"]]
    xi = np.arange(len(labels_t))
    plt.bar(xi - width/2, gat_t, width, label="GAT")
    plt.bar(xi + width/2, sage_t, width, label="GraphSAGE")
    plt.xticks(xi, labels_t)
    plt.ylabel("Seconds per node")
    plt.title("Decision Time per Node Comparison")
    plt.grid(alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_decision_time.png", dpi=160)
    plt.close()


# ================= 额外图：剩余时间 vs 功率选择 =================

def plot_power_vs_time(agent_summary: Dict, out_dir: Path, tag: str, nbins: int = 15):
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = agent_summary.get("agent_ref").power_log if agent_summary.get("agent_ref") else []
    if not logs:
        return
    arr = np.array(logs, dtype=np.float32)
    time_left = arr[:, 0]
    pidx = arr[:, 1].astype(int)
    limit = float(agent_summary.get("agent_ref").env.V2V_limit)
    time_left = np.clip(time_left, 0, limit)
    edges = np.linspace(0, limit, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    probs = np.zeros((nbins, 3), dtype=np.float32)
    counts = np.zeros(nbins, dtype=np.int32)
    for b in range(nbins):
        mask = (time_left >= edges[b]) & (time_left < edges[b + 1])
        cnt = int(np.sum(mask))
        counts[b] = cnt
        if cnt > 0:
            for k in [0, 1, 2]:
                probs[b, k] = np.sum(mask & (pidx == k)) / cnt

    # 线图
    plt.figure(figsize=(8, 5))
    plt.plot(mids, probs[:, 0], '-o', label='P0 23dB')
    plt.plot(mids, probs[:, 1], '-s', label='P1 10dB')
    plt.plot(mids, probs[:, 2], '-^', label='P2 5dB')
    plt.xlabel("Time left for V2V (s)")
    plt.ylabel("P(power | time-bin)")
    plt.title(f"Power vs Time-left ({tag})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"power_vs_time_{tag}.png", dpi=160)
    plt.close()

    # 热力图
    plt.figure(figsize=(8, 4))
    plt.imshow(probs.T, aspect='auto', origin='lower',
               extent=[edges[0], edges[-1], -0.5, 2.5], cmap='viridis')
    plt.yticks([0, 1, 2], ['P0 23dB', 'P1 10dB', 'P2 5dB'])
    plt.xlabel("Time left (s)")
    plt.title(f"Power vs Time-left Heatmap ({tag})")
    plt.colorbar(label='Probability')
    plt.tight_layout()
    plt.savefig(out_dir / f"power_vs_time_heatmap_{tag}.png", dpi=160)
    plt.close()


# ================= 额外图：完全图 vs 非完全图 决策耗时 =================

def measure_complete_vs_pruned(agent_summary: Dict, out_dir: Path, tag: str):
    agent = agent_summary.get("agent_ref")
    if agent is None or not hasattr(agent, "G"):
        return
    G = agent.G
    N = 3 * len(agent.env.vehicles)
    # 默认图整图前向
    try:
        t0 = time.perf_counter()
        if hasattr(G, "_forward_all"):
            emb = G._forward_all(training=False)
            emb = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
        else:
            emb = G.use_GraphSAGE(agent.channel_reward, agent.step, list(range(N)), False)
        t1 = time.perf_counter()
        pruned_gnn = (t1 - t0) / max(1, N)
    except Exception:
        pruned_gnn = None

    # 默认图 full decision
    try:
        t2 = time.perf_counter()
        for i in range(len(agent.env.vehicles)):
            for j in range(3):
                s = agent.get_state([i, j])
                e = emb[3 * i + j] if isinstance(emb, np.ndarray) else np.array(emb)[3 * i + j]
                _ = agent.predict(np.concatenate((e, s), axis=0), agent.step)
        t3 = time.perf_counter()
        pruned_full = (t3 - t2) / max(1, N)
    except Exception:
        pruned_full = None

    # 完全图：需要 G.load_graph 与 G.order_nodes
    try:
        import networkx as nx
        order_nodes = list(G.order_nodes) if hasattr(G, "order_nodes") and G.order_nodes is not None else list(range(N))
        complete_graph = nx.Graph()
        complete_graph.add_nodes_from(order_nodes)
        for a in range(N):
            for b in range(a + 1, N):
                complete_graph.add_edge(order_nodes[a], order_nodes[b])
        if hasattr(G, "load_graph"):
            G.load_graph(complete_graph, order_nodes)
        # 重新整图前向计时
        t4 = time.perf_counter()
        if hasattr(G, "_forward_all"):
            emb_c = G._forward_all(training=False)
            emb_c = emb_c.numpy() if hasattr(emb_c, "numpy") else np.asarray(emb_c)
        else:
            emb_c = G.use_GraphSAGE(agent.channel_reward, agent.step, list(range(N)), False)
        t5 = time.perf_counter()
        complete_gnn = (t5 - t4) / max(1, N)
        # 完全图 full decision
        t6 = time.perf_counter()
        for i in range(len(agent.env.vehicles)):
            for j in range(3):
                s = agent.get_state([i, j])
                e = emb_c[3 * i + j] if isinstance(emb_c, np.ndarray) else np.array(emb_c)[3 * i + j]
                _ = agent.predict(np.concatenate((e, s), axis=0), agent.step)
        t7 = time.perf_counter()
        complete_full = (t7 - t6) / max(1, N)
    except Exception:
        complete_gnn, complete_full = None, None

    # 画柱状图
    plt.figure(figsize=(8, 5))
    labels = ["Pruned GNN", "Pruned Full", "Complete GNN", "Complete Full"]
    vals = [
        pruned_gnn if pruned_gnn is not None else 0.0,
        pruned_full if pruned_full is not None else 0.0,
        complete_gnn if complete_gnn is not None else 0.0,
        complete_full if complete_full is not None else 0.0,
    ]
    colors = ["#42a5f5", "#1e88e5", "#ffb74d", "#f57c00"]
    plt.bar(range(4), vals, color=colors)
    plt.xticks(range(4), labels, rotation=15)
    plt.ylabel("Seconds per node")
    plt.title(f"Decision Time: Complete vs Pruned graph ({tag})")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / f"decision_time_complete_vs_pruned_{tag}.png", dpi=160)
    plt.close()


# ================= 额外图：动态环境时序 =================

def plot_dynamic_timeseries(agent_summary: Dict, out_dir: Path, tag: str):
    ts = agent_summary.get("last_timeseries")
    if not ts:
        return
    t = ts.get("t", [])
    v2i = ts.get("v2i", [])
    v2v = ts.get("v2v_succ", [])
    nveh = ts.get("nveh", [])
    if not t:
        return
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ln1 = ax1.plot(t, v2v, color="tab:blue", label="V2V Success")
    ax1.set_xlabel("Time (arb. units)")
    ax1.set_ylabel("V2V Success", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ln2 = ax2.plot(t, v2i, color="tab:orange", alpha=0.7, label="V2I Rate")
    ax2.set_ylabel("V2I Rate", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ln3 = ax3.plot(t, nveh, color="tab:green", alpha=0.6, label="#Vehicles")
    ax3.set_ylabel("#Vehicles", color="tab:green")
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    ax1.grid(alpha=0.25)
    plt.title(f"Dynamic series (V2I, V2V, #Veh) - {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"timeseries_dynamic_{tag}.png", dpi=160)
    plt.close()


# ================= 车辆规模扫描（V2I/V2V vs 车辆数） =================

def run_sweep_vehicles(args: argparse.Namespace, out_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    veh_list = parse_list_int_strict(getattr(args, "veh_list", ""))  # 防止 AttributeError
    if not veh_list:
        return {}
    results = {"gat": {"veh": [], "v2i": [], "v2v": []},
               "sage": {"veh": [], "v2i": [], "v2v": []}}
    for nveh in veh_list:
        n_up = int(np.ceil(nveh / 2))
        n_down = int(nveh - n_up)
        for model in ["gat", "sage"]:
            env = HighwayTopoEnv(
                n_up=n_up, n_down=n_down, lanes_per_dir=args.lanes, spacing=args.spacing,
                base_y=args.base_y, height=args.height, topology_type=args.topo,
                v2i_mode=args.v2i_mode, bs_layout=args.bs_layout, bs_spacing=args.bs_spacing,
                seed=args.seed
            )
            env.new_random_game()
            ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

            # 注入奖励超参
            env.beta_urgency_pos = float(args.beta_urgency_pos)
            env.beta_urgency_neg = float(args.beta_urgency_neg)
            env.urgency_threshold = float(args.urgency_threshold)
            env.rb_anti_conc_alpha = float(args.rb_anti_conc_alpha)
            env.rb_hot_threshold = float(args.rb_hot_threshold)
            env.rb_softmask_alpha = float(args.rb_softmask_alpha)
            env.rb_softmask_window = int(args.rb_softmask_window)

            ag = InstrumentedAgent([], env, gnn_type=model, warmup_steps=max(200, args.warmup_steps // 2),
                                   epsilon_decay_steps=args.epsilon_decay_steps, speed_mode=True, plot_dpi=args.dpi)
            ag.train(max_steps=max(400, args.train_steps // 3), test_every_steps=max(100, args.test_every // 2),
                     test_sample=max(40, args.test_sample // 2))
            v2i_final = ag.v2i_curve_vals[-1] if ag.v2i_curve_vals else 0.0
            v2v_final = ag.v2v_curve_vals[-1] if ag.v2v_curve_vals else 0.0
            results[model]["veh"].append(nveh)
            results[model]["v2i"].append(float(v2i_final))
            results[model]["v2v"].append(float(v2v_final))
    # 画图与保存
    if results["gat"]["veh"]:
        plt.figure(figsize=(7.5, 5))
        plt.plot(results["gat"]["veh"], results["gat"]["v2i"], '-o', label="GAT")
        plt.plot(results["sage"]["veh"], results["sage"]["v2i"], '-s', label="GraphSAGE")
        plt.xlabel("#Vehicles"); plt.ylabel("V2I Mean"); plt.title("V2I Mean vs #Vehicles")
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "sweep_v2i_vs_vehicles.png", dpi=160); plt.close()

        plt.figure(figsize=(7.5, 5))
        plt.plot(results["gat"]["veh"], results["gat"]["v2v"], '-o', label="GAT")
        plt.plot(results["sage"]["veh"], results["sage"]["v2v"], '-s', label="GraphSAGE")
        plt.xlabel("#Vehicles"); plt.ylabel("V2V Success"); plt.title("V2V Success vs #Vehicles")
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "sweep_v2v_vs_vehicles.png", dpi=160); plt.close()

        with (out_dir / "sweep_vehicles.csv").open("w", encoding="utf-8") as f:
            f.write("model,n_veh,v2i_mean,v2v_success\n")
            for model in ["gat", "sage"]:
                for n, a, b in zip(results[model]["veh"], results[model]["v2i"], results[model]["v2v"]):
                    f.write(f"{model},{n},{a},{b}\n")
    return results


# ================= 多随机种子箱线图 =================

def run_multi_seed_box(args: argparse.Namespace, out_dir: Path):
    seeds = parse_list_int_strict(getattr(args, "seeds", ""))  # 防止 AttributeError
    if not seeds or len(seeds) < 2:
        return
    vals = {"gat": {"v2i": [], "v2v": [], "entropy": [], "gini": []},
            "sage": {"v2i": [], "v2v": [], "entropy": [], "gini": []}}
    for sd in seeds:
        for model in ["gat", "sage"]:
            env = HighwayTopoEnv(
                n_up=args.n_up, n_down=args.n_down, lanes_per_dir=args.lanes, spacing=args.spacing,
                base_y=args.base_y, height=args.height, topology_type=args.topo,
                v2i_mode=args.v2i_mode, bs_layout=args.bs_layout, bs_spacing=args.bs_spacing,
                seed=sd
            )
            env.new_random_game()
            ensure_env_difficulty(env, args.demand_amount, args.v2v_limit)

            # 注入奖励超参
            env.beta_urgency_pos = float(args.beta_urgency_pos)
            env.beta_urgency_neg = float(args.beta_urgency_neg)
            env.urgency_threshold = float(args.urgency_threshold)
            env.rb_anti_conc_alpha = float(args.rb_anti_conc_alpha)
            env.rb_hot_threshold = float(args.rb_hot_threshold)
            env.rb_softmask_alpha = float(args.rb_softmask_alpha)
            env.rb_softmask_window = int(args.rb_softmask_window)

            ag = InstrumentedAgent([], env, gnn_type=model, warmup_steps=args.warmup_steps,
                                   epsilon_decay_steps=args.epsilon_decay_steps, speed_mode=False, plot_dpi=args.dpi)
            ag.train(max_steps=args.train_steps, test_every_steps=args.test_every, test_sample=args.test_sample)
            v2i_final = ag.v2i_curve_vals[-1] if ag.v2i_curve_vals else 0.0
            v2v_final = ag.v2v_curve_vals[-1] if ag.v2v_curve_vals else 0.0
            power_counts = np.zeros(3, dtype=np.int64)
            for _, pw in ag.power_log:
                if 0 <= pw < 3:
                    power_counts[pw] += 1
            power_entropy = compute_entropy(power_counts)
            rb_gini = compute_gini(ag.rb_usage_counts)
            vals[model]["v2i"].append(float(v2i_final))
            vals[model]["v2v"].append(float(v2v_final))
            vals[model]["entropy"].append(float(power_entropy))
            vals[model]["gini"].append(float(rb_gini))

    # 画箱线图
    def boxplot_pair(mkey: str, ylabel: str, fname: str):
        data = [vals["gat"][mkey], vals["sage"][mkey]]
        plt.figure(figsize=(7.2, 5))
        bp = plt.boxplot(data, labels=["GAT", "GraphSAGE"], patch_artist=True)
        colors = ["#42a5f5", "#ffb74d"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        plt.ylabel(ylabel); plt.title(f"{ylabel} (boxplot over seeds)")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig(out_dir / fname, dpi=160); plt.close()

    boxplot_pair("v2i", "V2I Mean Final", "box_v2i_final.png")
    boxplot_pair("v2v", "V2V Success Final", "box_v2v_final.png")
    boxplot_pair("entropy", "Power Selection Entropy", "box_power_entropy.png")
    boxplot_pair("gini", "RB Gini (lower better)", "box_rb_gini.png")

    with (out_dir / "box_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(vals, f, indent=2, ensure_ascii=False)


# ================= 主流程 =================

def main():
    ap = argparse.ArgumentParser(description="Compare GAT vs GraphSAGE with flat output.")
    # 场景
    ap.add_argument("--n-up", type=int, default=12)
    ap.add_argument("--n-down", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=25.0)
    ap.add_argument("--base-y", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1200.0)
    ap.add_argument("--topo", type=str, default="tree", choices=["star", "tree"])
    # 领导车
    ap.add_argument("--leader-front", action="store_true")
    ap.add_argument("--leader-lane-up", type=int, default=None)
    ap.add_argument("--leader-lane-down", type=int, default=None)
    ap.add_argument("--leader-dynamic", action="store_true")
    # 运动
    ap.add_argument("--use-move-speed", type=float, default=0.0)
    # V2I/RSU
    ap.add_argument("--v2i-mode", type=str, default="rsu", choices=["rsu", "single"])
    ap.add_argument("--bs-layout", type=str, default="median", choices=["median", "dual-roadside"])
    ap.add_argument("--bs-spacing", type=float, default=250.0)
    ap.add_argument("--bs-min-stay", type=int, default=5)
    ap.add_argument("--bs-hyst", type=float, default=15.0)
    # 训练
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--test-every", type=int, default=200)
    ap.add_argument("--test-sample", type=int, default=80)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--epsilon-decay-steps", type=int, default=800)
    # 难度
    ap.add_argument("--demand-amount", type=float, default=130.0)
    ap.add_argument("--v2v-limit", type=float, default=0.045)
    # 其他
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=224)
    ap.add_argument("--out-dir", type=str, default="runs/compare_flat")

    # 新增：GNN/Agent CLI 透传参数
    ap.add_argument("--gnn-type", type=str, choices=["gat", "sage"], default="gat")

    # GraphGAT（注意力剪枝）配置（将透传到 gnn_factory.build_gnn -> GraphGAT）
    ap.add_argument("--reprune-every", type=int, default=300)
    ap.add_argument("--hysteresis-keep", type=float, default=0.5)
    ap.add_argument("--reprune-start-step", type=int, default=600)
    ap.add_argument("--reg-attn-w", type=float, default=1e-3)

    # Agent/Env（奖励与反集中化/软掩码/紧急度）
    ap.add_argument("--beta-urgency-pos", type=float, default=0.018)
    ap.add_argument("--beta-urgency-neg", type=float, default=0.025)
    ap.add_argument("--urgency-threshold", type=float, default=0.25)
    ap.add_argument("--rb-anti-conc-alpha", type=float, default=0.012)
    ap.add_argument("--rb-hot-threshold", type=float, default=0.22)
    ap.add_argument("--rb-softmask-alpha", type=float, default=0.15)
    ap.add_argument("--rb-softmask-window", type=int, default=50)

    # 可选：车辆规模扫描与多随机种子箱线图（防止 AttributeError）
    ap.add_argument("--veh-list", type=str, default="", help='e.g., "8,12,16,24"')
    ap.add_argument("--seeds", type=str, default="", help='e.g., "123,456,789" (>=2) for boxplots')

    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 跑两种模型（按 CLI 指定也可以只跑一种，当前默认两种）
    gat_sum = run_one("gat", args, out_dir) if args.gnn_type in ("gat", "sage") else run_one("gat", args, out_dir)
    sage_sum = run_one("sage", args, out_dir)

    # 基础对比图
    plot_basic_comparison(gat_sum, sage_sum, out_dir)

    # 剩余时间 vs 功率选择（两模型）
    plot_power_vs_time(gat_sum, out_dir, "gat")
    plot_power_vs_time(sage_sum, out_dir, "graphsage")

    # 完全图 vs 非完全图决策耗时对比（两模型各一张）
    measure_complete_vs_pruned(gat_sum, out_dir, "gat")
    measure_complete_vs_pruned(sage_sum, out_dir, "graphsage")

    # 动态时序图
    plot_dynamic_timeseries(gat_sum, out_dir, "gat")
    plot_dynamic_timeseries(sage_sum, out_dir, "graphsage")

    # 车辆规模扫描（仅在传入 --veh-list 时执行）
    if getattr(args, "veh_list", ""):
        run_sweep_vehicles(args, out_dir)

    # 多随机种子箱线图（仅在传入 --seeds 时执行）
    if getattr(args, "seeds", ""):
        run_multi_seed_box(args, out_dir)

    # 最终汇总 JSON/CSV（仍在同一目录）
    compare_csv = out_dir / "comparison_metrics.csv"
    with compare_csv.open("w", encoding="utf-8") as f:
        f.write("model,v2i_final,v2v_final,v2i_conv_step,power_entropy,rb_unique_ratio,rb_gini,"
                "decision_time_full_s,decision_time_gnn_only_s\n")
        for s in [gat_sum, sage_sum]:
            f.write(f"{s['model']},{s['v2i_final']},{s['v2v_final']},{s['v2i_conv_step']},"
                    f"{s['power_entropy']},{s['rb_unique_ratio']},{s['rb_gini']},"
                    f"{s['decision_time_full_s']},{s['decision_time_gnn_only_s']}\n")

    combo_summary = {
        "settings": {
            "n_up": args.n_up, "n_down": args.n_down, "lanes": args.lanes, "spacing": args.spacing,
            "base_y": args.base_y, "height": args.height, "topology": args.topo,
            "leader_front": bool(args.leader_front), "leader_lane_up": args.leader_lane_up,
            "leader_lane_down": args.leader_lane_down, "leader_dynamic": bool(args.leader_dynamic),
            "use_move_speed": args.use_move_speed,
            "v2i_mode": args.v2i_mode, "bs_layout": args.bs_layout, "bs_spacing": args.bs_spacing,
            "bs_min_stay": args.bs_min_stay, "bs_hyst": args.bs_hyst,
            "train_steps": args.train_steps, "test_every": args.test_every, "test_sample": args.test_sample,
            "warmup_steps": args.warmup_steps, "epsilon_decay_steps": args.epsilon_decay_steps,
            "demand_amount": args.demand_amount, "v2v_limit": args.v2v_limit,
            "seed": args.seed,
            "gnn_type": args.gnn_type,
            "reprune_every": args.reprune_every,
            "hysteresis_keep": args.hysteresis_keep,
            "reprune_start_step": args.reprune_start_step,
            "reg_attn_w": args.reg_attn_w,
            "beta_urgency_pos": args.beta_urgency_pos,
            "beta_urgency_neg": args.beta_urgency_neg,
            "urgency_threshold": args.urgency_threshold,
            "rb_anti_conc_alpha": args.rb_anti_conc_alpha,
            "rb_hot_threshold": args.rb_hot_threshold,
            "rb_softmask_alpha": args.rb_softmask_alpha,
            "rb_softmask_window": args.rb_softmask_window,
            "veh_list": getattr(args, "veh_list", ""),
            "seeds": getattr(args, "seeds", ""),
        },
        "gat": {k: v for k, v in gat_sum.items() if k != "agent_ref"},
        "graphsage": {k: v for k, v in sage_sum.items() if k != "agent_ref"}
    }
    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(combo_summary, f, indent=2, ensure_ascii=False)

    print("[OK] Flat comparison complete.")
    print("Summary JSON:", (out_dir / "comparison_summary.json").resolve())
    print("Metrics CSV :", compare_csv.resolve())
    print("Files       :", [p.name for p in out_dir.glob('*.*')])

if __name__ == "__main__":
    main()