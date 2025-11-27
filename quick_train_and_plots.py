#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from highway_environment import HighwayTopoEnv
from agent import Agent

# 可选：tqdm 进度条
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def safe_make_agent(env):
    try:
        return Agent([], env, warmup_steps=600, epsilon_decay_steps=5000, speed_mode=True)
    except TypeError:
        return Agent([], env)


def get_loss_list(G):
    if hasattr(G, "gs_losses") and isinstance(G.gs_losses, list):
        return G.gs_losses
    if hasattr(G, "loss") and isinstance(G.loss, list):
        return G.loss
    return None


def main():
    ap = argparse.ArgumentParser(description="Few-steps quick run + plots (multi-RSU, used_blocks, losses).")
    # 几何/车队
    ap.add_argument("--topo", default="star", choices=["star", "tree"])
    ap.add_argument("--n-up", type=int, default=10)
    ap.add_argument("--n-down", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=20.0)
    ap.add_argument("--base-y", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1000.0)
    ap.add_argument("--leader-front", action="store_true")
    # 多 RSU
    ap.add_argument("--bs-layout", default="median", choices=["median", "dual-roadside"])
    ap.add_argument("--bs-spacing", type=float, default=250.0)
    ap.add_argument("--bs-min-stay", type=int, default=5)
    ap.add_argument("--bs-hyst", type=float, default=15.0)
    # 步数与显示
    ap.add_argument("--steps", type=int, default=80, help="total steps (few)")
    ap.add_argument("--margin", type=int, default=10, help="kept tail steps after reaching end in speed calc")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=224)
    ap.add_argument("--out-dir", default="quick_plots", help="directory to save figures")
    # 实时进度显示
    ap.add_argument("--progress", action="store_true", help="show real-time progress (tqdm if available)")
    ap.add_argument("--progress-interval", type=int, default=1, help="fallback printing interval when no tqdm")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    effective_len = max(0.0, args.height - 2.0 * args.base_y)
    steps_for_move = max(1, args.steps - args.margin)
    move_speed = effective_len / steps_for_move if effective_len > 0 else 0.0

    env = HighwayTopoEnv(
        n_up=args.n_up,
        n_down=args.n_down,
        lanes_per_dir=args.lanes,
        spacing=args.spacing,
        base_y=args.base_y,
        height=args.height,
        topology_type=args.topo,
        move_speed=float(move_speed),
        leader_at_front=bool(args.leader_front),
        v2i_mode="rsu",
        bs_layout=args.bs_layout,
        bs_spacing=float(args.bs_spacing),
        bs_min_stay_steps=int(args.bs_min_stay),
        bs_handover_hyst_m=float(args.bs_hyst),
    )
    env.new_random_game()

    agent = safe_make_agent(env)
    agent.training = True
    agent.GraphSAGE = True
    try:
        agent.dqn.update_target_network()
    except Exception:
        pass

    _ = agent.initial_better_state(0, agent.GraphSAGE)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_axis = []
    gat_losses = []
    loss_steps = []
    v2i_rate_ts = []
    v2v_success_ts = []
    power_timeleft_samples = {0: [], 1: [], 2: []}
    used_blocks_count_ts = []
    rb_usage_counter = np.zeros(getattr(agent, "RB_number", 20), dtype=int)
    summary_marks = [max(1, args.steps // 2), args.steps]
    summary_points = []

    iterator = range(1, args.steps + 1)
    pbar = None
    use_tqdm = args.progress and (tqdm is not None)
    if use_tqdm:
        pbar = tqdm(iterator, desc="Running steps", ncols=100)
    else:
        pbar = iterator

    for step in pbar:
        # 清空 RB 选择占位
        for i in range(len(env.vehicles)):
            agent.action_all_with_power_training[i, :, 0] = -1

        # 按每车3条链路选择动作
        for i in range(len(env.vehicles)):
            sorted_idx = np.argsort(env.individual_time_limit[i, :])
            for j in sorted_idx:
                idx = [3 * i + j]
                state_old = agent.get_state([i, j])
                agent.G.features[3 * i + j, :] = state_old[:60]

                loss_list_before = get_loss_list(agent.G) or []
                before_len = len(loss_list_before)

                node_emb = agent.G.use_GraphSAGE(agent.channel_reward, step, idx, agent.GraphSAGE)

                loss_list_after = get_loss_list(agent.G) or []
                if len(loss_list_after) > before_len:
                    gat_losses.append(loss_list_after[-1])
                    loss_steps.append(step)

                node_emb = np.squeeze(node_emb)
                better_state_old = np.concatenate((node_emb, state_old), axis=0)

                action = agent.predict(better_state_old, step, test_ep=False)
                rb = int(action % agent.RB_number)
                pw = int(action // agent.RB_number)
                pw = max(0, min(pw, 2))

                agent.action_all_with_power_training[i, j, 0] = rb
                agent.action_all_with_power_training[i, j, 1] = pw

                power_timeleft_samples[pw].append(float(state_old[-1]))

        used_blocks = agent.action_all_with_power_training[:, :, 0].astype(int).flatten()
        used_blocks = used_blocks[used_blocks >= 0]
        unique_rb = len(np.unique(used_blocks))
        used_blocks_count_ts.append(unique_rb)

        rb_counts = np.bincount(used_blocks, minlength=agent.RB_number)
        rb_usage_counter[:len(rb_counts)] += rb_counts

        action_temp = agent.action_all_with_power_training.copy()
        v2i_rate_vec, fail_percent = env.act_asyn(action_temp)
        v2i_rate = float(np.sum(v2i_rate_vec))
        v2v_success = 1.0 - float(fail_percent)

        v2i_rate_ts.append(v2i_rate)
        v2v_success_ts.append(v2v_success)
        steps_axis.append(step)

        if use_tqdm:
            # 动态显示关键指标
            pbar.set_postfix(v2i=f"{v2i_rate:.1f}", v2v=f"{v2v_success:.2f}", rb=unique_rb)
        elif args.progress and args.progress_interval > 0 and (step % args.progress_interval == 0):
            print(f"[step {step}] v2i={v2i_rate:.1f}, v2v={v2v_success:.3f}, unique_rb={unique_rb}")

        if step in summary_marks:
            summary_points.append((step, v2v_success, v2i_rate))

    # ========== 画图 ==========
    plt.rcParams.update({"figure.dpi": args.dpi})

    if gat_losses:
        plt.figure(figsize=(8, 4.6))
        plt.plot(loss_steps, gat_losses, color="#f57c00", label="GAT loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("GAT Training Loss over Steps")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "fig1_gat_loss.png")
        plt.close()

    bins = np.linspace(0.0, 0.10, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    probs = {}
    for pw in [0, 1, 2]:
        hist, _ = np.histogram(power_timeleft_samples[pw], bins=bins)
        probs[pw] = hist
    denom = probs[0] + probs[1] + probs[2]
    denom = np.maximum(denom, 1)
    p0 = probs[0] / denom
    p1 = probs[1] / denom
    p2 = probs[2] / denom

    plt.figure(figsize=(9, 5))
    plt.plot(bin_centers, p0, "o-", label="Power 23 dB")
    plt.plot(bin_centers, p1, "s-", label="Power 10 dB")
    plt.plot(bin_centers, p2, "^-", label="Power 5 dB")
    plt.xlim(0.0, 0.10)
    plt.xlabel("Time left for V2V transmission (s)")
    plt.ylabel("Probability of power selection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_power_vs_timeleft.png")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps_axis, v2v_success_ts, label="Inst. V2V Success", color="#1f77b4")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("V2V Success")
    ax1.set_ylim(0.0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(steps_axis, v2i_rate_ts, label="Inst. V2I Rate", color="#ff7f0e")
    ax2.set_ylabel("V2I Rate")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_timeseries_v2v_v2i.png")
    plt.close()

    if summary_points:
        xs = [p[0] for p in summary_points]
        v2v = [p[1] for p in summary_points]
        v2i = [p[2] for p in summary_points]
        plt.figure(figsize=(9, 5))
        ax1 = plt.gca()
        ax1.plot(xs, v2v, "o-", label="V2V Success Rate")
        ax1.set_xlabel("step")
        ax1.set_ylabel("V2V Success Rate")
        ax2 = ax1.twinx()
        ax2.plot(xs, v2i, "s-", color="#ff7f0e", label="V2I Rate")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "fig4_step_summary.png")
        plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(steps_axis, used_blocks_count_ts, "o-", color="#6a1b9a")
    plt.xlabel("step")
    plt.ylabel("Number of unique RBs used")
    plt.title("Used RB kinds per step")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_used_blocks_per_step.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    xs = np.arange(agent.RB_number)
    plt.bar(xs, rb_usage_counter, color="#00897b")
    plt.xlabel("RB index")
    plt.ylabel("Usage count")
    plt.title("RB usage histogram (whole run)")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_rb_usage_hist.png")
    plt.close()

    np.savetxt(out_dir / "ts_v2i_rate.csv", np.vstack([steps_axis, v2i_rate_ts]).T, delimiter=",", header="step,v2i_rate", comments="")
    np.savetxt(out_dir / "ts_v2v_success.csv", np.vstack([steps_axis, v2v_success_ts]).T, delimiter=",", header="step,v2v_success", comments="")
    np.savetxt(out_dir / "ts_used_blocks_kinds.csv", np.vstack([steps_axis, used_blocks_count_ts]).T, delimiter=",", header="step,unique_rbs", comments="")

    print("[OK] Figures saved to:", out_dir.resolve())
    print(f"[INFO] steps={args.steps}, inferred move_speed={move_speed:.3f} m/step, bs_layout={args.bs_layout}, bs_spacing={args.bs_spacing}")


if __name__ == "__main__":
    main()