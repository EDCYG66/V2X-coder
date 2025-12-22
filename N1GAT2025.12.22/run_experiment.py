#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiment.py
Lightweight launcher to run your Agent experiments with tunable parameters
without editing library files.

Features:
- Pass beta_urgency_pos/neg, rb_anti_conc_alpha, rb_hot_threshold, rb_softmask_alpha
  to Agent constructor.
- After Agent creation, set Graph_GAT pruning params reprune_every and hysteresis_keep.
- Support single-seed and multi-seed runs. For multi-seed each run writes to its own out dir.

Usage examples (see below):
  python run_experiment.py --seed 123 --train-steps 2400 --beta-urgency-pos 0.018 ...
  python run_experiment.py --seeds 123,456,789 --train-steps 2400 --rb-anti-conc-alpha 0.012 ...
"""

import os
import argparse
import numpy as np
from datetime import datetime

# Adjust these imports to match your project layout
from highway_environment import HighwayTopoEnv      # or Environment.HighwayTopoEnv as appropriate
from agent import Agent

def make_env(args, seed):
    env = HighwayTopoEnv(
        n_up=args.n_up, n_down=args.n_down, lanes_per_dir=args.lanes,
        spacing=args.spacing, base_y=args.base_y, height=args.height,
        topology_type=args.topo, v2i_mode=args.v2i_mode,
        bs_layout=args.bs_layout, bs_spacing=args.bs_spacing,
        seed=seed
    )
    env.new_random_game()
    env.demand_amount = float(args.demand_amount)
    env.demand = env.demand_amount * np.ones((env.n_Veh, 3))
    env.V2V_limit = float(args.v2v_limit)
    env.individual_time_limit = env.V2V_limit * np.ones((env.n_Veh, 3))
    if hasattr(env, "activate_links"):
        env.activate_links[:] = False
    return env

def run_once(seed, args):
    np.random.seed(seed)
    out_base = args.out_dir or "runs/experiment"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_base, f"seed{seed}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    env = make_env(args, seed)

    # Create Agent with tunable params (these map to the constructor in your agent.py)
    agent = Agent(
        config=None,
        environment=env,
        gnn_type=args.gnn_type,
        warmup_steps=args.warmup_steps,
        epsilon_decay_steps=args.epsilon_decay_steps,
        speed_mode=False,
        gat_train_interval=args.gat_train_interval,
        plot_dpi=args.dpi,
        replay_size=args.replay_size,
        power_log_stride=args.power_log_stride,
        power_log_max=args.power_log_max,
        soft_update_tau=args.soft_update_tau,
        power_cost_weight=args.power_cost_weight,
        conflict_cost_weight=args.conflict_cost_weight,
        skip_embedding_steps=args.skip_embedding_steps,
        batch_decay_step=args.batch_decay_step,
        batch_decay_factor=args.batch_decay_factor,
        beta_urgency_pos=args.beta_urgency_pos,
        beta_urgency_neg=args.beta_urgency_neg,
        urgency_threshold=args.urgency_threshold,
        conflict_penalty_weight=args.conflict_penalty_weight,
        conflict_window_steps=args.conflict_window_steps,
        rb_anti_conc_alpha=args.rb_anti_conc_alpha,
        rb_hot_threshold=args.rb_hot_threshold,
        rb_softmask_alpha=args.rb_softmask_alpha,
        rb_softmask_window=args.rb_softmask_window
    )

    # Set reprune/hysteresis at runtime (no code edit required)
    if hasattr(agent, "G"):
        if args.reprune_every is not None:
            agent.G.reprune_every = int(args.reprune_every)
        if args.hysteresis_keep is not None:
            agent.G.hysteresis_keep = float(args.hysteresis_keep)

    # Set agent log/export dir (optional)
    agent.export_dir = out_dir

    # Train
    agent.train(max_steps=args.train_steps, test_every_steps=args.test_every, test_sample=args.test_sample)

    print(f"[DONE] seed={seed} outputs -> {out_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    # environment / run settings
    p.add_argument("--seeds", type=str, default="", help="comma separated seeds (e.g. 123,456) or single --seed")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--train-steps", type=int, default=2400)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--test-every", type=int, default=200)
    p.add_argument("--test-sample", type=int, default=80)
    p.add_argument("--n-up", type=int, default=12)
    p.add_argument("--n-down", type=int, default=20)
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--spacing", type=float, default=25.0)
    p.add_argument("--height", type=float, default=1200.0)
    p.add_argument("--base-y", type=float, default=0.0)
    p.add_argument("--topo", type=str, default="tree")
    p.add_argument("--v2i-mode", type=str, default="rsu")
    p.add_argument("--bs-layout", type=str, default="median")
    p.add_argument("--bs-spacing", type=float, default=250.0)
    p.add_argument("--demand-amount", type=float, default=130.0)
    p.add_argument("--v2v-limit", type=float, default=0.045)
    p.add_argument("--out-dir", type=str, default="runs/experiment")
    p.add_argument("--gnn-type", type=str, default="gat")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--replay-size", type=int, default=200000)
    p.add_argument("--power-log-stride", type=int, default=4)
    p.add_argument("--power-log-max", type=int, default=200000)
    p.add_argument("--gat-train-interval", type=int, default=20)

    # agent / patch params (the ones you will scan)
    p.add_argument("--beta-urgency-pos", type=float, default=0.018)
    p.add_argument("--beta-urgency-neg", type=float, default=0.025)
    p.add_argument("--urgency-threshold", type=float, default=0.25)
    p.add_argument("--conflict-penalty-weight", type=float, default=0.02)
    p.add_argument("--conflict-window-steps", type=int, default=50)
    p.add_argument("--rb-anti-conc-alpha", type=float, default=0.012)
    p.add_argument("--rb-hot-threshold", type=float, default=0.22)
    p.add_argument("--rb-softmask-alpha", type=float, default=0.15)
    p.add_argument("--rb-softmask-window", type=int, default=50)

    # attention reprune tune (set at runtime on agent.G)
    p.add_argument("--reprune-every", type=int, default=400)
    p.add_argument("--hysteresis-keep", type=float, default=0.4)

    # other lower-level params (optional)
    p.add_argument("--soft-update-tau", type=float, default=0.005)
    p.add_argument("--power-cost-weight", type=float, default=0.01)
    p.add_argument("--conflict-cost-weight", type=float, default=0.02)
    p.add_argument("--skip-embedding-steps", type=int, default=9)
    p.add_argument("--batch-decay-step", type=int, default=None)
    p.add_argument("--batch-decay-factor", type=float, default=0.5)
    p.add_argument("--epsilon-decay-steps", type=int, default=800)

    return p.parse_args()

def main():
    args = parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    for s in seeds:
        run_once(s, args)

if __name__ == "__main__":
    main()