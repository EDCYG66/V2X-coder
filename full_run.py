#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from highway_environment import HighwayTopoEnv
from agent import Agent


def main():
    ap = argparse.ArgumentParser(description="Full run: training + analysis plots (GATv2 / SAGE).")
    ap.add_argument("--gnn-type", default="gat", choices=["gat", "sage"])

    # Geometry / traffic
    ap.add_argument("--topo", default="star", choices=["star", "tree"])
    ap.add_argument("--n-up", type=int, default=10)
    ap.add_argument("--n-down", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=20.0)
    ap.add_argument("--base-y", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1000.0)
    ap.add_argument("--leader-front", action="store_true")

    # V2I multi-RSU
    ap.add_argument("--bs-layout", default="median", choices=["median", "dual-roadside"])
    ap.add_argument("--bs-spacing", type=float, default=250.0)
    ap.add_argument("--bs-min-stay", type=int, default=5)
    ap.add_argument("--bs-hyst", type=float, default=15.0)

    # Train/eval
    ap.add_argument("--train-steps", type=int, default=5000)
    ap.add_argument("--test-every", type=int, default=1000)
    ap.add_argument("--test-sample", type=int, default=200)

    # Plots
    ap.add_argument("--dpi", type=int, default=224, help="Matplotlib savefig dpi")

    args = ap.parse_args()

    env = HighwayTopoEnv(
        n_up=args.n_up,
        n_down=args.n_down,
        lanes_per_dir=args.lanes,
        spacing=args.spacing,
        base_y=args.base_y,
        height=args.height,
        topology_type=args.topo,
        leader_at_front=bool(args.leader_front),
        v2i_mode="rsu",
        bs_layout=args.bs_layout,
        bs_spacing=float(args.bs_spacing),
        bs_min_stay_steps=int(args.bs_min_stay),
        bs_handover_hyst_m=float(args.bs_hyst),
    )
    env.new_random_game()

    agent = Agent(
        [],
        env,
        gnn_type=args.gnn_type,
        warmup_steps=1000,
        epsilon_decay_steps=30000,
        speed_mode=True,
        gat_train_interval=None,
        plot_dpi=args.dpi,
    )
    print(f"[RUN] Training with {args.gnn_type.upper()} ...")
    agent.train(max_steps=args.train_steps, test_every_steps=args.test_every, test_sample=args.test_sample)
    print("[DONE] Run dir:", agent.export_dir)


if __name__ == "__main__":
    main()