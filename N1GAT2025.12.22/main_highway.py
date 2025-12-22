# -*- coding: utf-8 -*-
"""
Dual 4-lane per direction; STAR/TREE per direction; RSU/V2I enabled.
- New CLI: --bs-layout, --bs-spacing, --bs-min-stay, --bs-hyst, --show-v2i
"""
import argparse
import inspect
import numpy as np
import random
import tensorflow as tf

from highway_environment import HighwayTopoEnv
from agent import Agent


ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=123)
# Vehicles & topology
ap.add_argument('--n-up', type=int, default=10, help='number of vehicles (up)')
ap.add_argument('--n-down', type=int, default=20, help='number of vehicles (down)')
ap.add_argument('--lanes', type=int, default=4, help='lanes per direction')
ap.add_argument('--spacing', type=float, default=20.0)
ap.add_argument('--topo', type=str, default='star', choices=['star', 'tree'], help='initial topology type')
ap.add_argument('--switch-at', type=int, default=None, help='switch topology once before training (0=immediately)')
# Geometry / motion / leader
ap.add_argument('--base-y', dest='base_y', type=float, default=None, help='starting Y (m), e.g., 0')
ap.add_argument('--height', type=float, default=None, help='road length (m)')
ap.add_argument('--leader-front', dest='leader_front', action='store_true', help='leader is foremost')
ap.add_argument('--leader-lane-up', dest='leader_lane_up', type=int, default=None, help='leader lane index for up (0..lanes-1)')
ap.add_argument('--leader-lane-down', dest='leader_lane_down', type=int, default=None, help='leader lane index for down (0..lanes-1)')
ap.add_argument('--leader-dynamic', dest='leader_dynamic', action='store_true', help='switch leader when foremost changes')
ap.add_argument('--use-move-speed', type=float, default=0.0, help='unified displacement (m/step), 0=per-vehicle speed')
# RSU / V2I
ap.add_argument('--bs-layout', dest='bs_layout', type=str, default='median', choices=['median', 'dual-roadside'], help='RSU layout strategy')
ap.add_argument('--bs-spacing', dest='bs_spacing', type=float, default=250.0, help='RSU spacing (m); <=0 disables RSUs')
ap.add_argument('--bs-min-stay', dest='bs_min_stay', type=int, default=5, help='min steps to stay before handover')
ap.add_argument('--bs-hyst', dest='bs_hyst', type=float, default=15.0, help='handover hysteresis (m)')
ap.add_argument('--show-v2i', dest='show_v2i', action='store_true', help='draw V2I dashed lines in exported images')
# Training
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--test-every', type=int, default=1000)
ap.add_argument('--test-sample', type=int, default=200)
ap.add_argument('--quick', action='store_true', help='quick mode (fewer steps)')
args = ap.parse_args()


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


def _build_env_from_args(a: argparse.Namespace) -> HighwayTopoEnv:
    sig = inspect.signature(HighwayTopoEnv.__init__)
    kw = {}

    def maybe(name_ctor: str, value):
        if name_ctor in sig.parameters and value is not None:
            kw[name_ctor] = value

    maybe('n_up', a.n_up)
    maybe('n_down', a.n_down)
    maybe('lanes_per_dir', a.lanes)
    maybe('spacing', a.spacing)
    maybe('topology_type', a.topo)
    maybe('base_y', a.base_y)
    maybe('height', a.height)
    maybe('move_speed', float(a.use_move_speed))
    maybe('leader_at_front', bool(a.leader_front))
    maybe('leader_lane_up', a.leader_lane_up)
    maybe('leader_lane_down', a.leader_lane_down)
    maybe('leader_dynamic', bool(a.leader_dynamic))
    # RSU/V2I
    maybe('bs_layout', a.bs_layout)
    maybe('bs_spacing', a.bs_spacing)
    maybe('bs_min_stay_steps', a.bs_min_stay)
    maybe('bs_handover_hyst_m', a.bs_hyst)

    return HighwayTopoEnv(**kw)


def main():
    set_seed(args.seed)
    setup_gpu()

    Env = _build_env_from_args(args)
    Env.new_random_game()

    # Export initial topology image (with V2I if requested)
    Env.visualize(save_path=f'highway_{args.topo}_dual4_{args.n_up}_{args.n_down}.png',
                  show_destinations=True, annotate_power=False, show_v2i=args.show_v2i)

    # Optional: quick comparison of the other topology
    if args.switch_at is not None and args.switch_at == 0:
        other = 'tree' if args.topo == 'star' else 'star'
        Env.set_topology(other)
        Env.visualize(save_path=f'highway_{other}_dual4_{args.n_up}_{args.n_down}.png',
                      show_destinations=True, annotate_power=False, show_v2i=args.show_v2i)
        Env.set_topology(args.topo)

    # Training schedule
    if args.quick:
        steps = 2500
        test_every = 500
        test_sample = 120
        warmup = 600
        decay = 5000
        speed_mode = True
    else:
        steps = args.steps
        test_every = args.test_every
        test_sample = args.test_sample
        warmup = 1200
        decay = 20000
        speed_mode = False

    agent = Agent([], Env, warmup_steps=warmup, epsilon_decay_steps=decay, speed_mode=speed_mode)
    agent.train(max_steps=steps, test_every_steps=test_every, test_sample=test_sample)


if __name__ == "__main__":
    main()