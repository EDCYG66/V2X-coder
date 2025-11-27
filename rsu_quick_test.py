#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from highway_environment import HighwayTopoEnv

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def v2i_pathloss_db(d_m: float, h_bs: float = 25.0, h_ms: float = 1.5) -> float:
    d3d_km = math.sqrt(d_m**2 + (h_bs - h_ms) ** 2) / 1000.0
    return 128.1 + 37.6 * math.log10(max(d3d_km, 1e-9))


def snr_db_from_pl(pl_db, p_tx_dbm, veh_g, bs_g, bs_nf, noise_dbm):
    rx_dbm = p_tx_dbm + veh_g + bs_g - bs_nf - pl_db
    return rx_dbm - noise_dbm


def rate_bpshz_from_snr_db(snr_db: float) -> float:
    return math.log2(1.0 + 10.0 ** (snr_db / 10.0))


def collect_metrics(env: HighwayTopoEnv):
    bs_xy = env.bs_positions
    vehs = env.vehicles
    serve = getattr(env, "v2i_serving_idx", None)
    if not bs_xy or serve is None:
        raise RuntimeError("RSU/服务小区信息缺失，请确认 v2i_mode='rsu' 且已完成 _assoc_v2i。")

    rows = []
    p_tx_dbm = float(env.V2I_power_dB)
    veh_g = float(env.vehAntGain)
    bs_g = float(env.bsAntGain)
    bs_nf = float(env.bsNoiseFigure)
    noise_dbm = float(env.sig2_dB)

    snrs = []
    rates = []
    for i, v in enumerate(vehs):
        bi = int(serve[i])
        bx, by = bs_xy[bi]
        d_m = float(np.hypot(v.position[0] - bx, v.position[1] - by))
        pl_db = v2i_pathloss_db(d_m)
        snr_db = snr_db_from_pl(pl_db, p_tx_dbm, veh_g, bs_g, bs_nf, noise_dbm)
        rate = rate_bpshz_from_snr_db(snr_db)
        snrs.append(snr_db)
        rates.append(rate)
        rows.append(dict(
            vid=i, dir=v.direction, x=float(v.position[0]), y=float(v.position[1]),
            serving_rsu=bi, dist_m=round(d_m, 3), pl_db=round(pl_db, 3),
            snr_db=round(snr_db, 3), rate_bpshz=round(rate, 6),
        ))

    summary = dict(
        n_veh=len(vehs),
        n_rsu=len(bs_xy),
        snr_db_mean=float(np.mean(snrs)),
        snr_db_median=float(np.median(snrs)),
        rate_bpshz_mean=float(np.mean(rates)),
        rate_bpshz_median=float(np.median(rates)),
    )
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Quick RSU test with optional progress.")
    ap.add_argument("--topo", default="star", choices=["star", "tree"])
    ap.add_argument("--n-up", type=int, default=10)
    ap.add_argument("--n-down", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=20.0)
    ap.add_argument("--base-y", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1000.0)
    ap.add_argument("--leader-front", action="store_true")
    ap.add_argument("--bs-layout", default="median", choices=["median", "dual-roadside"])
    ap.add_argument("--bs-spacing", type=float, default=250.0)
    ap.add_argument("--bs-min-stay", type=int, default=5)
    ap.add_argument("--bs-hyst", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=224)
    ap.add_argument("--progress", action="store_true", help="show real-time progress (tqdm if available)")
    args = ap.parse_args()

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

    out_dir = Path("quick_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    img0 = out_dir / "rsu_quick_initial.png"
    env.visualize(save_path=str(img0), show_destinations=False, show_v2i=True, dpi=args.dpi)

    # 进度条
    iterator = range(args.steps)
    if args.progress and (tqdm is not None):
        iterator = tqdm(iterator, desc="Stepping", ncols=100)

    for _ in iterator:
        env.renew_positions()

    img1 = out_dir / "rsu_quick_final.png"
    env.visualize(save_path=str(img1), show_destinations=False, show_v2i=True, dpi=args.dpi)

    rows0, summary0 = collect_metrics(env)  # 注意：这里示例只导出终态，若需初始也可在前面 collect 一次

    # 简单保存 JSON（终态摘要）
    import json
    with (out_dir / "rsu_quick_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dict(final=summary0), f, indent=2, ensure_ascii=False)

    print("[OK] Saved:", img0, img1, out_dir / "rsu_quick_summary.json")


if __name__ == "__main__":
    main()