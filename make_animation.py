#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

from highway_environment import HighwayTopoEnv

def auto_steps(height, base_y, move_speed, margin):
    effective = max(0.0, float(height) - 2.0 * float(base_y))
    if move_speed <= 0:
        return 1
    return int(math.ceil(effective / move_speed)) + int(margin)

def main():
    ap = argparse.ArgumentParser(description="Render highway animation (dual-direction, multi-RSU on median).")
    # 地图/行驶
    ap.add_argument("--topo", default="tree", choices=["star", "tree"], help="cluster topology per direction")
    ap.add_argument("--base-y", type=float, default=0.0, help="start Y (m)")
    ap.add_argument("--height", type=float, default=1000.0, help="map height (m)")
    ap.add_argument("--leader-front", action="store_true", help="leader at front")
    ap.add_argument("--use-move-speed", type=float, default=1.5, help="movement per step (m/step)")
    ap.add_argument("--auto-steps", action="store_true", help="infer steps from road length and move speed")
    ap.add_argument("--auto-steps-margin", type=int, default=20, help="extra steps after reaching end")
    # RSU/布局
    ap.add_argument("--bs-layout", default="median", choices=["median", "dual-roadside"], help="RSU layout")
    ap.add_argument("--bs-spacing", type=float, default=250.0, help="RSU spacing (m)")
    ap.add_argument("--v2i-mode", default="rsu", choices=["rsu", "single"], help="V2I mode (default: rsu)")
    ap.add_argument("--show-v2i", action="store_true", help="draw vehicle->serving RSU dashed lines")
    # 画面/导出
    ap.add_argument("--fps", type=int, default=20, help="video FPS")
    ap.add_argument("--dpi", type=int, default=224, help="figure DPI")
    ap.add_argument("--out", default="both_ends.mp4", help="output mp4 filename")
    # 可选：手动步数（当未用 --auto-steps 时）
    ap.add_argument("--steps", type=int, default=300, help="steps when not using --auto-steps")
    args = ap.parse_args()

    # 1) 构造环境（关键：v2i_mode=rsu 方可生成多RSU）
    env = HighwayTopoEnv(
        topology_type=args.topo,
        base_y=float(args.base_y),
        height=float(args.height),
        move_speed=float(args.use_move_speed),
        leader_at_front=bool(args.leader_front),
        v2i_mode=args.v2i_mode,
        bs_layout=args.bs_layout,
        bs_spacing=float(args.bs_spacing),
    )
    env.new_random_game()

    # 2) 推断步数（注意：这里使用下划线属性名）
    if args.auto_steps:
        steps = auto_steps(env.height, env.base_y, args.use_move_speed, args.auto_steps_margin)
    else:
        steps = int(args.steps)

    # 3) 渲染逐帧 PNG
    out_dir = Path(".")
    frames_dir = out_dir / "_frames_tmp"
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 首帧
    env.visualize(save_path=str(frames_dir / f"frame_{0:05d}.png"),
                  show_destinations=False, show_v2i=bool(args.show_v2i), dpi=int(args.dpi))

    # 后续帧
    for t in range(1, steps):
        env.renew_positions()
        env.visualize(save_path=str(frames_dir / f"frame_{t:05d}.png"),
                      show_destinations=False, show_v2i=bool(args.show_v2i), dpi=int(args.dpi))

    # 4) 用 ffmpeg 合成 MP4
    pattern = str(frames_dir / "frame_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_dir / args.out),
    ]
    env_vars = os.environ.copy()
    env_vars.pop("LD_LIBRARY_PATH", None)
    env_vars.pop("LD_PRELOAD", None)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env_vars)
    except subprocess.CalledProcessError as e:
        # 打印 ffmpeg 输出便于排错
        print(e.stdout.decode(errors="ignore"))
        print(e.stderr.decode(errors="ignore"))
        raise

    # 5) 清理帧
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"[OK] Video saved -> {args.out}")
    print(f"[INFO] RSU count = {len(getattr(env, 'bs_positions', []))}, layout={args.bs_layout}, spacing={args.bs_spacing}, v2i_mode={args.v2i_mode}")

if __name__ == "__main__":
    main()