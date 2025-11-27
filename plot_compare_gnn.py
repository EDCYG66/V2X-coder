#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read two trained runs (GAT/SAGE) + sweep results and draw all comparison figures:
- Training loss (GNN loss) with tail-zero cleanup
- Power selection vs time-left
- Training-effect (V2V success & V2I rate over steps) with smoothing
- Dynamic timeseries (vehicles, V2I, V2V) with rolling mean/std
- V2I mean vs vehicles (supports error bars if *_std/CI columns exist)
- V2V success vs vehicles (supports error bars)
- Decision time (complete vs incomplete) vs vehicles
  * If 'kind' column exists -> plot FULL and GNN-only separately
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_safe(p: Path):
    return pd.read_csv(p) if p.exists() else None


def drop_tail_zeros(df: pd.DataFrame, col: str, eps: float = 1e-8) -> pd.DataFrame:
    if df is None or col not in df:
        return df
    idx = len(df)
    while idx > 0 and (pd.isna(df.iloc[idx - 1][col]) or df.iloc[idx - 1][col] <= eps):
        idx -= 1
    return df.iloc[:idx].reset_index(drop=True)


def plot_two_curves(df1, df2, x, y, label1, label2, title, xlab, ylab, out_png, smooth_window: int = 1, tail_zero_cleanup: bool = False):
    plt.figure(figsize=(8, 5))
    if tail_zero_cleanup:
        df1 = drop_tail_zeros(df1, y)
        df2 = drop_tail_zeros(df2, y)
    if df1 is not None and x in df1 and y in df1:
        y1 = df1[y].rolling(window=smooth_window, min_periods=1).mean() if smooth_window > 1 else df1[y]
        plt.plot(df1[x], y1, label=label1)
    if df2 is not None and x in df2 and y in df2:
        y2 = df2[y].rolling(window=smooth_window, min_periods=1).mean() if smooth_window > 1 else df2[y]
        plt.plot(df2[x], y2, label=label2)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_power(df1, df2, label1, label2, out_png):
    plt.figure(figsize=(8, 5))
    if df1 is not None and "time_left" in df1:
        plt.plot(df1["time_left"], df1.get("prob_p0", 0), "-o", label=f"{label1} P0")
        plt.plot(df1["time_left"], df1.get("prob_p1", 0), "-s", label=f"{label1} P1")
        plt.plot(df1["time_left"], df1.get("prob_p2", 0), "-^", label=f"{label1} P2")
    if df2 is not None and "time_left" in df2:
        plt.plot(df2["time_left"], df2.get("prob_p0", 0), "--o", label=f"{label2} P0")
        plt.plot(df2["time_left"], df2.get("prob_p1", 0), "--s", label=f"{label2} P1")
        plt.plot(df2["time_left"], df2.get("prob_p2", 0), "--^", label=f"{label2} P2")
    plt.xlabel("Time left for V2V (s)")
    plt.ylabel("Probability of power selection")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_sweep(df, metric, ylabel, out_png, std_col=None, lo_col=None, hi_col=None):
    plt.figure(figsize=(8, 5))
    for m, g in df.groupby("model"):
        g = g.sort_values("n_veh")
        y = g[metric]
        x = g["n_veh"]
        _std_col = std_col or (metric + "_std")
        _lo_col = lo_col or (metric + "_lo")
        _hi_col = hi_col or (metric + "_hi")
        has_std = _std_col in g.columns
        has_ci = _lo_col in g.columns and _hi_col in g.columns
        if has_std:
            plt.errorbar(x, y, yerr=g[_std_col], fmt="-o", capsize=4, label=m.upper())
        elif has_ci:
            yerr = np.vstack([y - g[_lo_col], g[_hi_col] - y])
            plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=4, label=m.upper())
        else:
            plt.plot(x, y, "-o", label=m.upper())
    plt.xlabel("Number of Participating Vehicles")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Vehicles")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_decision_time(df, out_png, kind_filter=None):
    plt.figure(figsize=(8, 5))
    if kind_filter and "kind" in df.columns:
        df = df[df["kind"] == kind_filter]
        title_suffix = f" ({kind_filter.upper()})"
    else:
        title_suffix = ""
    models = sorted(df["model"].unique())
    graphs = ["complete", "incomplete"]
    for m in models:
        for g in graphs:
            sub = df[(df["model"] == m) & (df["graph"] == g)].sort_values("n_veh")
            if len(sub) > 0:
                plt.plot(sub["n_veh"], sub["decision_time_s"], "-o", label=f"{m.upper()} {g}")
    plt.xlabel("Number of Participating Vehicles")
    plt.ylabel("Decision Time (s)")
    plt.title(f"Decision Time: Complete vs Incomplete Graph{title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def assert_same_eval(df_sw):
    try:
        ok = True
        msgs = []
        for n in sorted(df_sw["n_veh"].unique()):
            sub = df_sw[df_sw["n_veh"] == n]
            seeds = sub.groupby("model")[["seed_eval"]].first().to_dict().get("seed_eval", {})
            if len(seeds) == 2:
                vals = list(seeds.values())
                if vals[0] != vals[1]:
                    ok = False
                    msgs.append(f"n_veh={n}: seed_eval not equal across models -> {seeds}")
        if not ok:
            print("[WARN] Two models did not reuse the same eval seeds for some n_veh:")
            for m in msgs:
                print("  -", m)
        else:
            print("[OK] Two models reused the same eval seeds for all n_veh.")
    except Exception:
        print("[INFO] sweep_results.csv has no seed_eval column; skip eval-seed consistency check.")


def plot_timeseries_with_rolling(ts: pd.DataFrame, out_png: Path, tag: str):
    if ts is None:
        return
    req_cols = {"t", "v2v_success", "v2i_rate", "num_vehicles"}
    if not req_cols.issubset(set(ts.columns)):
        return
    win = 5
    v2v_m = ts["v2v_success"].rolling(win, min_periods=1).mean()
    v2v_s = ts["v2v_success"].rolling(win, min_periods=2).std().fillna(0.0)
    v2i_m = ts["v2i_rate"].rolling(win, min_periods=1).mean()
    v2i_s = ts["v2i_rate"].rolling(win, min_periods=2).std().fillna(0.0)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ts["t"], v2v_m, color="tab:blue", label="V2V Success (roll mean)")
    ax1.fill_between(ts["t"], (v2v_m - v2v_s).clip(0, 1), (v2v_m + v2v_s).clip(0, 1),
                     color="tab:blue", alpha=0.15, label="V2V Std band")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("V2V Success")
    ax2 = ax1.twinx()
    ax2.plot(ts["t"], v2i_m, color="tab:orange", label="V2I Rate (roll mean)", alpha=0.9)
    ax2.fill_between(ts["t"], (v2i_m - v2i_s), (v2i_m + v2i_s),
                     color="tab:orange", alpha=0.15, label="V2I Std band")
    ax2.set_ylabel("V2I Rate")
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(ts["t"], ts["num_vehicles"], color="tab:green", alpha=0.6, label="Vehicles")
    ax3.set_ylabel("Vehicles")
    ax1.set_xlabel("Time (s)")
    ax1.set_title(f"Dynamic Timeseries ({tag.upper()})")
    lines, labels = [], []
    for a in (ax1, ax2, ax3):
        ls, lb = a.get_legend_handles_labels()
        lines += ls
        labels += lb
    ax1.legend(lines, labels, loc="best")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot comparisons for GAT vs GraphSAGE.")
    ap.add_argument("--gat-run", required=True, help="Path to GAT run dir (runs/v2v_gat_xxx)")
    ap.add_argument("--sage-run", required=True, help="Path to SAGE run dir (runs/v2v_sage_xxx)")
    ap.add_argument("--sweep-csv", required=True, help="Path to sweep_results.csv")
    ap.add_argument("--decision-csv", default=None, help="Path to decision_time.csv")
    ap.add_argument("--out-dir", default="PLOT")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gat_exp = Path(args.gat_run) / "exports"
    sage_exp = Path(args.sage_run) / "exports"

    # 训练损失
    gat_loss = read_csv_safe(gat_exp / "gnn_loss_gat.csv")
    sage_loss = read_csv_safe(sage_exp / "gnn_loss_sage.csv")
    plot_two_curves(gat_loss, sage_loss, "step", "loss", "GAT", "GraphSAGE",
                    "Training Loss", "Step", "Loss", out_dir / "compare_loss.png",
                    smooth_window=1, tail_zero_cleanup=False)

    # 训练期 V2I/V2V
    gat_te = read_csv_safe(gat_exp / "test_history_gat.csv")
    sage_te = read_csv_safe(sage_exp / "test_history_sage.csv")
    plot_two_curves(gat_te, sage_te, "step", "v2i_mean", "GAT", "GraphSAGE",
                    "V2I Mean over Steps", "Step", "V2I Mean", out_dir / "compare_v2i_over_steps.png",
                    smooth_window=3, tail_zero_cleanup=True)
    plot_two_curves(gat_te, sage_te, "step", "v2v_success", "GAT", "GraphSAGE",
                    "V2V Success over Steps", "Step", "V2V Success", out_dir / "compare_v2v_over_steps.png",
                    smooth_window=5, tail_zero_cleanup=False)

    # 功率选择
    gat_power = read_csv_safe(gat_exp / "power_select_gat.csv")
    sage_power = read_csv_safe(sage_exp / "power_select_sage.csv")
    plot_power(gat_power, sage_power, "GAT", "GraphSAGE", out_dir / "compare_power_select.png")

    # 动态时序
    for tag, exp in [("gat", gat_exp), ("sage", sage_exp)]:
        ts = read_csv_safe(exp / f"timeseries_{tag}.csv")
        plot_timeseries_with_rolling(ts, out_dir / f"timeseries_{tag}.png", tag)

    # 车辆数 Sweep：一致性检查 + 画图
    df_sw = pd.read_csv(args.sweep_csv)
    assert_same_eval(df_sw)
    plot_sweep(df_sw, "v2i_mean", "V2I Communication Rate", out_dir / "v2i_vs_vehicles.png")
    plot_sweep(df_sw, "v2v_mean", "V2V Communication Success Rate", out_dir / "v2v_vs_vehicles.png")

    # 决策时间：兼容 kind 列
    if args.decision_csv:
        df_dt = pd.read_csv(args.decision_csv)
        if "kind" in df_dt.columns:
            plot_decision_time(df_dt, out_dir / "decision_time_full.png", kind_filter="full")
            plot_decision_time(df_dt, out_dir / "decision_time_gnn.png", kind_filter="gnn")
        else:
            plot_decision_time(df_dt, out_dir / "decision_time_compare.png", kind_filter=None)

    print("[OK] Plots saved to", out_dir.resolve())


if __name__ == "__main__":
    main()