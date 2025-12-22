#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_compare.py
Statistical analysis script for multiple comparison_metrics.csv files.

Reads:
  - Glob pattern over subdirectories: e.g. runs/compare_high_difficulty*/comparison_metrics.csv
  - Each CSV expected format:
      model,v2i_final,v2v_final,v2i_conv_step,power_entropy,rb_unique_ratio,rb_gini,decision_time_full_s,decision_time_gnn_only_s

Outputs:
  - Per-metric Welch t-test (GAT vs GraphSAGE)
  - Bootstrap mean difference with 95% CI
  - Summary table (printed + JSON export)
  - Optional effect size (Cohen's d)

Usage:
  python stats_compare.py --glob 'runs/compare_*/*/comparison_metrics.csv' --out stats_summary.json
  (Wrap in quotes to avoid shell expansion issues.)

Dependencies: numpy, pandas, scipy (for ttest_ind)
If scipy not available, fallback simple t-test approximation.

"""

import argparse
import glob
import json
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_ind
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


METRICS = [
    "v2i_final",
    "v2v_final",
    "power_entropy",
    "rb_gini",
    "decision_time_full_s",
    "decision_time_gnn_only_s"
]


def read_all(glob_pattern: str):
    files = glob.glob(glob_pattern)
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Expect two rows: gat + graphsage
            for _, r in df.iterrows():
                rows.append(dict(file=f, **r.to_dict()))
        except Exception as e:
            print(f"[WARN] Cannot read {f}: {e}")
    if not rows:
        raise RuntimeError(f"No comparison_metrics.csv matched pattern {glob_pattern}")
    return pd.DataFrame(rows)


def welch_t_test(a: np.ndarray, b: np.ndarray):
    if SCIPY_OK:
        t_stat, p_val = ttest_ind(a, b, equal_var=False)
        return float(t_stat), float(p_val)
    # Fallback manual approximation
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    n_a, n_b = len(a), len(b)
    t_stat = (mean_a - mean_b) / np.sqrt(var_a / n_a + var_b / n_b)
    # Approximate df
    df = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a ** 2) / (n_a ** 2 * (n_a - 1)) + (var_b ** 2) / (n_b ** 2 * (n_b - 1))
    )
    # Two-sided p (approx normal if scipy missing)
    from math import erf, sqrt
    p_val = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / np.sqrt(2))))
    return float(t_stat), float(p_val)


def cohen_d(a: np.ndarray, b: np.ndarray):
    # Pooled standard deviation
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot=5000, ci=0.95):
    rng = np.random.default_rng(12345)
    diffs = []
    na, nb = len(a), len(b)
    for _ in range(n_boot):
        sa = rng.choice(a, na, replace=True)
        sb = rng.choice(b, nb, replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    lower = np.percentile(diffs, (1 - ci) / 2 * 100)
    upper = np.percentile(diffs, (1 + ci) / 2 * 100)
    return float(diffs.mean()), float(lower), float(upper)


def summarize(df: pd.DataFrame):
    out = {}
    # Group by model
    for m in ["gat", "graphsage"]:
        sub = df[df.model == m]
        out[m] = {}
        for metric in METRICS:
            if metric in sub:
                vals = sub[metric].values.astype(float)
                out[m][metric] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)),
                    "n": int(len(vals))
                }
    # Comparative stats
    comp = {}
    for metric in METRICS:
        if metric not in df.columns:
            continue
        ag = df[df.model == "gat"][metric].values.astype(float)
        sg = df[df.model == "graphsage"][metric].values.astype(float)
        if len(ag) == 0 or len(sg) == 0:
            continue
        t, p = welch_t_test(ag, sg)
        d = cohen_d(ag, sg)
        diff_mean, diff_l, diff_u = bootstrap_diff(ag, sg)
        comp[metric] = {
            "gat_mean": float(ag.mean()),
            "sage_mean": float(sg.mean()),
            "mean_diff_gat_minus_sage": diff_mean,
            "boot_ci_lower": diff_l,
            "boot_ci_upper": diff_u,
            "t_stat": t,
            "p_val": p,
            "cohen_d": d
        }
    return out, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True,
                    help="Glob pattern for comparison_metrics.csv files, e.g. 'runs/compare_*/*/comparison_metrics.csv'")
    ap.add_argument("--out", type=str, default="stats_summary.json")
    ap.add_argument("--p-threshold", type=float, default=0.05)
    args = ap.parse_args()

    df = read_all(args.glob)
    agg, comp = summarize(df)

    # Print table
    print("=== Aggregate Means (per model) ===")
    for m, metrics in agg.items():
        print(f"[{m}]")
        for k, v in metrics.items():
            print(f"  {k}: mean={v['mean']:.4f} std={v['std']:.4f} n={v['n']}")

    print("\n=== Comparative Statistics (GAT - GraphSAGE) ===")
    for k, v in comp.items():
        sig = "YES" if v["p_val"] < args.p_threshold else "NO"
        print(f"{k}: diff={v['mean_diff_gat_minus_sage']:.4f} "
              f"95%CI=({v['boot_ci_lower']:.4f},{v['boot_ci_upper']:.4f}) "
              f"t={v['t_stat']:.3f} p={v['p_val']:.3e} d={v['cohen_d']:.3f} significant={sig}")

    # JSON export
    out_json = {
        "aggregate": agg,
        "comparison": comp,
        "p_threshold": args.p_threshold,
        "glob_used": args.glob
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved statistical summary -> {args.out}")


if __name__ == "__main__":
    main()