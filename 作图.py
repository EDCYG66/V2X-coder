# -*- coding: utf-8 -*-
"""
通信侧指标对比：雷达图 + 小表（同时导出两张图片）
用法：
1) 直接运行：python plot_comm_metrics_radar_table.py
2) 按真实数值修改 metrics_dict 中三项：latency_ms, aoi_ms, success_rate
3) 可用 'show_algos' 控制雷达图里显示哪些算法，避免过度拥挤
4) 颜色与你前面的柱状图匹配（可按需微调）
输出：
- comm_metrics_radar.png
- comm_metrics_table.png
"""

import numpy as np
import matplotlib.pyplot as plt

# 你的真实数据填这里（示例占位值）
# latency_ms 和 aoi_ms：越低越好；success_rate：越高越好（0~1）
metrics_dict = {
    "Random":    {"latency_ms": 110, "aoi_ms": 140, "success_rate": 0.90},
    "MLP":       {"latency_ms":  85, "aoi_ms": 110, "success_rate": 0.93},
    "GCN":       {"latency_ms":  72, "aoi_ms":  95, "success_rate": 0.95},
    "DQN":       {"latency_ms":  68, "aoi_ms":  88, "success_rate": 0.96},
    "GraphSage": {"latency_ms":  60, "aoi_ms":  80, "success_rate": 0.97},
    "GATv2":     {"latency_ms":  52, "aoi_ms":  68, "success_rate": 0.985},
}

# 雷达图展示哪些算法（可选子集，避免太挤）
show_algos = ["Random","MLP","GCN","DQN","GraphSage","GATv2"]
# show_algos = ["DQN","GraphSage","GATv2"]  # 只展示关键三条

# 颜色（与你的柱状图保持一致的近似色）
colors = {
    "Random":    "#bfbfbf",  # 灰
    "MLP":       "#d8eaf9",  # 浅蓝
    "GCN":       "#c5e7d3",  # 浅绿
    "DQN":       "#f6e1b4",  # 米黄
    "GraphSage": "#b9d4f6",  # 偏蓝
    "GATv2":     "#f7c6c9",  # 浅粉
}

# 归一化到[0,1]并统一为“越高越好”
def normalize_higher_better(values, reverse=False):
    arr = np.array(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        out = np.ones_like(arr) * 0.5
    else:
        out = (arr - vmin) / (vmax - vmin)
    return 1 - out if reverse else out

algos = list(metrics_dict.keys())
latency_vals = [metrics_dict[a]["latency_ms"] for a in algos]
aoi_vals     = [metrics_dict[a]["aoi_ms"]     for a in algos]
succ_vals    = [metrics_dict[a]["success_rate"] for a in algos]

# 归一化：延迟、AoI 取反（越低越好）
lat_norm = normalize_higher_better(latency_vals, reverse=True)
aoi_norm = normalize_higher_better(aoi_vals,     reverse=True)
suc_norm = normalize_higher_better(succ_vals,    reverse=False)

# 为了表格显示，构造一个排序：按“综合分数”从低到高
score = 0.4*lat_norm + 0.3*aoi_norm + 0.3*suc_norm
sorted_idx = np.argsort(score)  # 低->高
sorted_algos = [algos[i] for i in sorted_idx]

# ---------- 雷达图 ----------
labels = ["延迟(↓)", "AoI(↓)", "成功率(↑)"]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])  # 闭合

plt.figure(figsize=(6.2, 5.8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)    # 让第一个轴在上方
ax.set_theta_direction(-1)

# 雷达图网格
ax.set_thetagrids(angles[:-1] * 180/np.pi, labels, fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2","0.4","0.6","0.8"], fontsize=9)
ax.grid(True, linestyle="--", alpha=0.35)

for name in show_algos:
    i = algos.index(name)
    vals = [lat_norm[i], aoi_norm[i], suc_norm[i]]
    vals = np.array(vals + [vals[0]])  # 闭合
    ax.plot(angles, vals, color=colors.get(name, None), linewidth=2.2, label=name)
    ax.fill(angles, vals, color=colors.get(name, None), alpha=0.15)

# 图例放右侧
ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False, title="算法")
plt.tight_layout()
plt.savefig("comm_metrics_radar.png", dpi=240, bbox_inches="tight")

# ---------- 小表（绘制成图片） ----------
# 表头顺序：算法、延迟(ms)、AoI(ms)、成功率、综合分(0-1)
table_cols = ["算法","延迟(ms)","AoI(ms)","成功率","综合分(0-1)"]
table_data = []
for a in sorted_algos:
    i = algos.index(a)
    table_data.append([
        a,
        int(round(latency_vals[i])),
        int(round(aoi_vals[i])),
        f"{succ_vals[i]*100:.1f}%",
        f"{score[i]:.2f}",
    ])

fig, ax = plt.subplots(figsize=(6.6, 1.6))
ax.axis("off")
the_table = ax.table(cellText=table_data, colLabels=table_cols,
                     cellLoc="center", colLoc="center", loc="center")
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1, 1.4)  # 拉高行距
# 加一点浅色背景提升可读性
for (row, col), cell in the_table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#f0f0f0")
plt.tight_layout()
plt.savefig("comm_metrics_table.png", dpi=240, bbox_inches="tight")
print("已保存：comm_metrics_radar.png, comm_metrics_table.png")