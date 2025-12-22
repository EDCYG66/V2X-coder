import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# ================= 配置区域 =================
# 定义你的实验文件夹路径和对应的图例名称
experiments = {
    "runs/final_optimized_v3": "Proposed GATv2",
    "runs/ablation_no_pruning": "w/o Adaptive Pruning",
    "runs/ablation_no_urgency": "w/o Urgency Penalty",
    "runs/ablation_single_head": "w/o Multi-Head Attn"
}

output_dir = "runs/ablation_plots_final"
os.makedirs(output_dir, exist_ok=True)
# ===========================================

def load_data(folder, label):
    # 读取 V2V 数据
    v2v_path = os.path.join(folder, "gat_v2v_curve.csv")
    v2i_path = os.path.join(folder, "gat_v2i_curve.csv")
    
    data_v2v = None
    data_v2i = None
    
    if os.path.exists(v2v_path):
        df = pd.read_csv(v2v_path)
        df['Label'] = label
        data_v2v = df
    else:
        print(f"警告: 找不到 {v2v_path}")

    if os.path.exists(v2i_path):
        df = pd.read_csv(v2i_path)
        df['Label'] = label
        data_v2i = df
        
    return data_v2v, data_v2i

# 收集数据
all_v2v = []
all_v2i = []

print("正在读取数据...")
for folder, label in experiments.items():
    v2v, v2i = load_data(folder, label)
    if v2v is not None: all_v2v.append(v2v)
    if v2i is not None: all_v2i.append(v2i)

if not all_v2v:
    print("没有找到任何数据！请确保你已经运行了所有消融实验脚本。")
    exit()

df_v2v = pd.concat(all_v2v)
df_v2i = pd.concat(all_v2i)

# 设置绘图风格 (仿论文风格)
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'serif' # 使用衬线字体，更像论文

# --- 绘图 1: V2V 成功率对比 ---
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_v2v, x="step", y="v2v_success", hue="Label", style="Label", linewidth=2.5)
plt.title("Ablation Study: V2V Reliability", fontsize=16, fontweight='bold')
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("V2V Success Rate", fontsize=14)
plt.ylim(0.5, 1.05) # 聚焦于 50% - 100% 区间
plt.legend(title="", loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(f"{output_dir}/ablation_v2v.png", dpi=300)
plt.savefig(f"{output_dir}/ablation_v2v.pdf") # PDF 格式适合插入 LaTeX
print(f"生成图片: {output_dir}/ablation_v2v.png")

# --- 绘图 2: V2I 速率对比 ---
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_v2i, x="step", y="v2i_mean", hue="Label", style="Label", linewidth=2.5)
plt.title("Ablation Study: V2I Throughput", fontsize=16, fontweight='bold')
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("V2I Rate (Mbps)", fontsize=14)
plt.legend(title="", loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(f"{output_dir}/ablation_v2i.png", dpi=300)
plt.savefig(f"{output_dir}/ablation_v2i.pdf")
print(f"生成图片: {output_dir}/ablation_v2i.png")

print(">>> 所有消融实验对比图绘制完成！")

