#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnn_factory.py
Factory to build GNN implementations for the Agent.
Supports:
- 'gat': GraphGAT (GATv2-like multi-head attention with optional Top-K pruning)
- 'sage': GraphSAGE (placeholder; if you have an existing GraphSAGE implementation, import it here)

该版本支持从外部（CLI/脚本）透传 GraphGAT 的注意力重剪枝配置：
- reprune_every, hysteresis_keep, reprune_start_step, reg_attn_w

并保留 TB writer与常用超参（lr、gat_train_interval、grad_clip、distance_threshold）的设置。
"""

from typing import Any

def build_gnn(environment: Any,
              gnn_type: str = "gat",
              distance_threshold: float = 150.0,
              lr: float = 5e-4,
              gat_train_interval: int = 20,
              grad_clip: float = 5.0,
              # ---- 新增：可透传到 GraphGAT 的构造入参 ----
              reprune_every: int = 300,
              hysteresis_keep: float = 0.5,
              reprune_start_step: int = 600,
              reg_attn_w: float = 1e-3):
    gnn_type = (gnn_type or "gat").lower()

    if gnn_type == "gat":
        # Import GraphGAT from Graph_GAT.py
        from Graph_GAT import GraphGAT
        # 构建图大小与特征维度：节点数=车辆数*3，输入维=60（Agent.get_state 前60维）
        num_nodes = 3 * len(environment.vehicles)
        
        # --- 核心参数配置 ---
        in_dim = 60
        hidden_dim = 32
        out_dim = 32
        heads = 8         # 建议使用 8 头注意力以提升稳定性
        top_k = 6

        G = GraphGAT(
            num_nodes=num_nodes,
            in_dim=in_dim,           # <--- 之前报错缺失的参数
            hidden_dim=hidden_dim,   # <--- 之前报错缺失的参数
            out_dim=out_dim,         # <--- 之前报错缺失的参数
            heads=heads,
            top_k=top_k,
            prune_mode="distance",   # 或 "attention"
            add_self_loop=True,
            attn_dropout=0.0,
            use_low_rank=False,
            low_rank_k=16,
            # 透传：注意力重剪枝配置
            reprune_every=int(reprune_every),
            hysteresis_keep=float(hysteresis_keep),
            reprune_start_step=int(reprune_start_step),
            reg_attn_w=float(reg_attn_w),
        )

        # 附一些环境信息（位置）便于距离剪枝
        try:
            positions = []
            for v in environment.vehicles:
                positions.append([float(v.position[0]), float(v.position[1])])
            import numpy as np
            positions_nodes = np.repeat(np.asarray(positions, dtype=float), 3, axis=0)
            G.update_positions(positions_nodes)
        except Exception:
            pass

        # 保留工厂传入的训练间隔（给 Agent 读取用）
        G.gat_train_interval = int(gat_train_interval)
        G.lr = float(lr)
        G.grad_clip = float(grad_clip)
        G.distance_threshold = float(distance_threshold)
        return G

    elif gnn_type == "sage":
        # 如果已有 GraphSAGE 的实现模块，请引入；否则用 GraphGAT 作为占位
        try:
            from Graph_SAGE import GraphSAGE
            G = GraphSAGE(environment, in_dim=60, out_dim=32, lr=lr)
            G.gat_train_interval = int(gat_train_interval)
            return G
        except Exception:
            from Graph_GAT import GraphGAT
            num_nodes = 3 * len(environment.vehicles)
            # 这里也需要补全 GraphSAGE fallback 的参数
            G = GraphGAT(
                num_nodes=num_nodes, 
                in_dim=60,          # 补全
                hidden_dim=32,      # 补全
                out_dim=32,         # 补全
                heads=1, 
                top_k=6,
                reprune_every=int(reprune_every),
                hysteresis_keep=float(hysteresis_keep),
                reprune_start_step=int(reprune_start_step),
                reg_attn_w=float(reg_attn_w),
            )
            G.gat_train_interval = int(gat_train_interval)
            return G

    else:
        raise ValueError(f"Unknown gnn_type: {gnn_type}. Use 'gat' or 'sage'.")