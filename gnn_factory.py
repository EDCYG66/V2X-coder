#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def build_gnn(env, gnn_type: str = "gat", **kwargs):
    """
    Factory to build a graph model wrapper with unified interface use_GraphSAGE.
    - gnn_type: "gat" -> Graph_GAT.GraphSAGE_sup (已固定为 GATv2 实现)
                "sage" -> Graph_SAGE.GraphSAGE_sup
    kwargs will be passed to constructor if supported.
    """
    lt = (gnn_type or "gat").lower()
    if lt == "gat":
        from Graph_GAT import GraphSAGE_sup as Impl
        # 精简：不再接受/透传 attn_version
        allowed = {
            "distance_threshold",
            "lr",
            "gat_train_interval",
            "grad_clip",
        }
        return Impl(env, **{k: v for k, v in kwargs.items() if k in allowed})
    elif lt == "sage":
        from Graph_SAGE import GraphSAGE_sup as Impl
        return Impl(env)
    else:
        raise ValueError(f"Unknown gnn_type={gnn_type}")