# -*- coding: utf-8 -*-
"""
Graph_SAGE backend (unified interface with Graph_GAT.GraphSAGE_sup)

Key features
- Same public methods used by Agent:
  * build_graph, load_graph, use_GraphSAGE
  * Optional _forward_all (batch inference for all nodes)
  * Attributes: gat_train_interval, tb_writer, gat_loss_history
- Robust dtypes when mixed precision is enabled elsewhere:
  * Force SAGE to work in float32 (set global policy on construction)
  * Cast embeddings/labels to float32 during loss computation
- Loss logging:
  * gat_loss_history: list of (step, loss) to align with GAT plots
  * gs_losses: legacy list[loss] kept for compatibility
"""
from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx
import tensorflow as tf

from Environment import *  # noqa: F401 - used by other parts of the project
from model_Graph import GraphModel


class GraphSAGE_sup(object):
    def __init__(self,
                 environment,
                 distance_threshold: float = 150.0,
                 lr: float = 5e-4,
                 gat_train_interval: Optional[int] = None,  # keep name for interface consistency
                 grad_clip: float = 5.0):
        self.env = environment
        self.weight_dir = 'weight'
        os.makedirs(self.weight_dir, exist_ok=True)

        # Important: ensure SAGE runs in float32 even if GAT enabled mixed_float16 elsewhere
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy('float32')
        except Exception:
            pass

        # Hyper-params
        self.distance_threshold = float(distance_threshold)
        self.lr = float(lr)
        self.grad_clip = float(grad_clip)
        self.gat_train_interval = int(gat_train_interval) if gat_train_interval is not None else 50

        # Graph model (GCN/SAGE-style); output dims must match channel_reward dims (=20)
        self.G_model = GraphModel(sample_num=5, depth=2, dims=20, gcn=True, concat=True)
        self.G_model_target = GraphModel(sample_num=5, depth=2, dims=20, gcn=True, concat=True)
        self.update_target_network()  # init target = online

        # Runtime tensors/buffers
        self.num_vehicle = len(getattr(self.env, "vehicles", []))
        self.features = np.zeros((max(1, 3 * self.num_vehicle), 60), dtype=np.float32)

        # Book-keeping
        self.order_nodes: List[int] = list(range(max(1, 3 * self.num_vehicle)))
        self.link = np.zeros((max(1, 3 * self.num_vehicle), 2), dtype=np.int32)
        self.graph = nx.Graph()
        self.gs_losses: List[float] = []            # legacy, kept for compatibility
        self.gat_loss_history: List[Tuple[int, float]] = []  # aligned with GAT: (step, loss)
        self.tb_writer = None  # will be set by Agent if available

        # LR decay config
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 2000
        self.learning_rate_minimum = 1e-4

        self.compile_model()

        # Placeholder used by Agent before first build_graph call
        self.num_V2V_list = np.zeros((self.num_vehicle, self.num_vehicle), dtype=np.float32)

    # -------------------- Keras compile --------------------
    def compile_model(self):
        # Base schedule
        base = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=int(self.learning_rate_decay_step),
            decay_rate=float(self.learning_rate_decay),
            staircase=True,
        )

        # Minimum LR clipping
        class ClippedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, inner, min_lr: float):
                self.inner = inner
                self.min_lr = tf.constant(min_lr, dtype=tf.float32)

            def __call__(self, step):
                return tf.maximum(self.inner(step), self.min_lr)

            def get_config(self):
                return {
                    "inner": tf.keras.saving.serialize_keras_object(self.inner),
                    "min_lr": float(self.min_lr.numpy())
                }

        lr_schedule = ClippedSchedule(base, float(self.learning_rate_minimum))

        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=lr_schedule,
            rho=0.95,
            epsilon=1e-7
        )

        self.G_model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    def update_target_network(self):
        try:
            self.G_model_target.set_weights(self.G_model.get_weights())
        except Exception:
            pass

    # -------------------- Graph building/loading --------------------
    def build_graph(self, num_V2V_list: np.ndarray) -> Tuple[nx.Graph, List[int], None]:
        vehicles = getattr(self.env, "vehicles", [])
        nveh = len(vehicles)
        n_nodes = 3 * nveh
        G = nx.Graph()
        order_nodes = list(range(n_nodes))
        G.add_nodes_from(order_nodes)

        # Use vehicle positions (if available) to build distance-based edges
        pos = [np.asarray(getattr(v, "position", (0.0, 0.0)), dtype=np.float32) for v in vehicles]

        for i in range(nveh):
            for k in range(i + 1, nveh):
                d = float(np.linalg.norm(pos[i] - pos[k]))
                if d <= self.distance_threshold:
                    # connect all link-nodes (3x3) between vehicle i and k
                    for j in range(3):
                        for t in range(3):
                            u = 3 * i + j
                            v = 3 * k + t
                            G.add_edge(u, v)

        return G, order_nodes, None

    def load_graph(self, graph: nx.Graph, order_nodes: List[int]):
        self.graph = graph
        self.order_nodes = list(order_nodes)
        n_nodes = len(self.order_nodes)
        link = np.zeros((n_nodes, 2), dtype=np.int32)
        for nid in range(n_nodes):
            link[nid, 0] = nid // 3
            link[nid, 1] = nid % 3
        self.link = link

    # -------------------- Neighborhood sampling --------------------
    def fetch_batch(self, order_nodes: List[int], idx_list: List[int]):
        """
        For each node in idx_list, sample up to S1 first-order neighbors,
        and for each first neighbor sample up to S2 second-order neighbors.

        Returns:
        - first_order_neighs:  [B, S1] int32
        - second_order_neighs: [B, S1, S2] int32
        - s1_weights:          [B, S1] float32 (row-uniform)
        - s2_weights:          [B, S1, S2] float32 (row-uniform per first neighbor)
        """
        S1 = getattr(self.G_model, "sample_num", 5) or 5
        S2 = S1  # same width for 2nd hop

        index_map = {n: i for i, n in enumerate(order_nodes)}

        B = len(idx_list)
        f1 = np.zeros((B, S1), dtype=np.int32)
        f2 = np.zeros((B, S1, S2), dtype=np.int32)
        w1 = np.zeros((B, S1), dtype=np.float32)
        w2 = np.zeros((B, S1, S2), dtype=np.float32)

        for bi, nid in enumerate(idx_list):
            node_label = order_nodes[nid]
            neighs = list(nx.neighbors(self.graph, node_label))
            neigh_idx = [index_map.get(x, nid) for x in neighs] or [nid]

            # First-order cap/pad to S1
            if len(neigh_idx) >= S1:
                chosen = neigh_idx[:S1]
            else:
                chosen = neigh_idx + [nid] * (S1 - len(neigh_idx))
            f1[bi, :] = np.asarray(chosen, dtype=np.int32)
            w1_count = max(1, len(neighs))
            w1_val = 1.0 / float(w1_count)
            w1[bi, :] = w1_val

            # Second-order per chosen first neighbor
            for s, n1 in enumerate(chosen):
                n1_label = order_nodes[n1]
                neighs2 = list(nx.neighbors(self.graph, n1_label))
                neigh2_idx = [index_map.get(x, n1) for x in neighs2] or [n1]
                if len(neigh2_idx) >= S2:
                    chosen2 = neigh2_idx[:S2]
                else:
                    chosen2 = neigh2_idx + [n1] * (S2 - len(neigh2_idx))
                f2[bi, s, :] = np.asarray(chosen2, dtype=np.int32)
                w2_count = max(1, len(neighs2))
                w2_val = 1.0 / float(w2_count)
                w2[bi, s, :] = w2_val

        return f1, f2, w1, w2

    # -------------------- Optional: whole-graph forward --------------------
    def _forward_all(self, training: bool = False) -> tf.Tensor:
        """
        Batch inference for all nodes. Not used during training in this SAGE backend,
        but provided for compatibility/performance with callers that prefer it.
        """
        if self.features is None:
            raise ValueError("features is None")
        if self.graph is None or self.order_nodes is None:
            raise ValueError("graph/order_nodes is None (call load_graph first)")
        N = int(self.features.shape[0])
        idx_all = list(range(N))
        f1, f2, w1, w2 = self.fetch_batch(self.order_nodes, idx_all)
        inputs = (
            tf.convert_to_tensor(self.features, dtype=tf.float32),
            tf.convert_to_tensor(idx_all, dtype=tf.int32),
            tf.convert_to_tensor(f1, dtype=tf.int32),
            tf.convert_to_tensor(f2, dtype=tf.int32),
            tf.convert_to_tensor(w1, dtype=tf.float32),
            tf.convert_to_tensor(w2, dtype=tf.float32),
        )
        out = self.G_model(inputs, training=training)
        return tf.cast(out, tf.float32)

    # -------------------- Forward/Train entry --------------------
    def use_GraphSAGE(self,
                      channel_reward: np.ndarray,
                      step: int,
                      idx: List[int],
                      train_flag: bool = True):
        """
        Compute embeddings for node indices in idx.

        Training objective (light supervision):
            minimize MSE( emb, 0.5*emb_target + 0.5*label )
        where label = channel_reward[idx, :] (shape [B, dims]).
        """
        # Prepare sampled neighborhoods
        first_order_neighs, second_order_neighs, s1_weights, s2_weights = \
            self.fetch_batch(self.order_nodes, idx)

        # Build tuple inputs expected by GraphModel, enforce dtypes explicitly
        inputs = (
            tf.convert_to_tensor(self.features, dtype=tf.float32),          # [N, 60]
            tf.convert_to_tensor(idx, dtype=tf.int32),                      # [B]
            tf.convert_to_tensor(first_order_neighs, dtype=tf.int32),       # [B, S1]
            tf.convert_to_tensor(second_order_neighs, dtype=tf.int32),      # [B, S1, S2]
            tf.convert_to_tensor(s1_weights, dtype=tf.float32),             # [B, S1]
            tf.convert_to_tensor(s2_weights, dtype=tf.float32),             # [B, S1, S2]
        )

        if train_flag and step % self.gat_train_interval == 0 and step > 0:
            with tf.GradientTape() as tape:
                emb = self.G_model(inputs, training=True)            # may be float16 if policy changes externally
                emb_t = self.G_model_target(inputs, training=False)
                # Ensure float32 math when mixing with labels
                emb = tf.cast(emb, tf.float32)
                emb_t = tf.cast(emb_t, tf.float32)

                labels = tf.gather(tf.convert_to_tensor(channel_reward, tf.float32),
                                   tf.convert_to_tensor(idx, tf.int32), axis=0)   # float32
                target = 0.5 * emb_t + 0.5 * labels
                loss = tf.reduce_mean(tf.square(emb - target))

            vars_ = self.G_model.trainable_variables
            grads = tape.gradient(loss, vars_)
            if self.grad_clip and self.grad_clip > 0:
                grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            self.G_model.optimizer.apply_gradients(zip(grads, vars_))

            lv = float(loss.numpy())
            self.gs_losses.append(lv)                   # legacy list
            self.gat_loss_history.append((int(step), lv))  # preferred: (step, loss)

            # TB (optional)
            if self.tb_writer is not None and (step % 100 == 0):
                with self.tb_writer.as_default():
                    tf.summary.scalar('SAGE/loss', lv, step=step)

            emb_out = emb  # already float32
        else:
            emb_out = self.G_model(inputs, training=False)
            emb_out = tf.cast(emb_out, tf.float32)

        # Periodically update target and save weights
        if step % 100 == 99:
            self.update_target_network()
            try:
                self.G_model.save_weights(os.path.join(self.weight_dir, 'GNN_weights_sage.h5'))
            except Exception:
                pass

        return emb_out.numpy()