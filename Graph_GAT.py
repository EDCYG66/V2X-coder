from __future__ import annotations
import numpy as np
import tensorflow as tf
import networkx as nx
from typing import List, Tuple, Optional

# -------------------- Utilities --------------------

def _get_adamw(learning_rate, weight_decay=1e-4):
    """
    Return an AdamW optimizer that works across TF 2.12+ variants.
    """
    try:
        # TF 2.13+
        from tensorflow.keras.optimizers import AdamW  # type: ignore
        return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    except Exception:
        try:
            # TF 2.12 experimental
            from tensorflow.keras.optimizers.experimental import AdamW  # type: ignore
            return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        except Exception:
            # Fallback to Adam (no weight decay)
            from tensorflow.keras.optimizers import Adam  # type: ignore
            return Adam(learning_rate=learning_rate)


class WarmupThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, warmup_steps: int = 500, decay_steps: int = 2000, decay_rate: float = 0.96):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(1, warmup_steps))
        self.decay = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.base_lr,
            decay_steps=int(decay_steps),
            decay_rate=float(decay_rate),
            staircase=True,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.minimum(1.0, step / float(self.warmup_steps))
        return warm * self.decay(step)


# -------------------- GAT v2 layer (only) --------------------

class GATv2Layer(tf.keras.layers.Layer):
    """
    GAT v2: a^T( LeakyReLU( W_attn [Wh_i || Wh_j] ) ) / temperature
    - presorted=True 表示 edge_index 已按 dst 升序（load_graph 中完成）
    - 使用 unsorted_segment_*；注意力 logits/softmax 在 float32 计算以提升数值稳定性
    """
    def __init__(self, out_dim: int, num_heads: int = 1, concat: bool = False,
                 dropout: float = 0.1, alpha: float = 0.2,
                 attn_topk: int = 6, attn_temperature: float = 0.55,
                 presorted: bool = True, attn_hidden: Optional[int] = None):
        super().__init__()
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.concat = bool(concat)
        self.dropout = float(dropout)
        self.alpha = float(alpha)
        self.attn_topk = int(attn_topk)
        self.attn_temperature = float(attn_temperature)
        self.presorted = bool(presorted)
        self.attn_hidden = int(attn_hidden) if attn_hidden else int(out_dim)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="W", shape=(in_dim, self.num_heads, self.out_dim),
            initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.W_attn = self.add_weight(
            name="W_attn", shape=(self.num_heads, 2 * self.out_dim, self.attn_hidden),
            initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.a_vec = self.add_weight(
            name="a_vec", shape=(self.num_heads, self.attn_hidden),
            initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.leaky = tf.keras.layers.LeakyReLU(self.alpha)
        self.drop = tf.keras.layers.Dropout(self.dropout)
        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def _segment_softmax(self, scores32, seg_ids, n_seg):
        max_per = tf.math.unsorted_segment_max(scores32, seg_ids, n_seg)
        scores32 = scores32 - tf.gather(max_per, seg_ids)
        exp = tf.exp(scores32)
        sum_per = tf.math.unsorted_segment_sum(exp, seg_ids, n_seg)
        denom = tf.gather(sum_per, seg_ids) + 1e-9
        return exp / denom

    def call(self, x: tf.Tensor, edge_index: tf.Tensor, training=False):
        N = tf.shape(x)[0]
        Wh = tf.tensordot(x, self.W, axes=1)  # (N, heads, Fout)

        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        if self.presorted:
            src_s, dst_s = src, dst
        else:
            sort_idx = tf.argsort(dst, stable=True)
            src_s = tf.gather(src, sort_idx)
            dst_s = tf.gather(dst, sort_idx)

        Wh_src = tf.gather(Wh, src_s)  # (E, heads, Fout)
        Wh_dst = tf.gather(Wh, dst_s)  # (E, heads, Fout)

        # [Wh_i || Wh_j] -> per-head MLP -> a_vec
        z = tf.concat([Wh_src, Wh_dst], axis=-1)            # (E, heads, 2*Fout)
        z_lin = tf.einsum('ehf,hfc->ehc', z, self.W_attn)   # (E, heads, H)
        z_act = self.leaky(z_lin)
        logits32 = tf.cast(tf.einsum('ehc,hc->eh', z_act, self.a_vec), tf.float32)
        e32 = logits32 / self.attn_temperature

        alphas32_list = []
        for h in range(self.num_heads):
            alphas32_list.append(self._segment_softmax(e32[:, h], dst_s, N))
        alphas32 = tf.stack(alphas32_list, axis=1)
        alphas = tf.cast(alphas32, Wh_src.dtype)

        msg = Wh_src * tf.expand_dims(alphas, axis=-1)
        out = tf.math.unsorted_segment_sum(msg, dst_s, N)  # (N, heads, Fout)

        out = self.drop(out, training=training)
        if self.concat:
            out = tf.reshape(out, (N, self.num_heads * self.out_dim))
        else:
            out = tf.reduce_mean(out, axis=1)
        return out


# -------------------- Model（v2 only） --------------------

class GATModel(tf.keras.Model):
    def __init__(self,
                 in_dim=60, hidden=32, heads1=1, heads2=1, out_dim=20, dropout=0.1,
                 use_residual=True, attn_topk=6, attn_temperature=0.55,
                 presorted=True):
        super().__init__()
        self.use_residual = bool(use_residual)
        self.out_dim = int(out_dim)
        self.norm_in = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        Layer = GATv2Layer  # 固定使用 v2

        self.gat1 = Layer(out_dim=hidden, num_heads=heads1, concat=False, dropout=dropout,
                          attn_topk=attn_topk, attn_temperature=attn_temperature, presorted=presorted)
        self.act1 = tf.keras.layers.ELU()
        self.gat2 = Layer(out_dim=out_dim, num_heads=heads2, concat=False, dropout=dropout,
                          attn_topk=attn_topk, attn_temperature=attn_temperature, presorted=presorted)
        self.norm_out = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self._res_proj: Optional[tf.keras.layers.Layer] = None

    def call(self, x, edge_index, training=False):
        h0 = x
        x = self.norm_in(tf.convert_to_tensor(x))
        h = self.gat1(x, edge_index, training=training)
        h = self.act1(h)
        h = self.gat2(h, edge_index, training=training)
        if self.use_residual:
            in_ch = h0.shape[-1]
            out_ch = h.shape[-1]
            if (in_ch is not None) and (out_ch is not None) and (in_ch == out_ch):
                h = h + 0.3 * tf.cast(h0, h.dtype)
            else:
                if self._res_proj is None:
                    self._res_proj = tf.keras.layers.Dense(int(out_ch), use_bias=False, name="residual_proj")
                h = h + 0.3 * self._res_proj(tf.cast(h0, h.dtype))
        return self.norm_out(h)


# -------------------- Wrapper（统一接口，GAT v2） --------------------

class GraphSAGE_sup:
    """
    GAT v2 包装器，提供与 Graph_SAGE 相同的外部接口：
    - build_graph, load_graph, use_GraphSAGE
    - _forward_all（整图前向）
    - 属性：gat_train_interval, tb_writer, gat_loss_history
    训练改进默认值：
    - AdamW + weight_decay=1e-4
    - 目标平滑：0.2 * target + 0.8 * labels（更信任监督）
    - Dropout warmup：前 500 步 0.03，之后 0.08
    - EMA τ = 0.10
    - Top-K 稀疏默认 4，温度 0.50
    """
    def __init__(self, env,
                 distance_threshold=150.0,
                 lr=1e-3,                         # 提升初始 LR（原 5e-4）
                 gat_train_interval=20,           # 更高训练频率
                 grad_clip=5.0,
                 attn_topk: int = 4,
                 attn_temperature: float = 0.50,
                 ema_tau: float = 0.10,
                 warmup_steps: int = 500,
                 allow_xla: bool = False,
                 use_mixed_precision: bool = True):
        self.env = env
        self.distance_threshold = float(distance_threshold)
        self.features: Optional[np.ndarray] = None
        self.graph: Optional[nx.Graph] = None
        self.order_nodes: Optional[List[int]] = None
        self.edge_index: Optional[tf.Tensor] = None
        self.link: Optional[np.ndarray] = None
        self.N: Optional[int] = None
        self._built = False
        self.tb_writer = None

        self.attn_topk = int(attn_topk)
        self.attn_temperature = float(attn_temperature)
        self.ema_tau = float(ema_tau)
        self.warmup_steps = int(warmup_steps)
        self.grad_clip = float(grad_clip) if grad_clip is not None else None
        self.gat_train_interval = int(gat_train_interval)

        if allow_xla:
            try:
                tf.config.optimizer.set_jit(True)
                print("[GAT] XLA JIT enabled")
            except Exception:
                pass
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
        except Exception:
            pass

        self.use_mixed_precision = bool(use_mixed_precision)
        if self.use_mixed_precision:
            try:
                from tensorflow.keras import mixed_precision as mp
                mp.set_global_policy('mixed_float16')
                print("[GAT] Mixed precision: mixed_float16")
            except Exception:
                self.use_mixed_precision = False

        # Models (presorted=True)
        self.G_model = GATModel(in_dim=60, hidden=32, heads1=1, heads2=1, out_dim=20, dropout=0.1,
                                use_residual=True, attn_topk=self.attn_topk,
                                attn_temperature=self.attn_temperature, presorted=True)
        self.G_model_target = GATModel(in_dim=60, hidden=32, heads1=1, heads2=1, out_dim=20, dropout=0.1,
                                       use_residual=True, attn_topk=self.attn_topk,
                                       attn_temperature=self.attn_temperature, presorted=True)
        self.G_model_target.set_weights(self.G_model.get_weights())

        # Optimizer: AdamW (+ LossScale if mixed precision)
        lr_sched = WarmupThenDecay(base_lr=lr, warmup_steps=self.warmup_steps, decay_steps=2000, decay_rate=0.96)
        base_opt = _get_adamw(lr_sched, weight_decay=1e-4)
        if self.use_mixed_precision:
            from tensorflow.keras.mixed_precision import LossScaleOptimizer
            self.optimizer = LossScaleOptimizer(base_opt)
        else:
            self.optimizer = base_opt

        self.gat_loss_history: List[Tuple[int, float]] = []

        @tf.function(
            reduce_retracing=True,
            input_signature=[
                tf.TensorSpec(shape=[None, 60], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 20], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.float32),
            ],
        )
        def _train_step(features, edge_index, labels, mask):
            with tf.GradientTape() as tape:
                preds = self.G_model(features, edge_index, training=True)
                preds_t = self.G_model_target(features, edge_index, training=False)

                preds32 = tf.cast(preds, tf.float32)
                preds_t32 = tf.cast(preds_t, tf.float32)
                labels32 = tf.cast(labels, tf.float32)

                # 更强监督：0.2 * target + 0.8 * labels
                smoothed = 0.2 * preds_t32 + 0.8 * labels32
                diff2 = tf.square(preds32 - smoothed)

                masked = diff2 * tf.expand_dims(mask, 1)
                denom_nodes = tf.reduce_sum(mask) + 1e-6
                denom_feats = tf.cast(tf.shape(labels)[-1], tf.float32)
                loss = tf.reduce_sum(masked) / (denom_nodes * denom_feats)

                if self.use_mixed_precision:
                    loss = self.optimizer.get_scaled_loss(loss)

            grads = tape.gradient(loss, self.G_model.trainable_variables)
            if self.use_mixed_precision:
                grads = self.optimizer.get_unscaled_gradients(grads)
            if self.grad_clip is not None:
                grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            self.optimizer.apply_gradients(zip(grads, self.G_model.trainable_variables))

            if self.use_mixed_precision:
                # unscale back for logging
                loss = tf.reduce_sum(masked) / (denom_nodes * denom_feats)
            return preds, loss

        self._tf_train_step = _train_step

    # -------------------- Graph build / load --------------------

    def build_graph(self, num_V2V_list: np.ndarray):
        n_veh = len(self.env.vehicles)
        n_nodes = 3 * n_veh

        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        order_nodes = list(range(n_nodes))
        link = np.zeros((n_nodes, 2), dtype=np.int32)

        for i in range(n_veh):
            for j in range(3):
                k = 3 * i + j
                link[k, 0] = i
                link[k, 1] = k

        # 同车 fully connect
        for i in range(n_veh):
            ks = [3 * i + 0, 3 * i + 1, 3 * i + 2]
            for a in range(3):
                for b in range(a + 1, 3):
                    g.add_edge(ks[a], ks[b])

        # 距离阈值内车辆 fully connect (3x3)
        D = self.env.Distance
        thr = self.distance_threshold
        for i in range(n_veh):
            for j in range(i + 1, n_veh):
                if D[i, j] <= thr:
                    for a in range(3):
                        for b in range(3):
                            g.add_edge(3 * i + a, 3 * j + b)

        self.graph = g
        self.order_nodes = order_nodes
        self.link = link
        self.N = n_nodes
        return g, order_nodes, None

    def load_graph(self, graph: nx.Graph, order_nodes: List[int]):
        self.graph = graph
        self.order_nodes = order_nodes
        edges = list(graph.edges())
        directed = []
        for u, v in edges:
            directed.append([u, v])
            directed.append([v, u])
        if not directed:
            n = len(order_nodes)
            directed = [[i, i] for i in range(n)]

        # Top-K 入边裁剪（按车辆级距离）
        if self.attn_topk and self.attn_topk > 0:
            D = self.env.Distance
            k = int(self.attn_topk)
            per_dst = {}
            pruned = []
            for u, v in directed:
                per_dst.setdefault(v, []).append(u)
            for v, srcs in per_dst.items():
                ve = v // 3
                srcs_unique = list(dict.fromkeys(srcs))
                srcs_sorted = sorted(srcs_unique, key=lambda s: D[ve, (s // 3)])
                for s in srcs_sorted[:k]:
                    pruned.append([s, v])
            directed = pruned

        arr = np.asarray(directed, dtype=np.int32)
        order = np.argsort(arr[:, 1], kind="mergesort")  # stable sort by dst
        arr = arr[order]
        self.edge_index = tf.convert_to_tensor(arr)

        # 标记 presorted=True
        for layer in (self.G_model.gat1, self.G_model.gat2, self.G_model_target.gat1, self.G_model_target.gat2):
            layer.presorted = True

        self._maybe_build()

    def _maybe_build(self):
        if (not self._built) and (self.features is not None) and (self.edge_index is not None):
            _ = self.G_model(tf.convert_to_tensor(self.features, tf.float32), self.edge_index, training=False)
            _ = self.G_model_target(tf.convert_to_tensor(self.features, tf.float32), self.edge_index, training=False)
            self._built = True

    def _forward_all(self, training=False):
        if self.features is None:
            raise ValueError("features is None")
        if self.edge_index is None:
            raise ValueError("edge_index is None")
        self._maybe_build()
        return self.G_model(tf.convert_to_tensor(self.features, tf.float32), self.edge_index, training=training)

    # -------------------- Train / Infer entry --------------------

    def _set_dropout_by_step(self, step: int):
        # 较弱的 dropout，有助于早期快速下降
        rate = 0.03 if int(step) < int(self.warmup_steps) else 0.08
        try:
            self.G_model.gat1.drop.rate = rate
            self.G_model.gat2.drop.rate = rate
            self.G_model_target.gat1.drop.rate = rate
            self.G_model_target.gat2.drop.rate = rate
        except Exception:
            pass

    def use_GraphSAGE(self, channel_reward: np.ndarray, step: int,
                      idx: List[int], Graph_SAGE_label: bool):
        if self.features is None or self.edge_index is None:
            raise ValueError("Graph not initialized")
        x = tf.convert_to_tensor(self.features, tf.float32)
        labels_full = tf.convert_to_tensor(channel_reward, tf.float32)
        N = x.shape[0]

        do_train = Graph_SAGE_label and (step % self.gat_train_interval == 0) and (step > 0)
        if do_train:
            self._set_dropout_by_step(step)
            mask = np.zeros((N,), dtype=np.float32)
            mask[np.asarray(idx, dtype=np.int32)] = 1.0
            mask_tf = tf.convert_to_tensor(mask, tf.float32)

            preds, loss = self._tf_train_step(x, self.edge_index, labels_full, mask_tf)

            # EMA 目标网络
            if self.ema_tau > 0:
                tgt_w = self.G_model_target.get_weights()
                src_w = self.G_model.get_weights()
                new_w = [(1.0 - self.ema_tau) * tw + self.ema_tau * sw for tw, sw in zip(tgt_w, src_w)]
                self.G_model_target.set_weights(new_w)
            else:
                if step % 100 == 99:
                    self.G_model_target.set_weights(self.G_model.get_weights())

            if self.tb_writer is not None:
                with self.tb_writer.as_default():
                    tf.summary.scalar('GAT/loss', float(loss.numpy()), step=step)

            self.gat_loss_history.append((int(step), float(loss.numpy())))
            preds_all = preds
        else:
            preds_all = self._forward_all(training=False)

        return tf.gather(preds_all, tf.convert_to_tensor(idx, tf.int32)).numpy()

    # -------------------- IO --------------------

    def save_graph_network_weights(self, path='weight/GNN_weights.h5'):
        self._maybe_build()
        self.G_model.save_weights(path)

    def load_weights(self, path='weight/GNN_weights.h5'):
        if self.features is None or self.edge_index is None:
            self.features = np.zeros((1, 60), dtype=np.float32)
            self.graph = nx.Graph()
            self.graph.add_node(0)
            self.load_graph(self.graph, [0])
        _ = self._forward_all(False)
        self.G_model.load_weights(path)
        self.G_model_target.set_weights(self.G_model.get_weights())