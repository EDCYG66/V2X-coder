# -*- coding: utf-8 -*-
"""
Agent（平滑版 + 合并补丁 + 紧急度功率奖励 + 邻居RB冲突目标混合
      + RB去集中化惩罚 + 动作阶段RB热度软掩码）

新增内容概览：
- RB去集中化惩罚（anti-concentration）：根据当前步的RB分配集中度（如Gini或热点比例）
  对 channel_reward 做小幅指数衰减，抑制“超级热点RB”，提升均匀性与后期V2V稳定。
- 动作阶段RB热度软掩码（soft mask）：在预测动作时，对近期“高热度”的RB对应Q值施加温度惩罚，
  软性引导策略避开拥挤RB，而不是强制禁止。

保持内容：
- 目标网络软更新、整图嵌入缓存、邻居RB冲突惩罚、功率成本惩罚、紧急度功率奖励、
  邻居RB冲突图混入GAT训练目标。
"""

from __future__ import print_function, division
import os
import random
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

from Environment import *
from base import BaseModel
from dqn_model import DQNModel
from replay_memory import ReplayMemory
from gnn_factory import build_gnn

__all__ = ["Agent"]

class Agent(BaseModel):
    def __init__(self,
                 config,
                 environment,
                 gnn_type: str = "gat",
                 warmup_steps: int = 1000,
                 epsilon_min: float = 0.05,
                 epsilon_decay_steps: int = 30000,
                 speed_mode: bool = False,
                 gat_train_interval: int = None,
                 plot_dpi: int = 224,
                 replay_size: int = 200000,
                 power_log_stride: int = 4,
                 power_log_max: int = 200000,
                 # --- 补丁参数（已有） ---
                 soft_update_tau: float = 0.005,
                 power_cost_weight: float = 0.01,
                 conflict_cost_weight: float = 0.02,
                 skip_embedding_steps: int = 9,
                 batch_decay_step: int = None,
                 batch_decay_factor: float = 0.5,
                 # --- 紧急度功率奖励与冲突图混合（已有，可调） ---
                 beta_urgency_pos: float = 0.02,
                 beta_urgency_neg: float = 0.02,
                 urgency_threshold: float = 0.25,
                 conflict_penalty_weight: float = 0.02,
                 conflict_window_steps: int = 50,
                 # --- 新增：RB去集中化与软掩码 ---
                 rb_anti_conc_alpha: float = 0.01,     # RB去集中化惩罚强度（指数系数）
                 rb_hot_threshold: float = 0.20,        # 软掩码阈值：最近窗口占用比例>阈值视为热点
                 rb_softmask_alpha: float = 0.15,       # 软掩码温度：越大惩罚越重
                 rb_softmask_window: int = 50):         # 软掩码滚动窗口步数
        self.weight_dir = 'weight'
        os.makedirs(self.weight_dir, exist_ok=True)
        self.env = environment

        self.gnn_type = (gnn_type or "gat").lower()
        self.G = build_gnn(
            environment,
            gnn_type=self.gnn_type,
            distance_threshold=150.0,
            lr=5e-4,
            gat_train_interval=gat_train_interval if gat_train_interval else 20,
            grad_clip=5.0,
        )

        self.dqn = DQNModel(input_dim=114, output_dim=60,
                            learning_rate=0.001, decay_steps=500000,
                            decay_rate=0.96, min_lr=0.0005,
                            grad_clip_norm=5.0)

        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir, memory_size=int(replay_size), state_dim=114, batch_size=512)

        self.max_step = 100000
        self.RB_number = 20

        self.num_vehicle = getattr(self.env, 'n_Veh', len(getattr(self.env, 'vehicles', [])) or 20)
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2], dtype='int32')
        self.action_all_with_power_training = np.zeros([self.num_vehicle, 3, 2], dtype='int32')

        self.discount = 0.9
        self.double_q = True
        self.training = True
        self.GraphSAGE = True

        self.channel_reward = np.zeros((3 * self.num_vehicle, self.RB_number), dtype=np.float32)
        self.neighbor_nodes = []

        self.train_every_n_steps = 25
        self.target_q_update_step = 200

        self.warmup_steps = warmup_steps
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps

        if speed_mode:
            self.warmup_steps = min(self.warmup_steps, 800)
            self.train_every_n_steps = 10
            self.memory.batch_size = 256

        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.logdir = os.path.join('runs', f'v2v_{self.gnn_type}_{ts}')
        os.makedirs(self.logdir, exist_ok=True)
        self.tb_dqn = tf.summary.create_file_writer(os.path.join(self.logdir, 'dqn'))
        self.tb_gnn = tf.summary.create_file_writer(os.path.join(self.logdir, 'gnn'))
        self.export_dir = os.path.join(self.logdir, 'exports')
        os.makedirs(self.export_dir, exist_ok=True)
        if hasattr(self.G, "tb_writer"):
            self.G.tb_writer = self.tb_gnn

        self.dqn_loss_history = []
        self.dqn_qmean_history = []
        self.used_blocks_history = []
        self.test_history = []
        self.power_log = []
        self._last_test_detailed = None
        self.POWER_DB = {0: 23, 1: 10, 2: 5}
        self.plot_dpi = int(plot_dpi)

        self.power_log_stride = max(1, int(power_log_stride)) if int(power_log_stride) > 0 else 0
        self.power_log_max = max(1000, int(power_log_max))

        # 补丁参数存储
        self.soft_update_tau = float(soft_update_tau)
        self.power_cost_weight = float(power_cost_weight)
        self.conflict_cost_weight = float(conflict_cost_weight)
        self.skip_embedding_steps = int(skip_embedding_steps)
        self.batch_decay_step = batch_decay_step
        self.batch_decay_factor = float(batch_decay_factor)
        self._cached_emb_step = -1

        # 紧急度/冲突图
        self.beta_urgency_pos = float(beta_urgency_pos)
        self.beta_urgency_neg = float(beta_urgency_neg)
        self.urgency_threshold = float(urgency_threshold)
        self.conflict_penalty_weight = float(conflict_penalty_weight)
        self.conflict_window_steps = int(conflict_window_steps)
        self._rb_neighbor_hits = np.zeros(self.RB_number, dtype=np.int32)
        self._rb_neighbor_hist_buffer = []

        # 新增：RB去集中化与软掩码
        self.rb_anti_conc_alpha = float(rb_anti_conc_alpha)
        self.rb_hot_threshold = float(rb_hot_threshold)
        self.rb_softmask_alpha = float(rb_softmask_alpha)
        self.rb_softmask_window = int(rb_softmask_window)
        self._rb_softmask_hist = []  # 滚动记录每步各RB被选择次数

    # ----------------- 现有方法 -----------------

    def _ensure_action_buffers(self):
        n = getattr(self.env, 'n_Veh', len(getattr(self.env, 'vehicles', [])))
        if n and n != self.num_vehicle:
            self.num_vehicle = n
            self.action_all_with_power = np.zeros([n, 3, 2], dtype='int32')
            self.action_all_with_power_training = np.zeros([n, 3, 2], dtype='int32')

    def _epsilon(self, step: int) -> float:
        if step < self.warmup_steps:
            return 1.0
        decay_progress = (step - self.warmup_steps) / max(1, self.epsilon_decay_steps)
        return max(self.epsilon_min, 1.0 - decay_progress)

    def merge_action(self, idx, action: int):
        a = int(action)
        self.action_all_with_power[idx[0], idx[1], 0] = a % self.RB_number
        self.action_all_with_power[idx[0], idx[1], 1] = a // self.RB_number

    def get_state(self, idx):
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0],
                            self.env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
        V2V_interference = (-self.env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
        NeiSelection = np.zeros(self.RB_number, dtype=np.float32)
        if len(self.neighbor_nodes) > 3 * idx[0] + idx[1]:
            neighs_list = self.neighbor_nodes[3 * idx[0] + idx[1]][0]
            for neigh_idx in neighs_list:
                rb = self.action_all_with_power_training[self.G.link[neigh_idx, 0],
                                                         self.G.link[neigh_idx, 1] % 3, 0]
                if 0 <= rb < self.RB_number:
                    NeiSelection[rb] = 1
        time_remaining = np.asarray(
            [self.env.demand[idx[0], idx[1]] / self.env.demand_amount], dtype=np.float32)
        load_remaining = np.asarray(
            [self.env.individual_time_limit[idx[0], idx[1]] / self.env.V2V_limit], dtype=np.float32)
        return np.concatenate((V2I_channel, V2V_interference, V2V_channel,
                               NeiSelection, time_remaining, load_remaining))

    # ----------------- 动作阶段RB热度软掩码（新增） -----------------

    def _softmask_q_values(self, q_vals: np.ndarray) -> np.ndarray:
        """
        对Q值应用RB热度软掩码：
        - 统计最近 rb_softmask_window 步各RB平均占用比例（来自 _rb_softmask_hist）
        - 对占用比例 > rb_hot_threshold 的 RB 降低对应动作Q值（温度惩罚）
        动作空间为 60（20 RB * 3 Power），RB映射: a % RB_number，Power映射: a // RB_number
        """
        if len(self._rb_softmask_hist) == 0:
            return q_vals
        hist = np.stack(self._rb_softmask_hist, axis=0)  # [T, RB]
        avg_hits = hist.mean(axis=0)                     # [RB]
        total = avg_hits.sum()
        if total <= 0:
            return q_vals
        rb_ratio = avg_hits / (total + 1e-8)             # [RB] 归一化占比
        hot_mask = (rb_ratio > self.rb_hot_threshold)    # True=热点
        # 构造每个动作的温度因子
        action_temp = np.ones_like(q_vals, dtype=np.float32)
        for a in range(q_vals.shape[-1]):
            rb = a % self.RB_number
            if hot_mask[rb]:
                # 温度惩罚：降低Q，鼓励探索冷RB
                action_temp[a] = (1.0 - self.rb_softmask_alpha)
        return q_vals * action_temp

    def predict(self, s_t, step: int, test_ep=False):
        ep = 0.0 if test_ep else self._epsilon(step)
        if (not test_ep) and (random.random() < ep):
            return int(np.random.randint(60))
        q = self.dqn.forward(s_t)
        q_np = q.numpy().reshape(-1) if hasattr(q, "numpy") else np.asarray(q).reshape(-1)
        # 应用软掩码
        q_masked = self._softmask_q_values(q_np)
        return int(np.argmax(q_masked))

    def observe(self, prestate, poststate, reward, action, step):
        self.memory.add(prestate, poststate, reward, action)
        if step >= self.warmup_steps and step % self.train_every_n_steps == 0:
            self.q_learning_mini_batch()

    def _update_channel_reward(self):
        raw = self.G.features[:, 0:20] + self.G.features[:, 20:40] - self.G.features[:, 40:60]
        scale = float(np.max(np.abs(raw))) + 1e-6
        self.channel_reward = raw / scale

    def initial_better_state(self, step, Graph_SAGE_label=True):
        self._ensure_action_buffers()
        self.G.num_V2V_list = np.zeros((len(self.env.vehicles), len(self.env.vehicles)))
        self.neighbor_nodes = []
        state_old = np.zeros((3 * len(self.env.vehicles), 82), dtype=np.float32)
        idx_list = []

        for i in range(len(self.env.vehicles)):
            for j in range(3):
                self.G.num_V2V_list[i, self.env.vehicles[i].destinations[j]] = 1

        graph, order_nodes, _ = self.G.build_graph(self.G.num_V2V_list)
        self.G.features = np.zeros((3 * len(self.env.vehicles), 60), dtype=np.float32)

        for i in range(len(self.env.vehicles)):
            for j in range(3):
                neigh_idx_holder = []
                node_label = order_nodes[3 * i + j]
                neighs = list(nx.neighbors(graph, node_label))
                neigh_idx_holder.append([order_nodes.index(neigh) for neigh in neighs])
                self.neighbor_nodes.append(neigh_idx_holder)

        for i in range(len(self.env.vehicles)):
            for j in range(3):
                idx = 3 * i + j
                s = self.get_state([i, j])
                state_old[idx] = s
                self.G.features[idx, :] = s[:60]
                idx_list.append(idx)

        self._update_channel_reward()
        self.G.load_graph(graph, order_nodes)
        node_embeddings = self.G.use_GraphSAGE(self.channel_reward, step, idx_list, Graph_SAGE_label)
        emb_scale = float(np.max(np.abs(node_embeddings))) + 1e-4
        node_embeddings = node_embeddings / emb_scale
        return np.concatenate((node_embeddings, state_old), axis=1)

    def warmup(self):
        if self.memory.size() >= self.warmup_steps:
            return
        self._ensure_action_buffers()
        print(f"[WarmUp] collecting {self.warmup_steps} transitions ...")
        self.env.new_random_game()
        self.initial_better_state(0, True)

        while self.memory.size() < self.warmup_steps:
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    s_old = self.get_state([i, j])
                    self.G.features[3*i + j, :] = s_old[:60]
                    self._update_channel_reward()
                    emb = self.G.use_GraphSAGE(self.channel_reward, 0, [3*i+j], False).squeeze()
                    emb = emb / (np.max(np.abs(emb)) + 1e-4)
                    full_old = np.concatenate((emb, s_old), axis=0)
                    action = np.random.randint(60)
                    self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                    self.action_all_with_power_training[i, j, 1] = action // self.RB_number
            reward_matrix, _, _ = self.env.batch_reward_all(self.action_all_with_power_training)
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    s_new = self.get_state([i, j])
                    self.G.features[3*i + j, :] = s_new[:60]
                    emb_new = self.G.use_GraphSAGE(self.channel_reward, 0, [3*i+j], False).squeeze()
                    emb_new = emb_new / (np.max(np.abs(emb_new)) + 1e-4)
                    full_new = np.concatenate((emb_new, s_new), axis=0)
                    act_int = int(self.action_all_with_power_training[i, j, 0] + self.RB_number * self.action_all_with_power_training[i, j, 1])
                    self.memory.add(full_old, full_new, float(reward_matrix[i, j]), act_int)
                    if self.memory.size() >= self.warmup_steps:
                        break
                if self.memory.size() >= self.warmup_steps:
                    break
        print("[WarmUp] done.")

    # ----------------- 紧急度奖励/冲突图/惩罚/缓存/衰减 -----------------

    def _extract_time_left_ratio(self, i: int, j: int) -> float:
        s = self.get_state([i, j])
        time_left_ratio = float(s[-2])  # 最后两维：[time_remaining_ratio, load_remaining_ratio]
        return max(0.0, min(1.0, time_left_ratio))

    def apply_urgency_power_shaping(self):
        actions = self.action_all_with_power_training
        pw = actions[:, :, 1]  # 0:23dB, 1:10dB, 2:5dB
        score = 0.0
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                urgent = (self._extract_time_left_ratio(i, j) < self.urgency_threshold)
                if urgent:
                    if pw[i, j] == 0:
                        score += self.beta_urgency_pos
                    elif pw[i, j] == 2:
                        score -= self.beta_urgency_neg
        shaping_factor = np.exp(score / (3.0 * len(self.env.vehicles) + 1e-6))
        self.channel_reward *= shaping_factor

    def compute_neighbor_rb_conflict_map(self) -> np.ndarray:
        hits = np.zeros(self.RB_number, dtype=np.int32)
        if hasattr(self, "neighbor_nodes") and self.neighbor_nodes:
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    link_index = 3 * i + j
                    neigh_info = self.neighbor_nodes[link_index][0] if link_index < len(self.neighbor_nodes) else []
                    for nid in neigh_info:
                        v2 = self.G.link[nid, 0]
                        l2 = self.G.link[nid, 1] % 3
                        rb_choice = self.action_all_with_power_training[v2, l2, 0]
                        if 0 <= rb_choice < self.RB_number:
                            hits[rb_choice] += 1
        self._rb_neighbor_hist_buffer.append(hits)
        if len(self._rb_neighbor_hist_buffer) > self.conflict_window_steps:
            self._rb_neighbor_hist_buffer = self._rb_neighbor_hist_buffer[-self.conflict_window_steps:]
        agg = np.sum(np.stack(self._rb_neighbor_hist_buffer, axis=0), axis=0) if self._rb_neighbor_hist_buffer else hits
        self._rb_neighbor_hits = agg.astype(np.int32)
        m = float(np.max(self._rb_neighbor_hits)) if np.max(self._rb_neighbor_hits) > 0 else 1.0
        return (self._rb_neighbor_hits.astype(np.float32) / m)

    def soft_update_target(self):
        for tw, ow in zip(self.dqn.target_model.weights, self.dqn.model.weights):
            tw.assign((1.0 - self.soft_update_tau) * tw + self.soft_update_tau * ow)

    def compute_penalty(self) -> float:
        actions = self.action_all_with_power_training
        rb = actions[:, :, 0]
        pw = actions[:, :, 1]
        penalty_total = 0.0
        penalty_total += self.power_cost_weight * float(np.sum(pw))
        if hasattr(self, "neighbor_nodes") and self.neighbor_nodes:
            for veh_id in range(len(self.env.vehicles)):
                for link_j in range(3):
                    link_index = 3 * veh_id + link_j
                    rb_ij = rb[veh_id, link_j]
                    neigh_info = self.neighbor_nodes[link_index][0] if link_index < len(self.neighbor_nodes) else []
                    conflict = 0
                    for nid in neigh_info:
                        v2 = self.G.link[nid, 0]
                        l2 = self.G.link[nid, 1] % 3
                        if rb[v2, l2] == rb_ij:
                            conflict += 1
                    if conflict > 0:
                        penalty_total += self.conflict_cost_weight * conflict
        return float(penalty_total)

    def forward_embeddings(self, force=False):
        need_update = force or (getattr(self.env, "n_step", 0) % (self.skip_embedding_steps + 1) == 0) or (self._cached_emb_step < 0)
        if need_update and hasattr(self.G, "_forward_all"):
            emb_all_t = self.G._forward_all(training=False)
            emb_all = emb_all_t.numpy() if hasattr(emb_all_t, "numpy") else np.asarray(emb_all_t)
            self._cached_emb_step = getattr(self.env, "n_step", 0)
        else:
            emb_all = self.G._cache_emb if hasattr(self.G, "_cache_emb") else None
        if emb_all is None:
            N = 3 * len(self.env.vehicles)
            idx_all = list(range(N))
            emb_all = self.G.use_GraphSAGE(self.channel_reward, getattr(self, "step", 0), idx_all, False)
            self._cached_emb_step = getattr(self.env, "n_step", 0)
        emb_all = emb_all / (np.max(np.abs(emb_all)) + 1e-4)
        return np.asarray(emb_all)

    def maybe_decay_batch_size(self):
        if self.batch_decay_step is not None and getattr(self, "step", 0) >= self.batch_decay_step:
            self.memory.batch_size = int(max(64, int(self.memory.batch_size * self.batch_decay_factor)))

    # ----------------- 新增：RB去集中化惩罚（每步） -----------------

    def _rb_anti_concentration_penalty(self):
        """
        计算当前步的 RB 集中度，并对 channel_reward 做指数衰减：
        使用 max_rb_count / total_links 作为热点比例（比 Gini 更轻量）。
        """
        rb = self.action_all_with_power_training[:, :, 0].reshape(-1)  # [num_links]
        total_links = rb.shape[0]
        counts = np.bincount(rb, minlength=self.RB_number).astype(np.float32)
        max_rb = float(np.max(counts))
        hot_ratio = max_rb / (float(total_links) + 1e-6)
        # 指数衰减，抑制热点
        factor = np.exp(-self.rb_anti_conc_alpha * hot_ratio)
        self.channel_reward *= factor
        # 记录到软掩码滚动窗口
        self._rb_softmask_hist.append(counts)
        if len(self._rb_softmask_hist) > self.rb_softmask_window:
            self._rb_softmask_hist = self._rb_softmask_hist[-self.rb_softmask_window:]

    # ----------------- 每步训练主体 -----------------

    def train_loop_step(self, gnn_train_interval=20, base_batch_size=512):
        # 低频 GNN 训练 + 冲突图混合
        if self.GraphSAGE and (self.step % gnn_train_interval == 0) and self.training:
            idx_all = list(range(3 * len(self.env.vehicles)))
            try:
                conflict_map = self.compute_neighbor_rb_conflict_map()
                if hasattr(self.G, "set_conflict_map"):
                    self.G.set_conflict_map(conflict_map, weight=self.conflict_penalty_weight)
            except Exception:
                pass
            _ = self.G.use_GraphSAGE(self.channel_reward, self.step, idx_all, True)

        emb_all = self.forward_embeddings(force=(self.step == 1))

        # 动作选择（带软掩码）
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                s_old = self.get_state([i, j])
                emb_old = emb_all[3 * i + j]
                better_old = np.concatenate((emb_old, s_old), axis=0)
                action = self.predict(better_old, self.step)
                self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                self.action_all_with_power_training[i, j, 1] = action // self.RB_number
                if self.power_log_stride and (self.step % self.power_log_stride == 0):
                    time_left_s = float(s_old[-1]) * float(self.env.V2V_limit)
                    self.power_log.append((time_left_s, action // self.RB_number))
                    if len(self.power_log) > self.power_log_max:
                        self.power_log = self.power_log[-self.power_log_max:]

        # 批量奖励
        reward_matrix, v2i_rate_total, fail_percent = self.env.batch_reward_all(self.action_all_with_power_training)

        # 紧急度 shaping
        self.apply_urgency_power_shaping()

        # RB去集中化惩罚（新增）
        self._rb_anti_concentration_penalty()

        # 写 replay + DQN 更新
        self.maybe_decay_batch_size()
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                s_new = self.get_state([i, j])
                emb_new = emb_all[3*i + j]
                better_old = np.concatenate((emb_new, s_new), axis=0)
                better_new = better_old
                act_int = int(self.action_all_with_power_training[i, j, 0] + self.RB_number * self.action_all_with_power_training[i, j, 1])
                self.observe(better_old, better_new, float(reward_matrix[i, j]), act_int, self.step)

        # 软更新
        if self.step >= self.warmup_steps and self.step % self.train_every_n_steps == 0:
            self.soft_update_target()

        # 记录使用的 RB 种类
        used_blocks = np.unique(self.action_all_with_power_training[:, :, 0])
        self.used_blocks_history.append((self.step, len(used_blocks)))

        # 额外惩罚融合（已有）
        penalty_val = self.compute_penalty()
        self.channel_reward *= np.exp(-0.001 * penalty_val)

        if self.tb_dqn is not None and self.step % 25 == 0:
            with self.tb_dqn.as_default():
                tf.summary.scalar('Env/used_blocks', float(len(used_blocks)), step=self.step)
                tf.summary.scalar('Env/epsilon', float(self._epsilon(self.step)), step=self.step)
                tf.summary.scalar('Patch/penalty', float(penalty_val), step=self.step)

    # ----------------- 训练主循环 -----------------

    def train(self, max_steps=50000, test_every_steps=2000, test_sample=200):
        self.dqn.update_target_network()
        self.warmup()

        self.env.new_random_game()
        _ = self.initial_better_state(0, True)

        for self.step in range(1, max_steps + 1):
            is_test_step = (test_every_steps > 0 and self.step % test_every_steps == 0)

            if is_test_step:
                self.training = False
                mean_v2i, fail = self.test_environment(test_sample=test_sample, detailed=False)
                self.test_history.append((self.step, float(mean_v2i), float(fail)))
                print(f"[TEST] step={self.step} v2i={mean_v2i:.3f} fail={fail:.3f} eps={self._epsilon(self.step):.3f}")
                self.training = True

            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    s_all = self.get_state([i, j])
                    self.G.features[3 * i + j, :] = s_all[:60]
            self._update_channel_reward()

            self.train_loop_step(gnn_train_interval=getattr(self.G, "gat_train_interval", 20), base_batch_size=self.memory.batch_size)

            if self.step % self.target_q_update_step == 0:
                self.dqn.update_target_network()
                print(f"[Target] updated at step {self.step}")

        final_v2i, final_fail = self.test_environment(test_sample=test_sample, detailed=True)
        self.test_history.append((self.step, float(final_v2i), float(final_fail)))
        self._export_results(final_v2i, final_fail)
        print(f"[TRAIN DONE] steps={self.step} final_v2i={final_v2i:.4f} fail={final_fail:.4f}")

    # ----------------- 测试/更新/导出 -----------------

    def test_environment(self, test_sample=200, detailed=False):
        V2I_Rate_list = []
        fail_list = []
        inst_v2i, inst_v2v, inst_nveh, inst_t = [], [], [], []
        t_idx = 0

        for _ in range(test_sample):
            action_temp = self.action_all_with_power_training.copy()
            reward_vec, fail = self.env.act_asyn(action_temp)
            V2I_Rate_list.append(np.sum(reward_vec))
            fail_list.append(float(fail))
            if detailed:
                inst_v2i.append(float(np.sum(reward_vec)))
                inst_v2v.append(float(1.0 - fail))
                inst_nveh.append(int(len(self.env.vehicles)))
                inst_t.append(t_idx)
                t_idx += 1

        mean_v2i = float(np.mean(V2I_Rate_list)) if V2I_Rate_list else 0.0
        fail_percent = float(np.mean(fail_list)) if fail_list else 0.0
        if detailed:
            self._last_test_detailed = dict(
                t=inst_t,
                v2i=inst_v2i,
                v2v_succ=inst_v2v,
                nveh=inst_nveh
            )
        return mean_v2i, fail_percent

    def q_learning_mini_batch(self):
        if self.memory.size() < self.memory.batch_size:
            return
        s_t, s_tp1, actions, rewards = self.memory.sample()
        s_t = tf.convert_to_tensor(s_t, tf.float32)
        s_tp1 = tf.convert_to_tensor(s_tp1, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)

        if self.double_q:
            q_next_online = self.dqn.forward(s_tp1)
            next_actions = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
            q_next_target = self.dqn.forward_target(s_tp1)
            idxs = tf.stack([tf.range(tf.shape(next_actions)[0], dtype=tf.int32), next_actions], axis=1)
            q_tp1 = tf.gather_nd(q_next_target, idxs)
        else:
            q_next_target = self.dqn.forward_target(s_tp1)
            q_tp1 = tf.reduce_max(q_next_target, axis=1)

        q_tp1 = tf.cast(q_tp1, rewards.dtype)
        target = rewards + tf.cast(self.discount, rewards.dtype) * q_tp1

        loss, q_values = self.dqn.train_step(s_t, target, actions)
        loss_v = float(loss.numpy())
        q_mean = float(tf.reduce_mean(tf.cast(q_values, tf.float32)).numpy())
        self.dqn_loss_history.append((self.step, loss_v))
        self.dqn_qmean_history.append((self.step, q_mean))

        if self.tb_dqn is not None and self.step % 25 == 0:
            with self.tb_dqn.as_default():
                tf.summary.scalar('DQN/loss', loss_v, step=self.step)
                tf.summary.scalar('DQN/q_mean', q_mean, step=self.step)

    def _export_results(self, final_mean_rate: float, final_fail_percent: float):
        os.makedirs(self.export_dir, exist_ok=True)
        tag = self.gnn_type

        if self.dqn_loss_history:
            steps = [s for s, _ in self.dqn_loss_history]
            losses = [l for _, l in self.dqn_loss_history]
            qmeans = [qm for (_, qm) in self.dqn_qmean_history]
            with open(os.path.join(self.export_dir, f'dqn_metrics_{tag}.csv'), 'w') as f:
                f.write('step,loss,q_mean\n')
                for s, l, qm in zip(steps, losses, qmeans):
                    f.write(f'{s},{l},{qm}\n')
            plt.figure(figsize=(7, 4))
            try:
                import pandas as pd
                df = pd.DataFrame({'step': steps, 'loss': losses})
                smooth = df['loss'].rolling(5, min_periods=1).mean()
                plt.plot(steps, smooth, label=f'DQN loss ({tag})')
            except Exception:
                plt.plot(steps, losses, label=f'DQN loss ({tag})')
            plt.xlabel('step'); plt.ylabel('loss'); plt.title('DQN Loss over Steps')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'dqn_loss_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        if self.used_blocks_history:
            with open(os.path.join(self.export_dir, f'env_metrics_{tag}.csv'), 'w') as f:
                f.write('step,used_blocks\n')
                for s, ub in self.used_blocks_history:
                    f.write(f'{s},{ub}\n')
            steps = [s for s, _ in self.used_blocks_history]
            ubs = [ub for _, ub in self.used_blocks_history]
            plt.figure(figsize=(7, 4))
            plt.plot(steps, ubs, label=f'Used RB Blocks ({tag})')
            plt.xlabel('step'); plt.ylabel('num_used_blocks')
            plt.title('Used RB Blocks over Steps')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'used_blocks_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        steps_gl, losses_gl = [], []
        if hasattr(self.G, 'gat_loss_history') and self.G.gat_loss_history:
            steps_gl = [s for s, _ in self.G.gat_loss_history]
            losses_gl = [l for _, l in self.G.gat_loss_history]

        if losses_gl:
            with open(os.path.join(self.export_dir, f'gnn_loss_{tag}.csv'), 'w') as f:
                f.write('step,loss\n')
                for s, l in zip(steps_gl, losses_gl):
                    f.write(f'{s},{l}\n')
            plt.figure(figsize=(7, 4))
            plt.plot(steps_gl, losses_gl, label=f'{self.gnn_type.upper()} loss')
            plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{self.gnn_type.upper()} Training Loss over Steps')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'gnn_loss_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        if self.test_history:
            th = self.test_history
            with open(os.path.join(self.export_dir, f'test_history_{tag}.csv'), 'w') as f:
                f.write('step,v2i_mean,fail_percent,v2v_success\n')
                for s, v2i, fail in th:
                    f.write(f'{s},{v2i},{fail},{1.0 - fail}\n')
            steps = [s for s, _, _ in th]
            v2i_vals = [v for _, v, _ in th]
            v2v_succ = [1.0 - f for _, _, f in th]
            fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
            ln1 = ax1.plot(steps, v2v_succ, '-o', label='V2V Success Rate', color='tab:blue')
            ax1.set_xlabel('step'); ax1.set_ylabel('V2V Success Rate', color='tab:blue')
            ax2 = ax1.twinx()
            ln2 = ax2.plot(steps, v2i_vals, '-s', label='V2I Rate', color='tab:orange')
            ax2.set_ylabel('V2I Rate', color='tab:orange')
            lines = ln1 + ln2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')
            ax1.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'training_effect_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        if self.power_log:
            arr = np.array(self.power_log, dtype=np.float32)
            times = arr[:, 0]; pidx = arr[:, 1].astype(int)
            limit = self.env.V2V_limit if hasattr(self.env, 'V2V_limit') else max(0.1, times.max())
            times = np.clip(times, 0, limit)
            nbins = 12
            edges = np.linspace(0, limit, nbins + 1)
            mids = 0.5 * (edges[:-1] + edges[1:])
            probs = np.zeros((nbins, 3), dtype=np.float32)
            for b in range(nbins):
                mask = (times >= edges[b]) & (times < edges[b + 1])
                cnt = mask.sum()
                if cnt > 0:
                    for k in [0, 1, 2]:
                        probs[b, k] = np.sum(mask & (pidx == k)) / cnt
            with open(os.path.join(self.export_dir, f'power_select_{tag}.csv'), 'w') as f:
                f.write('time_left,prob_p0,prob_p1,prob_p2\n')
                for m, (p0, p1, p2) in zip(mids, probs):
                    f.write(f'{m},{p0},{p1},{p2}\n')
            plt.figure(figsize=(7.2, 4.6))
            plt.plot(mids, probs[:, 0], '-o', label=f'Power {self.POWER_DB.get(0,"0")} dB')
            plt.plot(mids, probs[:, 1], '-s', label=f'Power {self.POWER_DB.get(1,"1")} dB')
            plt.plot(mids, probs[:, 2], '-^', label=f'Power {self.POWER_DB.get(2,"2")} dB')
            plt.xlabel('Time left for V2V transmission (s)')
            plt.ylabel('Probability of power selection')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'power_select_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        summary = {
            'model': self.gnn_type,
            'final_mean_v2i_rate': float(final_mean_rate),
            'final_fail_percent': float(final_fail_percent),
            'steps_trained': int(getattr(self, 'step', -1)),
            'epsilon_final': float(self._epsilon(getattr(self, 'step', 0))),
            'files': sorted(os.listdir(self.export_dir))
        }
        with open(os.path.join(self.export_dir, f'summary_{tag}.json'), 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Export] Files saved -> {self.export_dir} (tag={tag})")