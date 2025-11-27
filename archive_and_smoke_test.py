# -*- coding: utf-8 -*-
"""
Agent with switchable GNN backend (GAT v1/v2 or GraphSAGE) + full metric export
- Skip GNN training on test steps to avoid distribution shift during evaluation
- One GNN train per interval step, then cache embeddings for decisions
- Compatible fallback when _forward_all is not available
"""
from __future__ import print_function, division
import os
import random
import json
import time
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
                 warmup_steps: int = 4000,
                 epsilon_min: float = 0.05,
                 epsilon_decay_steps: int = 30000,
                 speed_mode: bool = False,
                 gat_train_interval: int = None,
                 attn_version: str = "v2",
                 plot_dpi: int = 224):
        self.weight_dir = 'weight'
        os.makedirs(self.weight_dir, exist_ok=True)
        self.env = environment

        self.gnn_type = (gnn_type or "gat").lower()
        # GNN 实例
        self.G = build_gnn(
            environment,
            gnn_type=self.gnn_type,
            distance_threshold=150.0,
            lr=5e-4,
            gat_train_interval=gat_train_interval if gat_train_interval else 20,
            grad_clip=5.0,
            attn_version=attn_version,
        )

        # DQN
        self.dqn = DQNModel(input_dim=102, output_dim=60,
                            learning_rate=0.01, decay_steps=500000,
                            decay_rate=0.96, min_lr=0.0005,
                            grad_clip_norm=5.0)

        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir)

        self.max_step = 100000
        self.RB_number = 20

        # 动态车辆数与动作缓存
        self.num_vehicle = getattr(self.env, 'n_Veh', len(getattr(self.env, 'vehicles', [])) or 20)
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2], dtype='int32')
        self.action_all_with_power_training = np.zeros([self.num_vehicle, 3, 2], dtype='int32')

        self.discount = 0.5
        self.double_q = True
        self.training = True
        self.GraphSAGE = True

        self.channel_reward = np.zeros((60, 20), dtype=np.float32)
        self.channel_reward_save = np.zeros((60, 20), dtype=np.float32)
        self.neighbor_nodes = []

        self.train_every_n_steps = 50
        self.target_q_update_step = 100

        self.warmup_steps = warmup_steps
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps

        if speed_mode:
            self.warmup_steps = min(self.warmup_steps, 1000)
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

    # 确保动作缓存尺寸匹配环境
    def _ensure_action_buffers(self):
        n = getattr(self.env, 'n_Veh', len(getattr(self.env, 'vehicles', [])))
        if n and n != self.num_vehicle:
            self.num_vehicle = n
            self.action_all_with_power = np.zeros([n, 3, 2], dtype='int32')
            self.action_all_with_power_training = np.zeros([n, 3, 2], dtype='int32')

    # Epsilon
    def _epsilon(self, step: int) -> float:
        if step < self.warmup_steps:
            return 1.0
        decay_progress = (step - self.warmup_steps) / max(1, self.epsilon_decay_steps)
        return max(self.epsilon_min, 1.0 - decay_progress)

    # 动作/状态
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

    def predict(self, s_t, step: int, test_ep=False):
        ep = 0.0 if test_ep else self._epsilon(step)
        if (not test_ep) and (random.random() < ep):
            return int(np.random.randint(60))
        q = self.dqn.forward(s_t)
        return int(tf.argmax(q, axis=1).numpy()[0])

    def observe(self, prestate, poststate, reward, action, step):
        self.memory.add(prestate, poststate, reward, action)
        if step >= self.warmup_steps and step % self.train_every_n_steps == 0:
            self.q_learning_mini_batch()

    # 初始图 + 嵌入
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

        raw = state_old[:, 0:20] + state_old[:, 20:40] - state_old[:, 40:60]
        scale = float(np.max(np.abs(raw))) + 1e-6
        self.channel_reward = raw / scale

        self.G.load_graph(graph, order_nodes)
        node_embeddings = self.G.use_GraphSAGE(self.channel_reward, step, idx_list, Graph_SAGE_label)
        emb_scale = float(np.max(np.abs(node_embeddings))) + 1e-4
        node_embeddings = node_embeddings / emb_scale
        return np.concatenate((node_embeddings, state_old), axis=1)

    # Warm-up
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
                    idx = 3 * i + j
                    st = self.get_state([i, j])
                    self.G.features[idx, :] = st[:60]
                    emb = self.G.use_GraphSAGE(self.channel_reward, 0, [idx], False).squeeze()
                    emb = emb / (np.max(np.abs(emb)) + 1e-4)
                    full_old = np.concatenate((emb, st), axis=0)
                    action = np.random.randint(60)
                    self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                    self.action_all_with_power_training[i, j, 1] = action // self.RB_number
                    reward = self.env.act_for_training(self.action_all_with_power_training, [i, j])
                    st_new = self.get_state([i, j])
                    self.G.features[idx, :] = st_new[:60]
                    emb_new = self.G.use_GraphSAGE(self.channel_reward, 0, [idx], False).squeeze()
                    emb_new = emb_new / (np.max(np.abs(emb_new)) + 1e-4)
                    full_new = np.concatenate((emb_new, st_new), axis=0)
                    self.memory.add(full_old, full_new, reward, action)
                    if self.memory.size() >= self.warmup_steps:
                        break
                if self.memory.size() >= self.warmup_steps:
                    break
        print("[WarmUp] done.")

    # 训练循环（测试步跳过 GNN 训练 + 嵌入缓存，且兼容无 _forward_all 的实现）
    def train(self, max_steps=50000, test_every_steps=2000, test_sample=200):
        self.dqn.update_target_network()
        self.warmup()

        self.env.new_random_game()
        _ = self.initial_better_state(0, True)

        for self.step in range(0, max_steps + 1):
            is_test_step = (test_every_steps > 0 and self.step % test_every_steps == 0 and self.step > 0)

            if is_test_step:
                self.training = False
                mean_v2i, fail = self.test_environment(test_sample=test_sample, detailed=False)
                self.test_history.append((self.step, float(mean_v2i), float(fail)))
                print(f"[TEST] step={self.step} v2i={mean_v2i:.3f} fail={fail:.3f} eps={self._epsilon(self.step):.3f}")
                self.training = True

            # 1) 刷新全部节点特征
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    s_all = self.get_state([i, j])
                    self.G.features[3 * i + j, :] = s_all[:60]

            # 2) 到训练间隔则整步训练（测试步跳过）
            interval = getattr(self.G, "gat_train_interval", 50)
            if self.GraphSAGE and (self.step % interval == 0) and (self.step > 0) and (not is_test_step):
                idx_all = list(range(3 * len(self.env.vehicles)))
                _ = self.G.use_GraphSAGE(self.channel_reward, self.step, idx_all, True)

            # 3) 缓存全图嵌入（优先 _forward_all；否则退化为全量 use_GraphSAGE）
            N = 3 * len(self.env.vehicles)
            if hasattr(self.G, "_forward_all"):
                emb_all_t = self.G._forward_all(training=False)
                emb_all = emb_all_t.numpy() if hasattr(emb_all_t, "numpy") else np.asarray(emb_all_t)
            else:
                idx_all = list(range(N))
                emb_all = self.G.use_GraphSAGE(self.channel_reward, self.step, idx_all, False)
            emb_all = emb_all / (np.max(np.abs(emb_all)) + 1e-4)

            # 4) 逐节点决策（仅 DQN），写经验
            self.training = True
            for i in range(len(self.env.vehicles)):
                for j in range(3):
                    idx = [3 * i + j]
                    s_old_raw = self.get_state([i, j])
                    self.G.features[3 * i + j, :] = s_old_raw[:60]

                    emb_old = emb_all[3 * i + j]
                    better_old = np.concatenate((emb_old, s_old_raw), axis=0)

                    t0 = time.perf_counter()
                    action = self.predict(better_old, self.step)
                    t1 = time.perf_counter()
                    if self.step % 50 == 0:
                        with open(os.path.join(self.export_dir, f"decision_time_stream_{self.gnn_type}.csv"), "a") as f:
                            f.write(f"{self.step},{i},{j},{t1 - t0}\n")

                    time_left_s = float(s_old_raw[-1]) * float(self.env.V2V_limit)
                    self.power_log.append((time_left_s, action // self.RB_number))

                    self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                    self.action_all_with_power_training[i, j, 1] = action // self.RB_number

                    reward_train = self.env.act_for_training(self.action_all_with_power_training, [i, j])

                    s_new_raw = self.get_state([i, j])
                    self.G.features[3 * i + j, :] = s_new_raw[:60]
                    if hasattr(self.G, "_forward_all"):
                        emb_new_all_t = self.G._forward_all(training=False)
                        emb_new_all = emb_new_all_t.numpy() if hasattr(emb_new_all_t, "numpy") else np.asarray(emb_new_all_t)
                        emb_new = emb_new_all[3 * i + j]
                    else:
                        emb_new = self.G.use_GraphSAGE(self.channel_reward, self.step, idx, False).squeeze()
                    emb_new = emb_new / (np.max(np.abs(emb_new)) + 1e-4)
                    better_new = np.concatenate((emb_new, s_new_raw), axis=0)

                    self.observe(better_old, better_new, reward_train, action, self.step)

            if self.step % self.target_q_update_step == 0 and self.step > 0:
                self.dqn.update_target_network()
                print(f"[Target] updated at step {self.step}")

            used_blocks = np.unique(self.action_all_with_power_training[:, :, 0])
            self.used_blocks_history.append((self.step, len(used_blocks)))

            if self.tb_dqn is not None and self.step % 25 == 0:
                with self.tb_dqn.as_default():
                    tf.summary.scalar('Env/used_blocks', float(len(used_blocks)), step=self.step)
                    tf.summary.scalar('Env/epsilon', float(self._epsilon(self.step)), step=self.step)

        final_v2i, final_fail = self.test_environment(test_sample=test_sample, detailed=True)
        self.test_history.append((self.step, float(final_v2i), float(final_fail)))
        self._export_results(final_v2i, final_fail)
        print(f"[TRAIN DONE] steps={self.step} final_v2i={final_v2i:.4f} fail={final_fail:.4f}")

    # 测试（修复：返回 fail_percent 平均值）
    def test_environment(self, test_sample=200, detailed=False):
        V2I_Rate_list = []
        fail_list = []
        inst_v2i, inst_v2v, inst_nveh, inst_t = [], [], [], []
        t_idx = 0

        for k in range(test_sample):
            action_temp = self.action_all_with_power.copy()
            for i in range(len(self.env.vehicles)):
                self.action_all_with_power[i, :, 0] = -1
                sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                for j in sorted_idx:
                    idx_node = [3 * i + j]
                    st = self.get_state([i, j])
                    self.G.features[3 * i + j, :] = st[:60]
                    emb = self.G.use_GraphSAGE(self.channel_reward, 0, idx_node, False).squeeze()
                    emb = emb / (np.max(np.abs(emb)) + 1e-4)
                    better = np.concatenate((emb, st), axis=0)
                    action = self.predict(better, 0, True)
                    self.merge_action([i, j], action)

                if i % max(1, len(self.env.vehicles)//10) == 1:
                    action_temp = self.action_all_with_power.copy()
                    reward, fail = self.env.act_asyn(action_temp)
                    V2I_Rate_list.append(np.sum(reward))
                    fail_list.append(float(fail))
                    if detailed:
                        inst_v2i.append(float(np.sum(reward)))
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

    # DQN 更新
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
            target = rewards + self.discount * q_tp1
        else:
            q_next_target = self.dqn.forward_target(s_tp1)
            q_tp1 = tf.reduce_max(q_next_target, axis=1)
            target = rewards + self.discount * q_tp1

        loss, q_values = self.dqn.train_step(s_t, target, actions)
        loss_v = float(loss.numpy())
        q_mean = float(tf.reduce_mean(q_values).numpy())
        self.dqn_loss_history.append((self.step, loss_v))
        self.dqn_qmean_history.append((self.step, q_mean))

        if self.tb_dqn is not None and self.step % 25 == 0:
            with self.tb_dqn.as_default():
                tf.summary.scalar('DQN/loss', loss_v, step=self.step)
                tf.summary.scalar('DQN/q_mean', q_mean, step=self.step)

    # 导出
    def _export_results(self, final_mean_rate: float, final_fail_percent: float):
        os.makedirs(self.export_dir, exist_ok=True)
        tag = self.gnn_type

        # 1) DQN metrics
        if self.dqn_loss_history:
            steps = [s for s, _ in self.dqn_loss_history]
            losses = [l for _, l in self.dqn_loss_history]
            qmeans = [qm for (_, qm) in self.dqn_qmean_history]
            with open(os.path.join(self.export_dir, f'dqn_metrics_{tag}.csv'), 'w') as f:
                f.write('step,loss,q_mean\n')
                for s, l, qm in zip(steps, losses, qmeans):
                    f.write(f'{s},{l},{qm}\n')
            plt.figure(figsize=(7, 4))
            plt.plot(steps, losses, label=f'DQN loss ({tag})')
            plt.xlabel('step'); plt.ylabel('loss'); plt.title('DQN Loss over Steps')
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'dqn_loss_{tag}.png'), dpi=self.plot_dpi)
            plt.close()

        # 2) Used RB blocks
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

        # 3) GNN loss
        steps_gl, losses_gl = [], []
        if hasattr(self.G, 'gat_loss_history') and self.G.gat_loss_history:
            steps_gl = [s for s, _ in self.G.gat_loss_history]
            losses_gl = [l for _, l in self.G.gat_loss_history]
        elif hasattr(self.G, 'gs_losses') and self.G.gs_losses:
            steps_gl = list(range(1, len(self.G.gs_losses)+1))
            losses_gl = list(self.G.gs_losses)
        elif hasattr(self.G, 'loss') and isinstance(self.G.loss, list) and self.G.loss:
            steps_gl = list(range(1, len(self.G.loss)+1))
            losses_gl = list(self.G.loss)

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

        # 4) Train effect curves
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

        # 5) Power select vs time-left
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

        # 6) Detailed timeseries
        if self._last_test_detailed and len(self._last_test_detailed.get('t', [])) > 0:
            d = self._last_test_detailed
            with open(os.path.join(self.export_dir, f'timeseries_{tag}.csv'), 'w') as f:
                f.write('t,v2i_rate,v2v_success,num_vehicles\n')
                for a, b, c, e in zip(d['t'], d['v2i'], d['v2v_succ'], d['nveh']):
                    f.write(f'{a},{b},{c},{e}\n')
            fig, ax1 = plt.subplots(figsize=(7.8, 4.8))
            ln1 = ax1.plot(d['t'], d['v2v_succ'], color='tab:blue', label='Inst. V2V Success')
            ax1.set_xlabel('Time (s)'); ax1.set_ylabel('V2V Success', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax1.twinx()
            ln2 = ax2.plot(d['t'], d['v2i'], color='tab:orange', alpha=0.6, label='Inst. V2I Rate')
            ax2.set_ylabel('V2I Rate', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.1))
            ln3 = ax3.plot(d['t'], d['nveh'], color='tab:green', alpha=0.6, label='Number of Vehicles')
            ax3.set_ylabel('Number of Vehicles', color='tab:green')
            lines = ln1 + ln2 + ln3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')
            ax1.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(self.export_dir, f'timeseries_{tag}.png'), dpi=self.plot_dpi)
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


if __name__ == "__main__":
    # 冒烟测试（可选）
    Env = Environ([100.0], [80.0], [0.0], [0.0], 180.0, 600.0)
    Env.new_random_game(12)
    agent = Agent([], Env, gnn_type="gat", warmup_steps=200, epsilon_decay_steps=2000)
    agent.train(max_steps=60, test_every_steps=20, test_sample=10)