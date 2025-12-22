from __future__ import division
import numpy as np
import time
import random
import math

# ==================== Channel Models ====================

class V2Vchannels:
    def __init__(self, n_Veh, n_RB):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros((len(self.positions), len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])

    def update_shadow(self, delta_distance_list):
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        if len(delta_distance_list) == 0:
            self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
        else:
            self.Shadow = (
                np.exp(-delta_distance / self.decorrelation_distance) * self.Shadow +
                np.sqrt(1 - np.exp(-2 * delta_distance / self.decorrelation_distance)) *
                np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
            )

    def update_fast_fading(self):
        h = (np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB)) +
             1j * np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB))) / np.sqrt(2)
        self.FastFading = 20 * np.log10(np.abs(h))

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(dv):
            if dv <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if dv < d_bp:
                    return 22.7 * np.log10(dv) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return (40.0 * np.log10(dv) + 9.45 - 17.3 * np.log10(self.h_bs)
                            - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5))

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
            self.ifLOS = False
            self.shadow_std = 4
        return PL


class V2Ichannels:
    def __init__(self, n_Veh, n_RB):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]
        self.shadow_std = 8
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1, d2)
            self.PathLoss[i] = 128.1 + 37.6 * np.log10(
                math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)

    def update_shadow(self, delta_distance_list):
        if len(delta_distance_list) == 0:
            self.Shadow = np.random.normal(0, self.shadow_std, self.n_Veh)
        else:
            delta_distance = np.asarray(delta_distance_list)
            self.Shadow = (
                np.exp(-delta_distance / self.Decorrelation_distance) * self.Shadow +
                np.sqrt(1 - np.exp(-2 * delta_distance / self.Decorrelation_distance)) *
                np.random.normal(0, self.shadow_std, self.n_Veh)
            )

    def update_fast_fading(self):
        h = (np.random.normal(size=(self.n_Veh, self.n_RB)) +
             1j * np.random.normal(size=(self.n_Veh, self.n_RB))) / np.sqrt(2)
        self.FastFading = 20 * np.log10(np.abs(h))


# ==================== Vehicle / Environment ====================

class Vehicle:
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height):
        self.timestep = 0.01
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height
        self.vehicles = []
        self.demands = []
        self.V2V_power_dB = 23
        self.V2I_power_dB = 23
        self.V2V_power_dB_List = [23, 10, 5]
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.n_RB = 20
        self.n_Veh = 20
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.n_step = 0
        self.Distance = np.zeros((self.n_Veh, self.n_Veh))

    # ---------- Vehicle generation ----------

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):
        for _ in range(n):
            ind = np.random.randint(0, len(self.down_lanes))
            self.add_new_vehicles([self.down_lanes[ind], random.randint(0, self.height)], 'd', random.randint(10, 15))
            self.add_new_vehicles([self.up_lanes[ind], random.randint(0, self.height)], 'u', random.randint(10, 15))
            self.add_new_vehicles([random.randint(0, self.width), self.left_lanes[ind]], 'l', random.randint(10, 15))
            self.add_new_vehicles([random.randint(0, self.width), self.right_lanes[ind]], 'r', random.randint(10, 15))
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity for c in self.vehicles])

    # ---------- Mobility ----------

    def renew_positions(self):
        i = 0
        while i < len(self.vehicles):
            delta_distance = self.vehicles[i].velocity * self.timestep
            change_direction = False
            v = self.vehicles[i]
            # (原始方向变更逻辑保持)
            if v.direction == 'u':
                for y in self.left_lanes:
                    if v.position[1] <= y <= v.position[1] + delta_distance:
                        if random.uniform(0, 1) < 0.4:
                            v.position = [v.position[0] - (delta_distance - (y - v.position[1])), y]
                            v.direction = 'l'; change_direction = True; break
                if not change_direction:
                    for y in self.right_lanes:
                        if v.position[1] <= y <= v.position[1] + delta_distance:
                            if random.uniform(0, 1) < 0.4:
                                v.position = [v.position[0] + (delta_distance + (y - v.position[1])), y]
                                v.direction = 'r'; change_direction = True; break
                if not change_direction:
                    v.position[1] += delta_distance
            elif v.direction == 'd':
                for y in self.left_lanes:
                    if v.position[1] >= y >= v.position[1] - delta_distance:
                        if random.uniform(0, 1) < 0.4:
                            v.position = [v.position[0] - (delta_distance - (v.position[1] - y)), y]
                            v.direction = 'l'; change_direction = True; break
                if not change_direction:
                    for y in self.right_lanes:
                        if v.position[1] >= y >= v.position[1] - delta_distance:
                            if random.uniform(0, 1) < 0.4:
                                v.position = [v.position[0] + (delta_distance + (v.position[1] - y)), y]
                                v.direction = 'r'; change_direction = True; break
                if not change_direction:
                    v.position[1] -= delta_distance
            elif v.direction == 'r':
                for x in self.up_lanes:
                    if v.position[0] <= x <= v.position[0] + delta_distance:
                        if random.uniform(0, 1) < 0.4:
                            v.position = [x, v.position[1] + (delta_distance - (x - v.position[0]))]
                            v.direction = 'u'; change_direction = True; break
                if not change_direction:
                    for x in self.down_lanes:
                        if v.position[0] <= x <= v.position[0] + delta_distance:
                            if random.uniform(0, 1) < 0.4:
                                v.position = [x, v.position[1] - (delta_distance - (x - v.position[0]))]
                                v.direction = 'd'; change_direction = True; break
                if not change_direction:
                    v.position[0] += delta_distance
            else:  # 'l'
                for x in self.up_lanes:
                    if v.position[0] >= x >= v.position[0] - delta_distance:
                        if random.uniform(0, 1) < 0.4:
                            v.position = [x, v.position[1] + (delta_distance - (v.position[0] - x))]
                            v.direction = 'u'; change_direction = True; break
                if not change_direction:
                    for x in self.down_lanes:
                        if v.position[0] >= x >= v.position[0] - delta_distance:
                            if random.uniform(0, 1) < 0.4:
                                v.position = [x, v.position[1] - (delta_distance - (v.position[0] - x))]
                                v.direction = 'd'; change_direction = True; break
                if not change_direction:
                    v.position[0] -= delta_distance

            # wrap-around strategy
            if (v.position[0] < 0) or (v.position[1] < 0) or (v.position[0] > self.width) or (v.position[1] > self.height):
                if v.direction == 'u':
                    v.direction = 'r'; v.position = [v.position[0], self.right_lanes[-1]]
                elif v.direction == 'd':
                    v.direction = 'l'; v.position = [v.position[0], self.left_lanes[0]]
                elif v.direction == 'l':
                    v.direction = 'u'; v.position = [self.up_lanes[0], v.position[1]]
                else:
                    v.direction = 'd'; v.position = [self.down_lanes[-1], v.position[1]]
            i += 1

    # ---------- Neighbor/Destinations (ROBUST) ----------

    def renew_neighbor(self):
        N = len(self.vehicles)
        for i in range(N):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
        if N == 0:
            self.Distance = np.zeros((0, 0))
            return
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T - z)
        self.Distance = Distance

        for i in range(N):
            sort_idx = np.argsort(Distance[:, i])
            others = sort_idx[1:]
            # nearest neighbors (up to 3)
            self.vehicles[i].neighbors = list(others[:min(3, len(others))])

            if len(others) == 0:
                self.vehicles[i].destinations = [i, i, i]
                continue
            pool_limit = max(3, min(len(others), int(np.ceil(N / 3))))
            candidate_pool = others[:pool_limit]
            if len(candidate_pool) >= 3:
                dest = np.random.choice(candidate_pool, 3, replace=False)
            else:
                dest = np.random.choice(others, 3, replace=True)
            self.vehicles[i].destinations = dest

    # ---------- Channels ----------

    def renew_channel(self):
        positions = [c.position for c in self.vehicles]
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = 0.002 * np.asarray([c.velocity for c in self.vehicles])
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)
        self.V2V_channels_abs = self.V2Vchannels.PathLoss + self.V2Vchannels.Shadow + 50 * np.identity(len(self.vehicles))
        self.V2I_channels_abs = self.V2Ichannels.PathLoss + self.V2Ichannels.Shadow

    def renew_channels_fastfading(self):
        self.renew_channel()
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        V2V_cf = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_cf - self.V2Vchannels.FastFading
        V2I_cf = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_cf - self.V2Ichannels.FastFading

    # ---------- Performance / Reward Helpers ----------

    def Compute_Performance_Reward_fast_fading_with_power_asyn(self, actions_power):
        actions = actions_power[:, :, 0]
        power_selection = actions_power[:, :, 1]
        Interference = np.zeros(self.n_RB)
        for i in range(len(self.vehicles)):
            for j in range(actions.shape[1]):
                if not self.activate_links[i, j]:
                    continue
                Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] -
                                                       self.V2I_channels_with_fastfading[i, actions[i, j]] +
                                                       self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = Interference + self.sig2
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        actions_mask = actions.copy()
        actions_mask[(np.logical_not(self.activate_links))] = -1
        for rb in range(self.n_RB):
            idxs = np.argwhere(actions_mask == rb)
            for a in range(len(idxs)):
                tx_i, link_j = idxs[a]
                rx_j = self.vehicles[tx_i].destinations[link_j]
                V2V_Signal[tx_i, link_j] = 10 ** ((self.V2V_power_dB_List[power_selection[tx_i, link_j]] -
                                                   self.V2V_channels_with_fastfading[tx_i, rx_j, rb] +
                                                   2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                if rb < self.n_Veh:
                    V2V_Interference[tx_i, link_j] += 10 ** ((self.V2I_power_dB -
                                                              self.V2V_channels_with_fastfading[rb, rx_j, rb] +
                                                              2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                for b in range(a + 1, len(idxs)):
                    tx_k, link_l = idxs[b]
                    rx_l = self.vehicles[tx_k].destinations[link_l]
                    V2V_Interference[tx_i, link_j] += 10 ** ((self.V2V_power_dB_List[power_selection[tx_k, link_l]] -
                                                              self.V2V_channels_with_fastfading[tx_k, rx_j, rb] +
                                                              2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[tx_k, link_l] += 10 ** ((self.V2V_power_dB_List[power_selection[tx_i, link_j]] -
                                                              self.V2V_channels_with_fastfading[tx_i, rx_l, rb] +
                                                              2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + V2V_Signal / self.V2V_Interference)
        V2I_Signals = (self.V2I_power_dB - self.V2I_channels_abs[0:min(self.n_RB, self.n_Veh)]
                       + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)
        V2I_Rate = np.log2(1 + 10 ** (V2I_Signals / 10) / self.V2I_Interference[0:min(self.n_RB, self.n_Veh)])

        # Latency & demand updates
        self.demand -= V2V_Rate * self.update_time_asyn * 1500
        self.test_time_count -= self.update_time_asyn
        self.individual_time_limit -= self.update_time_asyn
        self.individual_time_interval -= self.update_time_asyn
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(
            0.02, self.individual_time_interval[new_active].shape) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount
        early_finish = (self.demand <= 0) & self.activate_links
        unqualified = (self.individual_time_limit <= 0) & self.activate_links
        self.activate_links[early_finish | unqualified] = False
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqualified)
        fail_percent = self.failed_transmission / (self.failed_transmission + self.success_transmission + 1e-4)
        return V2I_Rate, fail_percent

    # ----------- Training reward (stabilized) -----------

    def Compute_Performance_Reward_Batch(self, actions_power, idx):
        # (保持结构但不返回所有列表；仅计算指定 idx 的所有 RB/功率组合收益)
        actions = actions_power[:, :, 0]
        power_selection = actions_power[:, :, 1]
        actions_copy = actions.copy()
        origin_rb = actions[idx[0], idx[1]]

        # Compute baseline sets for each RB/power quickly
        V2I_reward_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))
        V2V_reward_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))

        # Pre-calc other transmitters effect (粗略+近似，就沿用原推法)
        for rb in range(self.n_RB):
            for p_i, p_dB in enumerate(self.V2V_power_dB_List):
                # Proxy：把 rb/power 替换进单个链路其余固定
                rx = self.vehicles[idx[0]].destinations[idx[1]]
                signal = 10 ** ((p_dB - self.V2V_channels_with_fastfading[idx[0], rx, rb] +
                                  2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                interf_v2v = 0.0
                interf_v2v += 10 ** ((self.V2I_power_dB -
                                       self.V2V_channels_with_fastfading[rb % len(self.vehicles), rx, rb] +
                                       2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                v2v_rate = np.log2(1 + signal / (interf_v2v + self.sig2))
                v2i_signal = 10 ** ((self.V2I_power_dB + self.vehAntGain + self.bsAntGain -
                                      self.bsNoiseFigure - self.V2I_channels_abs[min(rb, self.n_Veh - 1)]) / 10)
                extra_i = 10 ** ((p_dB - self.V2I_channels_with_fastfading[idx[0], rb] +
                                   self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                v2i_rate = np.log2(1 + v2i_signal / (self.sig2 + extra_i))
                V2V_reward_list[rb, p_i] = v2v_rate
                V2I_reward_list[rb, p_i] = v2i_rate

        v2i_scaled = np.tanh(V2I_reward_list / 50.0)
        v2v_scaled = np.tanh(V2V_reward_list / 50.0)
        lam = 0.1
        combo = lam * v2i_scaled + (1 - lam) * v2v_scaled
        if self.demand[idx[0], idx[1]] < 0:
            time_left = self.V2V_limit
        else:
            time_left = self.individual_time_limit[idx[0], idx[1]]
        penalty = (self.V2V_limit - time_left) / self.V2V_limit
        return combo, -penalty, time_left

    # ========== 新增：批量奖励，供 Agent 统一计算使用 ==========
    def batch_reward_all(self, actions_power):
        """
        批量奖励计算。统一对所有车辆/三条链路的 (RB, power) 决策计算 V2V/V2I 指标与即时奖励。
        actions_power: shape [nVeh, 3, 2] -> (RB_index, power_index)
        返回：
          - reward_matrix: [nVeh,3] 每条链路即时奖励（速率归一 + 时间惩罚 + 反集中化 + 功率-时间耦合）
          - v2i_rate_total: float 全部RB上的 V2I 速率总和
          - fail_percent: float 失败比例
        """
        # 位置/信道更新（降低频率以平衡性能）
        self.n_step += 1
        if self.n_step % 10 == 0:
            self.renew_positions()
            self.renew_channels_fastfading()

        actions = actions_power[:, :, 0]
        powers = actions_power[:, :, 1]

        # V2I 干扰
        Interference_RB = np.zeros(self.n_RB)
        for i in range(len(self.vehicles)):
            for j in range(3):
                rb = actions[i, j]
                if rb < 0 or rb >= self.n_RB:
                    continue
                p_idx = powers[i, j]
                Interference_RB[rb] += 10 ** ((self.V2V_power_dB_List[p_idx] -
                                               self.V2I_channels_with_fastfading[i, rb] +
                                               self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = Interference_RB + self.sig2

        # V2V 信号与互扰
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        V2V_Interf = np.zeros((len(self.vehicles), 3))
        for rb in range(self.n_RB):
            idxs = np.argwhere(actions == rb)
            for a in range(len(idxs)):
                tx_i, link_j = idxs[a]
                rx_j = self.vehicles[tx_i].destinations[link_j]
                p_idx = powers[tx_i, link_j]
                sig = 10 ** ((self.V2V_power_dB_List[p_idx] -
                              self.V2V_channels_with_fastfading[tx_i, rx_j, rb] +
                              2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                V2V_Signal[tx_i, link_j] = sig
                for b in range(a + 1, len(idxs)):
                    tx_k, link_l = idxs[b]
                    rx_l = self.vehicles[tx_k].destinations[link_l]
                    p_idx2 = powers[tx_k, link_l]
                    interf_ik = 10 ** ((self.V2V_power_dB_List[p_idx2] -
                                        self.V2V_channels_with_fastfading[tx_k, rx_j, rb] +
                                        2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    interf_ki = 10 ** ((self.V2V_power_dB_List[p_idx] -
                                        self.V2V_channels_with_fastfading[tx_i, rx_l, rb] +
                                        2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interf[tx_i, link_j] += interf_ik
                    V2V_Interf[tx_k, link_l] += interf_ki

        V2V_Interf = V2V_Interf + self.sig2
        V2V_Rate = np.log2(1 + V2V_Signal / np.maximum(V2V_Interf, 1e-12))

        # V2I 总速率
        V2I_Signals = (self.V2I_power_dB - self.V2I_channels_abs[0:min(self.n_RB, self.n_Veh)]
                       + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)
        V2I_Rate = np.log2(1 + 10 ** (V2I_Signals / 10) / self.V2I_Interference[0:min(self.n_RB, self.n_Veh)])
        v2i_rate_total = float(np.sum(V2I_Rate))

        # 更新需求与时限
        self.demand -= V2V_Rate * self.update_time_asyn * 1500
        self.test_time_count -= self.update_time_asyn
        self.individual_time_limit -= self.update_time_asyn
        early_finish = (self.demand <= 0)
        unqualified = (self.individual_time_limit <= 0) & (self.demand > 0)
        self.success_transmission += int(np.sum(early_finish))
        self.failed_transmission += int(np.sum(unqualified))
        fail_percent = self.failed_transmission / (self.failed_transmission + self.success_transmission + 1e-6)

        # 即时奖励：速率归一 + 时间惩罚（基础奖励）
        time_left_norm = np.clip(self.individual_time_limit / self.V2V_limit, 0, 1)
        rate_norm = np.tanh(V2V_Rate / 40.0)
        base_reward_matrix = 0.7 * rate_norm + 0.3 * (1 - time_left_norm)

        # ====== 新增：反集中化 + 功率-时间耦合 ======
        # 将 environment_reward_patch.py 放置与 Environment.py 同目录或在 PYTHONPATH 下
        from environment_reward_patch import apply_reward_adjustments
        reward_matrix = apply_reward_adjustments(
            base_reward_matrix=base_reward_matrix,
            actions_rb=actions,
            actions_pw=powers,
            individual_time_limit=self.individual_time_limit,
            V2V_limit=self.V2V_limit,
            rb_anti_conc_alpha=getattr(self, "rb_anti_conc_alpha", 0.02),
            rb_hot_threshold=getattr(self, "rb_hot_threshold", 0.18),
            rb_softmask_alpha=getattr(self, "rb_softmask_alpha", 0.25),
            urgency_threshold=getattr(self, "urgency_threshold", 0.30),
            beta_urgency_pos=getattr(self, "beta_urgency_pos", 0.02),
            beta_urgency_neg=getattr(self, "beta_urgency_neg", 0.03),
        )

        # 重置完成/失败的链路
        reset_mask = early_finish | unqualified
        self.demand[reset_mask] = self.demand_amount
        self.individual_time_limit[reset_mask] = self.V2V_limit

        return reward_matrix, v2i_rate_total, float(fail_percent)

    def act_for_training(self, actions, idx):
        # actions: [nVeh,3,2]
        rb = actions[idx[0], idx[1], 0]
        pw_idx = actions[idx[0], idx[1], 1]
        reward_table, penalty, _ = self.Compute_Performance_Reward_Batch(actions, idx)
        base = reward_table[rb, pw_idx]
        reward = base + penalty  # small penalty
        # 演化环境
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return float(reward)

    def Compute_Interference(self, actions):
        V2V_Interference = np.zeros((len(self.vehicles), 3, self.n_RB)) + self.sig2
        channels = actions[:, :, 0].copy()
        powers = actions[:, :, 1].copy()
        for i in range(len(self.vehicles)):
            for j in range(3):
                rb = channels[i, j]
                if rb < 0 or rb >= self.n_RB:
                    continue
                p_sel = self.V2V_power_dB_List[powers[i, j]]
                rx = self.vehicles[i].destinations[j]
                for k in range(len(self.vehicles)):
                    for m in range(3):
                        if k == i and m == j:
                            continue
                        rb2 = channels[k, m]
                        if rb2 == rb:
                            rx2 = self.vehicles[k].destinations[m]
                            interf = 10 ** ((p_sel - self.V2V_channels_with_fastfading[i, rx2, rb] +
                                              2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                            V2V_Interference[k, m, rb] += interf
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)

    def act_asyn(self, actions):
        self.n_step += 1
        if self.n_step % 10 == 0:
            self.renew_positions()
            self.renew_channels_fastfading()
        v2i, fail = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        self.Compute_Interference(actions)
        return v2i, fail

    def act(self, actions):
        self.n_step += 1
        v2i, fail = self.Compute_Performance_Reward_fast_fading_with_power(actions)
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return v2i, fail

    def Compute_Performance_Reward_fast_fading_with_power(self, actions_power):
        # 可保留原实现或复用 asyn；此处简化调用 asyn 逻辑
        return self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions_power)

    def new_random_game(self, n_Veh=0):
        self.n_step = 0
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.Distance = np.zeros((self.n_Veh, self.n_Veh))
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self.renew_channels_fastfading()
        self.renew_neighbor()
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.demand_amount = 30
        self.demand = self.demand_amount * np.ones((self.n_Veh, 3))
        self.test_time_count = 10
        self.V2V_limit = 0.1
        self.individual_time_limit = self.V2V_limit * np.ones((self.n_Veh, 3))
        self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh, 3))
        self.UnsuccessfulLink = np.zeros((self.n_Veh, 3))
        self.success_transmission = 0
        self.failed_transmission = 0
        self.update_time_train = 0.01
        self.update_time_test = 0.002
        self.update_time_asyn = 0.0002
        self.activate_links = np.zeros((self.n_Veh, 3), dtype='bool')