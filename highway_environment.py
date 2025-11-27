# -*- coding: utf-8 -*-
"""
HighwayTopoEnv (dual 4-lane per direction) with V2I modes:
- Geometry: two directions, each with N lanes (default 4), vehicles placed at equal longitudinal spacing
- Motion: up (+Y) from bottom, down (-Y) from top; stop at opposite end (no wrap)
- Topology per direction: STAR or TREE; leader can be foremost, lane-pinned, and dynamic
- V2I modes:
  * single: single eNB/BS using Environment.V2Ichannels formula (BS_position configurable)
  * rsu: multi-RSU layout with nearest attachment and handover
- Visualization: fixed axes (Y in [0, height]), draw lanes, vehicles, neighbors, BS/RSUs, optional V2I dashed lines
"""

from typing import List, Optional, Tuple
import numpy as np
from Environment import Environ, Vehicle


class HighwayTopoEnv(Environ):
    def __init__(self,
                 n_up: int = 10,
                 n_down: int = 20,
                 lanes_per_dir: int = 4,
                 lane_width: float = 3.5,
                 median_gap_factor: float = 1.2,
                 spacing: float = 20.0,
                 base_y: float = 50.0,
                 width: float = 180.0,
                 height: float = 1400.0,
                 topology_type: str = 'star',      # 'star' or 'tree'
                 jitter_std: float = 0.0,
                 move_speed: float = 0.0,          # >0: unified step (m/step); else per-vehicle velocity * timestep
                 # Leader policy
                 leader_at_front: bool = True,
                 leader_lane_up: Optional[int] = None,
                 leader_lane_down: Optional[int] = None,
                 leader_dynamic: bool = False,
                 # V2I mode
                 v2i_mode: str = "single",         # 'single' | 'rsu'
                 bs_single_position: Optional[Tuple[float, float]] = None,  # only for 'single' mode
                 # RSU / V2I (for 'rsu' mode)
                 bs_layout: str = "median",        # 'median' | 'dual-roadside'
                 bs_spacing: float = 250.0,        # meters; <=0 disables RSU placement
                 bs_min_stay_steps: int = 5,       # min steps to stay before handover
                 bs_handover_hyst_m: float = 15.0, # meters of hysteresis to avoid ping-pong
                 seed: int = 123):
        np.random.seed(seed)
        assert topology_type in ('star', 'tree')
        assert v2i_mode in ('single', 'rsu')
        assert bs_layout in ('median', 'dual-roadside')
        self.n_up = int(n_up)
        self.n_down = int(n_down)
        self.lanes_per_dir = int(lanes_per_dir)
        self.lane_width = float(lane_width)
        self.median_gap = float(median_gap_factor * lane_width)
        self.spacing = float(spacing)
        self.base_y = float(base_y)
        self.topology_type = topology_type
        self.jitter_std = float(jitter_std)
        self.move_speed = float(move_speed)
        self.leader_at_front = bool(leader_at_front)
        self.leader_lane_up = leader_lane_up
        self.leader_lane_down = leader_lane_down
        self.leader_dynamic = bool(leader_dynamic)
        self.seed = seed
        self.width = float(width)
        self.height = float(height)
        # V2I mode/config
        self.v2i_mode = v2i_mode
        self.bs_single_position = bs_single_position  # may be None -> center
        # RSU config (used only in 'rsu' mode)
        self.bs_layout = bs_layout
        self.bs_spacing = float(bs_spacing)
        self.bs_min_stay_steps = int(bs_min_stay_steps)
        self.bs_handover_hyst_m = float(bs_handover_hyst_m)

        # Lane x positions: up lanes on the left of center, down lanes on the right
        center_x = width / 2.0
        up_lanes = [center_x - self.median_gap / 2.0 - (k + 0.5) * self.lane_width
                    for k in range(self.lanes_per_dir)]
        down_lanes = [center_x + self.median_gap / 2.0 + (k + 0.5) * self.lane_width
                      for k in range(self.lanes_per_dir)]
        # Unused lateral placeholders (kept for Environ compatibility)
        left_lanes = [0.0]
        right_lanes = [0.0]

        super().__init__(down_lanes, up_lanes, left_lanes, right_lanes, width, height)

        self.true_up_lanes = up_lanes
        self.true_down_lanes = down_lanes

        # Initial placement: up from bottom upwards; down from top downwards
        self._build_positions_dual_4lane()

        # Place BS/RSUs and initial V2I association
        self._place_base_stations()
        self._assoc_v2i(initial=True)

        # Pick leaders (by lane preference and "foremost" rule)
        self._pick_leaders_by_policy()

        # Build topology per direction (leader as hub/root)
        self._build_topology_two_clusters()

        # Initialize a session (channels, demands, etc.)
        self._init_session()

    # ---------------- Placement ----------------
    def _even_split(self, total: int, k: int) -> List[int]:
        q, r = divmod(total, k)
        return [q + (1 if i < r else 0) for i in range(k)]

    def _build_positions_dual_4lane(self):
        self.vehicles = []
        self._lane_idx = []  # per-vehicle lane index (0..lanes-1), used to select leader

        dist_up = self._even_split(self.n_up, self.lanes_per_dir)
        dist_down = self._even_split(self.n_down, self.lanes_per_dir)

        # Upward 'u': start at base_y, go up (+Y)
        for lane_idx, cnt in enumerate(dist_up):
            x = self.true_up_lanes[lane_idx]
            for t in range(cnt):
                y = self.base_y + t * self.spacing
                v = max(0.0, 22.0 + np.random.normal(0.0, 2.5))
                self.vehicles.append(Vehicle([x, y], 'u', velocity=v))
                self._lane_idx.append(lane_idx)

        # Downward 'd': start at height - base_y, go down (-Y)
        for lane_idx, cnt in enumerate(dist_down):
            x = self.true_down_lanes[lane_idx]
            for t in range(cnt):
                y = (self.height - self.base_y) - t * self.spacing
                v = max(0.0, 22.0 + np.random.normal(0.0, 2.5))
                self.vehicles.append(Vehicle([x, y], 'd', velocity=v))
                self._lane_idx.append(lane_idx)

        self.n_Veh = len(self.vehicles)
        self._group_dir = ['u'] * self.n_up + ['d'] * self.n_down

    # ---------------- BS/RSU & V2I ----------------
    def _place_base_stations(self):
        """Generate base-station positions for visualization and association.

        - single mode: one BS at bs_single_position (or center if None)
        - rsu mode: a series of RSUs along the road as before
        """
        self.bs_positions: List[Tuple[float, float]] = []
        if self.v2i_mode == "single":
            pos = self.bs_single_position or (self.width / 2.0, self.height / 2.0)
            self.bs_positions = [pos]
            return

        # rsu mode
        if self.bs_spacing <= 0:
            return
        center_x = self.width / 2.0
        x_left = min(self.true_up_lanes) - self.lane_width
        x_right = max(self.true_down_lanes) + self.lane_width

        # y grid with half-spacing offset
        y_list: List[float] = []
        y = self.base_y + 0.5 * self.bs_spacing
        while y <= self.height - self.base_y:
            y_list.append(y)
            y += self.bs_spacing

        if self.bs_layout == "median":
            self.bs_positions = [(center_x, yy) for yy in y_list]
        else:  # dual-roadside
            for i, yy in enumerate(y_list):
                self.bs_positions.append((x_left if (i % 2 == 0) else x_right, yy))

    def _assoc_v2i(self, initial: bool = False):
        """
        Associate each vehicle to a serving BS/RSU.
        - single: everyone attaches to index 0; v2i_dist_m = distance to that BS
        - rsu: nearest-RSU with hysteresis + min-stay to avoid ping-pong
        """
        if self.n_Veh == 0:
            self.v2i_serving_idx = np.zeros((0,), dtype=int)
            self.v2i_stay_steps = np.zeros((0,), dtype=int)
            self.v2i_dist_m = np.zeros((0,), dtype=float)
            return

        if self.v2i_mode == "single":
            if not self.bs_positions:
                self.bs_positions = [self.bs_single_position or (self.width / 2.0, self.height / 2.0)]
            bx, by = self.bs_positions[0]
            dists = []
            for v in self.vehicles:
                dists.append(float(np.hypot(v.position[0] - bx, v.position[1] - by)))
            self.v2i_serving_idx = np.zeros((self.n_Veh,), dtype=int)  # always 0
            if initial or not hasattr(self, "v2i_stay_steps"):
                self.v2i_stay_steps = np.zeros((self.n_Veh,), dtype=int)
            else:
                self.v2i_stay_steps = self.v2i_stay_steps + 1
            self.v2i_dist_m = np.asarray(dists, dtype=float)
            return

        # rsu mode
        n_bs = len(self.bs_positions)
        if n_bs == 0:
            self.v2i_serving_idx = np.full((self.n_Veh,), -1, dtype=int)
            self.v2i_stay_steps = np.zeros((self.n_Veh,), dtype=int)
            self.v2i_dist_m = np.zeros((self.n_Veh,), dtype=float)
            return

        bs_xy = np.array(self.bs_positions, dtype=float)  # (n_bs, 2)
        veh_xy = np.array([v.position for v in self.vehicles], dtype=float)  # (n_Veh, 2)

        # Squared distances to every RSU: (n_Veh, n_bs)
        dx = veh_xy[:, 0, None] - bs_xy[None, :, 0]
        dy = veh_xy[:, 1, None] - bs_xy[None, :, 1]
        d2 = dx * dx + dy * dy
        best_idx = np.argmin(d2, axis=1)
        best_dist = np.sqrt(d2[np.arange(self.n_Veh), best_idx])

        if initial or not hasattr(self, "v2i_serving_idx"):
            self.v2i_serving_idx = best_idx.copy()
            self.v2i_stay_steps = np.zeros((self.n_Veh,), dtype=int)
            self.v2i_dist_m = best_dist.copy()
            return

        # With hysteresis and minimum stay steps
        new_serving = self.v2i_serving_idx.copy()
        new_dist = self.v2i_dist_m.copy()
        for i in range(self.n_Veh):
            curr = int(self.v2i_serving_idx[i])
            curr_dist = float(np.sqrt(d2[i, curr])) if (0 <= curr < n_bs) else float("inf")
            cand = int(best_idx[i])
            cand_dist = float(best_dist[i])

            do_switch = False
            if cand != curr:
                # Switch only if candidate is closer by hysteresis and we've stayed long enough
                if (cand_dist + self.bs_handover_hyst_m < curr_dist) and (self.v2i_stay_steps[i] >= self.bs_min_stay_steps):
                    do_switch = True

            if do_switch or curr == -1:
                new_serving[i] = cand
                new_dist[i] = cand_dist
                self.v2i_stay_steps[i] = 0
            else:
                new_serving[i] = curr
                new_dist[i] = curr_dist
                self.v2i_stay_steps[i] += 1

        self.v2i_serving_idx = new_serving
        self.v2i_dist_m = new_dist

    # ---------------- Leader selection ----------------
    def _default_center_lanes(self):
        # Even lanes: up uses "middle-left", down uses "middle-right"
        if self.lanes_per_dir % 2 == 0:
            up_idx = self.lanes_per_dir // 2 - 1
            down_idx = self.lanes_per_dir // 2
        else:
            up_idx = down_idx = self.lanes_per_dir // 2
        return up_idx, down_idx

    def _pick_leaders_by_policy(self):
        def clamp_lane(i: int) -> int:
            return int(max(0, min(self.lanes_per_dir - 1, i)))

        up_center, down_center = self._default_center_lanes()
        up_lane = clamp_lane(self.leader_lane_up if self.leader_lane_up is not None else up_center)
        down_lane = clamp_lane(self.leader_lane_down if self.leader_lane_down is not None else down_center)

        up_idxs = [i for i, d in enumerate(self._group_dir) if d == 'u']
        down_idxs = [i for i, d in enumerate(self._group_dir) if d == 'd']

        def pick_from_lane(cands: List[int], lane_idx: int, direction: str) -> Optional[int]:
            lane_cands = [i for i in cands if self._lane_idx[i] == lane_idx]
            key = (lambda i: self.vehicles[i].position[1])
            if not cands:
                return None
            if direction == 'u':  # foremost = max y
                return max(lane_cands, key=key, default=max(cands, key=key))
            else:                 # 'd' foremost = min y
                return min(lane_cands, key=key, default=min(cands, key=key))

        self.leader_idx_up = pick_from_lane(up_idxs, up_lane, 'u') if up_idxs else None
        self.leader_idx_down = pick_from_lane(down_idxs, down_lane, 'd') if down_idxs else None

    # ---------------- Topology ----------------
    def _order_by_front(self, indices: List[int], direction: str) -> List[int]:
        if direction == 'u':
            return sorted(indices, key=lambda i: self.vehicles[i].position[1], reverse=True)
        else:
            return sorted(indices, key=lambda i: self.vehicles[i].position[1])

    def _build_topology_two_clusters(self):
        for v in self.vehicles:
            v.destinations = []
            v.neighbors = []
        self.depth = np.zeros(self.n_Veh, dtype=int)

        up_idxs = [i for i, d in enumerate(self._group_dir) if d == 'u']
        down_idxs = [i for i, d in enumerate(self._group_dir) if d == 'd']

        if up_idxs:
            self._build_cluster(up_idxs, direction='u', fixed_leader=self.leader_idx_up)
        if down_idxs:
            self._build_cluster(down_idxs, direction='d', fixed_leader=self.leader_idx_down)

        max_depth = int(max(1, self.depth.max()))
        self.struct_features_per_vehicle = []
        deg = [len(self.vehicles[i].neighbors) for i in range(self.n_Veh)]
        up_deg_max = max([deg[i] for i in up_idxs]) if up_idxs else 1
        down_deg_max = max([deg[i] for i in down_idxs]) if down_idxs else 1
        for i in range(self.n_Veh):
            depth_norm = float(self.depth[i]) / max_depth
            if i in up_idxs:
                is_hub = 1.0 if deg[i] >= up_deg_max else 0.0
            elif i in down_idxs:
                is_hub = 1.0 if deg[i] >= down_deg_max else 0.0
            else:
                is_hub = 0.0
            self.struct_features_per_vehicle.append((depth_norm, is_hub))

    def _build_cluster(self, indices: List[int], direction: str, fixed_leader: Optional[int] = None):
        order = self._order_by_front(indices, direction)
        if fixed_leader is not None and fixed_leader in order:
            order.remove(fixed_leader)
            order = [fixed_leader] + order  # force leader first
        elif self.leader_at_front:
            pass  # already foremost

        if self.topology_type == 'star':
            hub = order[0]
            leaves = order[1:4] if len(order) > 1 else [hub, hub, hub]
            while len(leaves) < 3:
                leaves.append(leaves[-1])
            self.vehicles[hub].destinations = leaves  # leader broadcasts to rear
            for i in order[1:]:
                self.vehicles[i].destinations = [hub, hub, hub]  # allow uplink to leader
                self.vehicles[i].neighbors.append(hub)
                self.vehicles[hub].neighbors.append(i)
                self.depth[i] = 1
            self.depth[hub] = 0
            self.vehicles[hub].neighbors = sorted(list(set(self.vehicles[hub].neighbors)))
        else:
            L = len(order)
            for local_i, gidx in enumerate(order):
                left_local = 2 * local_i + 1
                right_local = 2 * local_i + 2
                parent_local = (local_i - 1) // 2 if local_i > 0 else 0
                if local_i == 0:
                    dest = []
                    if left_local < L:
                        dest.append(order[left_local])
                    if right_local < L:
                        dest.append(order[right_local])
                    if left_local < L:
                        dest.append(order[left_local])
                    while len(dest) < 3 and dest:
                        dest.append(dest[-1])
                    if not dest:
                        dest = [gidx, gidx, gidx]
                else:
                    children = []
                    if left_local < L:
                        children.append(order[left_local])
                    if right_local < L:
                        children.append(order[right_local])
                    if children:
                        while len(children) < 2:
                            children.append(children[0])
                        dest = [children[0], children[1], order[parent_local]]
                    else:
                        dest = [order[parent_local]] * 3
                self.vehicles[gidx].destinations = dest
                if left_local < L:
                    self.vehicles[gidx].neighbors.append(order[left_local])
                    self.vehicles[order[left_local]].neighbors.append(gidx)
                if right_local < L:
                    self.vehicles[gidx].neighbors.append(order[right_local])
                    self.vehicles[order[right_local]].neighbors.append(gidx)
                d = 0
                p = local_i
                while p > 0:
                    p = (p - 1) // 2
                    d += 1
                self.depth[gidx] = d

    # ---------------- Session and evolution ----------------
    def _init_session(self):
        self.Distance = np.zeros((self.n_Veh, self.n_Veh))
        self.V2Vchannels = self.V2Vchannels.__class__(self.n_Veh, self.n_RB)
        self.V2Ichannels = self.V2Ichannels.__class__(self.n_Veh, self.n_RB)

        # IMPORTANT: set BS_position for single-BS mode BEFORE channel renewal,
        # so V2I pathloss uses the correct coordinates.
        if self.v2i_mode == "single":
            pos = self.bs_single_position or (self.width / 2.0, self.height / 2.0)
            # Environment.V2Ichannels expects a list [x, y]
            self.V2Ichannels.BS_position = [float(pos[0]), float(pos[1])]

        self.renew_channels_fastfading()
        self._update_distance()

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
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.n_step = 0

    def _update_distance(self):
        for i in range(self.n_Veh):
            xi, yi = self.vehicles[i].position
            for j in range(i, self.n_Veh):
                xj, yj = self.vehicles[j].position
                d = float(np.hypot(xi - xj, yi - yj))
                self.Distance[i, j] = self.Distance[j, i] = d

    def renew_positions(self):
        """
        Up: +Y, Down: -Y; stop at boundaries (no wrap).
        Step priority: move_speed (m/step) > vehicle.velocity * timestep.
        If leader_dynamic=True and the foremost vehicle changes, rebuild topology to set new leader.
        Also updates V2I association (handover rules apply for 'rsu'; single just refreshes distances).
        """
        moved = False
        for v in self.vehicles:
            dy = self.move_speed if self.move_speed > 0 else float(getattr(v, 'velocity', 0.0)) * float(self.timestep)
            if dy <= 0.0 and self.jitter_std <= 0:
                continue

            y_old = v.position[1]
            if v.direction == 'u':
                v.position[1] = min(y_old + dy, self.height - self.base_y)
            elif v.direction == 'd':
                v.position[1] = max(y_old - dy, self.base_y)
            else:
                v.position[1] = min(y_old + dy, self.height - self.base_y)

            if self.jitter_std > 0:
                v.position[1] += np.random.normal(0.0, self.jitter_std)
                v.position[1] = float(np.clip(v.position[1], self.base_y, self.height - self.base_y))

            if v.position[1] != y_old:
                moved = True

        if moved:
            self._update_distance()
            # Dynamic leader
            if self.leader_dynamic:
                old_up, old_dn = self.leader_idx_up, self.leader_idx_down
                self._pick_leaders_by_policy()
                if self.leader_idx_up != old_up or self.leader_idx_down != old_dn:
                    self._build_topology_two_clusters()
            # V2I association/handover or distance refresh
            self._assoc_v2i(initial=False)

    def new_random_game(self, n_Veh: int = 0):
        self._init_session()

    def renew_neighbor(self):
        return  # fixed topology

    # ---------------- Topology switching ----------------
    def set_topology(self, topology_type: str):
        if topology_type not in ('star', 'tree'):
            raise ValueError("topology_type must be 'star' or 'tree'")
        self.topology_type = topology_type
        self._build_topology_two_clusters()
        self._init_session()
        print(f"[HighwayTopoEnv] Topology switched to: {self.topology_type}")

    def toggle_topology(self):
        self.set_topology('tree' if self.topology_type == 'star' else 'star')

    # ---------------- Visualization (fixed axes) ----------------
    def visualize(self,
                  save_path: str = 'highway_topology.png',
                  show_destinations: bool = False,
                  annotate_power: bool = False,
                  agent=None,
                  figsize=(8, 10),
                  dpi: int = 220,
                  show_v2i: bool = False):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#fafafa')

        # Lanes
        for x in self.true_up_lanes + self.true_down_lanes:
            ax.plot([x, x], [0, self.height], color='#bdbdbd', linewidth=1.0, zorder=0)

        xs = [v.position[0] for v in self.vehicles]
        ys = [v.position[1] for v in self.vehicles]

        # BS/RSUs
        if getattr(self, "bs_positions", None):
            bx = [p[0] for p in self.bs_positions]
            by = [p[1] for p in self.bs_positions]
            ax.scatter(bx, by, c="#2e7d32", marker="s", s=80, edgecolors='black', linewidths=0.6, zorder=3)
            for i, (x, y) in enumerate(self.bs_positions):
                label = "BS" if (self.v2i_mode == "single") else f"B{i}"
                ax.text(x + 0.6, y, label, fontsize=7, va='center', color="#1b5e20", zorder=4)

        # Vehicles: up '^' red, down 'v' blue; leaders emphasized
        color_map = {'u': '#d32f2f', 'd': '#1976d2'}
        marker_map = {'u': '^', 'd': 'v'}
        s_normal, s_leader = 60, 90
        for i, v in enumerate(self.vehicles):
            is_leader = (i == getattr(self, 'leader_idx_up', -1)) or (i == getattr(self, 'leader_idx_down', -1))
            ax.scatter(v.position[0], v.position[1],
                       c=color_map.get(v.direction, 'black'),
                       marker=marker_map.get(v.direction, 'o'),
                       s=(s_leader if is_leader else s_normal),
                       edgecolors='black',
                       linewidths=(1.2 if is_leader else 0.6),
                       zorder=3)
            depth_arr = getattr(self, 'depth', np.zeros(self.n_Veh))
            label = f"{i}(d{int(depth_arr[i])})"
            if is_leader:
                label = "L-" + label
            ax.text(v.position[0] + 0.6, v.position[1], label, fontsize=7, va='center', zorder=4)

        # Neighbor edges
        lines_main = []
        for i, v in enumerate(self.vehicles):
            for nb in getattr(v, 'neighbors', []):
                if i < nb:
                    lines_main.append([(xs[i], ys[i]), (xs[nb], ys[nb])])
        if lines_main:
            lc = LineCollection(lines_main, colors='#888888', linewidths=1.0, zorder=1)
            ax.add_collection(lc)

        # Destinations dashed
        if show_destinations:
            lines_dest = []
            for i, v in enumerate(self.vehicles):
                for d in getattr(v, 'destinations', []):
                    if 0 <= d < self.n_Veh and d != i:
                        lines_dest.append([(xs[i], ys[i]), (xs[d], ys[d])])
            if lines_dest:
                lcd = LineCollection(lines_dest, colors='tab:green', linewidths=0.6,
                                     linestyles='dashed', alpha=0.45, zorder=2)
                ax.add_collection(lcd)

        # V2I dashed (vehicle -> serving BS/RSU)
        if show_v2i and getattr(self, "v2i_serving_idx", None) is not None and len(self.bs_positions) > 0:
            lines_v2i = []
            for i in range(self.n_Veh):
                bi = int(self.v2i_serving_idx[i])
                if 0 <= bi < len(self.bs_positions):
                    bx, by = self.bs_positions[bi]
                    lines_v2i.append([(xs[i], ys[i]), (bx, by)])
            if lines_v2i:
                lcv = LineCollection(lines_v2i, colors='#9e9e9e', linewidths=0.7,
                                     linestyles='dashed', alpha=0.6, zorder=1)
                ax.add_collection(lcv)

        # Optional (RB, Power) annotations
        if annotate_power and agent is not None:
            acts = agent.action_all_with_power_training
            for i in range(min(self.n_Veh, acts.shape[0])):
                lines = []
                for j in range(3):
                    rb = int(acts[i, j, 0])
                    pw = int(acts[i, j, 1])
                    if rb >= 0:
                        lines.append(f"{j}:RB{rb}/P{pw}")
                if lines:
                    ax.text(xs[i] - 2.5, ys[i], "\n".join(lines),
                            fontsize=6, color='black', alpha=0.75, ha='right', va='center',
                            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', lw=0.4))

        # Fixed axes (map does not move)
        x_min = min(self.true_up_lanes + self.true_down_lanes) - 8
        x_max = max(self.true_up_lanes + self.true_down_lanes) + 8
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, self.height)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        title = f'Dual 4-lane | {self.topology_type.upper()} per dir | N_up={self.n_up}, N_down={self.n_down}'
        if len(getattr(self, "bs_positions", [])) > 0:
            if self.v2i_mode == "single":
                title += ' | Single BS'
            else:
                title += f' | RSU:{self.bs_layout}, Δ={int(self.bs_spacing)}m, K={len(self.bs_positions)}'
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle='--')

        # Legend patches (optional)
        rsu_patch = mpatches.Patch(color='#2e7d32', label=('BS' if self.v2i_mode == "single" else 'RSU'), alpha=0.8)
        ax.legend(handles=[rsu_patch], loc='upper right', fontsize=8, frameon=True)

        import matplotlib.pyplot as plt2  # prevent shadowing
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close(fig)
        print(f"[HighwayTopoEnv] Visualization saved -> {save_path}")