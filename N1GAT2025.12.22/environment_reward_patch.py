import numpy as np

def compute_rb_gini(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return 0.0
    s = counts.sum()
    if s <= 0:
        return 0.0
    diffs = np.abs(counts[:, None] - counts[None, :])
    return float(diffs.sum() / (2 * counts.size * s))

def apply_reward_adjustments(
    base_reward_matrix: np.ndarray,
    actions_rb: np.ndarray,
    actions_pw: np.ndarray,
    individual_time_limit: np.ndarray,
    V2V_limit: float,
    rb_anti_conc_alpha: float = 0.02,
    rb_hot_threshold: float = 0.18,
    rb_softmask_alpha: float = 0.25,
    urgency_threshold: float = 0.30,
    beta_urgency_pos: float = 0.02,
    beta_urgency_neg: float = 0.03,
):
    n_rb = int(np.max(actions_rb) + 1) if actions_rb.size > 0 else 0
    n_rb = max(n_rb, 20)  # fallback

    # 1) RB 反集中化惩罚
    rb_counts = np.zeros(n_rb, dtype=np.int32)
    for i in range(actions_rb.shape[0]):
        for j in range(actions_rb.shape[1]):
            rb = int(actions_rb[i, j])
            if 0 <= rb < n_rb:
                rb_counts[rb] += 1
    rb_gini = compute_rb_gini(rb_counts)
    penalty_gini = rb_anti_conc_alpha * rb_gini

    total_links = max(1, int(np.sum(actions_rb >= 0)))
    frac = rb_counts.astype(np.float32) / float(total_links)
    is_hot = (frac > rb_hot_threshold)
    hot_rb_set = set(np.where(is_hot)[0].tolist())

    # 2) 功率-剩余时间耦合
    time_left_norm = np.clip(individual_time_limit / float(V2V_limit), 0.0, 1.0)
    urgent_mask = (time_left_norm <= urgency_threshold)
    high_power_mask = (actions_pw == 0)  # 0=23dB

    reward = base_reward_matrix.copy()

    # 全局 gini 惩罚
    if penalty_gini > 0:
        reward -= penalty_gini

    # 每链路的 hot-RB 惩罚与功率-时间耦合
    for i in range(actions_rb.shape[0]):
        for j in range(actions_rb.shape[1]):
            rb = int(actions_rb[i, j])
            if rb < 0 or rb >= n_rb:
                continue
            if rb in hot_rb_set:
                reward[i, j] -= rb_softmask_alpha
            if urgent_mask[i, j]:
                if high_power_mask[i, j]:
                    reward[i, j] += beta_urgency_pos
                else:
                    reward[i, j] -= 0.5 * beta_urgency_pos
            else:
                if high_power_mask[i, j]:
                    reward[i, j] -= beta_urgency_neg
                else:
                    reward[i, j] += 0.5 * beta_urgency_neg

    return reward