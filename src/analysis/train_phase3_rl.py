"""
Phase 3 RL Training (FQI, unified config)

- 使用 CondorEnvAge3 (统一环境公式)
- 训练环境:  config_env_optimistic.yaml  （探索友好的乐观环境）
- 评估环境:  config_env_calibrated.yaml （真实校准环境）

输出:
- 训练好的 Q 函数 (RandomForestRegressor)
- 离散动作网格 action_grid
- 以字典形式保存: {"Q": Q, "action_grid": action_grid}
"""

import os
import sys
import yaml
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.env.condor_env_age3 import CondorEnvAge3


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_from_cfg(cfg, seed_override=None):
    env_cfg = cfg.get("env", {})
    rew_cfg = cfg.get("reward", {})
    merged = {**env_cfg, **rew_cfg}

    if seed_override is not None:
        merged["seed"] = seed_override
    else:
        sim_cfg = cfg.get("sim", {})
        if "seed" in sim_cfg:
            merged["seed"] = sim_cfg["seed"]

    return CondorEnvAge3(**merged)


def build_action_grid(actions_cfg):
    """根据 actions 段落构建离散动作网格"""
    rel_min = actions_cfg["release_min"]
    rel_max = actions_cfg["release_max"]
    rel_step = actions_cfg["release_step"]

    mit_min = actions_cfg["mitigation_min"]
    mit_max = actions_cfg["mitigation_max"]
    mit_step = actions_cfg["mitigation_step"]

    releases = np.arange(rel_min, rel_max + 1e-8, rel_step, dtype=float)
    # mitigation 可能是 0.1 步长，要注意浮点误差
    # Mitigation might be in steps of 0.1; be aware of floating-point errors.
    n_steps = int(round((mit_max - mit_min) / mit_step)) + 1
    mitigations = mit_min + mit_step * np.arange(n_steps, dtype=float)

    grid = np.array(
        [[r, m] for r in releases for m in mitigations],
        dtype=float,
    )
    return grid


# ------------------------------------------------------------------
# 数据采集 & FQI Data Acquisition & FQI
# ------------------------------------------------------------------
def collect_dataset(env, action_grid, n_episodes, horizon, rng):
    """
    用“行为策略”采集经验数据:
    这里使用简单的随机策略，在 action_grid 上均匀采样。
    返回: (states, actions, rewards, next_states)

    注意：这里对 reward 做了缩放 (reward shaping)，
    只影响 RL，不改变环境本身和 Phase-2 结果。
    Collecting empirical data using a "behavioral strategy":

    A simple random strategy is used here, sampling uniformly across the `action_grid`.
    Returns: (states, actions, rewards, next_states)
    Note: Reward shaping is applied here,
    This only affects RL and does not change the environment itself or the Phase-2 results.
    """

    # -------- reward 缩放系数（方案 B）--------
    reward_scale = 10000000.0

    states = []
    actions = []
    rewards = []
    next_states = []

    n_actions = len(action_grid)

    for ep in range(n_episodes):
        state = env.reset()
        for t in range(horizon):
            # 随机策略 random policy: uniform over action_grid
            a_idx = rng.randint(n_actions)
            action = action_grid[a_idx]

            next_state, reward_raw, done, info = env.step(action)

            # === RL reward shaping: 缩小尺度，避免 FQI 爆炸 ===
            # RL reward shaping: scale down to prevent FQI explosion
            reward = reward_raw / reward_scale

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

            state = next_state
            if done:
                break

    return (
        np.array(states, dtype=float),
        np.array(actions, dtype=float),
        np.array(rewards, dtype=float),
        np.array(next_states, dtype=float),
    )


def build_q_regressor(random_state):
    """构建 RandomForestRegressor，用于 FQI"""
    """Builds a RandomForestRegressor for FQI"""
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return rf


def fitted_q_iteration(
    states,
    actions,
    rewards,
    next_states,
    action_grid,
    gamma,
    n_iterations,
    random_state=0,
):
    """
    标准 FQI: standard Fitted Q Iteration (FQI)
    - feature: [state, action] 向量拼接 , concatenated [state, action] vectors
    - Q^0(s,a) = E[r | s,a]
    - Q^{k+1}(s,a) = E[r + gamma * max_{a'} Q^k(s', a') | s,a]
    """
    rng = np.random.RandomState(random_state)
    n_samples = states.shape[0]
    n_actions = action_grid.shape[0]
    state_dim = states.shape[1]
    action_dim = action_grid.shape[1]

    # 拼接 state-action 作为特征
    X = np.concatenate([states, actions], axis=1)

    Q = None

    for it in range(n_iterations):
        if it == 0 or Q is None:
            # 第一次迭代: 只拟合 immediate reward
            # first iteration: fit immediate reward only
            y = rewards.copy()
        else:
            # 计算 next_states 下的 V(s') = max_a' Q(s', a')
            # Compute V(s') = max_a' Q(s', a') for next_states
            sa_list = []
            for a in action_grid:
                a_tiled = np.repeat(a.reshape(1, action_dim), n_samples, axis=0)
                sa = np.concatenate([next_states, a_tiled], axis=1)
                sa_list.append(sa)
            sa_all = np.stack(sa_list, axis=0)  # [n_actions, n_samples, state+action]
            sa_all_flat = sa_all.reshape(-1, state_dim + action_dim)

            q_flat = Q.predict(sa_all_flat)
            q_all = q_flat.reshape(n_actions, n_samples)
            v_next = np.max(q_all, axis=0)

            y = rewards + gamma * v_next

        Q = build_q_regressor(random_state=rng.randint(10_000))
        Q.fit(X, y)

        y_mean = float(np.mean(y))
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        print(
            f"[FQI] iter={it+1}/{n_iterations} "
            f"target_mean={y_mean:.6f} min={y_min:.6f} max={y_max:.6f}"
        )

    return Q


# ------------------------------------------------------------------
# 策略 & 评估 Ploicy & Evaluation
# ------------------------------------------------------------------
def make_greedy_policy(Q, action_grid):
    """给定 Q 回归器与离散动作网格，构造贪婪策略 π(s)=argmax_a Q(s,a)"""
    """Given a Q regressor and a discrete action grid, construct a greedy policy π(s)=argmax_a Q(s,a)"""

    def policy_fn(state):
        n_actions = action_grid.shape[0]
        state_tiled = np.repeat(state.reshape(1, -1), n_actions, axis=0)
        sa = np.concatenate([state_tiled, action_grid], axis=1)
        q_vals = Q.predict(sa)
        best_idx = int(np.argmax(q_vals))
        return action_grid[best_idx]

    return policy_fn


def evaluate_policy(env_cfg, policy_fn, horizon, n_episodes, seed=0):
    """在给定环境配置上评估策略，返回平均回报和平均 N_total 轨迹

    注意：这里返回的是 *原始环境 reward* 的总和，没有做缩放，
    方便和 Phase-2 的尺度保持一致。
    Evaluate the policy on the given environment configuration, returning the average return and average N_total trajectory.    
    Please note: The total returned here is the *original environment reward* without scaling,
    to keep it consistent with the scale of Phase-2.
    """
    returns = []
    trajs = []

    for ep in range(n_episodes):
        env = make_env_from_cfg(env_cfg, seed_override=seed + ep)
        state = env.reset()
        total_reward = 0.0
        ep_traj = []

        for t in range(horizon):
            ep_traj.append(state)
            action = policy_fn(state)
            next_state, reward_raw, done, info = env.step(action)
            total_reward += reward_raw
            state = next_state
            if done:
                break

        returns.append(total_reward)
        trajs.append(np.vstack(ep_traj))

    returns = np.array(returns, dtype=float)
    trajs = np.stack(trajs, axis=0)  # [N, T, 5]
    mean_traj = np.mean(trajs, axis=0)

    print(
        f"[Eval] episodes={n_episodes}, "
        f"avg_return={returns.mean():.3f}, std={returns.std():.3f}"
    )
    return returns, mean_traj


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    print("=== Phase 3 RL Training (FQI, unified config) ===")

    # config paths
    #cfg_train_env_path = os.path.join(ROOT, "src", "config", "config_env_optimistic.yaml")
    cfg_calib_env_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    cfg_train_env_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    
    train_cfg = load_yaml(cfg_train_env_path)
    calib_cfg = load_yaml(cfg_calib_env_path)

    #  train / actions config
    train_section = train_cfg.get("train", {})
    actions_cfg = train_cfg.get("actions", {})

    gamma = train_section.get("gamma", 0.99)
    n_episodes = train_section.get("n_episodes", 300)
    horizon = train_section.get("horizon", 100)
    n_iterations = train_section.get("n_iterations", 25)
    random_seed = train_section.get("random_seed", 0)
    save_model_path = train_section.get("save_model_path", "trained_phase3_fqi.pkl")

    rng = np.random.RandomState(random_seed)

    print(f"[Config] gamma={gamma}, episodes={n_episodes}, horizon={horizon}, "
          f"n_iterations={n_iterations}, seed={random_seed}")
    print(f"[Config] model will be saved to: {save_model_path}")

    # 构建训练环境 & 动作网格
    # Build training environment & action grid
    train_env = make_env_from_cfg(train_cfg, seed_override=random_seed)
    action_grid = build_action_grid(actions_cfg)
    print(f"[ActionGrid] {len(action_grid)} actions in grid")

    # 采集离线数据
    # Collect offline dataset
    print("[Data] Collecting dataset with random behavior policy ...")
    states, actions, rewards, next_states = collect_dataset(
        train_env,
        action_grid,
        n_episodes=n_episodes,
        horizon=horizon,
        rng=rng,
    )
    print(f"[Data] Collected {states.shape[0]} transitions")

    # Fitted Q Iteration
    print("[Train] Starting FQI ...")
    Q = fitted_q_iteration(
        states,
        actions,
        rewards,
        next_states,
        action_grid,
        gamma=gamma,
        n_iterations=n_iterations,
        random_state=random_seed,
    )

    # save model
    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    payload = {"Q": Q, "action_grid": action_grid}
    joblib.dump(payload, save_model_path)
    print(f"[Save] Saved FQI policy to {save_model_path}")

    # # 在 Calibrated 环境上做评估（使用原始 reward）
    # print("[Eval] Evaluating greedy policy on calibrated environment ...")
    # greedy_pi = make_greedy_policy(Q, action_grid)
    # eval_horizon = calib_cfg.get("train", {}).get("horizon", horizon)
    # evaluate_policy(
    #     calib_cfg,
    #     greedy_pi,
    #     horizon=eval_horizon,
    #     n_episodes=50,
    #     seed=random_seed + 10_000,
    # )

    print("=== Done Phase 3 RL Training ===")


if __name__ == "__main__":
    main()
