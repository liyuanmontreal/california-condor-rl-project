"""
Phase 3: Age-structured trajectories under RL (FQI) policy
----------------------------------------------------------

在 Calibrated 环境上，用训练好的 FQI 贪婪策略
画出年龄结构轨迹 (Nj, Ns, Na, N_total) 随时间变化。
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.env.condor_env_age3 import CondorEnvAge3


# ------------------ 工具函数 Tools  ------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_from_cfg(cfg, seed_override=None):
    """根据统一 config 构建 CondorEnvAge3 实例（calibrated 环境）"""
    """Construct a CondorEnvAge3 instance based on the unified config (calibrated environment)"""
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


def make_greedy_policy(Q, action_grid):
    """给定 Q(s,a) 和离散动作网格，构建贪婪策略 π(s)"""
    """Given Q(s,a) and a discrete action grid, construct a greedy policy π(s)"""

    def policy(state):
        n_actions = action_grid.shape[0]
        state_tiled = np.repeat(state.reshape(1, -1), n_actions, axis=0)
        sa = np.concatenate([state_tiled, action_grid], axis=1)
        q_vals = Q.predict(sa)
        best_idx = int(np.argmax(q_vals))
        best_action = action_grid[best_idx]
        return float(best_action[0]), float(best_action[1])

    return policy


def rollout_single(env, policy_fn, horizon):
    """在给定环境和策略下，采样一条轨迹，返回 (T,5) 状态数组"""
    """ sample a trajectory under the given environment and policy, returning a (T,5) state array """
    states = []
    s = env.reset()
    states.append(s.copy())
    for t in range(horizon - 1):
        rel, mit = policy_fn(s)
        ns, r, done, info = env.step([rel, mit])
        states.append(ns.copy())
        s = ns
        if done:
            break
    return np.vstack(states)


# ------------------ 主函数 ------------------
def main():
    print("=== Phase 3: Age-structured trajectories under RL policy ===")

    # 1) read calibrated config
    cfg_calib_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    calib_cfg = load_yaml(cfg_calib_path)

    sim_cfg = calib_cfg.get("sim", {})
    horizon = sim_cfg.get("horizon", 120)
    seed = sim_cfg.get("seed", 0)

    # 2) load trained FQI model
    train_cfg = calib_cfg.get("train", {})
    rl_model_path = train_cfg.get("save_model_path", "trained_fqi_age3.npy")
    if not os.path.isabs(rl_model_path):
        rl_model_path = os.path.join(ROOT, rl_model_path)

    print(f"[INFO] Using calibrated config: {cfg_calib_path}")
    print(f"[INFO] horizon={horizon}, seed={seed}")
    print(f"[INFO] Loading RL model from: {rl_model_path}")

    payload = joblib.load(rl_model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]
    print(f"[INFO] Loaded RL model, action_grid shape={action_grid.shape}")

    policy_rl = make_greedy_policy(Q, action_grid)

    # 3) 在 calibrated 环境上 roll out 一条轨迹
    # roll out one trajectory in calibrated environment
    env = make_env_from_cfg(calib_cfg, seed_override=seed + 999)
    traj = rollout_single(env, policy_rl, horizon=horizon)

    # traj: [T, 5] = [Nj, Ns, Na, C, H]
    Nj = traj[:, 0]
    Ns = traj[:, 1]
    Na = traj[:, 2]
    N_total = Nj + Ns + Na  # only wild population

    T = len(traj)
    t = np.arange(T)

    # plot
    plt.figure(figsize=(10, 6))

    plt.plot(t, Nj, label="Juvenile (Nj)", color="C0")
    plt.plot(t, Ns, label="Subadult (Ns)", color="C1")
    plt.plot(t, Na, label="Adult (Na)", color="C2")
    plt.plot(t, N_total, label="Total wild (N_total)", color="C3", linestyle="--")

    plt.xlabel("Time step (year)")
    plt.ylabel("Population size")
    plt.title("Age-structured trajectories under RL (FQI) policy\nCalibrated environment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(ROOT,"outputs", "phase3_rl_age_trajectories_calibrated.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
