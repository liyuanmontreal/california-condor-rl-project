"""
Phase 3: RL Best Policy Action Trajectories
-------------------------------------------

画出 RL 最优策略 (FQI greedy policy) 的动作轨迹：
- release(t)
- mitigation(t)

环境：config_env_calibrated.yaml
策略：train.save_model_path 中保存的 FQI 模型

Plot the action trajectory of the optimal RL policy (FQI greedy policy):
- release(t)
- mitigation(t)
Environment: config_env_calibrated.yaml
Policy: FQI model saved in train.save_model_path
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


# ------------------ Utility ------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_from_cfg(cfg, seed_override=None):
    """构建环境实例"""
    """Construct an environment instance"""
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
    """FQI 贪婪策略"""
    """FQI greedy policy"""

    def policy(state):
        nA = action_grid.shape[0]
        s_rep = np.repeat(state.reshape(1, -1), nA, axis=0)
        sa = np.concatenate([s_rep, action_grid], axis=1)
        qvals = Q.predict(sa)
        idx = int(np.argmax(qvals))
        a = action_grid[idx]
        return float(a[0]), float(a[1])

    return policy


def rollout_actions(env, policy_fn, horizon):
    """Rollout 策略并记录动作 [release(t), mitigation(t)]"""
    releases = []
    mitigations = []

    s = env.reset()
    for t in range(horizon):
        rel, mit = policy_fn(s)
        releases.append(rel)
        mitigations.append(mit)
        s, r, done, info = env.step([rel, mit])
        if done:
            break

    return np.array(releases), np.array(mitigations)


# ------------------ Main ------------------
def main():
    print("=== Phase 3: RL Best Policy (Action Trajectories) ===")

    # read config
    cfg_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    cfg = load_yaml(cfg_path)

    horizon = cfg["sim"]["horizon"]
    seed = cfg["sim"]["seed"]

    # RL model path
    model_path = cfg["train"]["save_model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)

    print(f"[INFO] Using config: {cfg_path}")
    print(f"[INFO] Model: {model_path}")

    payload = joblib.load(model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]
    print(f"[INFO] Loaded model, action_grid shape={action_grid.shape}")

    policy = make_greedy_policy(Q, action_grid)

    # create env and  rollout
    env = make_env_from_cfg(cfg, seed_override=seed + 500)
    releases, mitigations = rollout_actions(env, policy, horizon)

    # ------------ Plot ------------
    t = np.arange(len(releases))

    plt.figure(figsize=(12, 6))

    plt.plot(t, releases, label="Release action", color="purple")
    plt.plot(t, mitigations, label="Mitigation action", color="teal")

    plt.xlabel("Time (year)")
    plt.ylabel("Action magnitude")
    plt.title("RL Best Policy: Action Trajectories (release & mitigation)\nCalibrated environment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(ROOT, "outputs","phase3_rl_best_policy_actions.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
