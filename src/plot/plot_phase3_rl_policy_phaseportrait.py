"""
Phase 3: Policy Phase Portrait (release vs N_total)
---------------------------------------------------

在 Calibrated 环境下，用训练好的 FQI 贪婪策略，
画出策略相图：release 动作随 N_total (Nj+Ns+Na) 的变化关系。

每个点代表在某一时刻、某一条轨迹上：
    x = N_total (Nj + Ns + Na)
    y = release action

In a Calibrated environment, using a trained FQI greedy policy,
plot the policy phase diagram: the relationship between the release action and N_total (Nj + Ns + Na).
Each point represents a specific moment on a specific trajectory:
x = N_total (Nj + Ns + Na)
y = release action
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


# ------------------ 工具函数 ------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_from_cfg(cfg, seed_override=None):
    """根据统一 config 构建环境实例"""
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
    """FQI 贪婪策略 π(s)"""

    def policy(state):
        nA = action_grid.shape[0]
        s_rep = np.repeat(state.reshape(1, -1), nA, axis=0)
        sa = np.concatenate([s_rep, action_grid], axis=1)
        qvals = Q.predict(sa)
        idx = int(np.argmax(qvals))
        a = action_grid[idx]
        return float(a[0]), float(a[1])  # release, mitigation

    return policy


def collect_phase_points(cfg, policy_fn, horizon, n_episodes):
    """
    在 calibrated 环境下，用 RL 策略跑 n_episodes 条轨迹，
    收集 (N_total, release) 点。

    返回：
        N_list: (K,)  所有时间步的 N_total
        R_list: (K,)  对应的 release 动作
    """
    N_list = []
    R_list = []

    base_seed = cfg.get("sim", {}).get("seed", 0)

    for ep in range(n_episodes):
        env = make_env_from_cfg(cfg, seed_override=base_seed + 1000 + ep)
        s = env.reset()  # [Nj, Ns, Na, C, H]
        for t in range(horizon):
            Nj, Ns, Na = s[0], s[1], s[2]
            N_total = Nj + Ns + Na

            rel, mit = policy_fn(s)

            N_list.append(float(N_total))
            R_list.append(float(rel))

            s, r, done, info = env.step([rel, mit])
            if done:
                break

    return np.array(N_list), np.array(R_list)


# ------------------ 主函数 ------------------
def main():
    print("=== Phase 3: Policy Phase Portrait (release vs N_total) ===")

    # 1) 读取 calibrated 配置
    cfg_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    cfg = load_yaml(cfg_path)

    sim_cfg = cfg.get("sim", {})
    horizon = sim_cfg.get("horizon", 120)
    n_rollouts = sim_cfg.get("N_rollouts", 20)

    # 2) 加载 RL 模型
    train_cfg = cfg.get("train", {})
    model_path = train_cfg.get("save_model_path", "trained_fqi_age3.npy")
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)

    print(f"[INFO] Using config: {cfg_path}")
    print(f"[INFO] RL model path: {model_path}")
    print(f"[INFO] horizon={horizon}, n_rollouts={n_rollouts}")

    payload = joblib.load(model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]
    print(f"[INFO] Loaded RL model, action_grid shape={action_grid.shape}")

    policy = make_greedy_policy(Q, action_grid)

    # 3) 收集 (N_total, release) 点
    N_vals, R_vals = collect_phase_points(cfg, policy, horizon, n_rollouts)
    print(f"[INFO] collected {len(N_vals)} state-action pairs")

    # 4) 绘制相图
    plt.figure(figsize=(8, 6))
    plt.scatter(N_vals, R_vals, alpha=0.3, s=15)

    plt.xlabel("Total wild population N_total (Nj + Ns + Na)")
    plt.ylabel("Release action")
    plt.title("Phase portrait of RL policy\nCalibrated environment: release vs N_total")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        ROOT, "outputs","phase3_rl_policy_phaseportrait_release_vs_Ntotal.png"
    )
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
