"""
Phase 3: Dual Policy Phase Portrait (release & mitigation vs N_total)
---------------------------------------------------------------------

在 Calibrated 环境下，用训练好的 FQI 贪婪策略，
同时绘制两个相图：

1) Release vs N_total
2) Mitigation vs N_total

每个点代表某次 rollout 某个时间步：
    x = N_total (Nj + Ns + Na)
    y = release 或 mitigation

In a Calibrated environment, using a trained FQI greedy policy,
two phase graphs are plotted simultaneously:
1) Release vs N_total
2) Mitigation vs N_total
Each point represents a rollout at a specific time step:
x = N_total (Nj + Ns + Na)
y = release or mitigation
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


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_from_cfg(cfg, seed_override=None):
    """Build calibrated environment"""
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
    """Construct greedy policy from FQI model"""

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
    Collect (N_total, release, mitigation) phase points across many rollouts.
    """

    N_list = []
    R_list = []
    M_list = []

    base_seed = cfg.get("sim", {}).get("seed", 0)

    for ep in range(n_episodes):
        env = make_env_from_cfg(cfg, seed_override=base_seed + 2000 + ep)
        s = env.reset()
        for t in range(horizon):
            Nj, Ns, Na = s[0], s[1], s[2]
            N_total = Nj + Ns + Na

            rel, mit = policy_fn(s)

            N_list.append(float(N_total))
            R_list.append(float(rel))
            M_list.append(float(mit))

            s, r, done, info = env.step([rel, mit])
            if done:
                break

    return np.array(N_list), np.array(R_list), np.array(M_list)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("=== Phase 3: Dual Policy Phase Portrait (release & mitigation vs N_total) ===")

    # Load config
    cfg_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    cfg = load_yaml(cfg_path)

    horizon = cfg["sim"]["horizon"]
    n_rollouts = cfg["sim"]["N_rollouts"]

    # Load RL model
    model_path = cfg["train"]["save_model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)

    print(f"[INFO] Using config: {cfg_path}")
    print(f"[INFO] Using RL model: {model_path}")

    payload = joblib.load(model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]

    policy = make_greedy_policy(Q, action_grid)

    print("[INFO] Collecting phase portrait points...")
    N_vals, R_vals, M_vals = collect_phase_points(cfg, policy, horizon, n_rollouts)
    print(f"[INFO] collected {len(N_vals)} points")

    # ---------------- Plot --------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1) Release vs N_total
    axes[0].scatter(N_vals, R_vals, alpha=0.3, s=15, color="purple")
    axes[0].set_xlabel("Total wild population N_total")
    axes[0].set_ylabel("Release action")
    axes[0].set_title("Phase Portrait: Release vs N_total")
    axes[0].grid(True, alpha=0.3)

    # 2) Mitigation vs N_total
    axes[1].scatter(N_vals, M_vals, alpha=0.3, s=15, color="teal")
    axes[1].set_xlabel("Total wild population N_total")
    axes[1].set_ylabel("Mitigation action")
    axes[1].set_title("Phase Portrait: Mitigation vs N_total")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(ROOT,"outputs", "phase3_rl_policy_phaseportrait_dual.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
