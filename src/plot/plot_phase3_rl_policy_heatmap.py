"""
Phase 3 RL Policy Heatmap
-------------------------

绘制两个策略热力图：
1) Release action as a function of N_total
2) Mitigation action as a function of N_total

横轴 = Total wild population (N_total)
纵轴 = 动作值
颜色 = 动作密度 / 使用频率
Draw twopolicy heatmaps:
1) Release action as a function of N_total
2) Mitigation action as a function of N_total
Horizontal axis = Total wild population (N_total)
Vertical axis = Action value
Color = Action density / Usage frequency
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.env.condor_env_age3 import CondorEnvAge3


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_from_cfg(cfg, seed_override=None):
    env_cfg = cfg["env"]
    rew_cfg = cfg["reward"]
    merged = {**env_cfg, **rew_cfg}

    if seed_override is not None:
        merged["seed"] = seed_override
    else:
        merged["seed"] = cfg["sim"].get("seed", 0)

    return CondorEnvAge3(**merged)


def make_policy(Q, action_grid):
    def policy_fn(state):
        nA = action_grid.shape[0]
        s_rep = np.repeat(state.reshape(1, -1), nA, axis=0)
        sa = np.concatenate([s_rep, action_grid], axis=1)
        qvals = Q.predict(sa)
        idx = int(np.argmax(qvals))
        return action_grid[idx]
    return policy_fn


def collect_data(cfg, policy, horizon, n_rollouts):
    N, R, M = [], [], []
    base_seed = cfg["sim"]["seed"]

    for ep in range(n_rollouts):
        env = make_env_from_cfg(cfg, seed_override=base_seed + 7000 + ep)
        s = env.reset()

        for _ in range(horizon):
            Nj, Ns, Na = s[0], s[1], s[2]
            N_total = Nj + Ns + Na
            N.append(N_total)

            rel, mit = policy(s)
            R.append(rel)
            M.append(mit)

            s, r, done, info = env.step([rel, mit])
            if done:
                break

    return np.array(N), np.array(R), np.array(M)


# ===================== MAIN =====================
def main():
    print("=== Phase 3 RL Policy Heatmap ===")

    cfg_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    cfg = load_yaml(cfg_path)

    model_path = cfg["train"]["save_model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)

    payload = joblib.load(model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]

    policy = make_policy(Q, action_grid)

    horizon = cfg["sim"]["horizon"]
    n_rollouts = cfg["sim"]["N_rollouts"]

    N_vals, R_vals, M_vals = collect_data(cfg, policy, horizon, n_rollouts)

    # -------------------- Heatmap Data --------------------
    bins = 40
    N_bins = np.linspace(min(N_vals), max(N_vals), bins+1)

    R_heat = np.zeros(bins)
    M_heat = np.zeros(bins)

    for i in range(bins):
        mask = (N_vals >= N_bins[i]) & (N_vals < N_bins[i+1])
        if np.sum(mask) > 0:
            R_heat[i] = np.mean(R_vals[mask])
            M_heat[i] = np.mean(M_vals[mask])
        else:
            R_heat[i] = np.nan
            M_heat[i] = np.nan

    N_centers = (N_bins[:-1] + N_bins[1:]) / 2

    # -------------------- Plot --------------------
    sns.set_style("white")
    sns.set_context("talk")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Release heatmap (smooth curve + shading)
    axes[0].plot(N_centers, R_heat, color="purple", linewidth=3)
    axes[0].fill_between(N_centers, R_heat, color="purple", alpha=0.25)
    axes[0].set_title("Policy Heatmap: Release vs N_total")
    axes[0].set_xlabel("Total wild population N_total")
    axes[0].set_ylabel("Avg release action")

    # Mitigation heatmap
    axes[1].plot(N_centers, M_heat, color="teal", linewidth=3)
    axes[1].fill_between(N_centers, M_heat, color="teal", alpha=0.25)
    axes[1].set_title("Policy Heatmap: Mitigation vs N_total")
    axes[1].set_xlabel("Total wild population N_total")
    axes[1].set_ylabel("Avg mitigation action")

    plt.tight_layout()

    out_path = os.path.join(ROOT,"outputs", "phase3_rl_policy_heatmap.png")
    plt.savefig(out_path, dpi=200)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
