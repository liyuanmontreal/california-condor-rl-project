"""
Phase 3: RL Policy Phase Portrait (KDE Smooth Version)
------------------------------------------------------

使用 seaborn KDE 得到平滑的 Release / Mitigation 策略相图。
展示“策略的连续结构与主趋势”。
use seaborn KDE to obtain smooth release/mitigation policy phase portraits.
Show "the continuous structure and main trends of the policy".

1) Release vs N_total (KDE)
2) Mitigation vs N_total (KDE)

"""

import os
import sys
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.env.condor_env_age3 import CondorEnvAge3


# ---------------- Utility ----------------
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
        merged["seed"] = cfg.get("sim", {}).get("seed", 0)

    return CondorEnvAge3(**merged)


def make_greedy_policy(Q, action_grid):
    def policy(state):
        nA = action_grid.shape[0]
        s_rep = np.repeat(state.reshape(1, -1), nA, axis=0)
        sa = np.concatenate([s_rep, action_grid], axis=1)
        qvals = Q.predict(sa)
        idx = int(np.argmax(qvals))
        a = action_grid[idx]
        return float(a[0]), float(a[1])
    return policy


def collect_phase_points(cfg, policy, horizon, n_rollouts):
    N_vals, R_vals, M_vals = [], [], []
    base_seed = cfg["sim"].get("seed", 0)

    for ep in range(n_rollouts):
        env = make_env_from_cfg(cfg, seed_override=base_seed + 5000 + ep)
        s = env.reset()

        for t in range(horizon):
            Nj, Ns, Na = s[0], s[1], s[2]
            N_total = Nj + Ns + Na

            rel, mit = policy(s)

            N_vals.append(N_total)
            R_vals.append(rel)
            M_vals.append(mit)

            s, r, done, info = env.step([rel, mit])
            if done:
                break

    return np.array(N_vals), np.array(R_vals), np.array(M_vals)


# ---------------- Main ----------------
def main():
    print("=== Phase 3: RL Policy KDE Portrait ===")

    cfg_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    cfg = load_yaml(cfg_path)

    model_path = cfg["train"]["save_model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)

    payload = joblib.load(model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]

    policy = make_greedy_policy(Q, action_grid)

    horizon = cfg["sim"]["horizon"]
    n_rollouts = cfg["sim"]["N_rollouts"]

    N_vals, R_vals, M_vals = collect_phase_points(cfg, policy, horizon, n_rollouts)

    print(f"[INFO] collected {len(N_vals)} points")

    sns.set_style("white")
    sns.set_context("talk")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---------------- KDE Plot: Release vs N_total ----------------
    sns.kdeplot(
        x=N_vals,
        y=R_vals,
        cmap="Purples",
        fill=True,
        thresh=0.05,
        levels=100,
        ax=axes[0]
    )
    axes[0].set_title("KDE Portrait: Release vs N_total")
    axes[0].set_xlabel("Total wild population N_total")
    axes[0].set_ylabel("Release action")

    # ---------------- KDE Plot: Mitigation vs N_total ----------------
    sns.kdeplot(
        x=N_vals,
        y=M_vals,
        cmap="GnBu",
        fill=True,
        thresh=0.05,
        levels=100,
        ax=axes[1]
    )
    axes[1].set_title("KDE Portrait: Mitigation vs N_total")
    axes[1].set_xlabel("Total wild population N_total")
    axes[1].set_ylabel("Mitigation action")

    plt.tight_layout()

    out_path = os.path.join(ROOT,"outputs", "phase3_rl_policy_phaseportrait_kde.png")
    plt.savefig(out_path, dpi=200)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
