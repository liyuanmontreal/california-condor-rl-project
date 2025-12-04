"""
Phase 3: RL Policy Phase Portrait (Hexbin Density Version)
----------------------------------------------------------


1) Release vs N_total
2) Mitigation vs N_total

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


# ---------------- 工具函数 ----------------
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


def collect_phase_points(cfg, policy_fn, horizon, n_episodes):
    N_vals, R_vals, M_vals = [], [], []

    base_seed = cfg.get("sim", {}).get("seed", 0)

    for ep in range(n_episodes):
        env = make_env_from_cfg(cfg, seed_override=base_seed + 3000 + ep)
        s = env.reset()

        for t in range(horizon):
            Nj, Ns, Na = s[0], s[1], s[2]
            N_total = Nj + Ns + Na

            rel, mit = policy_fn(s)

            N_vals.append(N_total)
            R_vals.append(rel)
            M_vals.append(mit)

            s, r, done, info = env.step([rel, mit])
            if done:
                break

    return np.array(N_vals), np.array(R_vals), np.array(M_vals)


# ----------------  ----------------
def main():
    print("=== Phase 3: RL Policy Phase Portrait (hexbin density) ===")

    cfg_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    cfg = load_yaml(cfg_path)

    horizon = cfg["sim"]["horizon"]
    n_rollouts = cfg["sim"]["N_rollouts"]

    model_path = cfg["train"]["save_model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)

    payload = joblib.load(model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]

    policy = make_greedy_policy(Q, action_grid)

    N_vals, R_vals, M_vals = collect_phase_points(cfg, policy, horizon, n_rollouts)
    print(f"[INFO] collected {len(N_vals)} points")

    # ---------- Hexbin Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Release vs N_total
    hb1 = axes[0].hexbin(
        N_vals, R_vals,
        gridsize=40,
        cmap="Purples",
        mincnt=1,
        bins="log"
    )
    axes[0].set_xlabel("Total wild population N_total")
    axes[0].set_ylabel("Release action")
    axes[0].set_title("Release vs N_total (hexbin density)")
    fig.colorbar(hb1, ax=axes[0], label="log density")

    # Mitigation vs N_total
    hb2 = axes[1].hexbin(
        N_vals, M_vals,
        gridsize=40,
        cmap="GnBu",
        mincnt=1,
        bins="log"
    )
    axes[1].set_xlabel("Total wild population N_total")
    axes[1].set_ylabel("Mitigation action")
    axes[1].set_title("Mitigation vs N_total (hexbin density)")
    fig.colorbar(hb2, ax=axes[1], label="log density")

    plt.tight_layout()

    out_path = os.path.join(ROOT,"outputs", "phase3_rl_policy_phaseportrait_hexbin.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
