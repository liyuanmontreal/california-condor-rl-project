"""
Phase-2 Policy Comparison (Unified Config Version)
--------------------------------------------------
对比 5 条策略曲线：
compare 5 curves:
A = mechanistic + zero
B = mechanistic + literature
C = calibrated + real-history
D = calibrated + literature
E = calibrated + zero
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.env.condor_env_age3 import CondorEnvAge3


# ============================================================
# Unified config loader
# ============================================================
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_from_cfg(cfg, seed_override=None):
    """
    - cfg["env"] + cfg["reward"] use for env
    - cfg["sim"] only use to rollout
    """
    env_cfg = cfg.get("env", {})
    reward_cfg = cfg.get("reward", {})

    if seed_override is not None:
        env_cfg["seed"] = seed_override
    else:
        if "sim" in cfg and "seed" in cfg["sim"]:
            env_cfg["seed"] = cfg["sim"]["seed"]

    merged = {**env_cfg, **reward_cfg}
    return CondorEnvAge3(**merged)


# ============================================================
# Rollout function
# ============================================================
def rollout(env, policy_fn, horizon):
    traj = []
    state = env.reset()
    traj.append(state.copy())

    for t in range(horizon - 1):
        release, mitigation = policy_fn(state, t)
        next_state, reward, done, info = env.step([release, mitigation])
        traj.append(next_state.copy())
        state = next_state
        if done:
            break

    return np.vstack(traj)  # (T,5)


# ============================================================
# Ensemble rollout (for ± std shading)
# ============================================================
def rollout_ensemble(cfg, policy_fn, N, horizon):
    all_traj = []

    for seed in range(N):
        env = make_env_from_cfg(cfg, seed_override=seed)
        tr = rollout(env, policy_fn, horizon)
        all_traj.append(tr)

    arr = np.stack(all_traj, axis=0)  # [N, T, 5]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    return arr, mean, std


# ============================================================
# Policies
# ============================================================
def policy_zero(state, t):
    """No release, no mitigation."""
    return 0.0, 0.0


def policy_literature(state, t):
    """Literature policy: fixed release=20, mitigation=0.7."""
    return 20.0, 0.7


def build_real_history_policy(real_df):
    releases = real_df["released_to_wild"].fillna(0).to_numpy(dtype=float)
    T = len(releases)

    if T == 0:
        last3 = 0.0
    elif T <= 3:
        last3 = float(np.mean(releases))
    else:
        last3 = float(np.mean(releases[-3:]))

    last3 = max(last3, 0.0)

    def policy(state, t):
        if t < T:
            rel = releases[t]
        else:
            rel = last3
        return float(max(rel, 0.0)), 0.0

    return policy


def load_real_condor_data(path="data/condor_real_data.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace("\ufeff", "")
    df = df.sort_values("year").reset_index(drop=True)
    return df


# ============================================================
# Main
# ============================================================
def main():
    print("=== Phase 2: Policy Comparison (Unified Config) ===")

    # Load configs
    mech_cfg = load_yaml("src/config/config_env_mechanistic.yaml")
    calib_cfg = load_yaml("src/config/config_env_calibrated.yaml")
 

    # Real condor data
    real_df = load_real_condor_data()

    # Horizon / N_rollouts 取自 mechanistic.sim（两者应一致）
    horizon = mech_cfg["sim"]["horizon"]
    N = mech_cfg["sim"]["N_rollouts"]

    print(f"[INFO] horizon={horizon}, N_rollouts={N}")

    # Build policies
    polA = policy_zero
    polB = policy_literature
    polC = build_real_history_policy(real_df)
    polD = policy_literature
    polE = policy_zero

    # Rollouts
    print("[Phase2] A: mechanistic + zero ...")
    _, meanA, stdA = rollout_ensemble(mech_cfg, polA, N, horizon)

    print("[Phase2] B: mechanistic + literature ...")
    _, meanB, stdB = rollout_ensemble(mech_cfg, polB, N, horizon)

    print("[Phase2] C: calibrated + real-history ...")
    _, meanC, stdC = rollout_ensemble(calib_cfg, polC, N, horizon)

    print("[Phase2] D: calibrated + literature ...")
    _, meanD, stdD = rollout_ensemble(calib_cfg, polD, N, horizon)

    print("[Phase2] E: calibrated + zero ...")
    _, meanE, stdE = rollout_ensemble(calib_cfg, polE, N, horizon)


    # Plot total N
    t = np.arange(horizon)
    plt.figure(figsize=(12, 7))

    def total(x):
        return x[:, 0] + x[:, 1] + x[:, 2]

    plt.plot(t, total(meanA), label="A: mech zero", color="black")
    plt.plot(t, total(meanB), label="B: mech literature", color="red")
    plt.plot(t, total(meanC), label="C: calib real-history", color="blue")
    plt.plot(t, total(meanD), label="D: calib literature", color="green")
    plt.plot(t, total(meanE), label="E: calib zero", color="orange")


    plt.title("Phase 2: Total Wild Population (Nj + Ns + Na)")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig("phase2_policy_comparison.png", dpi=150)
    out_path = os.path.join(
            ROOT, "outputs","phase2_policy_comparison.png"
        )
    plt.savefig(out_path, dpi=150)
    print("[Saved] phase2_policy_comparison.png")


if __name__ == "__main__":
    main()
