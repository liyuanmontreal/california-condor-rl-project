"""
Phase 3: RL Policy Comparison on Calibrated Environment
-------------------------------------------------------

compare four policy curve in Calibrated env:

A = zero policy          (no release, no mitigation)
B = literature policy    (release=20, mitigation=0.7)
C = real-history policy  (use historical release data)
D = RL policy (FQI)      (greedy policy from trained Q)

draw both wild and total population curves:
- Wild population: Nj + Ns + Na
- Total population: Nj + Ns + Na + C
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.env.condor_env_age3 import CondorEnvAge3


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_from_cfg(cfg, seed_override=None):
    """根据统一 config 构建环境实例"""
    """creates environment instance from unified config"""
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


def rollout(env, policy_fn, horizon):
    """单次 rollout，返回 (T,5) 的状态轨迹"""
    """returns: np.array shape [T, 5] for single rollout"""
    traj = []
    s = env.reset()
    traj.append(s.copy())
    for t in range(horizon - 1):
        a_release, a_mit = policy_fn(s, t)
        ns, r, done, info = env.step([a_release, a_mit])
        traj.append(ns.copy())
        s = ns
        if done:
            break
    return np.vstack(traj)


def rollout_ensemble(cfg, policy_fn, n_episodes, horizon):
    """多次 rollout，用于画均值曲线"""
    """returns: multiple rollouts array [N,T,5], mean [T,5], std [T,5] to draw shading"""
    all_traj = []
    for ep in range(n_episodes):
        env = make_env_from_cfg(cfg, seed_override=ep)
        tr = rollout(env, policy_fn, horizon)
        all_traj.append(tr)
    arr = np.stack(all_traj, axis=0)  # [N,T,5]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return arr, mean, std


# ----------------------------------------------------------------------
# Baseline Policies
# ----------------------------------------------------------------------
def policy_zero(state, t):
    """No release, no mitigation"""
    return 0.0, 0.0


def policy_literature(state, t):
    """Literature policy: always release 20 birds, mitigation=0.7"""
    return 20.0, 0.7


def load_real_condor_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace("\ufeff", "")
    df = df.sort_values("year").reset_index(drop=True)
    return df


def build_real_history_policy(real_df):
    """根据真实数据构建历史策略（只用 released_to_wild 列）"""
    """uses real data to create history policy column 'released_to_wild' from real_df"""
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


# ----------------------------------------------------------------------
# RL Policy (from trained Q)
# ----------------------------------------------------------------------
def make_greedy_policy(Q, action_grid):
    """给定 Q(s,a) 和动作网格，构建贪婪策略 π(s)"""
    """creates greedy policy function from Q and action_grid"""

    def policy(state, t):
        n_actions = action_grid.shape[0]
        state_tiled = np.repeat(state.reshape(1, -1), n_actions, axis=0)
        sa = np.concatenate([state_tiled, action_grid], axis=1)
        q_vals = Q.predict(sa)
        best_idx = int(np.argmax(q_vals))
        best_action = action_grid[best_idx]
        return float(best_action[0]), float(best_action[1])

    return policy


# ----------------------------------------------------------------------
# Main plotting
# ----------------------------------------------------------------------
def main():
    print("=== Phase 3: RL Policy Comparison on Calibrated Environment ===")

    # ---- Config & paths ----
    cfg_calib_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    calib_cfg = load_yaml(cfg_calib_path)

    train_cfg = calib_cfg.get("train", {})
    rl_model_path = train_cfg.get("save_model_path", "trained_fqi_age3.npy")
    if not os.path.isabs(rl_model_path):
        rl_model_path = os.path.join(ROOT, rl_model_path)

    real_data_path = os.path.join(ROOT, "data", "condor_real_data.csv")

    horizon = calib_cfg["sim"]["horizon"]
    n_rollouts = calib_cfg["sim"]["N_rollouts"]

    print(f"[INFO] Using calibrated config: {cfg_calib_path}")
    print(f"[INFO] horizon={horizon}, N_rollouts={n_rollouts}")
    print(f"[INFO] RL model path: {rl_model_path}")
    print(f"[INFO] Real data path: {real_data_path}")

    # ---- Load RL model ----
    payload = joblib.load(rl_model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]
    print(f"[INFO] Loaded RL policy, action_grid shape={action_grid.shape}")

    rl_policy = make_greedy_policy(Q, action_grid)

    # ---- Build baseline policies ----
    real_df = load_real_condor_data(real_data_path)
    hist_policy = build_real_history_policy(real_df)

    # ---- Rollouts ----
    print("[Rollout] A: calibrated + zero ...")
    _, meanA, stdA = rollout_ensemble(calib_cfg, policy_zero, n_rollouts, horizon)

    print("[Rollout] B: calibrated + literature ...")
    _, meanB, stdB = rollout_ensemble(calib_cfg, policy_literature, n_rollouts, horizon)

    print("[Rollout] C: calibrated + real-history ...")
    _, meanC, stdC = rollout_ensemble(calib_cfg, hist_policy, n_rollouts, horizon)

    print("[Rollout] D: calibrated + RL (FQI greedy) ...")
    _, meanD, stdD = rollout_ensemble(calib_cfg, rl_policy, n_rollouts, horizon)

    # ---- Compute wild & total ----
    def wild(x):   # x: [T,5]
        return x[:, 0] + x[:, 1] + x[:, 2]

    def total(x):  # wild + captive
        return x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]

    t = np.arange(horizon)

    wildA, totA = wild(meanA), total(meanA)
    wildB, totB = wild(meanB), total(meanB)
    wildC, totC = wild(meanC), total(meanC)
    wildD, totD = wild(meanD), total(meanD)

    # ---- Plot ----
    plt.figure(figsize=(12, 7))

    # A: zero
    plt.plot(t, wildA, label="A: calib zero (wild)", color="orange", linestyle="-")
    plt.plot(t, totA,  label="A: calib zero (total)", color="orange", linestyle="--", alpha=0.7)

    # B: literature
    plt.plot(t, wildB, label="B: calib literature (wild)", color="green", linestyle="-")
    plt.plot(t, totB,  label="B: calib literature (total)", color="green", linestyle="--", alpha=0.7)

    # C: real-history
    plt.plot(t, wildC, label="C: calib real-history (wild)", color="blue", linestyle="-")
    plt.plot(t, totC,  label="C: calib real-history (total)", color="blue", linestyle="--", alpha=0.7)

    # D: RL
    plt.plot(t, wildD, label="D: calib RL (FQI) (wild)", color="red", linestyle="-")
    plt.plot(t, totD,  label="D: calib RL (FQI) (total)", color="red", linestyle="--", alpha=0.7)

    # RL 的 wild 区间带
    # RL's wild shading
    plt.fill_between(
        t,
        wildD - wild(stdD),
        wildD + wild(stdD),
        color="red",
        alpha=0.15,
        linewidth=0,
    )

    plt.title("Phase 3: Calibrated Environment - Wild vs Total Population")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)  
    plt.tight_layout()

    out_path = os.path.join(ROOT,"outputs", "phase3_policy_comparison_calibrated_wild_total.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
