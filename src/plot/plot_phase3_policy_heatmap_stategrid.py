"""
Phase 3: Full Policy Heatmap over State Grid
--------------------------------------------

绘制 2D 热力图：
    release(N_total, C)
    mitigation(N_total, C)

横轴: N_total (wild)
纵轴: C (captive)
颜色: 动作值

展示 RL 策略结构。

Draw a 2D heatmap:
release(N_total, C)
mitigation(N_total, C)
Horizontal axis: N_total (wild)
Vertical axis: C (captive)
Color: Action value
Display the RL policy structure.
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


def make_policy(Q, action_grid):
    """Return greedy policy."""
    def policy_fn(state):
        # brute force Q(s,a)
        na = action_grid.shape[0]
        s_rep = np.repeat(state.reshape(1, -1), na, axis=0)
        sa = np.concatenate([s_rep, action_grid], axis=1)
        qvals = Q.predict(sa)
        idx = int(np.argmax(qvals))
        return action_grid[idx]  # (release, mitigation)
    return policy_fn


def main():

    print("=== Phase 3: 2D Policy Heatmap ===")

    cfg_path = os.path.join(ROOT, "src", "config", "config_env_calibrated.yaml")
    cfg = load_yaml(cfg_path)

    model_path = cfg["train"]["save_model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)

    payload = joblib.load(model_path)
    Q = payload["Q"]
    action_grid = payload["action_grid"]

    policy = make_policy(Q, action_grid)

    # ---- prepare a state grid ----
    N_vals = np.linspace(50, 800, 70)      # wild
    C_vals = np.linspace(0, 400, 60)       # captive
    H_fixed = 1.0

    Release_map = np.zeros((len(C_vals), len(N_vals)))
    Mit_map     = np.zeros((len(C_vals), len(N_vals)))

    print("[INFO] Evaluating policy over grid ...")

    for i, C in enumerate(C_vals):
        for j, N in enumerate(N_vals):

            # 分解野外成年结构比例
            # 使用 Calibrated 初始结构比例 Nj:Ns:Na = 74:74:221（共369）
            Nj = N * (74/369)
            Ns = N * (74/369)
            Na = N * (221/369)

            s = np.array([Nj, Ns, Na, C, H_fixed], dtype=float)

            a = policy(s)
            Release_map[i, j] = a[0]
            Mit_map[i, j]     = a[1]

    # ---- Plot ----
    sns.set_style("white")
    sns.set_context("talk")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ========== RELEASE HEATMAP ==========
    hm1 = axes[0].imshow(
        Release_map,
        origin="lower",
        aspect="auto",
        extent=[N_vals[0], N_vals[-1], C_vals[0], C_vals[-1]],
        cmap="viridis"
    )
    plt.colorbar(hm1, ax=axes[0], label="release")
    axes[0].set_title("Release policy (H=1.0)")
    axes[0].set_xlabel("N_total (wild)")
    axes[0].set_ylabel("C (captive)")

    # ========== MITIGATION HEATMAP ==========
    hm2 = axes[1].imshow(
        Mit_map,
        origin="lower",
        aspect="auto",
        extent=[N_vals[0], N_vals[-1], C_vals[0], C_vals[-1]],
        cmap="viridis"
    )
    plt.colorbar(hm2, ax=axes[1], label="mitigation")
    axes[1].set_title("Mitigation policy (H=1.0)")
    axes[1].set_xlabel("N_total (wild)")
    axes[1].set_ylabel("C (captive)")

    out_path = os.path.join(ROOT,"outputs", "phase3_policy_heatmap_stategrid.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
