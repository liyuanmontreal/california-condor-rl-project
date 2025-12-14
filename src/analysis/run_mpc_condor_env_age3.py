
"""
MPC baseline using the SAME calibrated environment as RL (CondorEnvAge3).

Purpose:
- Provide a fair, model-based baseline on identical dynamics
- Short-horizon MPC (deterministic rollout) for sanity-check and interpretation
- NOT tuned to beat RL; emphasizes stabilization and feasibility
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
import numpy as np
from copy import deepcopy

# === Import the latest calibrated environment ===
from src.env.condor_env_age3 import CondorEnvAge3


# -------------------------------
# MPC configuration
# -------------------------------
HORIZON = 10          # short horizon (years)
N_SAMPLES = 200       # candidate action sequences
DISCOUNT = 0.98

RELEASE_LEVELS = [0, 5, 10, 15, 20]
MITIGATION_LEVELS = [0.0, 0.3, 0.6, 1.0]

TARGET_POP = 600.0    # soft target for total wild population

# -------------------------------
# Cost / objective
# -------------------------------

def stage_cost(state, action):
    """Quadratic population deviation + linear intervention costs."""
    Nj, Ns, Na, C, H = state
    N_total = Nj + Ns + Na

    release, mitigation = action

    pop_cost = (N_total - TARGET_POP) ** 2
    action_cost = 5.0 * release + 200.0 * mitigation

    return pop_cost + action_cost


# -------------------------------
# Deterministic rollout (planner model)
# -------------------------------

def rollout_deterministic(env, action_seq):
    """Roll out using deterministic dynamics (no disasters, no noise)."""
    env_sim = deepcopy(env)
    # --- Disable stochasticity if supported by the environment ---
    if hasattr(env_sim, "disable_stochasticity"):
        env_sim.disable_stochasticity()
    elif hasattr(env_sim, "set_deterministic"):
        env_sim.set_deterministic(True)
    else:
        # Fallback: manually zero out stochastic components if present
        if hasattr(env_sim, "disaster_base_prob"):
            env_sim.disaster_base_prob = 0.0
        if hasattr(env_sim, "noise_std"):
            env_sim.noise_std = 0.0

    total_cost = 0.0
    gamma = 1.0

    for a in action_seq:
        s, _, _, _ = env_sim.step(a)
        total_cost += gamma * stage_cost(s, a)
        gamma *= DISCOUNT

    return total_cost


# -------------------------------
# MPC action selection
# -------------------------------

def mpc_action(env):
    """Sample-based MPC over short horizon."""
    best_cost = np.inf
    best_action = (0, 0.0)

    for _ in range(N_SAMPLES):
        action_seq = [
            (
                np.random.choice(RELEASE_LEVELS),
                np.random.choice(MITIGATION_LEVELS)
            )
            for _ in range(HORIZON)
        ]

        cost = rollout_deterministic(env, action_seq)

        if cost < best_cost:
            best_cost = cost
            best_action = action_seq[0]

    return best_action


# -------------------------------
# Main simulation loop
# -------------------------------

def run_mpc_episode(seed=0, T=120):
    np.random.seed(seed)

    env = CondorEnvAge3()
    env.reset()

    states = []
    actions = []

    for t in range(T):
        a = mpc_action(env)
        s, r, done, info = env.step(a)

        states.append(s)
        actions.append(a)

        if done:
            break

    return np.array(states), np.array(actions)


def plot_mpc_trajectory(states, actions):
    """Visualize MPC population and control actions."""
    import matplotlib.pyplot as plt

    Nj = states[:, 0]
    Ns = states[:, 1]
    Na = states[:, 2]
    N_total = Nj + Ns + Na

    releases = actions[:, 0]
    mitigations = actions[:, 1]

    t = np.arange(len(N_total))

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # --- Population ---
    axs[0].plot(t, N_total, label="Total wild population", color="black")
    axs[0].plot(t, Nj, "--", label="Juvenile", alpha=0.7)
    axs[0].plot(t, Ns, "--", label="Subadult", alpha=0.7)
    axs[0].plot(t, Na, "--", label="Adult", alpha=0.7)
    axs[0].set_ylabel("Population")
    axs[0].legend()
    axs[0].set_title("MPC rollout on calibrated Condor environment")

    # --- Release action ---
    axs[1].step(t, releases, where="post")
    axs[1].set_ylabel("Release")
    axs[1].set_title("MPC release actions")

    # --- Mitigation action ---
    axs[2].step(t, mitigations, where="post")
    axs[2].set_ylabel("Mitigation")
    axs[2].set_xlabel("Year")
    axs[2].set_title("MPC mitigation actions")

    plt.tight_layout()
  
    plt.savefig("outputs/fig_mpc_condor_env_age3_trajectory.png", dpi=300)
    plt.show()
    print("[INFO] Figure saved to outputs/fig_mpc_condor_env_age3_trajectory.png")

if __name__ == "__main__":
    states, actions = run_mpc_episode()
    plot_mpc_trajectory(states, actions)

    Nj = states[:, 0]
    Ns = states[:, 1]
    Na = states[:, 2]
    N_total = Nj + Ns + Na

    print("Final total wild population:", N_total[-1])
