"""
MPC Simulation with Dynamic Stable-Growth Target
------------------------------------------------
-------------------------------------------------------------------------------
Start from 2024 observed population (566) and simulate 30 years ahead.
Produces both real-aligned and future-projected figures.
"""
import sys,os
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(os.path.abspath(__file__))))))



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.tools.load_data import load_condor_data

# ============================================
def dynamic_target(t, base=600.0, K=700.0, k=0.08):
    """Future-oriented stable growth target (600→700)."""
    return base + (K - base) * (1 - np.exp(-k * t))

# ============================================
def gompertz_step(N, u, r=0.08, K=700.0, noise_std=5.0):
    """Population dynamics with control input and stochasticity."""
    N_next = N * np.exp(r * np.log(K / N)) + u
    N_next += np.random.normal(0, noise_std)
    return max(0, N_next)

class MPCConfig:
    def __init__(self, horizon=5, release_max=40.0, lambda_u=0.1):
        self.horizon = horizon
        self.release_max = release_max
        self.lambda_u = lambda_u

# ============================================
def run_mpc_dynamic_target():
    df = load_condor_data()
    df["Year"] = df["Year"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    df["Year"] = df["Year"].astype(int)

    N0 = float(df["Total"].iloc[-1])  # 566
    print(f"[INFO] Starting MPC from N0 = {N0:.1f} (2024 observed population)")

    cfg = MPCConfig(horizon=5, release_max=40.0, lambda_u=0.1)
    T = 30  # simulate 30 years ahead
    years_future = list(range(df["Year"].iloc[-1] + 1, df["Year"].iloc[-1] + 1 + T))
    N = N0
    traj, releases, targets = [N], [], []

    for t in range(T):
        N_star = dynamic_target(t)
        targets.append(N_star)
        error = N_star - N
        u = np.clip(0.3 * error, 0, cfg.release_max)
        N = gompertz_step(N, u, r=0.08, K=700.0)
        traj.append(N)
        releases.append(u)
        print(f"Year {years_future[t]} | Target={N_star:.1f} | Release={u:.1f} | Pop={N:.1f}")

    print(f"[RESULT] Final projected population after {T} years: {N:.1f}")

    # ============================================
    # Export MPC trajectory
    os.makedirs("outputs", exist_ok=True)
    mpc_results = pd.DataFrame({
        "Year": years_future,
        "MPC_Population": traj[1:],
        "Target": targets
    })
    mpc_results.to_csv("outputs/mpc_population_traj.csv", index=False)
    print("[INFO] MPC trajectory exported to outputs/mpc_population_traj.csv")

    # ============================================
    # Visualization
    years_past = df["Year"].values
    pop_past = df["Total"].values
    years_all = list(years_past) + years_future
    pop_all = list(pop_past) + traj[1:]

    plt.figure(figsize=(10,6))
    plt.plot(years_past, pop_past, "o-", color="#666666", label="Real Data (1982–2024, up to 566)")
    plt.plot(years_all, pop_all, "s--", color="#003366", label="MPC Projected Population (2025–2055)")
    plt.plot(years_all[len(years_past):], targets, "--", color="#cc0000", label="Forward-looking Target (600→700)")
    plt.axhline(700, color="gray", linestyle="--", label="Carrying Capacity K≈700")

    plt.title("MPC Simulation with Forward-looking Target (1982–2050 Projection)")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/fig_mpc_dynamic_target_projection.png", dpi=300)
    plt.show()

    print("[INFO] Figure saved to outputs/fig_mpc_dynamic_target_projection.png")

if __name__ == "__main__":
    run_mpc_dynamic_target()
