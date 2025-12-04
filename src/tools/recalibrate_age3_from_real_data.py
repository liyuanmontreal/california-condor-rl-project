"""
Recalibrate Age-3 CondorEnv parameters from real population data
-----------------------------------------------------------------

This script reads real historical condor data (CSV),
fits Age-3 demographic parameters, and outputs a complete
config_env_calibrated.yaml usable by CondorEnvAge3.

"""

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit


# ------------------------------
# 1. Load real data
# ------------------------------
def load_real_data(path):
    df = pd.read_csv(path)
    df = df.sort_values("Year").reset_index(drop=True)
    return df


# ------------------------------
# 2. Fit survival rates
# ------------------------------
def fit_survival(df):
    """
    Estimate juvenile, subadult, adult survival from
    year-to-year total change and age ratios.
    """

    # If age structure is unknown, assume literature ratios:
    # 20% juvenile, 20% subadult, 60% adult
    ratio_j = 0.20
    ratio_s = 0.20
    ratio_a = 0.60

    surv_estimates = []

    for i in range(len(df) - 1):
        N_t = df.loc[i, "Total"]
        N_t1 = df.loc[i + 1, "Total"]

        release = df.loc[i, "Released_to_Wild"]

        # expected additions from reproduction (rough approximation)
        births = 0.2 * (N_t * ratio_a)

        # observed survivors:
        survivors = N_t1 - release - births
        if survivors <= 0:
            continue

        survival_rate = survivors / N_t
        surv_estimates.append(survival_rate)

    surv_mean = np.mean(surv_estimates)

    # split aggregated survival into age classes
    sj = surv_mean * 0.95
    ss = surv_mean * 1.05
    sa = surv_mean * 1.15

    return float(sj), float(ss), float(sa)


# ------------------------------
# 3. Fit fecundity
# ------------------------------
def fit_fecundity(df):
    """
    Estimate fecundity from delta population and inferred adult numbers.
    """
    fecundity_list = []

    for i in range(len(df) - 1):
        N_t = df.loc[i, "Total"]
        N_t1 = df.loc[i + 1, "Total"]
        delta = N_t1 - N_t

        A_t = N_t * 0.60  # approx adult proportion

        births_est = delta * 0.4  # rough

        if A_t > 0:
            fec = births_est / A_t
            fecundity_list.append(fec)

    fec_mean = max(0.01, np.mean(fecundity_list))
    return float(fec_mean)


# ------------------------------
# 4. Fit disaster probability
# ------------------------------
def fit_disaster_probability(df, threshold_drop=0.25):
    """
    Detect years with sharp drops and estimate disaster probability.
    """
    crash_years = 0
    total_years = len(df) - 1

    for i in range(len(df) - 1):
        N_t = df.loc[i, "Total"]
        N_t1 = df.loc[i + 1, "Total"]

        if (N_t1 < N_t * (1 - threshold_drop)):
            crash_years += 1

    p = crash_years / total_years
    return float(p)


# ------------------------------
# 5. Write calibrated YAML
# ------------------------------
def write_yaml(path, params):
    cfg = {
        "env": params,
        "reward": {
            "target_low": 650,
            "target_high": 700,
            "w_low": 20,
            "w_high": 1,
            "lambda_release": 0.2,
            "lambda_mitigation": 0.1,
            "lambda_disaster": 20.0
        },
        "sim": {
            "seed": 0,
            "horizon": 120,
            "N_rollouts": 20
        },
        "actions": {
            "release_min": 0,
            "release_max": 20,
            "release_step": 1,
            "mitigation_min": 0.0,
            "mitigation_max": 1.0,
            "mitigation_step": 0.1
        },
        "train": {
            "gamma": 0.99,
            "n_iterations": 25,
            "n_episodes": 300,
            "horizon": 100,
            "random_seed": 0,
            "save_model_path": "trained_fqi_age3.npy"
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"[Saved calibrated config] {path}")


# ------------------------------
# 6. Main calibration function
# ------------------------------
def recalibrate(csv_path, output_yaml):

    df = load_real_data(csv_path)

    sj, ss, sa = fit_survival(df)
    fec = fit_fecundity(df)
    disaster_prob = fit_disaster_probability(df)

    params = {
        "K": 800,
        "Nj0": 74,
        "Ns0": 74,
        "Na0": 221,
        "C0": 197,
        "H0": 1.0,

        "sj_base": sj,
        "ss_base": ss,
        "sa_base": sa,

        "fecundity_per_adult": fec,
        "habitat_repro_sensitivity": 0.2,

        "lead_j_base": 0.06,
        "lead_s_base": 0.12,
        "lead_a_base": 0.05,
        "lead_mitigation_effect": 0.3,
        "habitat_lead_sensitivity": 0.8,

        "disaster_base_prob": disaster_prob,
        "disaster_mortality_frac": 0.40,
        "disaster_habitat_sensitivity": 0.8,

        "release_base_survival": 0.55,
        "release_density_sensitivity": 0.2,

        "captive_recruitment_rate": 0.05,
        "C_cap": 400,

        "habitat_baseline": 1.0,
        "habitat_recovery_rate": 0.03,
        "habitat_mitigation_effect": 0.08,
        "habitat_min": 0.3,
        "habitat_max": 1.3
    }

    write_yaml(output_yaml, params)



# ------------------------------
# Run example
# ------------------------------
if __name__ == "__main__":
    recalibrate(
        csv_path="data/condor_real_data.csv",
        output_yaml="src/config/config_env_calibrated_auto.yaml"
    )
