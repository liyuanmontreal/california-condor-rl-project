# Offline Reinforcement Learning for California Condor Conservation

This project studies long-term population management of the **California condor** using **offline reinforcement learning (RL)**. We formulate conservation as a sequential decision-making problem under uncertainty and compare RL-derived policies against historical, literature-based, and model predictive control (MPC) baselines in a calibrated ecological simulation.

The goal is **not population maximization**, but **long-term stabilization and cost-efficient management** under stochastic disturbances such as lead poisoning and rare disaster events.

---

## Project Overview

* **Species:** California condor (*Gymnogyps californianus*)
* **Problem:** Long-horizon ecological management under uncertainty
* **Approach:** Offline RL (Fitted Q-Iteration) on a calibrated age-structured population model
* **Baselines:**

  * Zero-intervention
  * Literature-inspired heuristic policy
  * Calibrated real-history replay
  * Short-horizon Model Predictive Control (MPC)

This project was developed as a final project for **IFT6162 – Reinforcement Learning**.

---

## Key Contributions

* A **calibrated, age-structured condor population environment** incorporating:

  * Survival and age transitions
  * Reproduction dependent on habitat quality
  * Lead-poisoning mortality and mitigation
  * Habitat degradation and recovery
  * Stochastic disaster events

* An **offline RL pipeline** using Fitted Q-Iteration (FQI) with Random Forest approximation

* A careful comparison between **adaptive RL policies** and **fixed, heuristic management strategies**

* Policy interpretation through **trajectory plots, heatmaps, and phase portraits**

---

## Environment Description

The environment models wild and captive condor populations with explicit age structure:

* **State variables:**

  * Juvenile, subadult, and adult wild populations
  * Captive population size
  * Habitat quality

* **Actions:**

  * Release of captive birds into the wild
  * Mitigation effort against lead poisoning

* **Dynamics include:**

  * Age-specific survival and transitions
  * Adult reproduction modulated by habitat
  * Logistic growth of the captive population
  * Lead-poisoning mortality reduced by mitigation
  * Habitat degradation and recovery
  * Rare, stochastic disaster events

Model parameters are calibrated using **published ecological studies** and **historical recovery trends** to reflect realistic long-term difficulty rather than optimistic growth.

---

## Reinforcement Learning Method

We adopt an **offline reinforcement learning** setting, motivated by the impracticality and ethical constraints of online exploration in real conservation systems.

* **Dataset generation:**

  * Random management policies simulated over 100-year horizons
  * Approximately 30,000 state–action transitions
  * Coverage of both typical dynamics and rare disaster events

* **Algorithm:**

  * Fitted Q-Iteration (FQI)
  * Random Forest regression as function approximator

* **Policy extraction:**

  * Greedy maximization of the learned Q-function over a discretized action space

FQI is chosen for its robustness to stochasticity, suitability for long-horizon problems, and interpretability.

---

## Baselines

### Zero-Intervention Policy

No releases or mitigation. Serves as a sanity check and consistently leads to population collapse.

### Literature Policy

A fixed, high-intensity heuristic inspired by conservation literature:

```python
release = 20
mitigation = 0.7
```

This policy assumes sustained, heavy intervention and serves as an **upper-bound reference**, not a cost-aware strategy.

### Calibrated Real-History Policy

Replays historically observed release numbers and then freezes intervention intensity:

* Uses real release data during historical years
* Afterward, applies the average of the last 3 historical years
* No mitigation

This baseline validates model realism but does not represent an optimized policy.

### Model Predictive Control (MPC)

A short-horizon, model-based baseline:

* Sample-based **single-shooting MPC**
* Deterministic planning, stochastic execution
* Receding-horizon re-planning

MPC stabilizes the population but relies on frequent, reactive interventions.

---

## Results Summary

* Zero-intervention policies fail to sustain the population
* Literature policy maintains higher populations via persistent, heavy intervention
* RL policy achieves **long-term stabilization and robustness** under stochastic dynamics
* Learned policies are:

  * Cost-sensitive
  * Multi-modal
  * Selective rather than aggressive

The RL policy prioritizes **sustainability and efficiency**, not population maximization.

---

## Interpretation and Insights

* Weak state dependence reflects a **net-growth ecological regime**, not RL failure
* Rare disasters dominate long-term variability; robustness is more important than maximizing the mean
* RL learns **when not to intervene**, exploiting intrinsic ecological recovery
* MPC highlights the contrast between short-horizon reactivity and long-horizon optimization

---

## Limitations and Future Work

* Simplified modeling of lead exposure
* No spatial structure
* Discontinuous policies induced by Random Forest approximation

Future directions include spatially explicit models, smoother function approximators, and uncertainty-aware planning.

---

## Repository Structure 
```text

data/                 # condor real data
figs/                 # real data ananlysis figure
outputs/              # plot ouput
report/               # project report and poster
src/
  env/                # Condor environment
  config/             # mechanisitic/calibrate config file
  analysis/           #  Offline RL (FQI), mpc , Evaluation and plotting

```

---

## How to Run

1. Set up a Python environment (Python ≥ 3.9 recommended)
2. Install dependencies (NumPy, SciPy, scikit-learn, matplotlib)
3. Run environment sanity checks and baselines
real data analysis
 ```bash
python -m src.analysis.real_data_analysis 
python -m src.analysis.growth_rate_estimator
python src/tools/recalibrate_age3_from_real_data.py
```
sanity check, policy + enviorment comparison
```bash
python src/analysis/run_phase2_policy_comparison.py
```
4. Train RL policy
```bash
python src/analysis/train_phase3_rl.py
```
5. Train mpc policy
```bash
python src/analysis/run_mpc_condor_env_age3.py
```
6. Evaluate and visualize results
```bash
python src/plot/plot_phase3_rl_results.py
python src/plot/plot_phase3_rl_age_trajectories.py
python src/plot/plot_phase3_rl_best_policy_actions.py
python src/plot/plot_phase3_rl_policy_phaseportrait.py
python src/plot/plot_phase3_rl_policy_phaseportrait_dual.py
python src/plot/plot_phase3_rl_policy_phaseportrait_hexbin.py
python src/plot/plot_phase3_rl_policy_heatmap.py
python src/plot/plot_phase3_policy_heatmap_stategrid.py
```

---

## Course Context

This project was completed as part of **IFT6162 – Reinforcement Learning** and emphasizes:

* Problem formulation
* Modeling assumptions
* Robust evaluation
* Interpretability over raw performance



    

