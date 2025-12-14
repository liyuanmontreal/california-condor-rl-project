How to Run

## Phase 1: Real Data Analysis

```bash
python -m src.analysis.real_data_analysis 
python -m src.analysis.growth_rate_estimator
python src/tools/recalibrate_age3_from_real_data.py



```
## phase 2: Policy + Enviorment Comparison
```bash

python src/analysis/run_phase2_policy_comparison.py

```
## phase 3 : training RL
```bash
python src/analysis/run_phase3_rl.py
```
## Plot
```bash
python src/analysis/plot_phase3_rl_results.py
python src/analysis/plot_phase3_rl_age_trajectories.py
python src/analysis/plot_phase3_rl_best_policy_actions.py
python src/analysis/plot_phase3_rl_policy_phaseportrait.py
python src/analysis/plot_phase3_rl_policy_phaseportrait_kde.py
python src/analysis/plot_phase3_rl_policy_phaseportrait_dual.py
python src/analysis/plot_phase3_rl_policy_phaseportrait_hexbin.py
python src/analysis/plot_phase3_rl_policy_heatmap.py
python src/analysis/plot_phase3_policy_heatmap_stategrid.py
```

## mpc
```bash
python src/analysis/run_mpc_condor_env_age3.py
```

project 

    data
        -condor_real_data.csv
    figs
    outputs
    report
    src
        analysis
        config
        data
        env 
        -utils.py
    
    

