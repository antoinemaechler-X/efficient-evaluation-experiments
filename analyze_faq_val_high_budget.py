'''
Task 1 Analysis: Pick best FAQ hyperparameters from the 25%-50% validation logs.
Includes all missingness settings (ms=0 and ms=1).
Compares with original 2.5%-25% best settings.

Reads individual CSV logs from logs/val/, concatenates, and selects best settings.
Saves to logs/val/best_settings_high_budget.csv

Usage: python analyze_faq_val_high_budget.py
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# ============================================================
# PART 1: Load and analyze high-budget (25%-50%) validation logs
# ============================================================
DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
GAMMA_VALS = [0.0, 0.05, 0.25, 0.5, 0.75]

dfs = []
for dataset in DATASETS:
    for gamma in GAMMA_VALS:
        for ms in [0, 1]:
            fname = f"logs/val/dataset={dataset}_gamma={gamma}_ms={ms}.csv"
            if os.path.exists(fname):
                df = pd.read_csv(fname)
                dfs.append(df)
                print(f"  Loaded {fname}: {len(df)} rows")
            else:
                print(f"  MISSING: {fname}")

logs = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows loaded: {len(logs)}")

# filter to 25%-50% budgets only
high_budget = logs[logs["prop_budget"] > 0.25].copy()
print(f"After filtering to >25% budgets: {len(high_budget)}")
print(f"Unique budgets: {sorted(high_budget['prop_budget'].unique())}")
print(f"Unique missingness settings: {sorted(high_budget['mcar_obs_prob'].unique())}")

# only use combos with all 5 seeds complete
group_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "beta0", "rho", "gamma", "tau"]
seed_counts = high_budget.groupby(group_cols, dropna=False).size().reset_index(name="n_seeds")
complete = seed_counts[seed_counts["n_seeds"] == 5]
high_complete = pd.merge(high_budget, complete[group_cols], on=group_cols, how="inner")

# average across 5 seeds
mean_logs = high_complete.groupby(group_cols, dropna=False).mean().reset_index().sort_values(by="mean_width")

# best settings per (dataset, n_full_obs, mcar_obs_prob, budget)
scenario_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"]
best_high = mean_logs.groupby(scenario_cols, dropna=False).first().reset_index()

# check completeness
expected_budgets = set(np.round(np.linspace(0.25, 0.5, 11)[1:], decimals=3))
missingness_settings = best_high.groupby(["n_full_obs", "mcar_obs_prob"]).size().reset_index(name="count")
print(f"\nMissingness settings found:")
for _, row in missingness_settings.iterrows():
    print(f"  n_full_obs={row['n_full_obs']}, mcar_obs_prob={row['mcar_obs_prob']}: {row['count']} rows")

for dataset in DATASETS:
    sub = best_high[best_high["dataset"] == dataset]
    fully_obs = sub[sub["mcar_obs_prob"] == 1.0]
    other = sub[sub["mcar_obs_prob"] != 1.0]
    found_fo = set(fully_obs["prop_budget"].values)
    missing_fo = expected_budgets - found_fo
    if missing_fo:
        print(f"\n  WARNING: {dataset} fully-obs missing budgets: {sorted(missing_fo)}")
    else:
        print(f"\n  {dataset}: fully-obs all 10 budgets complete")
    print(f"  {dataset}: {len(other)} non-fully-obs rows")

# save
best_high.to_csv("logs/val/best_settings_high_budget.csv", index=False)
print(f"\nSaved: logs/val/best_settings_high_budget.csv ({len(best_high)} rows)")

# ============================================================
# PART 2: Load original 2.5%-25% best settings for comparison
# ============================================================
best_low = pd.read_csv("logs/val/best_settings.csv")
# filter to fully-observed only for the comparison plots
best_low_fo = best_low[best_low["mcar_obs_prob"] == 1.0].copy()

# ============================================================
# PART 3: Combined display (fully-observed)
# ============================================================
best_high_fo = best_high[best_high["mcar_obs_prob"] == 1.0].copy()

print(f"\n{'='*90}")
print("BEST FAQ HYPERPARAMETERS: LOW (2.5%-25%) vs HIGH (25%-50%) BUDGETS [fully-observed]")
print(f"{'='*90}")

for dataset in DATASETS:
    print(f"\n--- {dataset} ---")
    print(f"  {'budget':>7}  {'beta0':>5}  {'rho':>5}  {'gamma':>5}  {'tau':>5}  {'val_width':>12}  {'coverage':>10}")
    print(f"  {'-'*70}")

    low = best_low_fo[best_low_fo["dataset"] == dataset].sort_values("prop_budget")
    high = best_high_fo[best_high_fo["dataset"] == dataset].sort_values("prop_budget")

    print(f"  Low budgets (2.5%-25%):")
    for _, row in low.iterrows():
        print(f"  {row['prop_budget']:>7.3f}  {row['beta0']:>5.2f}  {row['rho']:>5.2f}  "
              f"{row['gamma']:>5.2f}  {row['tau']:>5.2f}  {row['mean_width']:>12.6f}  "
              f"{row['coverage_full']:>10.4f}")

    print(f"\n  High budgets (25%-50%):")
    for _, row in high.iterrows():
        print(f"  {row['prop_budget']:>7.3f}  {row['beta0']:>5.2f}  {row['rho']:>5.2f}  "
              f"{row['gamma']:>5.2f}  {row['tau']:>5.2f}  {row['mean_width']:>12.6f}  "
              f"{row['coverage_full']:>10.4f}")

# ============================================================
# PART 4: Plots (fully-observed comparison)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, dataset in enumerate(DATASETS):
    ax = axes[i]

    low = best_low_fo[best_low_fo["dataset"] == dataset].sort_values("prop_budget")
    high = best_high_fo[best_high_fo["dataset"] == dataset].sort_values("prop_budget")

    ax.plot(low["prop_budget"], low["mean_width"], "o-", color="steelblue", label="2.5%-25%")
    ax.plot(high["prop_budget"], high["mean_width"], "s-", color="darkorange", label="25%-50%")
    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("Validation Mean CI Width", fontsize=12)
    ax.set_title(dataset, fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("FAQ Best Validation CI Width Across Budget Ranges", fontsize=14)
plt.tight_layout()
plt.savefig("figures/task1_width_vs_budget.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: figures/task1_width_vs_budget.png")

# Plot hyperparameter trends
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
param_names = ["beta0", "rho", "gamma", "tau"]

for idx, param in enumerate(param_names):
    ax = axes[idx // 2, idx % 2]
    for dataset, color in zip(DATASETS, ["steelblue", "darkorange"]):
        low = best_low_fo[best_low_fo["dataset"] == dataset].sort_values("prop_budget")
        high = best_high_fo[best_high_fo["dataset"] == dataset].sort_values("prop_budget")
        all_budgets = pd.concat([low, high]).sort_values("prop_budget")
        ax.plot(all_budgets["prop_budget"], all_budgets[param], "o-", color=color,
                label=dataset.split("+")[0], markersize=5)
    ax.set_xlabel("Budget Proportion")
    ax.set_ylabel(param)
    ax.set_title(f"Best {param} vs Budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("How Best Hyperparameters Change with Budget", fontsize=14)
plt.tight_layout()
plt.savefig("figures/task1_hyperparam_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figures/task1_hyperparam_trends.png")
