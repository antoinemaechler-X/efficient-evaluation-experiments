"""
WOR Validation Analyzer: Find best FAQ hyperparameters per (dataset, budget).

Reads all wor_faq_val_*.csv files, averages across seeds,
picks the setting with narrowest mean_width per (dataset, prop_budget).

Outputs: logs/val/wor_best_settings.csv
"""
import numpy as np
import pandas as pd
import os, glob

# Load all WOR validation log files
val_files = glob.glob("logs/val/wor_faq_val_*.csv")
logs = pd.concat([pd.read_csv(f) for f in val_files], ignore_index=True)

print(f"Loaded {len(logs)} rows from {len(val_files)} files.")

# The first 8 columns define a hyperparameter setting (before seed)
setting_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget",
                "beta0", "rho", "gamma", "tau"]

# Average results across seeds, sort by mean_width (narrowest first)
mean_logs = logs.groupby(setting_cols, dropna=False).mean().reset_index()
mean_logs = mean_logs.sort_values(by="mean_width")

# Get best setting per (dataset, prop_budget) — fully-observed only, so no missingness grouping needed
scenario_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"]
best_settings = mean_logs.groupby(scenario_cols, dropna=False).first().reset_index()

# Save
best_settings.to_csv("logs/val/wor_best_settings.csv", index=False)
print(f"Saved best settings: {len(best_settings)} rows to logs/val/wor_best_settings.csv")
print(best_settings[["dataset", "prop_budget", "beta0", "rho", "gamma", "tau", "mean_width", "coverage"]].to_string())
