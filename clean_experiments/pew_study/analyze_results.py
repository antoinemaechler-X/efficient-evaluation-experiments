"""
Analyze Pew ATP Wave 79 study results.

Loads WOR, Bernoulli, and Active Inference CSV files, computes ESS multipliers
relative to WOR-uniform and Classical, and builds a summary table.

Output: results/summary.csv
"""
import numpy as np
import pandas as pd
import os
import glob

# ====================================================================
# PART 1: Load all results
# ====================================================================
# Bernoulli is the only complete run; WOR and Active Inference are pending and
# get picked up automatically once their CSVs land in results/.
bernoulli_files = glob.glob("results/bernoulli_sl=*.csv")
wor_files = glob.glob("results/wor_sl=*.csv")
active_inf_files = glob.glob("results/active_inference_sl=*.csv")

assert bernoulli_files, "No Bernoulli result files found"

dfs = [pd.concat([pd.read_csv(f) for f in bernoulli_files], ignore_index=True)]
if wor_files:
    dfs.append(pd.concat([pd.read_csv(f) for f in wor_files], ignore_index=True))
if active_inf_files:
    dfs.append(pd.concat([pd.read_csv(f) for f in active_inf_files], ignore_index=True))

all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df.sort_values(by=["group", "prop_budget", "method", "seed"]).reset_index(drop=True)

print(f"Loaded {len(all_df)} rows from {sorted(all_df['method'].unique())}")
print(f"Budgets: {sorted(all_df['prop_budget'].unique())}")

# ====================================================================
# PART 2: Compute ESS multipliers relative to TWO baselines
# ====================================================================
merge_cols = ["group", "prop_budget", "seed"]

# Classical baseline (always available from the Bernoulli run).
classical = all_df.query("method == 'classical'")[merge_cols + ["width"]].copy()
classical = classical.rename(columns={"width": "width_ref_classical"})
merged = pd.merge(all_df, classical, on=merge_cols, how="left")
assert merged["width_ref_classical"].notna().all(), "Missing Classical reference for some seeds"
all_df["ess_multiplier_vs_classical"] = (merged["width_ref_classical"] / merged["width"]) ** 2

# WOR-uniform baseline only exists once WOR has run; otherwise leave it as NaN.
if "wor-uniform" in all_df["method"].unique():
    wor_unif = all_df.query("method == 'wor-uniform'")[merge_cols + ["width"]].copy()
    wor_unif = wor_unif.rename(columns={"width": "width_ref_wor"})
    merged_wor = pd.merge(all_df, wor_unif, on=merge_cols, how="left")
    all_df["ess_multiplier"] = (merged_wor["width_ref_wor"] / merged_wor["width"]) ** 2
else:
    all_df["ess_multiplier"] = np.nan

# ====================================================================
# PART 3: Build summary
# ====================================================================
def build_summary(df, group_cols):
    summary = df.groupby(group_cols, dropna=False).agg(
        ess_multiplier=("ess_multiplier", "mean"),
        ess_multiplier_serr=("ess_multiplier", lambda x: x.std() / np.sqrt(len(x))),
        ess_multiplier_vs_classical=("ess_multiplier_vs_classical", "mean"),
        ess_multiplier_vs_classical_serr=("ess_multiplier_vs_classical", lambda x: x.std() / np.sqrt(len(x))),
        coverage=("coverage", "mean"),
        coverage_serr=("coverage", lambda x: x.std() / np.sqrt(len(x))),
        mean_width=("width", "mean"),
        mean_width_serr=("width", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()
    return summary

summary = build_summary(all_df, ["group", "prop_budget", "method"])

# ====================================================================
# PART 4: Save
# ====================================================================
os.makedirs("results", exist_ok=True)

all_df.to_csv("results/all_per_seed.csv", index=False)
summary.to_csv("results/summary.csv", index=False)

print(f"\nSaved per-seed results: results/all_per_seed.csv")
print(f"Saved summary: results/summary.csv")

# ====================================================================
# PART 5: Print summary table
# ====================================================================
for group in sorted(summary["group"].unique()):
    print(f"\n{'='*80}")
    print(f"Group: {group}")
    print(f"{'='*80}")
    g = summary.query(f"group == '{group}'")
    for method in sorted(g["method"].unique()):
        m = g.query(f"method == '{method}'")
        avg_ess = m["ess_multiplier_vs_classical"].mean()
        avg_cov = m["coverage"].mean()
        print(f"  {method:25s}: avg ESS vs Classical = {avg_ess:.3f}, avg coverage = {avg_cov:.3f}")
