"""
Analyze AlphaFold study results.

Loads WOR and Bernoulli CSV files, computes ESS multipliers
relative to WOR-uniform, and builds a summary table.

Output: alphafold_study/logs/cleaned/summary.csv
"""
import numpy as np
import pandas as pd
import os
import glob

# ====================================================================
# PART 1: Load all results
# ====================================================================
wor_files = glob.glob("alphafold_study/logs/wor_sl=*.csv")
bernoulli_files = glob.glob("alphafold_study/logs/bernoulli_sl=*.csv")

assert wor_files, "No WOR result files found"
assert bernoulli_files, "No Bernoulli result files found"

wor_df = pd.concat([pd.read_csv(f) for f in wor_files], ignore_index=True)
bern_df = pd.concat([pd.read_csv(f) for f in bernoulli_files], ignore_index=True)

all_df = pd.concat([wor_df, bern_df], ignore_index=True)
all_df = all_df.sort_values(by=["group", "prop_budget", "method", "seed"]).reset_index(drop=True)

print(f"WOR rows: {len(wor_df)} from {len(wor_files)} files")
print(f"Bernoulli rows: {len(bern_df)} from {len(bernoulli_files)} files")
print(f"Total rows: {len(all_df)}")
print(f"Methods: {sorted(all_df['method'].unique())}")
print(f"Groups: {sorted(all_df['group'].unique())}")
print(f"Budgets: {sorted(all_df['prop_budget'].unique())}")

# ====================================================================
# PART 2: Compute ESS multiplier relative to WOR-uniform
# ====================================================================
merge_cols = ["group", "prop_budget", "seed"]

# Get WOR-uniform widths as reference
wor_unif = all_df.query("method == 'wor-uniform'")[merge_cols + ["width"]].copy()
wor_unif = wor_unif.rename(columns={"width": "width_ref"})

# Merge reference width into all rows
merged = pd.merge(all_df, wor_unif, on=merge_cols, how="left")
assert merged["width_ref"].notna().all(), "Missing WOR-uniform reference for some seeds"

merged["ess_multiplier"] = (merged["width_ref"] / merged["width"]) ** 2
all_df["ess_multiplier"] = merged["ess_multiplier"].values

# ====================================================================
# PART 3: Build summary
# ====================================================================
def build_summary(df, group_cols):
    summary = df.groupby(group_cols, dropna=False).agg(
        ess_multiplier=("ess_multiplier", "mean"),
        ess_multiplier_serr=("ess_multiplier", lambda x: x.std() / np.sqrt(len(x))),
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
os.makedirs("alphafold_study/logs/cleaned", exist_ok=True)

all_df.to_csv("alphafold_study/logs/cleaned/all_per_seed.csv", index=False)
summary.to_csv("alphafold_study/logs/cleaned/summary.csv", index=False)

print(f"\nSaved per-seed results: alphafold_study/logs/cleaned/all_per_seed.csv")
print(f"Saved summary: alphafold_study/logs/cleaned/summary.csv")

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
        avg_ess = m["ess_multiplier"].mean()
        avg_cov = m["coverage"].mean()
        print(f"  {method:25s}: avg ESS mult = {avg_ess:.3f}, avg coverage = {avg_cov:.3f}")
