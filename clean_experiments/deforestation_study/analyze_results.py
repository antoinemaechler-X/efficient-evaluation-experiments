"""
Analyze Amazon deforestation study results.

Loads WOR, Bernoulli, Cross-PPI, and Active Inference CSV files, computes ESS
multipliers relative to WOR-uniform and Classical, and builds a summary table.

Output: results/summary.csv
"""
import numpy as np
import pandas as pd
import os
import sys
import glob

# ====================================================================
# PART 1: Load whatever results exist
# ====================================================================
# All four experiments are pending; each is picked up here once its CSVs appear.
patterns = ["wor_sl=*.csv", "bernoulli_sl=*.csv", "cross_ppi_sl=*.csv", "active_inference_sl=*.csv"]
files = [f for p in patterns for f in glob.glob(f"results/{p}")]

if not files:
    print("No result CSVs in results/ yet. Run the experiments first (see README).")
    sys.exit(0)

all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
all_df = all_df.sort_values(by=["group", "prop_budget", "method", "seed"]).reset_index(drop=True)

print(f"Loaded {len(all_df)} rows from {sorted(all_df['method'].unique())}")
print(f"Budgets: {sorted(all_df['prop_budget'].unique())}")

# ====================================================================
# PART 2: Compute ESS multipliers
# ====================================================================
merge_cols = ["group", "prop_budget", "seed"]

# Classical baseline (needs run_bernoulli.py).
if "classical" in all_df["method"].unique():
    classical = all_df.query("method == 'classical'")[merge_cols + ["width"]].rename(
        columns={"width": "width_ref_classical"})
    merged = pd.merge(all_df, classical, on=merge_cols, how="left")
    all_df["ess_multiplier_vs_classical"] = (merged["width_ref_classical"] / merged["width"]) ** 2
else:
    all_df["ess_multiplier_vs_classical"] = np.nan

# WOR-uniform baseline (needs run_wor.py).
if "wor-uniform" in all_df["method"].unique():
    wor_unif = all_df.query("method == 'wor-uniform'")[merge_cols + ["width"]].rename(
        columns={"width": "width_ref_wor"})
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
