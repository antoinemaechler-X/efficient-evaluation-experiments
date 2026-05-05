import numpy as np
import pandas as pd
import os
import glob

# PART 1: Load all results
wor_files = glob.glob("logs/wor_sl=*.csv")
bernoulli_files = glob.glob("logs/bernoulli_sl=*.csv")

wor_df = pd.concat([pd.read_csv(f) for f in wor_files], ignore_index=True)
bern_df = pd.concat([pd.read_csv(f) for f in bernoulli_files], ignore_index=True)

all_df = pd.concat([wor_df, bern_df], ignore_index=True)
all_df = all_df.sort_values(["group", "prop_budget", "method", "seed"]).reset_index(drop=True)

print(f"WOR: {len(wor_df)} rows | Bernoulli: {len(bern_df)} rows | Total: {len(all_df)}")
print(f"Methods: {sorted(all_df['method'].unique())}")

# PART 2: Compute ESS multipliers relative to two baselines
merge_cols = ["group", "prop_budget", "seed"]

wor_unif = all_df.query("method == 'wor-uniform'")[merge_cols + ["width"]].rename(
    columns={"width": "width_ref_wor"})
classical = all_df.query("method == 'classical'")[merge_cols + ["width"]].rename(
    columns={"width": "width_ref_classical"})

merged = pd.merge(all_df, wor_unif, on=merge_cols, how="left")
merged = pd.merge(merged, classical, on=merge_cols, how="left")

all_df["ess_multiplier"] = (merged["width_ref_wor"] / merged["width"]) ** 2
all_df["ess_multiplier_vs_classical"] = (merged["width_ref_classical"] / merged["width"]) ** 2

# PART 3: Build summary
def build_summary(df, group_cols):
    return df.groupby(group_cols, dropna=False).agg(
        ess_multiplier=("ess_multiplier", "mean"),
        ess_multiplier_serr=("ess_multiplier", lambda x: x.std() / np.sqrt(len(x))),
        ess_multiplier_vs_classical=("ess_multiplier_vs_classical", "mean"),
        ess_multiplier_vs_classical_serr=("ess_multiplier_vs_classical", lambda x: x.std() / np.sqrt(len(x))),
        coverage=("coverage", "mean"),
        coverage_serr=("coverage", lambda x: x.std() / np.sqrt(len(x))),
        mean_width=("width", "mean"),
        mean_width_serr=("width", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

summary = build_summary(all_df, ["group", "prop_budget", "method"])

# PART 4: Save
os.makedirs("logs/cleaned", exist_ok=True)
all_df.to_csv("logs/cleaned/all_per_seed.csv", index=False)
summary.to_csv("logs/cleaned/summary.csv", index=False)
print("Saved logs/cleaned/all_per_seed.csv and logs/cleaned/summary.csv")

for group in sorted(summary["group"].unique()):
    print(f"\n{group}")
    g = summary.query(f"group == '{group}'")
    for method in sorted(g["method"].unique()):
        m = g.query(f"method == '{method}'")
        print(f"  {method:25s}: avg ESS×={m['ess_multiplier'].mean():.3f}, "
              f"avg cov={m['coverage'].mean():.3f}")
