"""
WOR Cleaning Results: Combine WOR-FAQ results with EXISTING baseline/ablation results.

The main comparison is WOR-FAQ vs. the original (with-replacement) baselines and ablation.
WOR baselines/ablation are saved separately as a curiosity.

Reads:
    - logs/final/wor_faq_final_sl=*.csv (WOR FAQ results)
    - logs/final/cleaned/best_baseline_df.csv (EXISTING best baselines, from original cleaning_results.py)
    - logs/final/cleaned/best_ablation_df.csv (EXISTING best ablation)
    - logs/final/cleaned/uniform_df.csv (EXISTING uniform)
    - (optional) logs/final/wor_baselines_dataset=*.csv, wor_ablation_dataset=*.csv (WOR curiosity)

Outputs to logs/final/cleaned/:
    - wor_faq_df.csv, wor_faq_summary.csv (WOR FAQ per-seed + summary)
    - wor_vs_orig_*.csv (WOR FAQ compared against original baselines)
"""
import numpy as np
import pandas as pd
import os, glob

N_SEEDS = 100

# ====================================================================
# PART 1: Load WOR FAQ results
# ====================================================================
faq_files = glob.glob("logs/final/wor_faq_final_sl=*.csv")
wor_faq_df = pd.concat([pd.read_csv(f) for f in faq_files], ignore_index=True)
wor_faq_df = wor_faq_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)
print(f"WOR FAQ: {len(wor_faq_df)} rows from {len(faq_files)} files")

# ====================================================================
# PART 2: Load EXISTING (original, with-replacement) baseline results
# ====================================================================
# These were already computed by the original pipeline (cleaning_results.py)
uniform_df = pd.read_csv("logs/final/cleaned/uniform_df.csv")
best_baseline_df = pd.read_csv("logs/final/cleaned/best_baseline_df.csv")
best_ablation_df = pd.read_csv("logs/final/cleaned/best_ablation_df.csv")

# Filter to fully-observed only (mcar_obs_prob == 1.0)
uniform_df = uniform_df.query("mcar_obs_prob == 1.0" if "mcar_obs_prob" in uniform_df.columns
                               else "True").copy()
best_baseline_df = best_baseline_df.query("mcar_obs_prob == 1.0").copy()
best_ablation_df = best_ablation_df.query("mcar_obs_prob == 1.0").copy()

uniform_df = uniform_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)
best_baseline_df = best_baseline_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)
best_ablation_df = best_ablation_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)

print(f"Original uniform: {len(uniform_df)} rows")
print(f"Original best baseline: {len(best_baseline_df)} rows")
print(f"Original best ablation: {len(best_ablation_df)} rows")

# ====================================================================
# PART 3: Compute ESS multipliers for WOR FAQ vs. ORIGINAL uniform
# ====================================================================
merge_cols = ["dataset", "prop_budget", "seed"]

# WOR FAQ ESS multiplier (relative to original uniform)
faq_merged = pd.merge(wor_faq_df, uniform_df[merge_cols + ["mean_width"]],
                       on=merge_cols, suffixes=("", "_unif"))
wor_faq_df["ess_multiplier"] = (faq_merged["mean_width_unif"] / faq_merged["mean_width"]) ** 2

# Original baselines already have ess_multiplier from cleaning_results.py
# but let's recompute to be safe
if "ess_multiplier" not in best_baseline_df.columns:
    bl_merged = pd.merge(best_baseline_df, uniform_df[merge_cols + ["mean_width"]],
                          on=merge_cols, suffixes=("", "_unif"))
    best_baseline_df["ess_multiplier"] = (bl_merged["mean_width_unif"] / bl_merged["mean_width"]) ** 2

if "ess_multiplier" not in best_ablation_df.columns:
    abl_merged = pd.merge(best_ablation_df, uniform_df[merge_cols + ["mean_width"]],
                           on=merge_cols, suffixes=("", "_unif"))
    best_ablation_df["ess_multiplier"] = (abl_merged["mean_width_unif"] / abl_merged["mean_width"]) ** 2

if "ess_multiplier" not in uniform_df.columns:
    uniform_df["ess_multiplier"] = 1.0

# ====================================================================
# PART 4: Build summaries
# ====================================================================
def build_summary(df, group_cols):
    summary = df.groupby(group_cols, dropna=False).agg(
        ess_multiplier=("ess_multiplier", "mean"),
        ess_multiplier_serr=("ess_multiplier", lambda x: x.std() / np.sqrt(len(x))),
        coverage=("coverage", "mean"),
        coverage_serr=("coverage", lambda x: x.std() / np.sqrt(len(x))),
        mean_width=("mean_width", "mean"),
        mean_width_serr=("mean_width", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()
    return summary

scenario_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"]
scenario_cols_unif = ["dataset", "prop_budget"]

wor_faq_summary = build_summary(wor_faq_df, scenario_cols)

# Use existing summaries for baselines, or rebuild from filtered data
orig_baseline_summary = build_summary(best_baseline_df, scenario_cols)
orig_ablation_summary = build_summary(best_ablation_df, scenario_cols)
orig_uniform_summary = build_summary(uniform_df, scenario_cols_unif)

# ====================================================================
# PART 5: Save everything
# ====================================================================
os.makedirs("logs/final/cleaned", exist_ok=True)

# WOR FAQ per-seed + summary
wor_faq_df.to_csv("logs/final/cleaned/wor_faq_df.csv", index=False)
wor_faq_summary.to_csv("logs/final/cleaned/wor_faq_summary.csv", index=False)

# Original baselines/ablation/uniform filtered to fully-observed (for plotting convenience)
orig_baseline_summary.to_csv("logs/final/cleaned/wor_vs_orig_best_baseline_summary.csv", index=False)
orig_ablation_summary.to_csv("logs/final/cleaned/wor_vs_orig_best_ablation_summary.csv", index=False)
orig_uniform_summary.to_csv("logs/final/cleaned/wor_vs_orig_uniform_summary.csv", index=False)

print("\nDone! Saved WOR FAQ + original baseline summaries.")
print(f"\nWOR FAQ summary:\n{wor_faq_summary.to_string()}")

# ====================================================================
# PART 6 (optional): Process WOR baselines/ablation if they exist
# ====================================================================
wor_bl_files = glob.glob("logs/final/wor_baselines_dataset=*_sl=*.csv")
wor_abl_files = glob.glob("logs/final/wor_ablation_dataset=*_sl=*.csv")

if wor_bl_files and wor_abl_files:
    print("\n--- WOR baselines/ablation (curiosity) ---")

    wor_bl_df = pd.concat([pd.read_csv(f) for f in wor_bl_files], ignore_index=True)
    wor_abl_df = pd.concat([pd.read_csv(f) for f in wor_abl_files], ignore_index=True)

    # WOR uniform
    wor_uniform_df = wor_bl_df.query("policy == 'unif' and f == 'zero'").copy()
    wor_uniform_df = wor_uniform_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)
    wor_uniform_df["ess_multiplier"] = 1.0

    # Post-hoc best WOR baseline
    variant_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "policy", "tau", "f"]
    bl_means = wor_bl_df.groupby(variant_cols, dropna=False).mean(numeric_only=True).reset_index()
    best_bl_settings = bl_means.loc[
        bl_means.groupby(scenario_cols, dropna=False)["mean_width"].idxmin(), variant_cols]
    wor_best_bl_df = pd.merge(wor_bl_df, best_bl_settings, how="inner", on=variant_cols)
    wor_best_bl_df = wor_best_bl_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)

    # Post-hoc best WOR ablation
    abl_variant_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau"]
    abl_means = wor_abl_df.groupby(abl_variant_cols, dropna=False).mean(numeric_only=True).reset_index()
    best_abl_settings = abl_means.loc[
        abl_means.groupby(scenario_cols, dropna=False)["mean_width"].idxmin(), abl_variant_cols]
    wor_best_abl_df = pd.merge(wor_abl_df, best_abl_settings, how="inner", on=abl_variant_cols)
    wor_best_abl_df = wor_best_abl_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)

    # ESS multipliers (relative to WOR uniform)
    wor_faq_merged2 = pd.merge(wor_faq_df, wor_uniform_df[merge_cols + ["mean_width"]],
                                on=merge_cols, suffixes=("", "_wunif"))
    wor_faq_df["ess_multiplier_wor"] = (wor_faq_merged2["mean_width_wunif"] / wor_faq_merged2["mean_width"]) ** 2

    bl_merged2 = pd.merge(wor_best_bl_df, wor_uniform_df[merge_cols + ["mean_width"]],
                           on=merge_cols, suffixes=("", "_wunif"))
    wor_best_bl_df["ess_multiplier"] = (bl_merged2["mean_width_wunif"] / bl_merged2["mean_width"]) ** 2

    abl_merged2 = pd.merge(wor_best_abl_df, wor_uniform_df[merge_cols + ["mean_width"]],
                            on=merge_cols, suffixes=("", "_wunif"))
    wor_best_abl_df["ess_multiplier"] = (abl_merged2["mean_width_wunif"] / abl_merged2["mean_width"]) ** 2

    # Save WOR curiosity summaries
    build_summary(wor_best_bl_df, scenario_cols).to_csv(
        "logs/final/cleaned/wor_best_baseline_summary.csv", index=False)
    build_summary(wor_best_abl_df, scenario_cols).to_csv(
        "logs/final/cleaned/wor_best_ablation_summary.csv", index=False)
    build_summary(wor_uniform_df, scenario_cols_unif).to_csv(
        "logs/final/cleaned/wor_uniform_summary.csv", index=False)

    print("Saved WOR curiosity summaries too.")
else:
    print("\nNo WOR baseline/ablation files found (curiosity experiment not yet run).")
