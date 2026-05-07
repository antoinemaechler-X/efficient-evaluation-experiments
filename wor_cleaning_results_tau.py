"""
WOR Cleaning Results (fixed-tau): Summarize results for tau=0.25 and tau=0.5 runs,
and compare with the original WOR-FAQ (best tau from validation).

Reads:
    - logs/final/wor_faq_final_tau025_sl=*.csv  (tau=0.25 fixed)
    - logs/final/wor_faq_final_tau05_sl=*.csv   (tau=0.5  fixed)
    - logs/final/cleaned/wor_faq_df.csv         (original WOR-FAQ, best tau)
    - logs/final/cleaned/uniform_df.csv         (original uniform, for ESS baseline)

Outputs to logs/final/cleaned/:
    - wor_faq_tau025_df.csv, wor_faq_tau025_summary.csv
    - wor_faq_tau05_df.csv,  wor_faq_tau05_summary.csv
"""
import numpy as np
import pandas as pd
import os, glob

# ====================================================================
# Load uniform baseline (for ESS multiplier computation)
# ====================================================================
uniform_df = pd.read_csv("logs/final/cleaned/uniform_df.csv")
uniform_df = uniform_df.query("mcar_obs_prob == 1.0").copy()
uniform_df = uniform_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)
print(f"Uniform baseline: {len(uniform_df)} rows")

merge_cols = ["dataset", "prop_budget", "seed"]

def build_summary(df, group_cols):
    return df.groupby(group_cols, dropna=False).agg(
        ess_multiplier=("ess_multiplier", "mean"),
        ess_multiplier_serr=("ess_multiplier", lambda x: x.std() / np.sqrt(len(x))),
        coverage=("coverage", "mean"),
        coverage_serr=("coverage", lambda x: x.std() / np.sqrt(len(x))),
        mean_width=("mean_width", "mean"),
        mean_width_serr=("mean_width", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

scenario_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"]

# ====================================================================
# Process each tau
# ====================================================================
os.makedirs("logs/final/cleaned", exist_ok=True)

for tau_tag, tau_label in [("025", "0.25"), ("05", "0.5")]:
    files = glob.glob(f"logs/final/wor_faq_final_tau{tau_tag}_sl=*.csv")
    if not files:
        print(f"No files found for tau={tau_label}, skipping.")
        continue

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)
    print(f"\ntau={tau_label}: {len(df)} rows from {len(files)} files")

    # ESS multiplier relative to original uniform
    merged = pd.merge(df, uniform_df[merge_cols + ["mean_width"]],
                      on=merge_cols, suffixes=("", "_unif"))
    df["ess_multiplier"] = (merged["mean_width_unif"] / merged["mean_width"]) ** 2

    summary = build_summary(df, scenario_cols)

    df.to_csv(f"logs/final/cleaned/wor_faq_tau{tau_tag}_df.csv", index=False)
    summary.to_csv(f"logs/final/cleaned/wor_faq_tau{tau_tag}_summary.csv", index=False)

    print(f"Saved wor_faq_tau{tau_tag}_df.csv and wor_faq_tau{tau_tag}_summary.csv")
    print(summary[["dataset", "prop_budget", "ess_multiplier", "coverage", "mean_width"]].to_string())

# ====================================================================
# Quick comparison: original WOR-FAQ vs tau=0.25 vs tau=0.5
# ====================================================================
print("\n\n=== Coverage comparison (mean across budgets) ===")
orig = pd.read_csv("logs/final/cleaned/wor_faq_summary.csv")
for tau_tag, tau_label in [("025", "0.25"), ("05", "0.5")]:
    fname = f"logs/final/cleaned/wor_faq_tau{tau_tag}_summary.csv"
    if not os.path.exists(fname):
        continue
    df_s = pd.read_csv(fname)
    print(f"\ntau={tau_label} vs original (best tau from val):")
    for ds in df_s.dataset.unique():
        orig_cov = orig.query(f"dataset == '{ds}' and mcar_obs_prob == 1.0")["coverage"].mean()
        new_cov  = df_s.query(f"dataset == '{ds}' and mcar_obs_prob == 1.0")["coverage"].mean()
        orig_ess = orig.query(f"dataset == '{ds}' and mcar_obs_prob == 1.0")["ess_multiplier"].mean()
        new_ess  = df_s.query(f"dataset == '{ds}' and mcar_obs_prob == 1.0")["ess_multiplier"].mean()
        print(f"  {ds}: coverage {orig_cov:.4f} → {new_cov:.4f}  |  ESS {orig_ess:.3f} → {new_ess:.3f}")
