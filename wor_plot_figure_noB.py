"""
WOR Figure noB: Compare WOR-FAQ variants (A+B, A-only, bias-corrected).

Reads:
    - logs/final/wor_faq_final_noB_sl=*.csv (noB raw results)
    - logs/final/wor_faq_final_corrected_sl=*.csv (bias-corrected raw results)
    - logs/final/cleaned/wor_faq_summary.csv (standard WOR FAQ, with B)
    - logs/final/cleaned/wor_vs_orig_best_baseline_summary.csv
    - logs/final/cleaned/wor_vs_orig_uniform_summary.csv

Outputs: figures/wor_noB_comparison.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os, glob

# --- Font/style configuration (matches paper) ---
TICK_SIZE = 6
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 11
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=TICK_SIZE)
plt.rc("ytick", labelsize=TICK_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)

MARKERSIZE = 3
LINEWIDTH = 0.5
plt.rc("lines", markersize=MARKERSIZE, linewidth=LINEWIDTH)
plt.rc("grid", linewidth=0.5, alpha=0.5)

colors = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
          "#984ea3", "#999999", "#e41a1c", "#dede00"]

N_QUESTIONS = {"bbh+gpqa+ifeval+math+musr": 9574, "mmlu-pro": 12032}

# ====================================================================
# Load and clean noB results
# ====================================================================
noB_files = glob.glob("logs/final/wor_faq_final_noB_sl=*.csv")
noB_df = pd.concat([pd.read_csv(f) for f in noB_files], ignore_index=True)
noB_df = noB_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)
print(f"WOR FAQ noB: {len(noB_df)} rows from {len(noB_files)} files")

# Load uniform for ESS computation
uniform_df = pd.read_csv("logs/final/cleaned/uniform_df.csv")
if "mcar_obs_prob" in uniform_df.columns:
    uniform_df = uniform_df.query("mcar_obs_prob == 1.0").copy()
uniform_df = uniform_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)

# Compute ESS multiplier for noB
merge_cols = ["dataset", "prop_budget", "seed"]
noB_merged = pd.merge(noB_df, uniform_df[merge_cols + ["mean_width"]],
                      on=merge_cols, suffixes=("", "_unif"))
noB_df["ess_multiplier"] = (noB_merged["mean_width_unif"] / noB_merged["mean_width"]) ** 2

# Build noB summary
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
noB_summary = build_summary(noB_df, scenario_cols)

# Save cleaned noB
noB_df.to_csv("logs/final/cleaned/wor_faq_noB_df.csv", index=False)
noB_summary.to_csv("logs/final/cleaned/wor_faq_noB_summary.csv", index=False)
print(f"Saved noB cleaned files.")

# ====================================================================
# Load and clean corrected (bias-corrected) results
# ====================================================================
corr_files = glob.glob("logs/final/wor_faq_final_corrected_sl=*.csv")
corr_df = pd.concat([pd.read_csv(f) for f in corr_files], ignore_index=True)
corr_df = corr_df.sort_values(by=["dataset", "prop_budget", "seed"]).reset_index(drop=True)
print(f"WOR FAQ corrected: {len(corr_df)} rows from {len(corr_files)} files")

# Compute ESS multiplier for corrected
corr_merged = pd.merge(corr_df, uniform_df[merge_cols + ["mean_width"]],
                       on=merge_cols, suffixes=("", "_unif"))
corr_df["ess_multiplier"] = (corr_merged["mean_width_unif"] / corr_merged["mean_width"]) ** 2

corr_summary = build_summary(corr_df, scenario_cols)

# Save cleaned corrected
corr_df.to_csv("logs/final/cleaned/wor_faq_corrected_df.csv", index=False)
corr_summary.to_csv("logs/final/cleaned/wor_faq_corrected_summary.csv", index=False)
print(f"Saved corrected cleaned files.")

# ====================================================================
# Load comparison data
# ====================================================================
faq_summary = pd.read_csv("logs/final/cleaned/wor_faq_summary.csv")
best_baseline_summary = pd.read_csv("logs/final/cleaned/wor_vs_orig_best_baseline_summary.csv")
uniform_summary = pd.read_csv("logs/final/cleaned/wor_vs_orig_uniform_summary.csv")

budgets = faq_summary.prop_budget.unique()

# ====================================================================
# Plot
# ====================================================================
fig = plt.figure(dpi=400, figsize=(6.5, 2.7))
gs = gridspec.GridSpec(3, 2)

datasets_config = [
    ("mmlu-pro", "MMLU-Pro", 0, f"Budget (Out of {N_QUESTIONS['mmlu-pro']} Total Questions)"),
    ("bbh+gpqa+ifeval+math+musr", "BBH+GPQA+IFEval+MATH+MuSR", 1,
     f"Budget (Out of {N_QUESTIONS['bbh+gpqa+ifeval+math+musr']} Total Questions)"),
]

for dataset, title, col, xlabel in datasets_config:
    nq = N_QUESTIONS[dataset]

    q_faq = faq_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_noB = noB_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_corr = corr_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_baseline = best_baseline_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_uniform = uniform_summary.query(f"dataset == '{dataset}'")

    # --- ESS subplot (top) ---
    ax_ess = fig.add_subplot(gs[:2, col])

    # WOR-FAQ (with B) — standard
    ax_ess.errorbar(
        q_faq.prop_budget * nq, q_faq.ess_multiplier * budgets * nq,
        yerr=q_faq.ess_multiplier_serr * budgets * nq,
        marker="o", capsize=MARKERSIZE, capthick=1.0, color=colors[0])
    for x, y, z in zip(q_faq.prop_budget * nq, q_faq.ess_multiplier,
                       q_faq.ess_multiplier * budgets * nq):
        ax_ess.annotate(f"{y:.2f}", xy=(x, z),
                        textcoords="offset points", xytext=(0, SMALL_SIZE // 2 - 1), ha="center")

    # WOR-FAQ corrected (bias-corrected)
    ax_ess.errorbar(
        q_corr.prop_budget * nq, q_corr.ess_multiplier * budgets * nq,
        yerr=q_corr.ess_multiplier_serr * budgets * nq,
        marker="s", capsize=MARKERSIZE, capthick=1.0, color=colors[5])

    # WOR-FAQ noB (A-only variance)
    ax_ess.errorbar(
        q_noB.prop_budget * nq, q_noB.ess_multiplier * budgets * nq,
        yerr=q_noB.ess_multiplier_serr * budgets * nq,
        marker="D", capsize=MARKERSIZE, capthick=1.0, color=colors[7])

    # Best baseline (original WR)
    ax_ess.errorbar(
        q_baseline.prop_budget * nq, q_baseline.ess_multiplier * budgets * nq,
        yerr=q_baseline.ess_multiplier_serr * budgets * nq,
        marker="x", capsize=MARKERSIZE, capthick=1.0, color=colors[1])

    # Uniform
    ax_ess.errorbar(
        q_uniform.prop_budget * nq, q_uniform.ess_multiplier * budgets * nq,
        yerr=q_uniform.ess_multiplier_serr * budgets * nq,
        marker="^", capsize=MARKERSIZE, capthick=1.0, color=colors[2])

    ax_ess.grid()
    ax_ess.set_title(title)
    if col == 0:
        ax_ess.set_ylabel("Effective Sample Size")
    ax_ess.tick_params(axis="x", labelbottom=False)

    # --- Coverage subplot (bottom) ---
    ax_cov = fig.add_subplot(gs[2:, col], sharex=ax_ess)

    ax_cov.errorbar(
        q_faq.prop_budget * nq, q_faq.coverage, yerr=q_faq.coverage_serr,
        marker="o", capsize=MARKERSIZE, capthick=1.0, color=colors[0])
    ax_cov.errorbar(
        q_corr.prop_budget * nq, q_corr.coverage, yerr=q_corr.coverage_serr,
        marker="s", capsize=MARKERSIZE, capthick=1.0, color=colors[5])
    ax_cov.errorbar(
        q_noB.prop_budget * nq, q_noB.coverage, yerr=q_noB.coverage_serr,
        marker="D", capsize=MARKERSIZE, capthick=1.0, color=colors[7])
    ax_cov.errorbar(
        q_baseline.prop_budget * nq, q_baseline.coverage, yerr=q_baseline.coverage_serr,
        marker="x", capsize=MARKERSIZE, capthick=1.0, color=colors[1])
    ax_cov.errorbar(
        q_uniform.prop_budget * nq, q_uniform.coverage, yerr=q_uniform.coverage_serr,
        marker="^", capsize=MARKERSIZE, capthick=1.0, color=colors[2])

    ax_cov.grid()
    ax_cov.set_ylim(bottom=0.85, top=1.0)
    ax_cov.axhline(y=0.95, color="black", linestyle="--")
    ax_cov.set_xlabel(xlabel)
    if col == 0:
        ax_cov.set_ylabel("Coverage")

# Legend
handles = [
    Line2D([], [], marker="o", color=colors[0], label="WOR-FAQ (A+B variance)"),
    Line2D([], [], marker="s", color=colors[5], label="WOR-FAQ (bias-corrected)"),
    Line2D([], [], marker="D", color=colors[7], label="WOR-FAQ (A-only, no B)"),
    Line2D([], [], marker="x", color=colors[1], label="Best Baseline (WR)"),
    Line2D([], [], marker="^", color=colors[2], label="Uniform (WR)"),
    Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
]
fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.10))

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/wor_noB_comparison.pdf", facecolor="white", bbox_inches="tight")
print("Saved to figures/wor_noB_comparison.pdf")
plt.close()
