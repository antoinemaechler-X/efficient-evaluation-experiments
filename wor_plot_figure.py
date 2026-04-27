"""
WOR Figure: ESS + Coverage for WOR-FAQ vs. ORIGINAL baselines and uniform.

Compares the new WOR-FAQ method against the existing (with-replacement) baselines,
since the improvement is to FAQ/PAI only.

Reads:
    - logs/final/cleaned/wor_faq_summary.csv (WOR FAQ)
    - logs/final/cleaned/wor_vs_orig_best_baseline_summary.csv (original baselines, fully-obs)
    - logs/final/cleaned/wor_vs_orig_uniform_summary.csv (original uniform)

Outputs: figures/wor_ess+coverage_fully-observed.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os

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

# --- Load data ---
# WOR FAQ (new method)
faq_summary = pd.read_csv("logs/final/cleaned/wor_faq_summary.csv")

# Original baselines (existing results, filtered to fully-observed)
best_baseline_summary = pd.read_csv("logs/final/cleaned/wor_vs_orig_best_baseline_summary.csv")
uniform_summary = pd.read_csv("logs/final/cleaned/wor_vs_orig_uniform_summary.csv")

budgets = faq_summary.prop_budget.unique()

# --- Create figure (same layout as paper Figure 1) ---
fig = plt.figure(dpi=400, figsize=(6.5, 2.7))
gs = gridspec.GridSpec(3, 2)

datasets_config = [
    ("mmlu-pro", "MMLU-Pro", 0, f"Budget (Out of {N_QUESTIONS['mmlu-pro']} Total Questions)"),
    ("bbh+gpqa+ifeval+math+musr", "BBH+GPQA+IFEval+MATH+MuSR", 1,
     f"Budget (Out of {N_QUESTIONS['bbh+gpqa+ifeval+math+musr']} Total Questions)"),
]

for dataset, title, col, xlabel in datasets_config:
    nq = N_QUESTIONS[dataset]

    # Query data
    q_faq = faq_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_baseline = best_baseline_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_uniform = uniform_summary.query(f"dataset == '{dataset}'")

    # --- ESS subplot (top) ---
    ax_ess = fig.add_subplot(gs[:2, col])

    # FAQ (WOR)
    ax_ess.errorbar(
        q_faq.prop_budget * nq, q_faq.ess_multiplier * budgets * nq,
        yerr=q_faq.ess_multiplier_serr * budgets * nq,
        marker="o", capsize=MARKERSIZE, capthick=1.0, color=colors[0])
    for x, y, z in zip(q_faq.prop_budget * nq, q_faq.ess_multiplier,
                       q_faq.ess_multiplier * budgets * nq):
        ax_ess.annotate(f"{y:.2f}", xy=(x, z),
                        textcoords="offset points", xytext=(0, SMALL_SIZE // 2 - 1), ha="center")

    # Best baseline (original, with-replacement)
    ax_ess.errorbar(
        q_baseline.prop_budget * nq, q_baseline.ess_multiplier * budgets * nq,
        yerr=q_baseline.ess_multiplier_serr * budgets * nq,
        marker="x", capsize=MARKERSIZE, capthick=1.0, color=colors[1])
    for x, y, z in zip(q_baseline.prop_budget * nq, q_baseline.ess_multiplier,
                       q_baseline.ess_multiplier * budgets * nq):
        ax_ess.annotate(f"{y:.2f}", xy=(x, z),
                        textcoords="offset points", xytext=(0, SMALL_SIZE // 2 - 1), ha="center")

    # Uniform (original)
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
    Line2D([], [], marker="o", color=colors[0], label="FAQ (Without Replacement)"),
    Line2D([], [], marker="x", color=colors[1], label="Best Baseline (Post-hoc Per Budget)"),
    Line2D([], [], marker="^", color=colors[2], label="Uniform"),
    Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
]
fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.07))

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/wor_ess+coverage_fully-observed.pdf", facecolor="white", bbox_inches="tight")
print("Saved to figures/wor_ess+coverage_fully-observed.pdf")
plt.show()
