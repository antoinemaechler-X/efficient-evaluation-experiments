"""
WOR Ablation Figure: ESS + Coverage for WOR-FAQ vs. WOR-Ablation (Z&C scoring).

Shows the contribution of full FAQ scoring on top of simpler Z&C (Zrnic & Candes) scoring,
both using without-replacement PAI.  Original WR baselines and uniform included for reference.
All ESS multipliers are relative to WR uniform.

Reads:
    - logs/final/cleaned/wor_faq_summary.csv (WOR FAQ)
    - logs/final/cleaned/wor_best_ablation_summary.csv (WOR Ablation, Z&C scoring)
    - logs/final/cleaned/wor_vs_orig_best_baseline_summary.csv (original baselines, fully-obs)
    - logs/final/cleaned/wor_vs_orig_uniform_summary.csv (original uniform)

Outputs: figures/wor_ess+coverage_ablation.pdf
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
faq_summary = pd.read_csv("logs/final/cleaned/wor_faq_summary.csv")
best_baseline_summary = pd.read_csv("logs/final/cleaned/wor_vs_orig_best_baseline_summary.csv")
uniform_summary = pd.read_csv("logs/final/cleaned/wor_vs_orig_uniform_summary.csv")

# WOR Ablation: recompute ESS relative to WR uniform (same scale as FAQ)
ablation_summary = pd.read_csv("logs/final/cleaned/wor_best_ablation_summary.csv")
abl_merged = pd.merge(
    ablation_summary,
    uniform_summary[["dataset", "prop_budget", "mean_width"]],
    on=["dataset", "prop_budget"], suffixes=("", "_unif"))
ablation_summary["ess_multiplier"] = (
    abl_merged["mean_width_unif"] / abl_merged["mean_width"]) ** 2
ablation_summary["ess_multiplier_serr"] = (
    2 * ablation_summary["ess_multiplier"]
    * ablation_summary["mean_width_serr"] / ablation_summary["mean_width"])

budgets = faq_summary.prop_budget.unique()

# --- Create figure ---
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
    q_abl = ablation_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
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

    # Ablation (WOR Z&C scoring)
    ax_ess.errorbar(
        q_abl.prop_budget * nq, q_abl.ess_multiplier * budgets * nq,
        yerr=q_abl.ess_multiplier_serr * budgets * nq,
        marker="v", capsize=MARKERSIZE, capthick=1.0, color=colors[3])
    for x, y, z in zip(q_abl.prop_budget * nq, q_abl.ess_multiplier,
                       q_abl.ess_multiplier * budgets * nq):
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
        q_abl.prop_budget * nq, q_abl.coverage, yerr=q_abl.coverage_serr,
        marker="v", capsize=MARKERSIZE, capthick=1.0, color=colors[3])
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
    Line2D([], [], marker="v", color=colors[3], label="Z&C Scoring (Without Replacement)"),
    Line2D([], [], marker="x", color=colors[1], label="Best Baseline (Post-hoc Per Budget)"),
    Line2D([], [], marker="^", color=colors[2], label="Uniform"),
    Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
]
fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.11))

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/wor_ess+coverage_ablation.pdf", facecolor="white", bbox_inches="tight")
print("Saved to figures/wor_ess+coverage_ablation.pdf")
plt.show()
