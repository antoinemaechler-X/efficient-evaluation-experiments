"""
WOR Figure (tau comparison): ESS + Coverage for 6 methods on the same plot:
  Uniform (WR), Best Baseline (WR), FAQ (WR), WOR-FAQ (best τ), WOR-FAQ τ=0.25, WOR-FAQ τ=0.5

Outputs: figures/wor_tau_comparison.pdf
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

colors = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3"]

N_QUESTIONS = {"bbh+gpqa+ifeval+math+musr": 9574, "mmlu-pro": 12032}

# --- Load data ---
uniform_summary    = pd.read_csv("logs/final/cleaned/wor_vs_orig_uniform_summary.csv")
baseline_summary   = pd.read_csv("logs/final/cleaned/wor_vs_orig_best_baseline_summary.csv")
wr_faq_summary     = pd.read_csv("logs/final/cleaned/faq_summary.csv")
wr_faq_summary     = wr_faq_summary.query("mcar_obs_prob == 1.0").copy()
wor_best_summary   = pd.read_csv("logs/final/cleaned/wor_faq_summary.csv")
wor_tau025_summary = pd.read_csv("logs/final/cleaned/wor_faq_tau025_summary.csv")
wor_tau05_summary  = pd.read_csv("logs/final/cleaned/wor_faq_tau05_summary.csv")

budgets = np.sort(wor_best_summary.prop_budget.unique())

fig = plt.figure(dpi=400, figsize=(6.5, 2.7))
gs = gridspec.GridSpec(3, 2)

datasets_config = [
    ("mmlu-pro", "MMLU-Pro", 0, f"Budget (Out of {N_QUESTIONS['mmlu-pro']} Total Questions)"),
    ("bbh+gpqa+ifeval+math+musr", "BBH+GPQA+IFEval+MATH+MuSR", 1,
     f"Budget (Out of {N_QUESTIONS['bbh+gpqa+ifeval+math+musr']} Total Questions)"),
]

for dataset, title, col, xlabel in datasets_config:
    nq = N_QUESTIONS[dataset]

    q_uniform  = uniform_summary.query(f"dataset == '{dataset}'").sort_values("prop_budget")
    q_baseline = baseline_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0").sort_values("prop_budget")
    q_wr_faq   = wr_faq_summary.query(f"dataset == '{dataset}'").sort_values("prop_budget")
    q_wor_best = wor_best_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0").sort_values("prop_budget")
    q_wor_025  = wor_tau025_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0").sort_values("prop_budget")
    q_wor_05   = wor_tau05_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0").sort_values("prop_budget")

    # --- ESS subplot (top) ---
    ax_ess = fig.add_subplot(gs[:2, col])

    # Uniform
    ax_ess.errorbar(
        q_uniform.prop_budget * nq, q_uniform.ess_multiplier * budgets * nq,
        yerr=q_uniform.ess_multiplier_serr * budgets * nq,
        marker="^", capsize=MARKERSIZE, capthick=1.0, color=colors[0])

    # Best baseline (WR)
    ax_ess.errorbar(
        q_baseline.prop_budget * nq, q_baseline.ess_multiplier * budgets * nq,
        yerr=q_baseline.ess_multiplier_serr * budgets * nq,
        marker="x", capsize=MARKERSIZE, capthick=1.0, color=colors[1])
    for x, y, z in zip(q_baseline.prop_budget * nq, q_baseline.ess_multiplier,
                       q_baseline.ess_multiplier * budgets * nq):
        ax_ess.annotate(f"{y:.2f}", xy=(x, z),
                        textcoords="offset points", xytext=(0, SMALL_SIZE // 2 - 1), ha="center")

    # FAQ (WR)
    ax_ess.errorbar(
        q_wr_faq.prop_budget * nq, q_wr_faq.ess_multiplier * budgets * nq,
        yerr=q_wr_faq.ess_multiplier_serr * budgets * nq,
        marker="s", capsize=MARKERSIZE, capthick=1.0, color=colors[2])
    for x, y, z in zip(q_wr_faq.prop_budget * nq, q_wr_faq.ess_multiplier,
                       q_wr_faq.ess_multiplier * budgets * nq):
        ax_ess.annotate(f"{y:.2f}", xy=(x, z),
                        textcoords="offset points", xytext=(0, SMALL_SIZE // 2 - 1), ha="center")

    # WOR-FAQ (best tau)
    ax_ess.errorbar(
        q_wor_best.prop_budget * nq, q_wor_best.ess_multiplier * budgets * nq,
        yerr=q_wor_best.ess_multiplier_serr * budgets * nq,
        marker="o", capsize=MARKERSIZE, capthick=1.0, color=colors[3])
    for x, y, z in zip(q_wor_best.prop_budget * nq, q_wor_best.ess_multiplier,
                       q_wor_best.ess_multiplier * budgets * nq):
        ax_ess.annotate(f"{y:.2f}", xy=(x, z),
                        textcoords="offset points", xytext=(0, SMALL_SIZE // 2 - 1), ha="center")

    # WOR-FAQ tau=0.25
    ax_ess.errorbar(
        q_wor_025.prop_budget * nq, q_wor_025.ess_multiplier * budgets * nq,
        yerr=q_wor_025.ess_multiplier_serr * budgets * nq,
        marker="D", capsize=MARKERSIZE, capthick=1.0, color=colors[4])

    # WOR-FAQ tau=0.5
    ax_ess.errorbar(
        q_wor_05.prop_budget * nq, q_wor_05.ess_multiplier * budgets * nq,
        yerr=q_wor_05.ess_multiplier_serr * budgets * nq,
        marker="P", capsize=MARKERSIZE, capthick=1.0, color=colors[5])

    ax_ess.grid()
    ax_ess.set_title(title)
    if col == 0:
        ax_ess.set_ylabel("Effective Sample Size")
    ax_ess.tick_params(axis="x", labelbottom=False)

    # --- Coverage subplot (bottom) ---
    ax_cov = fig.add_subplot(gs[2:, col], sharex=ax_ess)

    ax_cov.errorbar(
        q_uniform.prop_budget * nq, q_uniform.coverage, yerr=q_uniform.coverage_serr,
        marker="^", capsize=MARKERSIZE, capthick=1.0, color=colors[0])
    ax_cov.errorbar(
        q_baseline.prop_budget * nq, q_baseline.coverage, yerr=q_baseline.coverage_serr,
        marker="x", capsize=MARKERSIZE, capthick=1.0, color=colors[1])
    ax_cov.errorbar(
        q_wr_faq.prop_budget * nq, q_wr_faq.coverage, yerr=q_wr_faq.coverage_serr,
        marker="s", capsize=MARKERSIZE, capthick=1.0, color=colors[2])
    ax_cov.errorbar(
        q_wor_best.prop_budget * nq, q_wor_best.coverage, yerr=q_wor_best.coverage_serr,
        marker="o", capsize=MARKERSIZE, capthick=1.0, color=colors[3])
    ax_cov.errorbar(
        q_wor_025.prop_budget * nq, q_wor_025.coverage, yerr=q_wor_025.coverage_serr,
        marker="D", capsize=MARKERSIZE, capthick=1.0, color=colors[4])
    ax_cov.errorbar(
        q_wor_05.prop_budget * nq, q_wor_05.coverage, yerr=q_wor_05.coverage_serr,
        marker="P", capsize=MARKERSIZE, capthick=1.0, color=colors[5])

    ax_cov.grid()
    ax_cov.set_ylim(bottom=0.85, top=1.0)
    ax_cov.axhline(y=0.95, color="black", linestyle="--")
    ax_cov.set_xlabel(xlabel)
    if col == 0:
        ax_cov.set_ylabel("Coverage")

# Legend
handles = [
    Line2D([], [], marker="^", color=colors[0], label="Uniform (WR)"),
    Line2D([], [], marker="x", color=colors[1], label="Best Baseline (WR)"),
    Line2D([], [], marker="s", color=colors[2], label="FAQ (WR)"),
    Line2D([], [], marker="o", color=colors[3], label="WOR-FAQ (best τ)"),
    Line2D([], [], marker="D", color=colors[4], label="WOR-FAQ τ=0.25"),
    Line2D([], [], marker="P", color=colors[5], label="WOR-FAQ τ=0.5"),
    Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
]
fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/wor_tau_comparison.pdf", facecolor="white", bbox_inches="tight")
print("Saved to figures/wor_tau_comparison.pdf")
plt.show()
