import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os

TICK_SIZE = 6
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 11
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
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

RESULTS = "results"

# --- Load data ---
wor_faq_summary     = pd.read_csv(f"{RESULTS}/wor_faq_summary.csv")
wr_faq_summary      = pd.read_csv(f"{RESULTS}/wr_faq_summary.csv")
best_baseline_summary = pd.read_csv(f"{RESULTS}/wor_vs_orig_best_baseline_summary.csv")
uniform_summary     = pd.read_csv(f"{RESULTS}/wor_vs_orig_uniform_summary.csv")

datasets_config = [
    ("mmlu-pro", "MMLU-Pro", 0,
     f"Budget (Out of {N_QUESTIONS['mmlu-pro']:,} Total Questions)"),
    ("bbh+gpqa+ifeval+math+musr", "BBH+GPQA+IFEval+MATH+MuSR", 1,
     f"Budget (Out of {N_QUESTIONS['bbh+gpqa+ifeval+math+musr']:,} Total Questions)"),
]

legend_handles = [
    Line2D([], [], marker="o", color=colors[0], label="FAQ (Without Replacement)"),
    Line2D([], [], marker="s", color=colors[3], label="FAQ (With Replacement)"),
    Line2D([], [], marker="x", color=colors[1], label="Best Baseline (Post-hoc Per Budget)"),
    Line2D([], [], marker="^", color=colors[2], label="Uniform"),
    Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
]


def make_figure(budget_mask=None, suffix="full"):
    fig = plt.figure(dpi=400, figsize=(6.5, 2.7))
    gs = gridspec.GridSpec(3, 2)

    for dataset, title, col, xlabel in datasets_config:
        nq = N_QUESTIONS[dataset]

        q_wor  = wor_faq_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
        q_wr   = wr_faq_summary.query(f"dataset == '{dataset}'")
        q_bl   = best_baseline_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
        q_unif = uniform_summary.query(f"dataset == '{dataset}'")

        if budget_mask is not None:
            q_wor  = q_wor[q_wor["prop_budget"].isin(budget_mask)]
            q_wr   = q_wr[q_wr["prop_budget"].isin(budget_mask)]
            q_bl   = q_bl[q_bl["prop_budget"].isin(budget_mask)]
            q_unif = q_unif[q_unif["prop_budget"].isin(budget_mask)]

        ax_ess = fig.add_subplot(gs[:2, col])

        for q, marker, color, annotate in [
            (q_wor,  "o", colors[0], True),
            (q_wr,   "s", colors[3], True),
            (q_bl,   "x", colors[1], True),
            (q_unif, "^", colors[2], False),
        ]:
            q = q.sort_values("prop_budget")
            xs = q["prop_budget"].values * nq
            ys = q["ess_multiplier"].values * q["prop_budget"].values * nq
            yerrs = q["ess_multiplier_serr"].values * q["prop_budget"].values * nq
            ax_ess.errorbar(xs, ys, yerr=yerrs,
                            marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color)
            if annotate:
                for x, mult, y in zip(xs, q["ess_multiplier"].values, ys):
                    ax_ess.annotate(f"{mult:.2f}", xy=(x, y),
                                    textcoords="offset points",
                                    xytext=(0, SMALL_SIZE // 2 - 1),
                                    ha="center", fontsize=SMALL_SIZE - 1)

        ax_ess.grid()
        ax_ess.set_title(title)
        if col == 0:
            ax_ess.set_ylabel("Effective Sample Size")
        ax_ess.tick_params(axis="x", labelbottom=False)
        ylo, yhi = ax_ess.get_ylim()
        ax_ess.set_ylim(ylo, yhi * 1.08)

        ax_cov = fig.add_subplot(gs[2:, col], sharex=ax_ess)

        for q, marker, color in [
            (q_wor,  "o", colors[0]),
            (q_wr,   "s", colors[3]),
            (q_bl,   "x", colors[1]),
            (q_unif, "^", colors[2]),
        ]:
            q = q.sort_values("prop_budget")
            xs = q["prop_budget"].values * nq
            ax_cov.errorbar(xs, q["coverage"].values, yerr=q["coverage_serr"].values,
                            marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color)

        ax_cov.grid()
        ax_cov.set_ylim(bottom=0.85, top=1.0)
        ax_cov.axhline(y=0.95, color="black", linestyle="--")
        ax_cov.set_xlabel(xlabel)
        if col == 0:
            ax_cov.set_ylabel("Coverage")

    fig.legend(handles=legend_handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.11))
    plt.tight_layout()

    out = f"figures/wor_vs_wr_{suffix}.pdf"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


# Plot 1: all 10 budgets
make_figure(budget_mask=None, suffix="full")

# Plot 2: first 5 budgets only (2.5% to 12.5%)
budgets_5 = np.round(np.linspace(0.0, 0.25, 11)[1:6], decimals=3)
make_figure(budget_mask=budgets_5, suffix="low_budget")

# Plot 3: first 5 budgets, WOR-FAQ / Best Baseline / Uniform only (no WR FAQ)
legend_handles_no_wr = [
    Line2D([], [], marker="o", color=colors[0], label="FAQ (Without Replacement)"),
    Line2D([], [], marker="x", color=colors[1], label="Best Baseline (Post-hoc Per Budget)"),
    Line2D([], [], marker="^", color=colors[2], label="Uniform"),
    Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
]

def make_figure_no_wr(budget_mask=None, suffix="low_budget_no_wr"):
    fig = plt.figure(dpi=400, figsize=(6.5, 2.7))
    gs = gridspec.GridSpec(3, 2)

    for dataset, title, col, xlabel in datasets_config:
        nq = N_QUESTIONS[dataset]

        q_wor  = wor_faq_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
        q_bl   = best_baseline_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
        q_unif = uniform_summary.query(f"dataset == '{dataset}'")

        if budget_mask is not None:
            q_wor  = q_wor[q_wor["prop_budget"].isin(budget_mask)]
            q_bl   = q_bl[q_bl["prop_budget"].isin(budget_mask)]
            q_unif = q_unif[q_unif["prop_budget"].isin(budget_mask)]

        ax_ess = fig.add_subplot(gs[:2, col])

        for q, marker, color, annotate in [
            (q_wor,  "o", colors[0], True),
            (q_bl,   "x", colors[1], True),
            (q_unif, "^", colors[2], False),
        ]:
            q = q.sort_values("prop_budget")
            xs = q["prop_budget"].values * nq
            ys = q["ess_multiplier"].values * q["prop_budget"].values * nq
            yerrs = q["ess_multiplier_serr"].values * q["prop_budget"].values * nq
            ax_ess.errorbar(xs, ys, yerr=yerrs,
                            marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color)
            if annotate:
                for x, mult, y in zip(xs, q["ess_multiplier"].values, ys):
                    ax_ess.annotate(f"{mult:.2f}", xy=(x, y),
                                    textcoords="offset points",
                                    xytext=(0, SMALL_SIZE // 2 - 1),
                                    ha="center", fontsize=SMALL_SIZE - 1)

        ax_ess.grid()
        ax_ess.set_title(title)
        if col == 0:
            ax_ess.set_ylabel("Effective Sample Size")
        ax_ess.tick_params(axis="x", labelbottom=False)
        ylo, yhi = ax_ess.get_ylim()
        ax_ess.set_ylim(ylo, yhi * 1.08)

        ax_cov = fig.add_subplot(gs[2:, col], sharex=ax_ess)

        for q, marker, color in [
            (q_wor,  "o", colors[0]),
            (q_bl,   "x", colors[1]),
            (q_unif, "^", colors[2]),
        ]:
            q = q.sort_values("prop_budget")
            xs = q["prop_budget"].values * nq
            ax_cov.errorbar(xs, q["coverage"].values, yerr=q["coverage_serr"].values,
                            marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color)

        ax_cov.grid()
        ax_cov.set_ylim(bottom=0.85, top=1.0)
        ax_cov.axhline(y=0.95, color="black", linestyle="--")
        ax_cov.set_xlabel(xlabel)
        if col == 0:
            ax_cov.set_ylabel("Coverage")

    fig.legend(handles=legend_handles_no_wr, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout()

    out = f"figures/wor_vs_wr_{suffix}.pdf"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()

make_figure_no_wr(budget_mask=budgets_5, suffix="low_budget_no_wr")
