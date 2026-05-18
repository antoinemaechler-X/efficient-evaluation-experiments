"""
Plot Galaxy Zoo 2 study results.

Produces PDFs:
  1. Main: WOR Active vs Cross-PPI vs Classical (ESS + coverage)
  2. All methods: all 6 methods vs WOR-uniform
  3. All methods: all 6 methods vs Classical
  4. Main sparse: same as 1, fewer budget points

Single column (no groups), top row = ESS, bottom row = coverage.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os

# --- Style ---
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

# --- Load data ---
data_dir = "galaxy_study/data"
Y = np.load(f"{data_dir}/Y.npy").flatten()

N_GROUP = {
    "all": len(Y),
}

summary = pd.read_csv("galaxy_study/logs/cleaned/summary.csv")

# --- Method config: (method_name, marker, color, label, annotate) ---
method_config = [
    ("wor-active",         "o", colors[0], "WOR Active (Ours)",       True),
    ("wor-uniform",        "^", colors[2], "WOR Uniform (Ours)",      False),
    ("bernoulli-active",   "s", colors[3], "Bernoulli Active",        False),
    ("bernoulli-uniform",  "D", colors[1], "Bernoulli Uniform",       False),
    ("cross-ppi",          "P", colors[5], "Cross-PPI (Zrnic)",       True),
    ("classical",          "x", colors[4], "Classical",                False),
]

groups_config = [
    ("all", "Galaxy Zoo 2 — Spiral Fraction", 0),
]


def make_figure(ess_col, ess_serr_col, methods, out_path, legend_ncol=3, budget_subset=None):
    """
    Generate ESS + coverage figure for single-group setup.
    """
    handles = [
        Line2D([], [], marker=m, color=c, label=l, linewidth=LINEWIDTH, markersize=MARKERSIZE)
        for _, m, c, l, _ in methods
    ] + [
        Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
    ]

    fig = plt.figure(dpi=400, figsize=(4.0, 2.7))
    gs = gridspec.GridSpec(3, 1)

    for group_name, title, col in groups_config:
        nq = N_GROUP[group_name]

        ax_ess = fig.add_subplot(gs[:2, 0])

        for method, marker, color, label, annotate in methods:
            q = summary.query(f"group == '{group_name}' and method == '{method}'")
            q = q.sort_values("prop_budget")
            if budget_subset is not None:
                q = q[q["prop_budget"].round(6).isin(np.round(budget_subset, 6))]

            xs = q["prop_budget"].values * nq
            ess_vals = q[ess_col].values
            ess_serr = q[ess_serr_col].values

            # Filter out inf/nan for plotting
            valid = np.isfinite(ess_vals)
            if not valid.all():
                xs_v = xs[valid]
                budgets_v = q["prop_budget"].values[valid]
                ess_vals = ess_vals[valid]
                ess_serr = ess_serr[valid]
            else:
                xs_v = xs
                budgets_v = q["prop_budget"].values

            ys = ess_vals * budgets_v * nq
            yerrs = ess_serr * budgets_v * nq

            ax_ess.errorbar(xs_v, ys, yerr=yerrs,
                            marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color)

            if annotate:
                for x, mult, y in zip(xs_v, ess_vals, ys):
                    ax_ess.annotate(f"{mult:.2f}", xy=(x, y),
                                    textcoords="offset points",
                                    xytext=(0, SMALL_SIZE // 2 - 1),
                                    ha="center", fontsize=SMALL_SIZE - 1)

        ax_ess.grid()
        ax_ess.set_title(title)
        ax_ess.set_ylabel("Effective Sample Size")
        ax_ess.tick_params(axis="x", labelbottom=False)
        ylo, yhi = ax_ess.get_ylim()
        ax_ess.set_ylim(ylo, yhi * 1.08)

        # --- Coverage subplot ---
        ax_cov = fig.add_subplot(gs[2:, 0], sharex=ax_ess)

        for method, marker, color, label, _ in methods:
            q = summary.query(f"group == '{group_name}' and method == '{method}'")
            q = q.sort_values("prop_budget")
            if budget_subset is not None:
                q = q[q["prop_budget"].round(6).isin(np.round(budget_subset, 6))]
            xs = q["prop_budget"].values * nq
            ax_cov.errorbar(xs, q["coverage"].values, yerr=q["coverage_serr"].values,
                            marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color)

        ax_cov.grid()
        ax_cov.set_ylim(bottom=0.85, top=1.0)
        ax_cov.axhline(y=0.95, color="black", linestyle="--")
        ax_cov.set_xlabel(f"Budget (Out of {nq:,} Items)")
        ax_cov.set_ylabel("Coverage")

    fig.legend(handles=handles, ncol=legend_ncol, loc="lower center",
               bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()

    os.makedirs("galaxy_study/figures", exist_ok=True)
    plt.savefig(out_path, facecolor="white", bbox_inches="tight")
    png_path = out_path.replace(".pdf", ".png")
    plt.savefig(png_path, facecolor="white", bbox_inches="tight")
    print(f"Saved {out_path} + {png_path}")
    plt.close()


# Plot 1: All 6 methods, ESS relative to WOR-uniform
make_figure("ess_multiplier", "ess_multiplier_serr",
            method_config,
            "galaxy_study/figures/galaxy_ess+coverage_vs_wor_uniform.pdf")

# Plot 2: All 6 methods, ESS relative to Classical
make_figure("ess_multiplier_vs_classical", "ess_multiplier_vs_classical_serr",
            method_config,
            "galaxy_study/figures/galaxy_ess+coverage_vs_classical.pdf")

# Plot 3: WOR Active vs Active Inference vs Cross-PPI vs Classical, ESS relative to Classical
method_config_main = [
    ("wor-active",        "o", colors[0], "WOR Active (Ours)",        True),
    ("bernoulli-active",  "s", colors[3], "Active Inference (Zrnic)", True),
    ("cross-ppi",         "P", colors[5], "Cross-PPI (Zrnic)",       True),
    ("classical",         "^", colors[2], "Classical",                False),
]
make_figure("ess_multiplier_vs_classical", "ess_multiplier_vs_classical_serr",
            method_config_main,
            "galaxy_study/figures/galaxy_main.pdf",
            legend_ncol=4)

# Plot 4: Same as Plot 3, sparse budgets (0.04, 0.06, ..., 0.20)
budgets_10 = np.linspace(0.01, 0.2, 20)[3::2]  # 0.04, 0.06, ..., 0.20
make_figure("ess_multiplier_vs_classical", "ess_multiplier_vs_classical_serr",
            method_config_main,
            "galaxy_study/figures/galaxy_main_sparse.pdf",
            legend_ncol=4, budget_subset=budgets_10)
