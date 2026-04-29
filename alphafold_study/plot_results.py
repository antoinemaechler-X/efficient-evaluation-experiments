"""
Plot AlphaFold study results.

2 columns (non-phosphorylated, phosphorylated).
Top row: ESS (= ESS_mult × budget × N_group) with multiplier annotations.
Bottom row: Coverage with 95% reference line.

Output: alphafold_study/figures/alphafold_ess+coverage.pdf
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
data_dir = "alphafold_study/data"
Y = np.load(f"{data_dir}/Y.npy").flatten()
Z = np.load(f"{data_dir}/Z.npy").flatten()

N_GROUP = {
    "non-phosphorylated": int((Z == 0).sum()),
    "phosphorylated": int((Z == 1).sum()),
}

summary = pd.read_csv("alphafold_study/logs/cleaned/summary.csv")

# --- Method config: (method_name, marker, color, label, annotate) ---
method_config = [
    ("wor-active",         "o", colors[0], "WOR Active (Ours)",         True),
    ("wor-uniform",        "^", colors[2], "WOR Uniform (Ours)",        False),
    ("bernoulli-active",   "s", colors[3], "Bernoulli Active (Zrnic)",  True),
    ("bernoulli-uniform",  "D", colors[1], "Bernoulli Uniform",         False),
    ("classical",          "x", colors[4], "Classical",                  False),
]

groups_config = [
    ("non-phosphorylated", "Non-Phosphorylated", 0),
    ("phosphorylated",     "Phosphorylated",     1),
]

# --- Legend ---
legend_handles = [
    Line2D([], [], marker=m, color=c, label=l, linewidth=LINEWIDTH, markersize=MARKERSIZE)
    for _, m, c, l, _ in method_config
] + [
    Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
]

# --- Figure ---
fig = plt.figure(dpi=400, figsize=(6.5, 2.7))
gs = gridspec.GridSpec(3, 2)

for group_name, title, col in groups_config:
    nq = N_GROUP[group_name]

    ax_ess = fig.add_subplot(gs[:2, col])

    for method, marker, color, label, annotate in method_config:
        q = summary.query(f"group == '{group_name}' and method == '{method}'")
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

    # --- Coverage subplot ---
    ax_cov = fig.add_subplot(gs[2:, col], sharex=ax_ess)

    for method, marker, color, label, _ in method_config:
        q = summary.query(f"group == '{group_name}' and method == '{method}'")
        q = q.sort_values("prop_budget")

        xs = q["prop_budget"].values * nq
        ax_cov.errorbar(xs, q["coverage"].values, yerr=q["coverage_serr"].values,
                        marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color)

    ax_cov.grid()
    ax_cov.set_ylim(bottom=0.85, top=1.0)
    ax_cov.axhline(y=0.95, color="black", linestyle="--")
    ax_cov.set_xlabel(f"Budget (Out of {nq:,} Items)")
    if col == 0:
        ax_cov.set_ylabel("Coverage")

fig.legend(handles=legend_handles, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.14))
plt.tight_layout()

os.makedirs("alphafold_study/figures", exist_ok=True)
out = "alphafold_study/figures/alphafold_ess+coverage.pdf"
plt.savefig(out, facecolor="white", bbox_inches="tight")
print(f"Saved {out}")
plt.show()
