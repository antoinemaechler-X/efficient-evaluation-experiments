"""
Extended WOR comparison figure: adds 2 new curves to the base wor_vs_wr_full figure:
  - FAQ WOR Corrected (bias-corrected variance, best tau from validation)
  - FAQ WOR tau=0.25  (original Theorem 4 variance, tau forced to 0.25)

Reads:
  results/                          (existing summaries for WOR, WR, baseline, uniform)
  ../../logs/final/cleaned/         (tau=0.25 summary, uniform_df for ESS baseline)
  ../../logs/final/wor_faq_final_corrected_sl=*.csv  (raw corrected results)

Outputs:
  figures/wor_vs_wr_corrected.pdf
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── figure style (match existing plots) ──────────────────────────────────────
TICK_SIZE   = 6
SMALL_SIZE  = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 11
plt.rc("font",   size=SMALL_SIZE)
plt.rc("axes",   titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
plt.rc("xtick",  labelsize=TICK_SIZE)
plt.rc("ytick",  labelsize=TICK_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)
MARKERSIZE = 3
LINEWIDTH  = 0.5
plt.rc("lines", markersize=MARKERSIZE, linewidth=LINEWIDTH)
plt.rc("grid",  linewidth=0.5, alpha=0.5)

colors = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
          "#984ea3", "#999999", "#e41a1c", "#dede00"]

N_QUESTIONS = {
    "bbh+gpqa+ifeval+math+musr": 9574,
    "mmlu-pro": 12032,
}

RESULTS  = "results"
LOGS_CLN = "../../logs/final/cleaned"

# ── helpers ──────────────────────────────────────────────────────────────────
def build_summary(df, group_cols):
    return df.groupby(group_cols, dropna=False).agg(
        ess_multiplier      =("ess_multiplier", "mean"),
        ess_multiplier_serr =("ess_multiplier", lambda x: x.std() / np.sqrt(len(x))),
        coverage            =("coverage",        "mean"),
        coverage_serr       =("coverage",        lambda x: x.std() / np.sqrt(len(x))),
        mean_width          =("mean_width",       "mean"),
        mean_width_serr     =("mean_width",       lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()


# ── load existing summaries ───────────────────────────────────────────────────
wor_faq_summary      = pd.read_csv(f"{RESULTS}/wor_faq_summary.csv")
wr_faq_summary       = pd.read_csv(f"{RESULTS}/wr_faq_summary.csv")
best_baseline_summary = pd.read_csv(f"{RESULTS}/wor_vs_orig_best_baseline_summary.csv")
uniform_summary      = pd.read_csv(f"{RESULTS}/wor_vs_orig_uniform_summary.csv")

# ── tau=0.25 summary (already has ESS relative to original uniform) ───────────
tau025_summary = pd.read_csv(f"{LOGS_CLN}/wor_faq_tau025_summary.csv")

# ── build corrected summary on the fly ───────────────────────────────────────
corrected_files = sorted(glob.glob("../../logs/final/wor_faq_final_corrected_sl=*.csv"))
if not corrected_files:
    raise FileNotFoundError("No wor_faq_final_corrected_sl=*.csv files found. "
                            "Run from cleaned_processes/wor_study/")

corrected_df = pd.concat([pd.read_csv(f) for f in corrected_files], ignore_index=True)
corrected_df = corrected_df.sort_values(["dataset", "prop_budget", "seed"]).reset_index(drop=True)
print(f"Corrected: {len(corrected_df)} rows from {len(corrected_files)} files")

uniform_df = pd.read_csv(f"{LOGS_CLN}/uniform_df.csv")
uniform_df = uniform_df.query("mcar_obs_prob == 1.0").copy()

merge_cols = ["dataset", "prop_budget", "seed"]
corr_merged = pd.merge(corrected_df,
                       uniform_df[merge_cols + ["mean_width"]],
                       on=merge_cols, suffixes=("", "_unif"))
corrected_df["ess_multiplier"] = (corr_merged["mean_width_unif"] / corr_merged["mean_width"]) ** 2

scenario_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"]
corrected_summary = build_summary(corrected_df, scenario_cols)
print("Corrected summary built.")

# ── dataset layout ────────────────────────────────────────────────────────────
datasets_config = [
    ("mmlu-pro",                      "MMLU-Pro",                      0,
     f"Budget (Out of {N_QUESTIONS['mmlu-pro']:,} Total Questions)"),
    ("bbh+gpqa+ifeval+math+musr",     "BBH+GPQA+IFEval+MATH+MuSR",    1,
     f"Budget (Out of {N_QUESTIONS['bbh+gpqa+ifeval+math+musr']:,} Total Questions)"),
]

# ── legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([], [], marker="o", color=colors[0], label="FAQ WOR (best τ, Theorem 4)"),
    Line2D([], [], marker="D", color=colors[5], label="FAQ WOR Corrected (best τ, bias-corrected)"),
    Line2D([], [], marker="P", color=colors[4], label="FAQ WOR τ=0.25"),
    Line2D([], [], marker="s", color=colors[3], label="FAQ (With Replacement)"),
    Line2D([], [], marker="x", color=colors[1], label="Best Baseline (Post-hoc Per Budget)"),
    Line2D([], [], marker="^", color=colors[2], label="Uniform"),
    Line2D([], [], color="black", linestyle="--", label="95% Coverage"),
]

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(dpi=400, figsize=(6.5, 2.7))
gs  = gridspec.GridSpec(3, 2)

for dataset, title, col, xlabel in datasets_config:
    nq = N_QUESTIONS[dataset]

    q_wor  = wor_faq_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_wr   = wr_faq_summary.query(f"dataset == '{dataset}'")
    q_bl   = best_baseline_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_unif = uniform_summary.query(f"dataset == '{dataset}'")
    q_corr = corrected_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")
    q_t025 = tau025_summary.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0")

    # ── ESS panel ────────────────────────────────────────────────────────────
    ax_ess = fig.add_subplot(gs[:2, col])

    # (annotate=True, vert_offset in points for the label)
    curves_ess = [
        (q_wor,  "o", colors[0], True,  +4),
        (q_corr, "D", colors[5], True,  +1),
        (q_t025, "P", colors[4], False,  0),
        (q_wr,   "s", colors[3], True,  +4),
        (q_bl,   "x", colors[1], False,  0),
        (q_unif, "^", colors[2], False,  0),
    ]
    for q, marker, color, annotate, voff in curves_ess:
        q = q.sort_values("prop_budget")
        xs    = q["prop_budget"].values * nq
        ys    = q["ess_multiplier"].values * q["prop_budget"].values * nq
        yerrs = q["ess_multiplier_serr"].values * q["prop_budget"].values * nq
        ax_ess.errorbar(xs, ys, yerr=yerrs,
                        marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color)
        if annotate:
            for x, mult, y in zip(xs, q["ess_multiplier"].values, ys):
                ax_ess.annotate(f"{mult:.2f}", xy=(x, y),
                                textcoords="offset points",
                                xytext=(0, voff),
                                ha="center", va="bottom",
                                fontsize=SMALL_SIZE - 1, color="black")

    ax_ess.grid()
    ax_ess.set_title(title)
    if col == 0:
        ax_ess.set_ylabel("Effective Sample Size")
    ax_ess.tick_params(axis="x", labelbottom=False)
    ylo, yhi = ax_ess.get_ylim()
    ax_ess.set_ylim(ylo, yhi * 1.05)

    # ── Coverage panel ────────────────────────────────────────────────────────
    ax_cov = fig.add_subplot(gs[2:, col], sharex=ax_ess)

    curves_cov = [
        (q_wor,  "o", colors[0]),
        (q_corr, "D", colors[5]),
        (q_t025, "P", colors[4]),
        (q_wr,   "s", colors[3]),
        (q_bl,   "x", colors[1]),
        (q_unif, "^", colors[2]),
    ]
    for q, marker, color in curves_cov:
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

fig.legend(handles=legend_handles, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, -0.14))
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
out = "figures/wor_vs_wr_corrected.pdf"
plt.savefig(out, facecolor="white", bbox_inches="tight")
print(f"Saved {out}")
out_png = out.replace(".pdf", ".png")
plt.savefig(out_png, facecolor="white", bbox_inches="tight", dpi=300)
print(f"Saved {out_png}")
