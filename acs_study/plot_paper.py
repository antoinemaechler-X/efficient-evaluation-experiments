"""
Produce paper-style Figure 2 for ACS data.

Reads one or more CSVs from run_faq.py (must contain estimators:
classical, uniform+pai, faq) and plots:
  Top:    Effective Sample Size vs. budget
  Bottom: Coverage vs. budget

Same colors, markers, fonts, and layout as Main Text Figures.ipynb.

Usage:
    python plot_paper.py faq_all.csv --out acs_figure2.pdf
    python plot_paper.py faq_acs_D16.csv baselines_D16.csv --out acs_figure2.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ── Paper RC params (from Main Text Figures.ipynb) ──
TICK_SIZE = 6
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 11
MARKERSIZE = 3
LINEWIDTH = 0.5

plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=TICK_SIZE)
plt.rc("ytick", labelsize=TICK_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)
plt.rc("lines", markersize=MARKERSIZE, linewidth=LINEWIDTH)
plt.rc("grid", linewidth=0.5, alpha=0.5)

# colorblind-friendly palette (same as paper)
COLORS = ["#377eb8", "#ff7f00", "#4daf4a"]

# mapping from our estimator names → paper roles
ROLE_MAP = {
    "faq":          ("FAQ",                                   "o", COLORS[0]),
    "uniform+pai":  ("Best Baseline (Post-hoc Per Budget)",   "x", COLORS[1]),
    "classical":    ("Uniform",                               "^", COLORS[2]),
    # accept old naming (from earlier runs)
    "faq (full var)": ("FAQ",                                 "o", COLORS[0]),
    "faq (simp var)": None,  # skip
}


def load_and_merge(paths):
    """Read one or more CSVs and concatenate."""
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    # Handle old column names
    if "interval width" in df.columns and "mean_width" not in df.columns:
        df = df.rename(columns={"interval width": "mean_width"})

    # Add trial index if missing
    if "seed" not in df.columns:
        df["seed"] = df.groupby(["estimator", "$n_b$"]).cumcount()

    # Add prop_budget if missing
    if "prop_budget" not in df.columns:
        n_q = df["$n_b$"].max() / df.groupby("estimator")["$n_b$"].transform("max").max()
        # fallback: infer from data
        df["prop_budget"] = df["$n_b$"] / df["$n_b$"].max() * df.groupby("estimator")["$n_b$"].transform("max").max() / df["$n_b$"].max()
        # simpler: just use $n_b$ directly on x-axis

    # Drop estimators we don't want
    df = df[df["estimator"].map(lambda e: ROLE_MAP.get(e) is not None)].copy()
    return df


def compute_summaries(df, n_seeds):
    """Compute ESS multiplier and summary stats per (estimator, budget).

    ESS_multiplier = (uniform_width / method_width)^2, matched per trial.
    """
    budgets = sorted(df["$n_b$"].unique())

    # Build uniform (classical) width lookup: budget × trial → width
    uniform = df.query("estimator == 'classical'").copy()

    records = []
    for est in df["estimator"].unique():
        sub = df.query(f"estimator == '{est}'").copy()
        for nb in budgets:
            s_est = sub.query(f"`$n_b$` == {nb}").sort_values("seed")
            s_uni = uniform.query(f"`$n_b$` == {nb}").sort_values("seed")

            # per-trial ESS multiplier (paired)
            n = min(len(s_est), len(s_uni))
            if n == 0:
                continue
            ess_mult = (s_uni["mean_width"].values[:n] / s_est["mean_width"].values[:n]) ** 2

            records.append({
                "estimator": est,
                "$n_b$": nb,
                "ess_multiplier": ess_mult.mean(),
                "ess_multiplier_serr": ess_mult.std(ddof=1) / np.sqrt(n),
                "coverage": s_est["coverage"].mean(),
                "coverage_serr": s_est["coverage"].std(ddof=1) / np.sqrt(n),
                "mean_width": s_est["mean_width"].mean(),
                "mean_width_serr": s_est["mean_width"].std(ddof=1) / np.sqrt(n),
            })

    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Significance level (1-coverage target). Paper uses 0.05.")
    p.add_argument("--out", type=str, default="acs_figure2.pdf")
    p.add_argument("--title", type=str, default="ACS Census (CA 2019)")
    args = p.parse_args()

    df = load_and_merge(args.csvs)
    n_seeds = df.groupby(["estimator", "$n_b$"]).size().min()
    print(f"Loaded {len(df)} rows, {n_seeds} trials per (estimator, budget)")

    summary = compute_summaries(df, n_seeds)
    budgets = sorted(summary["$n_b$"].unique())
    N_QUESTIONS = int(budgets[-1] / summary.query(f"`$n_b$` == {budgets[-1]}")["$n_b$"].values[0] * budgets[-1])
    # N_QUESTIONS is the max n_b / max prop_budget, but we may not have prop_budget
    # Simpler: just use $n_b$ as x-axis directly.

    # ── Figure ──
    fig = plt.figure(dpi=400, figsize=(3.25, 2.7))
    gs = gridspec.GridSpec(3, 1)

    # ──── Top: ESS ────
    ax_ess = fig.add_subplot(gs[:2, 0])

    for est in ["faq", "uniform+pai", "classical"]:
        info = ROLE_MAP.get(est)
        if info is None:
            continue
        label, marker, color = info
        q = summary.query(f"estimator == '{est}'").sort_values("$n_b$")
        if q.empty:
            continue

        ess_y = q["ess_multiplier"].values * q["$n_b$"].values
        ess_yerr = q["ess_multiplier_serr"].values * q["$n_b$"].values

        ax_ess.errorbar(
            q["$n_b$"].values, ess_y, yerr=ess_yerr,
            marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color,
        )

        # annotate ESS multiplier on each point
        if est != "classical":  # uniform is always 1.0, skip
            for x, mult, y in zip(q["$n_b$"].values, q["ess_multiplier"].values, ess_y):
                ax_ess.annotate(
                    f"{mult:.2f}", xy=(x, y),
                    textcoords="offset points", xytext=(0, SMALL_SIZE // 2 - 1),
                    ha="center", fontsize=SMALL_SIZE - 1,
                )

    ax_ess.grid()
    ax_ess.set_ylabel("Effective Sample Size")
    ax_ess.set_title(args.title)
    ax_ess.tick_params(axis="x", labelbottom=False)

    # ──── Bottom: Coverage ────
    ax_cov = fig.add_subplot(gs[2:, 0], sharex=ax_ess)

    for est in ["faq", "uniform+pai", "classical"]:
        info = ROLE_MAP.get(est)
        if info is None:
            continue
        label, marker, color = info
        q = summary.query(f"estimator == '{est}'").sort_values("$n_b$")
        if q.empty:
            continue

        ax_cov.errorbar(
            q["$n_b$"].values, q["coverage"].values,
            yerr=q["coverage_serr"].values,
            marker=marker, capsize=MARKERSIZE, capthick=1.0, color=color,
        )

    ax_cov.grid()
    ax_cov.set_ylabel("Coverage")
    target = 1 - args.alpha
    ax_cov.axhline(y=target, color="black", linestyle="--")
    ax_cov.set_ylim(bottom=max(0.5, target - 0.15), top=1.0)
    nb_max = max(budgets)
    ax_cov.set_xlabel(f"Budget (Out of {nb_max:,} Total)")

    # ──── Legend ────
    handles = []
    for est in ["faq", "uniform+pai", "classical"]:
        info = ROLE_MAP.get(est)
        if info is None:
            continue
        label, marker, color = info
        handles.append(Line2D([], [], marker=marker, color=color, label=label))
    handles.append(
        Line2D([], [], color="black", linestyle="--",
               label=f"{target:.0%} Coverage Target"),
    )
    fig.legend(
        handles=handles, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, -0.12), fontsize=SMALL_SIZE,
    )

    plt.tight_layout()
    plt.savefig(args.out, facecolor="white", bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
