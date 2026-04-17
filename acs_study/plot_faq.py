"""Paper-style plot for FAQ results (width + coverage vs. budget).

Reads one or more CSVs from run_faq.py (any subset of estimators) and produces
a 2-panel figure matching the paper's Figure 5 style:
  * per-trial scatter (translucent dots)
  * median line per estimator
  * 10th-90th percentile shaded band
  * log-log scale on the width panel
  * dashed target line on the coverage panel
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Fixed palette so classical/uniform+pai/faq always look the same across plots.
PALETTE = {
    "classical":    "#777777",
    "uniform+pai":  "#1f77b4",
    "faq":          "#d62728",
}
ORDER = ["classical", "uniform+pai", "faq"]


def summarize(df):
    """Median + 10/90 percentile per (estimator, n_b)."""
    agg = df.groupby(['estimator', '$n_b$']).agg(
        width_med=('interval width', 'median'),
        width_q10=('interval width', lambda s: np.percentile(s, 10)),
        width_q90=('interval width', lambda s: np.percentile(s, 90)),
        cov_mean=('coverage', 'mean'),
        cov_sem=('coverage', lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=('coverage', 'size'),
    ).reset_index()
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csvs', nargs='+', help='CSV files from run_faq.py')
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--title', type=str, default=None)
    p.add_argument('--out', type=str, default='faq_paper.png')
    args = p.parse_args()

    dfs = []
    for path in args.csvs:
        d = pd.read_csv(path)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    # Keep only the three main estimators (drop any faq (simp var) rows etc.)
    df = df[df['estimator'].isin(ORDER)].copy()
    ests = [e for e in ORDER if e in df['estimator'].unique()]

    agg = summarize(df)

    sns.set_theme(context='paper', font_scale=1.25, style='whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    # ----- Panel 1: interval width -----
    ax = axs[0]
    for est in ests:
        color = PALETTE[est]
        d_est = df[df['estimator'] == est]
        a_est = agg[agg['estimator'] == est].sort_values('$n_b$')
        # per-trial scatter (translucent)
        ax.scatter(
            d_est['$n_b$'], d_est['interval width'],
            s=12, alpha=0.15, color=color, edgecolor='none', zorder=1,
        )
        # 10-90% band
        ax.fill_between(
            a_est['$n_b$'], a_est['width_q10'], a_est['width_q90'],
            color=color, alpha=0.18, zorder=2, linewidth=0,
        )
        # median line
        ax.plot(
            a_est['$n_b$'], a_est['width_med'],
            color=color, linewidth=2.2, label=est, zorder=3,
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$n_b$ (labelled samples)')
    ax.set_ylabel('CI half-width × 2')
    ax.set_title('Interval width')
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)

    # ----- Panel 2: coverage -----
    ax = axs[1]
    for est in ests:
        color = PALETTE[est]
        a_est = agg[agg['estimator'] == est].sort_values('$n_b$')
        # per-trial jittered points (coverage is 0/1, so jitter for visibility)
        d_est = df[df['estimator'] == est]
        jitter = 0.02 * (np.random.rand(len(d_est)) - 0.5)
        ax.scatter(
            d_est['$n_b$'],
            d_est['coverage'].to_numpy() + jitter,
            s=8, alpha=0.12, color=color, edgecolor='none', zorder=1,
        )
        # ±2 SEM band around mean coverage
        lo = a_est['cov_mean'] - 2 * a_est['cov_sem']
        hi = a_est['cov_mean'] + 2 * a_est['cov_sem']
        ax.fill_between(
            a_est['$n_b$'], lo, hi,
            color=color, alpha=0.18, zorder=2, linewidth=0,
        )
        ax.plot(
            a_est['$n_b$'], a_est['cov_mean'],
            color=color, linewidth=2.2, label=est, zorder=3,
        )
    ax.axhline(1 - args.alpha, color='black', linestyle='--',
               linewidth=1.0, alpha=0.7, label=f'target = {1 - args.alpha:.2f}')
    ax.set_xscale('log')
    ax.set_ylim([0.5, 1.05])
    ax.set_xlabel(r'$n_b$ (labelled samples)')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Coverage')
    ax.legend(frameon=True, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)

    if args.title:
        fig.suptitle(args.title, y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
