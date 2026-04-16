"""Plot FAQ interval widths + coverage from one or more CSVs."""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csvs', nargs='+', help='CSV files from run_faq.py')
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--out', type=str, default='faq_combined.png')
    args = p.parse_args()

    dfs = []
    for path in args.csvs:
        df = pd.read_csv(path)
        tag = os.path.splitext(os.path.basename(path))[0]
        df['run'] = tag
        df['label'] = df['estimator'] + ' | ' + tag
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    sns.set_theme(font_scale=1.2, style='white')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(ax=axs[0], data=df, x='$n_b$', y='interval width',
                 hue='label', alpha=0.9)
    axs[0].set(xscale='log', yscale='log')
    axs[0].set_title('Interval width vs. budget')
    axs[0].grid(True)
    axs[0].legend(fontsize=8)

    sns.lineplot(ax=axs[1], data=df, x='$n_b$', y='coverage',
                 hue='label', errorbar=None)
    axs[1].axhline(1 - args.alpha, color='gray', linestyle='--', alpha=0.7)
    axs[1].set_ylim([0.6, 1.02])
    axs[1].set_title(f'Coverage (target = {1 - args.alpha:.2f})')
    axs[1].grid(True)
    axs[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
