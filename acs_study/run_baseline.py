"""
Baseline census-analysis experiment.

Reproduces the active-inference experiment on ACS 2019 CA:
  - estimand: OLS coefficient of AGEP when regressing PINCP on (AGEP, SEX)
  - 3 estimators: classical, uniform, active
  - range of labelling budgets
  - many trials → CI width + coverage statistics

Requires predictions produced by train_model.py (default: predictions.npz).

Usage:
    # Quick smoke-test (~1 min): 100 trials, 10 budgets
    python run_baseline.py --predictions predictions_test.npz \
        --num_trials 100 --num_budgets 10 --out_csv baseline_test.csv

    # Full run (paper settings): 1000 trials, 20 budgets
    python run_baseline.py --predictions predictions.npz \
        --num_trials 1000 --num_budgets 20 --out_csv baseline.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, bernoulli
from tqdm import tqdm


def ols_and_ci(X, labels, Hessian_inv, n, alpha):
    """OLS point estimate + CI width via sandwich variance.

    Vectorized version of the census-analysis notebook's per-row gradient loop.
    """
    pointest = np.linalg.pinv(X).dot(labels)
    residuals = X.dot(pointest) - labels          # (n,)
    grads = residuals[:, None] * X                # (n, d)
    V = np.cov(grads.T)                           # (d, d)
    Sigma = Hessian_inv @ V @ Hessian_inv
    std = np.sqrt(Sigma[0, 0]) / np.sqrt(n)
    width = norm.ppf(1 - alpha / 2) * std
    return pointest[0], width


def run_experiment(Y, Yhat, X, uncertainty, theta_true,
                   budgets, num_trials, alpha, tau, seed=0):
    """Run the classical/uniform/active comparison across budgets."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    Hessian_inv = np.linalg.inv(1.0 / n * X.T @ X)

    rows = []
    for budget in tqdm(budgets, desc="budget"):
        # Sampling probabilities for active
        eta = budget / np.mean(uncertainty)
        probs = np.clip((1 - tau) * eta * uncertainty + tau * budget, 0, 1)

        nb = int(budget * n)  # nominal label count (for logging)

        for _ in range(num_trials):
            # --- active ---
            xi = rng.random(n) < probs
            active_labels = (Y - Yhat) * xi / probs + Yhat
            pt, w = ols_and_ci(X, active_labels, Hessian_inv, n, alpha)
            cov = (theta_true >= pt - w) and (theta_true <= pt + w)
            rows.append((pt - w, pt + w, 2 * w, int(cov), "active", nb))

            # --- uniform (prediction-powered) ---
            xi_u = rng.random(n) < budget
            unif_labels = (Y - Yhat) * xi_u / budget + Yhat
            pt, w = ols_and_ci(X, unif_labels, Hessian_inv, n, alpha)
            cov = (theta_true >= pt - w) and (theta_true <= pt + w)
            rows.append((pt - w, pt + w, 2 * w, int(cov), "uniform", nb))

            # --- classical (no ML prediction) ---
            class_labels = Y * xi_u / budget
            pt, w = ols_and_ci(X, class_labels, Hessian_inv, n, alpha)
            cov = (theta_true >= pt - w) and (theta_true <= pt + w)
            rows.append((pt - w, pt + w, 2 * w, int(cov), "classical", nb))

    df = pd.DataFrame(
        rows,
        columns=["lb", "ub", "interval width", "coverage", "estimator", "$n_b$"],
    )
    return df


def plot_width_coverage(df, theta_true, alpha, filename):
    """Simple width & coverage plots (matplotlib + seaborn)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(font_scale=1.2, style='white')
    col = [sns.color_palette("pastel")[1],
           sns.color_palette("pastel")[2],
           sns.color_palette("pastel")[0]]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.lineplot(ax=axs[0], data=df, x='$n_b$', y='interval width',
                 hue='estimator', palette=col, alpha=0.9)
    axs[0].set(xscale='log', yscale='log')
    axs[0].set_title('Interval width vs. budget')
    axs[0].grid(True)

    sns.lineplot(ax=axs[1], data=df, x='$n_b$', y='coverage',
                 hue='estimator', palette=col, errorbar=None)
    axs[1].axhline(1 - alpha, color='gray', linestyle='--', alpha=0.7)
    axs[1].set_ylim([0.6, 1.02])
    axs[1].set_title(f'Coverage (target = {1 - alpha:.2f})')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved plot to {filename}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=str, default='predictions.npz')
    p.add_argument('--num_trials', type=int, default=1000)
    p.add_argument('--num_budgets', type=int, default=11)
    p.add_argument('--budget_min', type=float, default=0.005)
    p.add_argument('--budget_max', type=float, default=0.10)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--tau', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_csv', type=str, default='baseline.csv')
    p.add_argument('--out_plot', type=str, default='baseline_width_coverage.png')
    args = p.parse_args()

    print(f"Loading predictions from {args.predictions}...")
    data = np.load(args.predictions)
    Y = data['Y']
    Yhat = data['Yhat']
    predicted_errs = data['predicted_errs']
    X = data['X'].astype(float)
    theta_true = float(data['theta_true'])
    print(f"n = {len(Y)}, theta_true = {theta_true:.4f}")

    # Uncertainty for active sampling: |h' x| * predicted_err, where h is the
    # influence direction for the target parameter (first row of H^-1 for AGEP).
    Hessian_inv = np.linalg.inv(1.0 / X.shape[0] * X.T @ X)
    h = Hessian_inv[:, 0]
    uncertainty = np.abs(h.dot(X.T)) * predicted_errs

    budgets = np.linspace(args.budget_min, args.budget_max, args.num_budgets)
    print(f"Budgets: {budgets[0]:.4f} ... {budgets[-1]:.4f} "
          f"({args.num_budgets} values), {args.num_trials} trials each")

    df = run_experiment(
        Y, Yhat, X, uncertainty, theta_true,
        budgets=budgets,
        num_trials=args.num_trials,
        alpha=args.alpha,
        tau=args.tau,
        seed=args.seed,
    )

    out_csv = os.path.abspath(args.out_csv)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Summary
    summary = df.groupby(['estimator', '$n_b$']).agg(
        width_mean=('interval width', 'mean'),
        coverage_mean=('coverage', 'mean'),
    ).reset_index()
    print("\n=== Summary (mean width & coverage) ===")
    print(summary.to_string(index=False))

    plot_width_coverage(df, theta_true, args.alpha, args.out_plot)


if __name__ == '__main__':
    main()
