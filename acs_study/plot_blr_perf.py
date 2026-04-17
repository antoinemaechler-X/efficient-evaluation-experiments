"""
Plot BLR model performance on ACS data for presentation slide.

Produces a 1x3 figure:
  Left:   Predicted vs actual z-scored income (unlabeled split, sampled)
  Middle: Residual distribution vs Normal(0, sigma2)
  Right:  R^2 as a function of number of labeled points (learning curve)

Usage:
    python plot_blr_perf.py --out blr_performance.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import scipy.sparse
import scipy.sparse.linalg
import sys

from utils import get_data, transform_features

FEATURES = [
    'AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
    'ANC1P', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P', 'SOCP', 'COW'
]
FT = np.array([
    "q", "q", "c", "c", "c", "c", "c", "c",
    "c", "c", "c", "c", "c", "c", "c", "c", "c"
])

TICK_SIZE  = 7
SMALL_SIZE = 8
MEDIUM_SIZE = 9

plt.rc("font",   size=SMALL_SIZE)
plt.rc("axes",   titlesize=MEDIUM_SIZE, labelsize=SMALL_SIZE)
plt.rc("xtick",  labelsize=TICK_SIZE)
plt.rc("ytick",  labelsize=TICK_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("lines",  linewidth=1.0)
plt.rc("grid",   linewidth=0.4, alpha=0.5)

BLUE = "#377eb8"
ORANGE = "#ff7f00"


def fit_blr(Phi, y, prior_tau=10.0):
    n, D = Phi.shape
    beta_ols = np.linalg.pinv(Phi) @ y
    residuals = y - Phi @ beta_ols
    sigma2 = float(np.var(residuals))
    prior_prec = np.eye(D) / (prior_tau ** 2)
    post_prec = prior_prec + (Phi.T @ Phi) / sigma2
    Sig_post = np.linalg.inv(post_prec)
    mu_post = Sig_post @ (Phi.T @ y / sigma2)
    return mu_post, Sig_post, sigma2, beta_ols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_scatter", type=int, default=3000,
                   help="Points to show in scatter plot (random subsample).")
    p.add_argument("--out", type=str, default="blr_performance.png")
    args = p.parse_args()

    print("Loading ACS data...")
    income_features, income, _ = get_data(year=2019, features=FEATURES, outcome='PINCP')
    n_all = len(income)
    n_tr = int(n_all * args.train_frac)

    feats_lab, feats_unlab, y_lab, y_unlab = train_test_split(
        income_features, income, train_size=n_tr, random_state=args.seed
    )
    y_lab   = y_lab.to_numpy().astype(np.float64)
    y_unlab = y_unlab.to_numpy().astype(np.float64)

    print("Encoding features...")
    _, enc = transform_features(income_features, FT)
    Phi_lab,   _ = transform_features(feats_lab,   FT, enc)
    Phi_unlab, _ = transform_features(feats_unlab, FT, enc)

    # z-score using labeled stats
    mu_Y, sd_Y = y_lab.mean(), y_lab.std()
    y_lab_n   = (y_lab   - mu_Y) / sd_Y
    y_unlab_n = (y_unlab - mu_Y) / sd_Y

    print(f"Truncated SVD (D={args.D})...")
    Phi_all = scipy.sparse.vstack([Phi_lab, Phi_unlab]).tocsr().astype(np.float32)
    U_svd, s_svd, Vt_svd = scipy.sparse.linalg.svds(Phi_all, k=args.D)
    order  = np.argsort(-s_svd)
    U_svd  = U_svd[:, order]
    s_svd  = s_svd[order]
    V_all  = (U_svd * s_svd).astype(np.float64)
    V_lab   = V_all[:n_tr]
    V_unlab = V_all[n_tr:]

    print("Fitting BLR on labeled split...")
    mu_post, Sig_post, sigma2, beta_ols = fit_blr(V_lab, y_lab_n)

    y_hat_lab   = V_lab   @ mu_post
    y_hat_unlab = V_unlab @ mu_post

    r2_lab   = r2_score(y_lab_n,   y_hat_lab)
    r2_unlab = r2_score(y_unlab_n, y_hat_unlab)
    rmse_unlab = np.sqrt(np.mean((y_unlab_n - y_hat_unlab) ** 2))
    print(f"R² labeled   = {r2_lab:.3f}")
    print(f"R² unlabeled = {r2_unlab:.3f}")
    print(f"RMSE unlabeled (z-scored) = {rmse_unlab:.3f}")
    print(f"sigma2 (from labeled residuals) = {sigma2:.4f}")

    # ── Learning curve: R² vs n_labeled ──────────────────────────────────────
    ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, n_tr]
    r2s = []
    for n in ns:
        idx = np.random.default_rng(0).choice(n_tr, size=min(n, n_tr), replace=False)
        mu_n, _, _, _ = fit_blr(V_lab[idx], y_lab_n[idx])
        r2s.append(r2_score(y_unlab_n, V_unlab @ mu_n))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 3), dpi=200)
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)

    # ── Left: predicted vs actual (unlabeled) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_unlab_n), size=args.n_scatter, replace=False)
    ax1.scatter(y_hat_unlab[idx], y_unlab_n[idx],
                alpha=0.15, s=3, color=BLUE, rasterized=True)
    lims = [-3.5, 3.5]
    ax1.plot(lims, lims, "k--", linewidth=0.8, label="y = ŷ")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel("Predicted income (z-score)")
    ax1.set_ylabel("Actual income (z-score)")
    ax1.set_title(f"Predictions (unlabeled)\nR² = {r2_unlab:.2f}")
    ax1.grid()

    # ── Middle: residual histogram ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    resid = y_unlab_n - y_hat_unlab
    ax2.hist(resid, bins=80, density=True, color=BLUE, alpha=0.6,
             label="Residuals")
    x = np.linspace(-5, 5, 300)
    ax2.plot(x, norm.pdf(x, 0, np.sqrt(sigma2)), color=ORANGE, linewidth=1.2,
             label=f"N(0, σ²={sigma2:.2f})")
    ax2.set_xlim(-5, 5)
    ax2.set_xlabel("Residual (z-score)")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual distribution")
    ax2.legend(fontsize=7)
    ax2.grid()

    # ── Right: learning curve ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.semilogx(ns, r2s, marker="o", color=BLUE, markersize=4)
    ax3.axvline(x=n_tr, color=ORANGE, linestyle="--", linewidth=0.9,
                label=f"Full labeled set\n(n={n_tr:,})")
    ax3.axhline(y=r2_unlab, color="gray", linestyle=":", linewidth=0.8)
    ax3.set_xlabel("Number of labeled points")
    ax3.set_ylabel("R² on unlabeled split")
    ax3.set_title("Learning curve")
    ax3.legend(fontsize=7)
    ax3.grid()

    plt.suptitle(
        f"BLR on D={args.D} SVD factors  |  "
        f"n_labeled={n_tr:,}  n_unlabeled={len(y_unlab_n):,}  "
        f"σ²={sigma2:.2f}  R²={r2_unlab:.2f}",
        fontsize=SMALL_SIZE, y=1.02,
    )

    plt.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
