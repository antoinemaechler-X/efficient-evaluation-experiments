"""
XGBoost + BLR hybrid model diagnostics for ACS data.

Produces a 1x4 figure:
  Left:    Predicted vs actual z-scored income (XGBoost, unlabeled split)
  Middle-left:  Residual distribution vs Normal(0, sigma2)
  Middle-right: Residual std by decile of predicted value (heteroscedasticity check)
  Right:   Learning curve R² vs n_labeled (XGBoost)

Usage:
    python plot_xgb_perf.py --out xgb_performance.png
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
import xgboost as xgb
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

TICK_SIZE   = 7
SMALL_SIZE  = 8
MEDIUM_SIZE = 9

plt.rc("font",   size=SMALL_SIZE)
plt.rc("axes",   titlesize=MEDIUM_SIZE, labelsize=SMALL_SIZE)
plt.rc("xtick",  labelsize=TICK_SIZE)
plt.rc("ytick",  labelsize=TICK_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("lines",  linewidth=1.0)
plt.rc("grid",   linewidth=0.4, alpha=0.5)

BLUE   = "#377eb8"
ORANGE = "#ff7f00"
GREEN  = "#4daf4a"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D",          type=int,   default=16)
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--xgb_rounds", type=int,   default=500)
    p.add_argument("--n_scatter",  type=int,   default=3000)
    p.add_argument("--n_deciles",  type=int,   default=10)
    p.add_argument("--out",        type=str,   default="xgb_performance.png")
    args = p.parse_args()

    print("Loading ACS data...")
    income_features, income, _ = get_data(year=2019, features=FEATURES, outcome='PINCP')
    n_all = len(income)
    n_tr  = int(n_all * args.train_frac)

    feats_lab, feats_unlab, y_lab, y_unlab = train_test_split(
        income_features, income, train_size=n_tr, random_state=args.seed
    )
    y_lab   = y_lab.to_numpy()
    y_unlab = y_unlab.to_numpy()

    print("Encoding features...")
    _, enc = transform_features(income_features, FT)
    Phi_lab,   _ = transform_features(feats_lab,   FT, enc)
    Phi_unlab, _ = transform_features(feats_unlab, FT, enc)

    mu_Y, sd_Y = y_lab.mean(), y_lab.std()
    y_lab_n   = ((y_lab   - mu_Y) / sd_Y).astype(np.float32)
    y_unlab_n = ((y_unlab - mu_Y) / sd_Y).astype(np.float32)

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    print(f"Training XGBoost ({args.xgb_rounds} rounds)...")
    dtrain = xgb.DMatrix(Phi_lab,   label=y_lab_n)
    dunlab = xgb.DMatrix(Phi_unlab)
    bst = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 6,
         "learning_rate": 0.05, "subsample": 0.8,
         "colsample_bytree": 0.8, "tree_method": "hist",
         "verbosity": 0},
        dtrain,
        num_boost_round=args.xgb_rounds,
        evals=[(dtrain, "train")],
        verbose_eval=100,
    )
    yhat_lab   = bst.predict(xgb.DMatrix(Phi_lab))
    yhat_unlab = bst.predict(dunlab)

    r2_lab   = r2_score(y_lab_n,   yhat_lab)
    r2_unlab = r2_score(y_unlab_n, yhat_unlab)
    resid    = y_unlab_n - yhat_unlab
    sigma2   = float(np.var(resid))
    print(f"XGBoost R² labeled={r2_lab:.3f}  unlabeled={r2_unlab:.3f}  sigma2={sigma2:.4f}")

    # ── BLR on residuals (D=16, full labeled) for comparison ─────────────────
    print(f"Truncated SVD (D={args.D})...")
    Phi_all = scipy.sparse.vstack([Phi_lab, Phi_unlab]).tocsr().astype(np.float32)
    U_svd, s_svd, _ = scipy.sparse.linalg.svds(Phi_all, k=args.D)
    order   = np.argsort(-s_svd)
    V_all   = (U_svd[:, order] * s_svd[order]).astype(np.float64)
    V_lab_np   = V_all[:n_tr]
    V_unlab_np = V_all[n_tr:]

    r_lab = (y_lab_n - yhat_lab).astype(np.float64)
    beta_ols = np.linalg.pinv(V_lab_np) @ r_lab
    rhat_unlab = V_unlab_np @ beta_ols
    r2_blr_residuals = r2_score(resid, rhat_unlab)
    r2_total = r2_score(y_unlab_n, yhat_unlab + rhat_unlab.astype(np.float32))
    print(f"BLR on residuals R²={r2_blr_residuals:.4f}  XGB+BLR total R²={r2_total:.3f}")

    # ── Heteroscedasticity: residual std per decile of predicted value ────────
    decile_labels = np.floor(
        np.argsort(np.argsort(yhat_unlab)) / len(yhat_unlab) * args.n_deciles
    ).astype(int).clip(0, args.n_deciles - 1)
    decile_std  = [resid[decile_labels == d].std() for d in range(args.n_deciles)]
    decile_mean = [yhat_unlab[decile_labels == d].mean() for d in range(args.n_deciles)]

    # ── Learning curve ────────────────────────────────────────────────────────
    ns   = [500, 1000, 2000, 5000, 10000, 30000, n_tr]
    r2s  = []
    rng  = np.random.default_rng(0)
    print("Computing learning curve...")
    for n in ns:
        idx = rng.choice(n_tr, size=min(n, n_tr), replace=False)
        b = xgb.train(
            {"objective": "reg:squarederror", "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8,
             "colsample_bytree": 0.8, "tree_method": "hist",
             "verbosity": 0},
            xgb.DMatrix(Phi_lab[idx] if hasattr(Phi_lab, '__getitem__') else
                        Phi_lab.tocsr()[idx], label=y_lab_n[idx]),
            num_boost_round=args.xgb_rounds,
        )
        r2s.append(r2_score(y_unlab_n, b.predict(dunlab)))
        print(f"  n={n:>7,}  R²={r2s[-1]:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 3), dpi=200)
    gs  = gridspec.GridSpec(1, 4, wspace=0.4)

    # Left: predicted vs actual
    ax1 = fig.add_subplot(gs[0])
    rng2 = np.random.default_rng(42)
    idx  = rng2.choice(len(y_unlab_n), size=args.n_scatter, replace=False)
    ax1.scatter(yhat_unlab[idx], y_unlab_n[idx],
                alpha=0.15, s=3, color=BLUE, rasterized=True)
    lims = [-3.5, 3.5]
    ax1.plot(lims, lims, "k--", linewidth=0.8)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel("Predicted income (z-score)")
    ax1.set_ylabel("Actual income (z-score)")
    ax1.set_title(f"XGBoost predictions\nR² = {r2_unlab:.3f}")
    ax1.grid()

    # Middle-left: residual distribution
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(resid, bins=80, density=True, color=BLUE, alpha=0.6, label="Residuals")
    x = np.linspace(-5, 5, 300)
    ax2.plot(x, norm.pdf(x, 0, np.sqrt(sigma2)), color=ORANGE, linewidth=1.2,
             label=f"N(0, σ²={sigma2:.2f})")
    ax2.set_xlim(-5, 5)
    ax2.set_xlabel("Residual (z-score)")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual distribution")
    ax2.legend(fontsize=7)
    ax2.grid()

    # Middle-right: heteroscedasticity check
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(decile_mean, decile_std, marker="o", color=BLUE, markersize=4)
    ax3.axhline(y=np.sqrt(sigma2), color=ORANGE, linestyle="--", linewidth=0.9,
                label=f"Global σ={np.sqrt(sigma2):.2f}")
    ax3.set_xlabel("Mean predicted income (z-score)\nby decile")
    ax3.set_ylabel("Residual std")
    ax3.set_title(f"Heteroscedasticity check\n(flat = homoscedastic → FAQ fails)")
    ax3.legend(fontsize=7)
    ax3.grid()

    # Right: learning curve
    ax4 = fig.add_subplot(gs[3])
    ax4.semilogx(ns, r2s, marker="o", color=BLUE, markersize=4)
    ax4.axvline(x=n_tr, color=ORANGE, linestyle="--", linewidth=0.9,
                label=f"Full split (n={n_tr:,})")
    ax4.axhline(y=r2_unlab, color="gray", linestyle=":", linewidth=0.8)
    ax4.set_xlabel("Number of labeled points")
    ax4.set_ylabel("R² on unlabeled split")
    ax4.set_title("XGBoost learning curve")
    ax4.legend(fontsize=7)
    ax4.grid()

    plt.suptitle(
        f"XGBoost + BLR on ACS  |  "
        f"XGB R²={r2_unlab:.3f}  XGB+BLR R²={r2_total:.3f}  "
        f"σ²={sigma2:.3f}  BLR adds {r2_blr_residuals:+.4f} on residuals",
        fontsize=SMALL_SIZE, y=1.02,
    )

    plt.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
