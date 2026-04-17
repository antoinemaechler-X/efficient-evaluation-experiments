"""
Hybrid FAQ on ACS: XGBoost predictions + BLR on residuals.

Design
------
XGBoost is trained on the FULL labeled split → best possible ŷ_j (R² ~0.45).
BLR is fitted on --n_labeled_blr points only → genuine posterior uncertainty
for FAQ's active-learning scores h_o and h_a.

This decouples the two roles that FAQ needs:
  • Prediction quality (AIPW variance reduction)  → XGBoost
  • Bayesian uncertainty for sampling              → BLR on residuals

AIPW estimator:
  φ_s = Σ_j (ŷ_xgb_j + ŷ_blr_j)  +  (y_Is − ŷ_total_Is) / q_s(Is)

BLR models r_j = y_j − ŷ_xgb_j  (residuals after XGBoost).
Sherman-Morrison updates use the same residuals.

Usage:
    python run_faq_xgb.py --n_labeled_blr 500 --out_csv faq_xgb_nblr500.csv
"""

import argparse
import os
import sys
import gc
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
import torch
import xgboost as xgb
from tqdm import tqdm
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from utils import get_data, transform_features


FEATURES = [
    'AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
    'ANC1P', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P', 'SOCP', 'COW'
]
FT = np.array([
    "q", "q", "c", "c", "c", "c", "c", "c",
    "c", "c", "c", "c", "c", "c", "c", "c", "c"
])


def fit_blr(Phi, y, prior_tau=10.0):
    """BLR closed-form posterior on (Phi, y)."""
    n, D = Phi.shape
    beta_ols = np.linalg.pinv(Phi) @ y
    residuals = y - Phi @ beta_ols
    sigma2 = float(np.var(residuals))
    prior_prec = np.eye(D) / (prior_tau ** 2)
    post_prec = prior_prec + (Phi.T @ Phi) / sigma2
    Sig_post = np.linalg.inv(post_prec)
    mu_post = Sig_post @ (Phi.T @ y / sigma2)
    return mu_post, Sig_post, sigma2


def trial(
    V, Y, Y_XGB, MU0, SIGMA0, sigma2,
    N_NEW, N_QUESTIONS, N_B,
    beta0, rho, gamma, tau,
    seed, device, counter,
):
    """
    FAQ trial with hybrid predictions.

    V      : (N_Q, D)  SVD factors for unlabeled points
    Y      : (N_Q,)    true z-scored income (unlabeled)
    Y_XGB  : (N_Q,)    XGBoost predictions (fixed throughout)
    MU0    : (D,)      BLR prior mean (fitted on residuals)
    SIGMA0 : (D, D)    BLR prior covariance
    sigma2 : scalar    residual variance after XGBoost + BLR fit

    BLR models R = Y - Y_XGB (residuals after XGBoost).
    AIPW uses total predictions: ŷ_total = Y_XGB + BLR_pred.
    """
    D = V.shape[1]
    R = Y - Y_XGB                                                         # (N_Q,) residuals

    Uhats    = MU0.unsqueeze(0).expand(N_NEW, -1).clone()                 # (N_NEW, D)
    Sigmahats = SIGMA0.unsqueeze(0).expand(N_NEW, -1, -1).clone()         # (N_NEW, D, D)

    thetahats      = torch.zeros((N_NEW, 1), dtype=torch.float32, device=device)
    varhats_main   = torch.zeros((N_NEW, 1), dtype=torch.float32, device=device)
    varhats_inner1 = torch.zeros((N_NEW, 1), dtype=torch.float32, device=device)
    varhats_inner2 = torch.zeros((N_NEW, N_B), dtype=torch.float32, device=device)

    v_bar      = V.mean(dim=0)
    inv_sigma2 = 1.0 / sigma2
    xgb_sum    = Y_XGB.sum()                                              # scalar, fixed

    torch.random.manual_seed(seed)

    for s in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=not sys.stderr.isatty()):

        # ── BLR residual predictions ──────────────────────────────────────────
        r_hat_js = Uhats @ V.T                                            # (N_NEW, N_Q)

        # ── Total predictions: XGBoost + BLR ─────────────────────────────────
        y_hat_js = Y_XGB.unsqueeze(0) + r_hat_js                         # (N_NEW, N_Q)

        # ── Symmetrise Sigma ──────────────────────────────────────────────────
        Sigmahats = (Sigmahats + Sigmahats.mT) / 2.0

        # ── Oracle score h_o ∝ predictive std ────────────────────────────────
        vtSigmav_js  = torch.einsum("ni,mij,nj->mn", V, Sigmahats, V)    # (N_NEW, N_Q)
        predictive_var = (sigma2 + vtSigmav_js).clamp_min(1e-12)
        sqrt_pred_std  = torch.sqrt(predictive_var)
        ho_js = sqrt_pred_std / sqrt_pred_std.sum(dim=1, keepdim=True)

        # ── Active score h_a ──────────────────────────────────────────────────
        Sigma_vbar  = torch.bmm(
            Sigmahats,
            v_bar.unsqueeze(0).expand(N_NEW, -1).unsqueeze(-1),
        ).squeeze(-1)                                                      # (N_NEW, D)
        sq_term     = Sigma_vbar @ V.T                                     # (N_NEW, N_Q)
        log_d_js    = 2 * torch.log(sq_term.abs().clamp_min(1e-12)) \
                      - torch.log(predictive_var)
        ha_js = torch.softmax(log_d_js, dim=1)

        # ── Tempering + uniform mix ───────────────────────────────────────────
        alpha_s = torch.maximum(
            torch.tensor(0.0),
            torch.tensor(1.0 - (s + 1.0) / (rho * N_B)),
        ) if rho != 0.0 else torch.tensor(0.0)
        beta_s = beta0 * torch.minimum(
            torch.tensor((s + 1.0) / (gamma * N_B)),
            torch.tensor(1.0),
        ) if gamma != 0.0 else torch.tensor(beta0)

        hcat_js = (((1.0 - alpha_s) * ho_js) + (alpha_s * ha_js)) ** beta_s
        q_js    = (hcat_js / hcat_js.sum(dim=1, keepdim=True)) * (1.0 - tau) \
                  + tau / N_QUESTIONS

        I_s = torch.multinomial(q_js, num_samples=1)                      # (N_NEW, 1)

        # ── AIPW (uses total predictions ŷ_total = XGB + BLR) ────────────────
        y_Is    = torch.gather(Y.unsqueeze(0).expand(N_NEW, -1), 1, I_s)  # true Y
        yhat_Is = torch.gather(y_hat_js, 1, I_s)                          # total pred
        q_Is    = torch.gather(q_js, 1, I_s)

        prob_sums = xgb_sum + (Uhats @ v_bar).unsqueeze(1) * N_QUESTIONS  # (N_NEW, 1)
        aipw_s    = (y_Is - yhat_Is) / q_Is
        phi_s     = prob_sums + aipw_s
        thetahats += phi_s

        varhats_main   += aipw_s ** 2
        varhats_inner1 += y_Is / q_Is
        varhats_inner2[:, s] = prob_sums.flatten()

        # ── Sherman-Morrison BLR update (on residuals R = Y - Y_XGB) ─────────
        v_Is          = V[I_s.flatten()]                                   # (N_NEW, D)
        Sigma_v_Is    = torch.bmm(Sigmahats, v_Is.unsqueeze(-1))           # (N_NEW, D, 1)
        vT_Sigma_v_Is = torch.bmm(v_Is.unsqueeze(-2), Sigma_v_Is).squeeze(-1)  # (N_NEW, 1)
        denominator   = 1.0 + inv_sigma2 * vT_Sigma_v_Is
        Sigmahats    -= inv_sigma2 * (Sigma_v_Is @ Sigma_v_Is.mT) \
                        / denominator.unsqueeze(-1)

        r_Is    = torch.gather(R.unsqueeze(0).expand(N_NEW, -1), 1, I_s)  # true residual
        rhat_Is = torch.gather(r_hat_js, 1, I_s)                          # BLR prediction
        Uhats  += torch.bmm(
            Sigmahats,
            ((r_Is - rhat_Is) * inv_sigma2 * v_Is).unsqueeze(-1),
        ).squeeze(-1)

    # ── Variance (FAQ formula) ────────────────────────────────────────────────
    thetahats_T  = thetahats / (N_B * N_QUESTIONS)
    v_T_sq_simp  = varhats_main / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_minus = (((varhats_inner1 / N_B) - varhats_inner2) ** 2).sum(
        dim=1, keepdim=True
    ) / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_full  = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

    return thetahats_T, v_T_sq_simp, v_T_sq_full


def trial_classical(Y, N_NEW, N_QUESTIONS, N_B, seed, device):
    torch.random.manual_seed(seed)
    I       = torch.randint(0, N_QUESTIONS, size=(N_NEW, N_B), device=device)
    Y_samp  = Y[I]
    thetahats = Y_samp.mean(dim=1, keepdim=True)
    sample_var = Y_samp.var(dim=1, keepdim=True, unbiased=True) / N_B
    return thetahats, sample_var


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--year',            type=int,   default=2019)
    p.add_argument('--train_frac',      type=float, default=0.5)
    p.add_argument('--seed',            type=int,   default=0)
    p.add_argument('--D',               type=int,   default=16)
    p.add_argument('--prior_tau',       type=float, default=10.0)
    p.add_argument('--n_labeled_blr',   type=int,   default=500,
                   help="Points used to initialise the BLR prior. "
                        "Smaller → more posterior uncertainty → FAQ has more signal.")
    p.add_argument('--xgb_rounds',      type=int,   default=500)
    p.add_argument('--xgb_depth',       type=int,   default=6)
    p.add_argument('--xgb_lr',          type=float, default=0.05)
    p.add_argument('--num_trials',      type=int,   default=100)
    p.add_argument('--num_budgets',     type=int,   default=11)
    p.add_argument('--budget_min',      type=float, default=0.005)
    p.add_argument('--budget_max',      type=float, default=0.10)
    p.add_argument('--alpha',           type=float, default=0.1)
    p.add_argument('--beta0',           type=float, default=1.0)
    p.add_argument('--rho',             type=float, default=0.05)
    p.add_argument('--gamma',           type=float, default=0.25)
    p.add_argument('--tau',             type=float, default=0.1)
    p.add_argument('--n_max',           type=int,   default=None)
    p.add_argument('--out_csv',         type=str,   default='faq_xgb.csv')
    p.add_argument('--out_plot',        type=str,   default='faq_xgb.png')
    p.add_argument('--estimators',      type=str,   default='classical,uniform+pai,faq')
    args = p.parse_args()

    est_set = set(s.strip() for s in args.estimators.split(','))
    valid   = {'classical', 'uniform+pai', 'faq'}
    if bad := est_set - valid:
        raise ValueError(f"Unknown estimators {bad}")
    print(f"Running estimators: {sorted(est_set)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load + split ──────────────────────────────────────────────────────────
    print("Loading ACS data...")
    np.random.seed(args.seed)
    income_features, income, _ = get_data(year=args.year, features=FEATURES, outcome='PINCP')
    n_all = len(income)
    n_tr  = int(n_all * args.train_frac)

    feats_lab, feats_unlab, y_lab, y_unlab = train_test_split(
        income_features, income, train_size=n_tr, random_state=args.seed
    )
    y_lab   = y_lab.to_numpy()
    y_unlab = y_unlab.to_numpy()

    # ── Encode + SVD ──────────────────────────────────────────────────────────
    print("Encoding features...")
    _, enc = transform_features(income_features, FT)
    Phi_lab,   _ = transform_features(feats_lab,   FT, enc)
    Phi_unlab, _ = transform_features(feats_unlab, FT, enc)

    if args.n_max is not None and args.n_max < Phi_unlab.shape[0]:
        Phi_unlab = Phi_unlab[:args.n_max]
        y_unlab   = y_unlab[:args.n_max]

    print(f"Truncated SVD (D={args.D})...")
    Phi_all = scipy.sparse.vstack([Phi_lab, Phi_unlab]).tocsr().astype(np.float32)
    U_svd, s_svd, _ = scipy.sparse.linalg.svds(Phi_all, k=args.D)
    order   = np.argsort(-s_svd)
    V_all   = (U_svd[:, order] * s_svd[order]).astype(np.float64)
    V_lab_np   = V_all[:n_tr]
    V_unlab_np = V_all[n_tr:]
    if args.n_max is not None:
        V_unlab_np = V_unlab_np[:args.n_max]

    # ── Normalise Y ───────────────────────────────────────────────────────────
    mu_Y, sd_Y = y_lab.mean(), y_lab.std()
    y_lab_n   = ((y_lab   - mu_Y) / sd_Y).astype(np.float32)
    y_unlab_n = ((y_unlab - mu_Y) / sd_Y).astype(np.float32)
    theta_true = float(y_unlab_n.mean())
    print(f"theta_true = {theta_true:.4f}")

    # ── Train XGBoost on FULL labeled split ───────────────────────────────────
    print(f"Training XGBoost ({args.xgb_rounds} rounds) on {n_tr:,} labeled points...")
    dtrain = xgb.DMatrix(Phi_lab, label=y_lab_n)
    dunlab = xgb.DMatrix(Phi_unlab)
    xgb_params = {
        "objective":        "reg:squarederror",
        "max_depth":        args.xgb_depth,
        "learning_rate":    args.xgb_lr,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "tree_method":      "hist",
        "device":           "cuda" if torch.cuda.is_available() else "cpu",
        "verbosity":        0,
    }
    bst = xgb.train(
        xgb_params, dtrain,
        num_boost_round=args.xgb_rounds,
        evals=[(dtrain, "train")],
        verbose_eval=100,
    )
    yhat_xgb_lab   = bst.predict(xgb.DMatrix(Phi_lab)).astype(np.float32)
    yhat_xgb_unlab = bst.predict(dunlab).astype(np.float32)

    from sklearn.metrics import r2_score
    r2_lab   = r2_score(y_lab_n,   yhat_xgb_lab)
    r2_unlab = r2_score(y_unlab_n, yhat_xgb_unlab)
    print(f"XGBoost R² labeled={r2_lab:.3f}  unlabeled={r2_unlab:.3f}")

    # ── BLR on residuals (only n_labeled_blr points) ──────────────────────────
    n_blr = min(args.n_labeled_blr, n_tr)
    print(f"Fitting BLR on residuals ({n_blr:,} labeled points, D={args.D})...")
    r_lab = (y_lab_n - yhat_xgb_lab).astype(np.float64)
    mu_post, Sig_post, sigma2 = fit_blr(
        V_lab_np[:n_blr].astype(np.float64),
        r_lab[:n_blr],
        prior_tau=args.prior_tau,
    )
    # Evaluate BLR residual predictions on unlabeled set
    r_hat_unlab = V_unlab_np @ mu_post
    y_hat_total = yhat_xgb_unlab + r_hat_unlab.astype(np.float32)
    r2_total = r2_score(y_unlab_n, y_hat_total)
    print(f"BLR sigma2 (on residuals) = {sigma2:.4f}")
    print(f"XGBoost+BLR R² unlabeled = {r2_total:.3f}  "
          f"(XGB alone: {r2_unlab:.3f}, BLR adds: {r2_total - r2_unlab:+.3f})")

    # ── Move to device ────────────────────────────────────────────────────────
    V      = torch.tensor(V_unlab_np, dtype=torch.float32, device=device)
    Y      = torch.tensor(y_unlab_n,  dtype=torch.float32, device=device)
    Y_XGB  = torch.tensor(yhat_xgb_unlab, dtype=torch.float32, device=device)
    MU0    = torch.tensor(mu_post,   dtype=torch.float32, device=device)
    SIGMA0 = torch.tensor(Sig_post,  dtype=torch.float32, device=device)

    N_QUESTIONS = V.shape[0]
    N_NEW       = args.num_trials
    budgets     = np.linspace(args.budget_min, args.budget_max, args.num_budgets)
    z_score     = norm.ppf(1 - args.alpha / 2)
    rng         = np.random.default_rng(args.seed)

    print(f"N_QUESTIONS={N_QUESTIONS:,}  N_NEW={N_NEW}  budgets={budgets.round(3)}")

    # ── Run trials ────────────────────────────────────────────────────────────
    rows    = []
    counter = 0

    def record(thetahats, var_tensor, name, N_B):
        half = z_score * torch.sqrt(var_tensor)
        lb   = (thetahats - half).cpu().numpy().flatten()
        ub   = (thetahats + half).cpu().numpy().flatten()
        w    = ub - lb
        cov  = ((lb <= theta_true) & (theta_true <= ub)).astype(int)
        prop = N_B / N_QUESTIONS
        for t in range(N_NEW):
            rows.append((lb[t], ub[t], w[t], cov[t], name, N_B, prop, t))
        print(f"  {name:<14s}  width={w.mean():.5f}  cov={cov.mean():.3f}")

    for b_idx, budget in enumerate(budgets):
        N_B = int(budget * N_QUESTIONS)
        if N_B < 1:
            continue
        seed_c = int(rng.integers(0, 2**31 - 1))
        seed_u = int(rng.integers(0, 2**31 - 1))
        seed_f = int(rng.integers(0, 2**31 - 1))
        print(f"\n=== Budget {budget:.4f}  (N_B={N_B}, {b_idx+1}/{len(budgets)}) ===")

        if 'classical' in est_set:
            th_c, var_c = trial_classical(Y, N_NEW, N_QUESTIONS, N_B, seed_c, device)
            record(th_c, var_c, "classical", N_B)
            del th_c, var_c

        if 'uniform+pai' in est_set:
            th_u, _, v_full_u = trial(
                V, Y, Y_XGB, MU0, SIGMA0, sigma2,
                N_NEW, N_QUESTIONS, N_B,
                args.beta0, args.rho, args.gamma, tau=1.0,
                seed=seed_u, device=device, counter=counter,
            )
            counter += 1
            record(th_u, v_full_u / N_B, "uniform+pai", N_B)
            del th_u, v_full_u

        if 'faq' in est_set:
            th_f, _, v_full_f = trial(
                V, Y, Y_XGB, MU0, SIGMA0, sigma2,
                N_NEW, N_QUESTIONS, N_B,
                args.beta0, args.rho, args.gamma, tau=args.tau,
                seed=seed_f, device=device, counter=counter,
            )
            counter += 1
            record(th_f, v_full_f / N_B, "faq", N_B)
            del th_f, v_full_f

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(
        rows,
        columns=["lb", "ub", "mean_width", "coverage", "estimator",
                 "$n_b$", "prop_budget", "seed"],
    )
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved {args.out_csv}")

    summary = df.groupby(['estimator', '$n_b$']).agg(
        width_mean=('mean_width', 'mean'),
        coverage_mean=('coverage', 'mean'),
    ).reset_index()
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
