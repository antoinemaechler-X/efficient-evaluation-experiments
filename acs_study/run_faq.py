"""
FAQ experiment on ACS Census data — direct port of faq_val.py / faq_final.py.

Structurally identical to the LLM-evaluation FAQ loop:
  - N_NEW parallel trials (different seeds share the same Y, V, priors)
  - N_B = int(budget * N_QUESTIONS) multinomial samples per trial
  - sampling probs = softmax tempered mixture of h_o (oracle) + h_a (active)
    with uniform tau mix
  - AIPW increment per step, Sherman-Morrison posterior update
  - Variance = v_T_sq_simp - v_T_sq_minus (FAQ formula)

The only changes vs. faq_val.py are the substitutions forced by the
continuous-regression setting:
  * binary z_ij -> continuous y_j (normalised PINCP)
  * sigmoid(U_i^T V_j) -> U_i^T V_j (Bayesian linear regression)
  * Bernoulli Fisher weight p(1-p) -> Gaussian precision 1/sigma2
  * factor V from truncated SVD of the encoded feature matrix (the
    census-analysis covariate matrix) instead of SVD of M1

Estimand: theta = mean(y_normalised) over the unlabeled split.

Usage:
    # Local smoke test (a few minutes on CPU)
    python run_faq.py --num_trials 5 --num_budgets 3 \
        --budget_min 0.025 --budget_max 0.10 \
        --n_max 10000 --D 8 \
        --out_csv faq_test.csv --out_plot faq_test.png

    # Full cluster run (GPU)
    python run_faq.py --num_trials 100 --num_budgets 11 \
        --budget_min 0.005 --budget_max 0.10 --D 16 \
        --out_csv faq.csv --out_plot faq.png
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
    """Closed-form Bayesian linear regression with a spherical Normal prior.

    Returns:
        mu_post  (D,)    posterior mean
        Sig_post (D, D)  posterior covariance
        sigma2    scalar noise variance estimate (from OLS residuals)
    """
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
    V, Y, MU0, SIGMA0, sigma2,
    N_NEW, N_QUESTIONS, N_B,
    beta0, rho, gamma, tau,
    seed, device, counter,
):
    """Single FAQ run parallelised over N_NEW seeds.

    Mirrors faq_val.py's trial() exactly; see comments there.
    """
    D = V.shape[1]

    # Per-trial factor-model posteriors
    Uhats = MU0.unsqueeze(0).expand(N_NEW, -1).clone().to(device)          # (N_NEW, D)
    Sigmahats = SIGMA0.unsqueeze(0).expand(N_NEW, -1, -1).clone().to(device)  # (N_NEW, D, D)

    thetahats = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_main = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_inner1 = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_inner2 = torch.zeros(size=(N_NEW, N_B), dtype=torch.float32, device=device)

    v_bar = V.mean(dim=0)          # (D,)
    inv_sigma2 = 1.0 / sigma2

    torch.random.manual_seed(seed)

    for s in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=not sys.stderr.isatty()):

        #### PART 1: sampling probabilities q_s(j)

        # a. BLR predictions (no sigmoid for continuous Y)
        y_hat_js = Uhats @ V.T                                           # (N_NEW, N_Q)

        # b. symmetrise Sigmahats
        Sigmahats = (Sigmahats + Sigmahats.mT) / 2.0

        # c. predictive variance at each point -> analog of p(1-p)
        vtSigmav_js = torch.einsum("ni,mij,nj->mn", V, Sigmahats, V)     # (N_NEW, N_Q)

        # diagnostic: print ratio at step 0 to check if posterior has signal
        if s == 0 and counter == 0:
            ratio = (vtSigmav_js / sigma2).mean().item()
            print(f"  [diag] vtSigmav/sigma2={ratio:.6f}  (<<1 = posterior collapsed, ~1 = FAQ has signal)")
        predictive_var = (sigma2 + vtSigmav_js).clamp_min(1e-12)         # (N_NEW, N_Q)
        sqrt_pred_std = torch.sqrt(predictive_var)

        # d. oracle score h_o ∝ predictive std (analog of √(p(1-p)))
        ho_js = sqrt_pred_std / sqrt_pred_std.sum(dim=1, keepdim=True)

        # e. active-learning score h_a:
        #    for BLR with constant noise, the FAQ formula simplifies to
        #    h_a ∝ |v_j^T (Σ v̄)|^2 / (σ² + v_j^T Σ v_j)
        Sigma_vbar = torch.bmm(
            Sigmahats,
            v_bar.unsqueeze(0).expand(N_NEW, -1).unsqueeze(-1),
        ).squeeze(-1)                                                    # (N_NEW, D)
        sq_term = Sigma_vbar @ V.T                                       # (N_NEW, N_Q)
        log_numerator = 2 * torch.log(sq_term.abs().clamp_min(1e-12))
        log_denominator = torch.log(predictive_var)
        log_d_js = log_numerator - log_denominator
        ha_js = torch.softmax(log_d_js, dim=1)

        # f. exploration + tempering governors
        alpha_s = torch.maximum(
            torch.tensor(0.0),
            torch.tensor(1.0 - ((s + 1.0) / (rho * N_B))),
        ) if rho != 0.0 else torch.tensor(0.0)
        beta_s = beta0 * torch.minimum(
            torch.tensor((s + 1.0) / (gamma * N_B)),
            torch.tensor(1.0),
        ) if gamma != 0.0 else torch.tensor(beta0)

        # g. combine + uniform mix
        hcat_js = (((1.0 - alpha_s) * ho_js) + (alpha_s * ha_js)) ** beta_s
        q_js = ((hcat_js / hcat_js.sum(dim=1, keepdim=True)) * (1.0 - tau)) \
               + (tau / N_QUESTIONS)

        # h. sample one index per trial (multinomial w/ replacement)
        I_s = torch.multinomial(input=q_js, num_samples=1)                # (N_NEW, 1)

        #### PART 2: AIPW increment (identical to faq_val.py)

        y_Is = torch.gather(
            Y.unsqueeze(0).expand(N_NEW, -1), dim=1, index=I_s
        )
        yhat_Is = torch.gather(y_hat_js, dim=1, index=I_s)
        q_Is = torch.gather(q_js, dim=1, index=I_s)

        prob_sums = y_hat_js.sum(dim=1, keepdim=True)
        aipw_s = (y_Is - yhat_Is) / q_Is
        phi_s = prob_sums + aipw_s
        thetahats += phi_s

        varhats_main += aipw_s ** 2
        varhats_inner1 += y_Is / q_Is
        varhats_inner2[:, s] = prob_sums.flatten()

        #### PART 3: Bayesian linear regression posterior update
        #           (Sherman-Morrison, identical form to faq_val.py with w=1/σ²)

        v_Is = V[I_s.flatten()]                                          # (N_NEW, D)
        Sigma_v_Is = torch.bmm(Sigmahats, v_Is.unsqueeze(-1))             # (N_NEW, D, 1)
        vT_Sigma_v_Is = torch.bmm(
            v_Is.unsqueeze(-2), Sigma_v_Is
        ).squeeze(-1)                                                    # (N_NEW, 1)
        denominator = 1.0 + inv_sigma2 * vT_Sigma_v_Is
        numerator = inv_sigma2 * (Sigma_v_Is @ Sigma_v_Is.mT)
        Sigmahats -= numerator / denominator.unsqueeze(-1)

        # mean vector update: beta += Sigma_new @ (1/σ²)(y - ŷ) v
        Uhats += torch.bmm(
            Sigmahats,
            ((y_Is - yhat_Is) * inv_sigma2 * v_Is).unsqueeze(-1),
        ).squeeze(-1)

    # ---- theta_hat + variance (FAQ formulas, unchanged)
    thetahats_T = thetahats / (N_B * N_QUESTIONS)
    v_T_sq_simp = varhats_main / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_minus = (((varhats_inner1 / N_B) - varhats_inner2) ** 2).sum(
        dim=1, keepdim=True
    )
    v_T_sq_minus /= (N_B * (N_QUESTIONS ** 2))
    v_T_sq_full = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

    return thetahats_T, v_T_sq_simp, v_T_sq_full


def trial_classical(Y, N_NEW, N_QUESTIONS, N_B, seed, device):
    """Classical baseline: uniform with-replacement sampling + sample mean.

    No ML model, no AIPW correction. This is the simplest possible estimator
    used as a reference point.
    """
    torch.random.manual_seed(seed)
    I = torch.randint(0, N_QUESTIONS, size=(N_NEW, N_B), device=device)  # (N_NEW, N_B)
    Y_samp = Y[I]                                                        # (N_NEW, N_B)
    thetahats = Y_samp.mean(dim=1, keepdim=True)                          # (N_NEW, 1)
    # Var(theta_hat) = Var(Y) / N_B with the unbiased sample variance.
    sample_var = Y_samp.var(dim=1, keepdim=True, unbiased=True) / N_B      # (N_NEW, 1)
    return thetahats, sample_var


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, default=2019)
    p.add_argument('--train_frac', type=float, default=0.5)
    p.add_argument('--n_labeled', type=int, default=None,
                   help="Absolute number of labeled points (overrides --train_frac).")
    p.add_argument('--seed', type=int, default=0,
                   help="Seed for the labeled/unlabeled split (match train_model.py).")
    p.add_argument('--D', type=int, default=16,
                   help="Truncated-SVD dimension for the factor matrix V.")
    p.add_argument('--prior_tau', type=float, default=10.0,
                   help="BLR spherical-Normal prior std on weights.")
    p.add_argument('--num_trials', type=int, default=100,
                   help="Number of FAQ trials per budget (parallelised as N_NEW).")
    p.add_argument('--num_budgets', type=int, default=11)
    p.add_argument('--budget_min', type=float, default=0.005)
    p.add_argument('--budget_max', type=float, default=0.10)
    p.add_argument('--alpha', type=float, default=0.1,
                   help="Significance level (1 - target coverage).")
    p.add_argument('--beta0', type=float, default=1.0)
    p.add_argument('--rho', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=0.25)
    p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--n_max', type=int, default=None,
                   help="Truncate the unlabeled stream for local testing.")
    p.add_argument('--out_csv', type=str, default='faq.csv')
    p.add_argument('--out_plot', type=str, default='faq_width_coverage.png')
    p.add_argument('--estimators', type=str, default='classical,uniform+pai,faq',
                   help="Comma-separated list. Pick any subset of "
                        "{classical, uniform+pai, faq}. Use this to rerun only "
                        "missing baselines without redoing FAQ.")
    args = p.parse_args()

    est_set = set(s.strip() for s in args.estimators.split(','))
    valid = {'classical', 'uniform+pai', 'faq'}
    bad = est_set - valid
    if bad:
        raise ValueError(f"Unknown estimators {bad}; valid: {valid}")
    print(f"Running estimators: {sorted(est_set)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load ACS + split (same logic as train_model.py for consistency) ---
    print(f"Loading ACS {args.year} CA data...")
    np.random.seed(args.seed)
    income_features, income, _ = get_data(
        year=args.year, features=FEATURES, outcome='PINCP'
    )
    n_all = len(income)
    n_tr = args.n_labeled if args.n_labeled is not None else int(n_all * args.train_frac)
    print(f"N = {n_all}, labeled = {n_tr}, unlabeled = {n_all - n_tr}")

    feats_lab, feats_unlab, y_lab, y_unlab = train_test_split(
        income_features, income, train_size=n_tr, random_state=args.seed
    )
    y_lab = y_lab.to_numpy()
    y_unlab = y_unlab.to_numpy()

    # --- Encode features (one-hot categoricals) ---
    print("Encoding features...")
    _, enc = transform_features(income_features, FT)
    Phi_lab, _ = transform_features(feats_lab, FT, enc)
    Phi_unlab, _ = transform_features(feats_unlab, FT, enc)

    if args.n_max is not None and args.n_max < Phi_unlab.shape[0]:
        print(f"Truncating unlabeled stream to {args.n_max} for testing.")
        Phi_unlab = Phi_unlab[:args.n_max]
        y_unlab = y_unlab[:args.n_max]

    print(f"Labeled encoded shape: {Phi_lab.shape}, nnz {Phi_lab.nnz}")
    print(f"Unlabeled encoded shape: {Phi_unlab.shape}, nnz {Phi_unlab.nnz}")

    # --- Truncated SVD to build the factor matrix V (same role as V in FAQ) ---
    # We fit SVD on the labeled+unlabeled encoded matrix stack so that V is
    # shared across splits.
    Phi_all = scipy.sparse.vstack([Phi_lab, Phi_unlab]).tocsr().astype(np.float32)
    print(f"Truncated SVD (D={args.D}) on {Phi_all.shape} matrix...")
    # svds returns singular values in ascending order
    U_svd, s_svd, Vt_svd = scipy.sparse.linalg.svds(Phi_all, k=args.D)
    order = np.argsort(-s_svd)
    U_svd = U_svd[:, order]
    s_svd = s_svd[order]
    Vt_svd = Vt_svd[order, :]
    # Factor per data point: (n, D) = U_svd * s_svd
    V_all = U_svd * s_svd                                    # (n_all, D)
    # Normalize columns to unit variance so posterior covariance is well-scaled.
    # Without this, large singular values make ||v_j||^2 >> 1 and the posterior
    # collapses to near-zero even with very few labeled points.
    V_col_std = V_all.std(axis=0, keepdims=True).clip(1e-8)
    V_all = V_all / V_col_std
    V_lab_np = V_all[:Phi_lab.shape[0]]
    V_unlab_np = V_all[Phi_lab.shape[0]:]
    if args.n_max is not None:
        V_unlab_np = V_unlab_np[:args.n_max]

    # --- Normalise Y using labeled stats (so theta is "mean of z-scored PINCP") ---
    mu_Y, sd_Y = y_lab.mean(), y_lab.std()
    y_lab_n = ((y_lab - mu_Y) / sd_Y).astype(np.float32)
    y_unlab_n = ((y_unlab - mu_Y) / sd_Y).astype(np.float32)
    theta_true = float(y_unlab_n.mean())
    print(f"y-normalisation: mean={mu_Y:.2f}, std={sd_Y:.2f}")
    print(f"theta_true (mean of normalised PINCP on unlabeled) = {theta_true:.4f}")

    # --- Fit BLR on the labeled factor matrix ---
    print(f"Fitting BLR on labeled split (n={V_lab_np.shape[0]}, D={args.D})...")
    mu_post, Sig_post, sigma2 = fit_blr(
        V_lab_np.astype(np.float64), y_lab_n.astype(np.float64),
        prior_tau=args.prior_tau,
    )
    print(f"BLR sigma2 = {sigma2:.4f}")

    # --- Move tensors to device ---
    V = torch.tensor(V_unlab_np, dtype=torch.float32, device=device)
    Y = torch.tensor(y_unlab_n, dtype=torch.float32, device=device)
    MU0 = torch.tensor(mu_post, dtype=torch.float32, device=device)
    SIGMA0 = torch.tensor(Sig_post, dtype=torch.float32, device=device)
    sigma2_t = float(sigma2)

    N_QUESTIONS = V.shape[0]
    N_NEW = args.num_trials
    print(f"N_QUESTIONS = {N_QUESTIONS}, N_NEW (trials) = {N_NEW}")

    budgets = np.linspace(args.budget_min, args.budget_max, args.num_budgets)
    z_score = norm.ppf(1 - args.alpha / 2)

    # --- Run experiment (classical, uniform+PAI, FAQ) ---
    rows = []
    counter = 0
    rng = np.random.default_rng(args.seed)

    def record(thetahats, var_tensor, name, N_B):
        half = z_score * torch.sqrt(var_tensor)
        lb = (thetahats - half).cpu().numpy().flatten()
        ub = (thetahats + half).cpu().numpy().flatten()
        w = ub - lb
        cov = ((lb <= theta_true) & (theta_true <= ub)).astype(int)
        prop = N_B / N_QUESTIONS
        for t in range(N_NEW):
            rows.append((lb[t], ub[t], w[t], cov[t], name, N_B, prop, t))
        print(f"  {name:<14s} width={w.mean():.5f}  cov={cov.mean():.3f}")

    for b_idx, budget in enumerate(budgets):
        N_B = int(budget * N_QUESTIONS)
        if N_B < 1:
            continue
        seed_c = int(rng.integers(0, 2**31 - 1))
        seed_u = int(rng.integers(0, 2**31 - 1))
        seed_f = int(rng.integers(0, 2**31 - 1))
        print(f"\n=== Budget {budget:.4f}  (N_B={N_B}, {b_idx+1}/{len(budgets)}) ===")

        # 1) CLASSICAL: uniform sampling + sample mean
        if 'classical' in est_set:
            th_c, var_c = trial_classical(
                Y=Y, N_NEW=N_NEW, N_QUESTIONS=N_QUESTIONS, N_B=N_B,
                seed=seed_c, device=device,
            )
            record(th_c, var_c, "classical", N_B)
            del th_c, var_c

        # 2) UNIFORM+PAI: trial() with tau=1 -> q_j = 1/N (pure uniform),
        #    AIPW + online BLR update still on.
        if 'uniform+pai' in est_set:
            th_u, v_simp_u, v_full_u = trial(
                V=V, Y=Y, MU0=MU0, SIGMA0=SIGMA0, sigma2=sigma2_t,
                N_NEW=N_NEW, N_QUESTIONS=N_QUESTIONS, N_B=N_B,
                beta0=args.beta0, rho=args.rho, gamma=args.gamma, tau=1.0,
                seed=seed_u, device=device, counter=counter,
            )
            counter += 1
            record(th_u, v_full_u / N_B, "uniform+pai", N_B)
            del th_u, v_simp_u, v_full_u

        # 3) FAQ: adaptive sampling
        if 'faq' in est_set:
            th_f, v_simp_f, v_full_f = trial(
                V=V, Y=Y, MU0=MU0, SIGMA0=SIGMA0, sigma2=sigma2_t,
                N_NEW=N_NEW, N_QUESTIONS=N_QUESTIONS, N_B=N_B,
                beta0=args.beta0, rho=args.rho, gamma=args.gamma, tau=args.tau,
                seed=seed_f, device=device, counter=counter,
            )
            counter += 1
            record(th_f, v_full_f / N_B, "faq", N_B)
            del th_f, v_simp_f, v_full_f

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # --- Save CSV ---
    df = pd.DataFrame(
        rows,
        columns=["lb", "ub", "mean_width", "coverage", "estimator", "$n_b$",
                 "prop_budget", "seed"],
    )
    out_csv = os.path.abspath(args.out_csv)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    summary = df.groupby(['estimator', '$n_b$']).agg(
        width_mean=('mean_width', 'mean'),
        coverage_mean=('coverage', 'mean'),
    ).reset_index()
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nUse  plot_paper.py {args.out_csv}  to generate paper-style figures.")


if __name__ == '__main__':
    main()
