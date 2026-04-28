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
    D = V.shape[1]

    Uhats = MU0.unsqueeze(0).expand(N_NEW, -1).clone().to(device)          # (N_NEW, D)
    Sigmahats = SIGMA0.unsqueeze(0).expand(N_NEW, -1, -1).clone().to(device)  # (N_NEW, D, D)

    thetahats = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_main = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_inner1 = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_inner2 = torch.zeros(size=(N_NEW, N_B), dtype=torch.float32, device=device)

    v_bar = V.mean(dim=0)
    inv_sigma2 = 1.0 / sigma2

    torch.random.manual_seed(seed)

    for s in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=not sys.stderr.isatty()):

        #### PART 1: Computing q_s(j) probabilities + sampling

        # a. BLR predictions
        y_hat_js = Uhats @ V.T                                           # (N_NEW, N_Q)

        # b. symmetrise Sigmahats
        Sigmahats = (Sigmahats + Sigmahats.mT) / 2.0

        # c. predictive variance -> analog of p(1-p) in the binary case
        vtSigmav_js = torch.einsum("ni,mij,nj->mn", V, Sigmahats, V)     # (N_NEW, N_Q)

        # diagnostic: vtSigmav/sigma2 << 1 means posterior collapsed -> FAQ = uniform+pai
        if s == 0 and counter == 0:
            ratio = (vtSigmav_js / sigma2).mean().item()
            print(f"  [diag] vtSigmav/sigma2={ratio:.6f}  (<<1 = posterior collapsed, ~1 = FAQ has signal)")
        predictive_var = (sigma2 + vtSigmav_js).clamp_min(1e-12)         # (N_NEW, N_Q)
        sqrt_pred_std = torch.sqrt(predictive_var)

        # d. oracle score h_o
        ho_js = sqrt_pred_std / sqrt_pred_std.sum(dim=1, keepdim=True)

        # e. active-learning score h_a
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

        # h. sample one index per trial
        I_s = torch.multinomial(input=q_js, num_samples=1)                # (N_NEW, 1)

        #### PART 2: Updating Our FAQ Estimators

        y_Is = torch.gather(Y.unsqueeze(0).expand(N_NEW, -1), dim=1, index=I_s)
        yhat_Is = torch.gather(y_hat_js, dim=1, index=I_s)
        q_Is = torch.gather(q_js, dim=1, index=I_s)

        prob_sums = y_hat_js.sum(dim=1, keepdim=True)
        aipw_s = (y_Is - yhat_Is) / q_Is
        phi_s = prob_sums + aipw_s
        thetahats += phi_s

        varhats_main += aipw_s ** 2
        varhats_inner1 += y_Is / q_Is
        varhats_inner2[:, s] = prob_sums.flatten()

        #### PART 3: BLR Posterior Update (Sherman-Morrison)

        v_Is = V[I_s.flatten()]                                          # (N_NEW, D)
        Sigma_v_Is = torch.bmm(Sigmahats, v_Is.unsqueeze(-1))             # (N_NEW, D, 1)
        vT_Sigma_v_Is = torch.bmm(
            v_Is.unsqueeze(-2), Sigma_v_Is
        ).squeeze(-1)                                                    # (N_NEW, 1)
        denominator = 1.0 + inv_sigma2 * vT_Sigma_v_Is
        numerator = inv_sigma2 * (Sigma_v_Is @ Sigma_v_Is.mT)
        Sigmahats -= numerator / denominator.unsqueeze(-1)

        Uhats += torch.bmm(
            Sigmahats,
            ((y_Is - yhat_Is) * inv_sigma2 * v_Is).unsqueeze(-1),
        ).squeeze(-1)

    # computing coverage + width metrics
    thetahats_T = thetahats / (N_B * N_QUESTIONS)
    v_T_sq_simp = varhats_main / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_minus = (((varhats_inner1 / N_B) - varhats_inner2) ** 2).sum(
        dim=1, keepdim=True
    )
    v_T_sq_minus /= (N_B * (N_QUESTIONS ** 2))
    v_T_sq_full = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

    return thetahats_T, v_T_sq_simp, v_T_sq_full


def trial_classical(Y, N_NEW, N_QUESTIONS, N_B, seed, device):
    torch.random.manual_seed(seed)
    I = torch.randint(0, N_QUESTIONS, size=(N_NEW, N_B), device=device)
    Y_samp = Y[I]
    thetahats = Y_samp.mean(dim=1, keepdim=True)
    sample_var = Y_samp.var(dim=1, keepdim=True, unbiased=True) / N_B
    return thetahats, sample_var


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, default=2019)
    p.add_argument('--train_frac', type=float, default=0.5)
    p.add_argument('--n_labeled', type=int, default=None,
                   help="Absolute number of labeled points (overrides --train_frac).")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--D', type=int, default=16)
    p.add_argument('--prior_tau', type=float, default=10.0)
    p.add_argument('--num_trials', type=int, default=100)
    p.add_argument('--num_budgets', type=int, default=11)
    p.add_argument('--budget_min', type=float, default=0.005)
    p.add_argument('--budget_max', type=float, default=0.10)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--beta0', type=float, default=1.0)
    p.add_argument('--rho', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=0.25)
    p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--n_max', type=int, default=None,
                   help="Truncate the unlabeled stream for local testing.")
    p.add_argument('--out_csv', type=str, default='faq.csv')
    p.add_argument('--out_plot', type=str, default='faq_width_coverage.png')
    p.add_argument('--estimators', type=str, default='classical,uniform+pai,faq')
    args = p.parse_args()

    est_set = set(s.strip() for s in args.estimators.split(','))
    valid = {'classical', 'uniform+pai', 'faq'}
    bad = est_set - valid
    if bad:
        raise ValueError(f"Unknown estimators {bad}; valid: {valid}")
    print(f"Running estimators: {sorted(est_set)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    np.random.seed(args.seed)
    income_features, income, _ = get_data(year=args.year, features=FEATURES, outcome='PINCP')
    n_all = len(income)
    n_tr = args.n_labeled if args.n_labeled is not None else int(n_all * args.train_frac)
    print(f"N = {n_all}, labeled = {n_tr}, unlabeled = {n_all - n_tr}")

    feats_lab, feats_unlab, y_lab, y_unlab = train_test_split(
        income_features, income, train_size=n_tr, random_state=args.seed
    )
    y_lab = y_lab.to_numpy()
    y_unlab = y_unlab.to_numpy()

    _, enc = transform_features(income_features, FT)
    Phi_lab, _ = transform_features(feats_lab, FT, enc)
    Phi_unlab, _ = transform_features(feats_unlab, FT, enc)

    if args.n_max is not None and args.n_max < Phi_unlab.shape[0]:
        Phi_unlab = Phi_unlab[:args.n_max]
        y_unlab = y_unlab[:args.n_max]

    # fit SVD on the full matrix so V is shared across labeled and unlabeled
    Phi_all = scipy.sparse.vstack([Phi_lab, Phi_unlab]).tocsr().astype(np.float32)
    print(f"SVD (D={args.D}) on {Phi_all.shape}...")
    U_svd, s_svd, Vt_svd = scipy.sparse.linalg.svds(Phi_all, k=args.D)
    order = np.argsort(-s_svd)
    U_svd = U_svd[:, order]; s_svd = s_svd[order]; Vt_svd = Vt_svd[order, :]
    V_all = U_svd * s_svd                                    # (n_all, D)
    V_lab_np = V_all[:Phi_lab.shape[0]]
    V_unlab_np = V_all[Phi_lab.shape[0]:]
    if args.n_max is not None:
        V_unlab_np = V_unlab_np[:args.n_max]

    # z-score Y using labeled stats
    mu_Y, sd_Y = y_lab.mean(), y_lab.std()
    y_lab_n = ((y_lab - mu_Y) / sd_Y).astype(np.float32)
    y_unlab_n = ((y_unlab - mu_Y) / sd_Y).astype(np.float32)
    theta_true = float(y_unlab_n.mean())
    print(f"theta_true = {theta_true:.4f}")

    mu_post, Sig_post, sigma2 = fit_blr(
        V_lab_np.astype(np.float64), y_lab_n.astype(np.float64), prior_tau=args.prior_tau,
    )
    print(f"BLR sigma2 = {sigma2:.4f}")

    V = torch.tensor(V_unlab_np, dtype=torch.float32, device=device)
    Y = torch.tensor(y_unlab_n, dtype=torch.float32, device=device)
    MU0 = torch.tensor(mu_post, dtype=torch.float32, device=device)
    SIGMA0 = torch.tensor(Sig_post, dtype=torch.float32, device=device)
    sigma2_t = float(sigma2)

    N_QUESTIONS = V.shape[0]
    N_NEW = args.num_trials
    budgets = np.linspace(args.budget_min, args.budget_max, args.num_budgets)
    z_score = norm.ppf(1 - args.alpha / 2)

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

        if 'classical' in est_set:
            th_c, var_c = trial_classical(
                Y=Y, N_NEW=N_NEW, N_QUESTIONS=N_QUESTIONS, N_B=N_B,
                seed=seed_c, device=device,
            )
            record(th_c, var_c, "classical", N_B)
            del th_c, var_c

        # uniform+pai = trial() with tau=1 (pure uniform sampling)
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

    df = pd.DataFrame(
        rows,
        columns=["lb", "ub", "mean_width", "coverage", "estimator", "$n_b$",
                 "prop_budget", "seed"],
    )
    df.to_csv(os.path.abspath(args.out_csv), index=False)
    print(f"\nSaved {args.out_csv}")

    summary = df.groupby(['estimator', '$n_b$']).agg(
        width_mean=('mean_width', 'mean'),
        coverage_mean=('coverage', 'mean'),
    ).reset_index()
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
