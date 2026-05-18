"""
Empirical verification of two assumptions in the bias correction analysis:

  1. T_1 >> |T_2|, T_3  in the exact decomposition  B_hat - B = T_1 + T_2 - T_3
  2. B_s << A_s  at every step s  (model calibration property)

Runs the full WOR-FAQ loop on one (dataset, budget) setting with best-tuned
hyperparameters, tracking per-step A_s, B_s and cumulative T_1, T_2, T_3.

Outputs:
  logs/verify_bias/decomposition_{dataset}_{budget}.csv
      One row per seed: T1, T2, T3, sigma_bar_sq, sanity check
  logs/verify_bias/profiles_{dataset}_{budget}.csv
      One row per step: A_s, B_s, ratio (averaged over seeds and models)

Usage:
    python verify_bias_decomposition.py --dataset mmlu-pro --budget 0.25 [--n_seeds 100]

See notes/wor_bias_correction_report.md §2-3 for the theory.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from scipy.special import expit as sigmoid
from scipy.stats import norm
from tqdm import tqdm


def trial_decomposition(M2, V, MU0, SIGMA0, N_NEW, N_QUESTIONS, N_B,
                        beta0, rho, gamma, tau, seed, device):
    """Run one WOR-FAQ trial tracking the bias decomposition quantities.

    Returns dict with:
        T1, T2, T3: floats (averaged across models)
        Bhat_minus_Bn: float (averaged, for sanity check: should = T1 + T2 - T3)
        sigma_bar_sq: float (true variance, averaged across models)
        A_profile: (N_B,) array of A_s averaged across models
        B_profile: (N_B,) array of B_s averaged across models
    """
    # ---- Initialization (identical to trial_faq_wor) ----
    Uhats = torch.tile(MU0, dims=(N_NEW, 1)).to(device)
    Sigmahats = torch.tile(SIGMA0, dims=(N_NEW, 1, 1)).to(device)
    observed = torch.zeros(N_NEW, N_QUESTIONS, dtype=torch.bool, device=device)
    thetahats = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_main = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_b = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)

    torch.random.manual_seed(seed)

    # ---- Extra tracking for decomposition ----
    # True mean per model: θ = (1/N) Σ y_i
    theta = M2.float().mean(dim=1, keepdim=True)  # (N_NEW, 1)

    # T_1 = (1/n) Σ_{t>=1} (θ̂_t - θ)²   [0-indexed: t=1..N_B-1]
    T1_acc = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    # T_2 = (2/n) Σ_{t>=1} (θ̂_t - θ)(θ - ψ_{t+1})
    T2_acc = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    # T_3 = (1/n)(θ - ψ_1)²  — computed at t=0
    T3_val = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    # Population B_n = (1/n) Σ_t (θ - ψ_t)²  — for sanity check
    Bn_acc = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)

    # Per-step profiles (averaged across models at each step)
    A_profile = np.zeros(N_B, dtype=np.float64)
    B_profile = np.zeros(N_B, dtype=np.float64)

    # ---- Main loop ----
    for t in range(N_B):
        # ==== PART 1: Scoring and sampling probabilities (same as wor_trial.py) ====
        p_hat_js = torch.sigmoid(Uhats @ V.T).clamp(min=1e-12, max=1.0 - 1e-12)
        p1mp_hat_js = p_hat_js * (1.0 - p_hat_js)
        sqrt_p1mp_hat_js = torch.sqrt(p1mp_hat_js)

        sqrt_p1mp_masked = sqrt_p1mp_hat_js.clone()
        sqrt_p1mp_masked[observed] = 0.0
        ho_js = sqrt_p1mp_masked / sqrt_p1mp_masked.sum(dim=1, keepdim=True).clamp(min=1e-12)

        Sigmahats = (Sigmahats + Sigmahats.mT) / 2.0

        _CHUNK_Q = 1024
        vtSigmav_js = torch.empty(N_NEW, N_QUESTIONS, device=device)
        for _q in range(0, N_QUESTIONS, _CHUNK_Q):
            _Vc = V[_q:_q + _CHUNK_Q]
            _SigV = Sigmahats @ _Vc.T
            _SigV.mul_(_Vc.T)
            vtSigmav_js[:, _q:_q + _CHUNK_Q] = _SigV.sum(dim=1)
            del _SigV
        log_denominator = torch.log1p(p1mp_hat_js * vtSigmav_js)
        sq_term = torch.bmm(
            Sigmahats, ((p1mp_hat_js @ V) / N_QUESTIONS).unsqueeze(-1)
        ).squeeze(-1) @ V.T
        log_numerator = (torch.log(p1mp_hat_js.clamp_min(1e-12))
                         + 2 * torch.log(sq_term.abs().clamp_min(1e-12)))
        log_d_js = log_numerator - log_denominator
        log_d_js[observed] = -float('inf')
        ha_js = torch.softmax(log_d_js, dim=1)

        alpha_s = (torch.maximum(
            torch.tensor(0.0),
            torch.tensor(1.0 - (t + 1.0) / (rho * N_B))
        ) if rho != 0.0 else 0.0)

        beta_s = (beta0 * torch.minimum(
            torch.tensor((t + 1.0) / (gamma * N_B)),
            torch.tensor(1.0)
        ) if gamma != 0.0 else beta0)

        hcat_js = (((1.0 - alpha_s) * ho_js) + (alpha_s * ha_js)) ** beta_s
        hcat_js[observed] = 0.0

        n_unobs = N_QUESTIONS - t
        hcat_sum = hcat_js.sum(dim=1, keepdim=True).clamp(min=1e-12)
        q_js = ((hcat_js / hcat_sum) * (1.0 - tau)) + (tau / n_unobs)
        q_js[observed] = 0.0

        # Free intermediate tensors
        del vtSigmav_js, log_denominator, sq_term, log_numerator, log_d_js
        del ha_js, ho_js, sqrt_p1mp_masked, hcat_js

        # ==== PART 2: Compute AIPW estimator ====
        I_t = torch.multinomial(input=q_js, num_samples=1)
        z_It = torch.gather(M2, dim=1, index=I_t)
        phat_It = torch.gather(p_hat_js, dim=1, index=I_t)
        q_It = torch.gather(q_js, dim=1, index=I_t)

        imputed_sum = (
            (observed.float() * M2).sum(dim=1, keepdim=True)
            + ((~observed).float() * p_hat_js).sum(dim=1, keepdim=True)
        )
        aipw_t = (z_It - phat_It) / q_It
        phi_t = imputed_sum + aipw_t

        # ==== PART 3: Track decomposition quantities ====
        # ψ_t = imputed_sum / N  (normalized)
        psi_t = imputed_sum / N_QUESTIONS  # (N_NEW, 1)

        # Accumulate population B_n: (θ - ψ_t)²
        Bn_acc += (theta - psi_t) ** 2

        # T_3: initial model error (only at t=0)
        if t == 0:
            T3_val = (theta - psi_t) ** 2

        # T_1 and T_2: for t >= 1 (paper: t >= 2)
        if t >= 1:
            theta_hat_prev = thetahats / (t * N_QUESTIONS)  # θ̂_t (normalized)
            T1_acc += (theta_hat_prev - theta) ** 2
            T2_acc += 2.0 * (theta_hat_prev - theta) * (theta - psi_t)

        # ==== PART 4: Per-step A_s, B_s ====
        unobs_mask = ~observed  # (N_NEW, N_QUESTIONS)
        residuals = (M2 - p_hat_js) * unobs_mask.float()
        q_safe = q_js.clamp(min=1e-12)
        A_s = ((residuals ** 2) / q_safe * unobs_mask.float()).sum(dim=1) / (N_QUESTIONS ** 2)
        B_s = (residuals.sum(dim=1) ** 2) / (N_QUESTIONS ** 2)
        A_profile[t] = A_s.mean().item()
        B_profile[t] = B_s.mean().item()

        # ==== PART 5: Update B_hat, A_hat, thetahats (same as wor_trial.py) ====
        if t >= 1:
            ntheta_prev = thetahats / t
            varhats_b += (ntheta_prev - imputed_sum) ** 2

        thetahats += phi_t
        varhats_main += (aipw_t ** 2)

        # ==== PART 6: Update observed set ====
        observed.scatter_(dim=1, index=I_t, value=True)

        # ==== PART 7: Update factor model (Sherman-Morrison) ====
        w_t = torch.gather(p1mp_hat_js, dim=1, index=I_t)
        v_It = V[I_t.flatten()]
        Sigma_v_It = torch.bmm(Sigmahats, v_It.unsqueeze(-1))
        vT_Sigma_v_It = torch.bmm(v_It.unsqueeze(-2), Sigma_v_It).squeeze(-1)
        denominator = 1.0 + (w_t * vT_Sigma_v_It)
        numerator = w_t.unsqueeze(-1) * (Sigma_v_It @ Sigma_v_It.mT)
        Sigmahats -= (numerator / denominator.unsqueeze(-1))
        Uhats += torch.bmm(
            Sigmahats, ((z_It - phat_It) * v_It).unsqueeze(-1)
        ).squeeze()

        # Free per-step tensors
        del residuals, q_safe, unobs_mask, p_hat_js, p1mp_hat_js, sqrt_p1mp_hat_js
        del q_js, I_t, z_It, phat_It, q_It, imputed_sum, aipw_t, phi_t, psi_t

    # ---- Finalize ----
    # Normalize by 1/n = 1/N_B
    T1 = (T1_acc / N_B).mean().item()
    T2 = (T2_acc / N_B).mean().item()
    T3 = (T3_val / N_B).mean().item()

    # Population B_n and estimator B_hat (for sanity check)
    Bn = (Bn_acc / (N_B * N_QUESTIONS ** 2)).mean().item()
    Bhat = (varhats_b / (N_B * N_QUESTIONS ** 2)).mean().item()
    Bhat_minus_Bn = Bhat - Bn

    sigma_bar_sq = float(np.mean(A_profile - B_profile))

    return {
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'Bhat_minus_Bn': Bhat_minus_Bn,
        'sigma_bar_sq': sigma_bar_sq,
        'A_profile': A_profile,
        'B_profile': B_profile,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mmlu-pro",
                        choices=["mmlu-pro", "bbh+gpqa+ifeval+math+musr"])
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--n_seeds", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_full_obs, mcar_obs_prob = None, 1.0

    # ---- Load data (same as verify_lambda.py) ----
    M2 = pd.read_csv(f"data/processed/{args.dataset}/M2.csv")
    M2 = torch.tensor(M2.iloc[:, 3:].to_numpy().astype(np.float32)).to(device)
    N_NEW, N_QUESTIONS = M2.shape

    U = torch.load(
        f"factor_models/final/{args.dataset}/U_nfobs={n_full_obs}_p={mcar_obs_prob}.pt"
    ).to(device)
    V = torch.load(
        f"factor_models/final/{args.dataset}/V_nfobs={n_full_obs}_p={mcar_obs_prob}.pt"
    ).to(device)
    MU0, SIGMA0 = U.mean(axis=0), torch.cov(U.T)

    N_B = int(N_QUESTIONS * args.budget)

    # ---- Load best hyperparameters ----
    best_settings = pd.read_csv("logs/val/wor_best_settings.csv")
    row = best_settings.query(
        f"dataset == '{args.dataset}'"
        f" and mcar_obs_prob == {mcar_obs_prob}"
        f" and prop_budget == {args.budget}"
    )
    if len(row) == 0:
        print(f"ERROR: No settings found for dataset={args.dataset}, budget={args.budget}")
        return
    beta0, rho, gamma, tau = row[["beta0", "rho", "gamma", "tau"]].values.flatten()

    print(f"Dataset: {args.dataset}, budget: {args.budget} (n={N_B}, N={N_QUESTIONS})")
    print(f"Hyperparameters: beta0={beta0}, rho={rho}, gamma={gamma}, tau={tau}")
    print(f"Running {args.n_seeds} seeds...")

    # ---- Run trials ----
    records = []
    # Online accumulators for per-step profiles (mean across seeds)
    A_sum = np.zeros(N_B, dtype=np.float64)
    B_sum = np.zeros(N_B, dtype=np.float64)
    A_sumsq = np.zeros(N_B, dtype=np.float64)
    B_sumsq = np.zeros(N_B, dtype=np.float64)

    for seed in tqdm(range(args.n_seeds), desc="seeds"):
        res = trial_decomposition(
            M2, V, MU0, SIGMA0,
            N_NEW, N_QUESTIONS, N_B,
            beta0, rho, gamma, tau, seed, device
        )

        # Sanity check: B_hat - B_n should equal T1 + T2 - T3
        check = res['T1'] + res['T2'] - res['T3']
        residual = abs(res['Bhat_minus_Bn'] - check)
        if residual > 1e-6 * max(abs(check), 1e-12):
            print(f"  WARNING seed={seed}: |sanity residual| = {residual:.2e} "
                  f"(Bhat-Bn={res['Bhat_minus_Bn']:.6e}, T1+T2-T3={check:.6e})")

        records.append({
            'seed': seed,
            'T1': res['T1'],
            'T2': res['T2'],
            'T3': res['T3'],
            'Bhat_minus_Bn': res['Bhat_minus_Bn'],
            'sigma_bar_sq': res['sigma_bar_sq'],
            'T1_over_sigma': res['T1'] / max(res['sigma_bar_sq'], 1e-15),
            'T2_over_sigma': res['T2'] / max(res['sigma_bar_sq'], 1e-15),
            'T3_over_sigma': res['T3'] / max(res['sigma_bar_sq'], 1e-15),
        })

        A_sum += res['A_profile']
        B_sum += res['B_profile']
        A_sumsq += res['A_profile'] ** 2
        B_sumsq += res['B_profile'] ** 2

    # ---- Save decomposition CSV ----
    outdir = "logs/verify_bias"
    os.makedirs(outdir, exist_ok=True)

    df = pd.DataFrame(records)
    tag = f"{args.dataset}_{args.budget}"
    df.to_csv(f"{outdir}/decomposition_{tag}.csv", index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DECOMPOSITION SUMMARY ({args.dataset}, budget={args.budget})")
    print(f"{'='*60}")
    print(f"  E[T1]       = {df['T1'].mean():.6e}  (se={df['T1'].std()/np.sqrt(len(df)):.2e})")
    print(f"  E[T2]       = {df['T2'].mean():.6e}  (se={df['T2'].std()/np.sqrt(len(df)):.2e})")
    print(f"  E[T3]       = {df['T3'].mean():.6e}  (se={df['T3'].std()/np.sqrt(len(df)):.2e})")
    print(f"  sigma_bar^2 = {df['sigma_bar_sq'].mean():.6e}")
    print(f"  delta=T1/s2 = {df['T1_over_sigma'].mean():.4f}")
    print(f"  T2/sigma^2  = {df['T2_over_sigma'].mean():.6e}")
    print(f"  T3/sigma^2  = {df['T3_over_sigma'].mean():.6e}")
    print(f"  |T2|/T1     = {abs(df['T2'].mean()) / max(df['T1'].mean(), 1e-15):.4e}")
    print(f"  T3/T1       = {df['T3'].mean() / max(df['T1'].mean(), 1e-15):.4e}")

    # ---- Save profile CSV ----
    ns = args.n_seeds
    A_mean = A_sum / ns
    B_mean = B_sum / ns
    A_se = np.sqrt((A_sumsq / ns - A_mean ** 2).clip(min=0) / ns)
    B_se = np.sqrt((B_sumsq / ns - B_mean ** 2).clip(min=0) / ns)
    v_mean = A_mean - B_mean
    ratio_mean = np.where(A_mean > 1e-15, B_mean / A_mean, 0.0)

    prof_df = pd.DataFrame({
        'step': np.arange(N_B),
        'A_s': A_mean,
        'A_s_se': A_se,
        'B_s': B_mean,
        'B_s_se': B_se,
        'v_s': v_mean,
        'B_over_A': ratio_mean,
    })
    prof_df.to_csv(f"{outdir}/profiles_{tag}.csv", index=False)

    print(f"\n  B_s/A_s ratio across steps:")
    print(f"    mean  = {ratio_mean.mean():.6f}")
    print(f"    max   = {ratio_mean.max():.6f}")
    print(f"    at s=0: {ratio_mean[0]:.6f}")
    if N_B > 1:
        print(f"    at s={N_B-1}: {ratio_mean[-1]:.6f}")

    print(f"\nSaved: {outdir}/decomposition_{tag}.csv")
    print(f"Saved: {outdir}/profiles_{tag}.csv")


if __name__ == "__main__":
    main()
