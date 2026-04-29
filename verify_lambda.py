"""
Verify the non-stationarity index Λ predicts WOR FAQ coverage.

Runs trial_faq_wor with log_profile=True on one (dataset, budget) setting,
computes Λ from the per-step variance profile v_s, and checks:

    predicted_coverage = 0.95 − 0.114 × (Λ − Λ₃) / n
    ≈ actual_coverage (within ± 0.005)

See notes/wor_coverage_analysis.md §4.3–4.5 for the derivation.

Usage:
    python verify_lambda.py [--dataset mmlu-pro] [--budget 0.25] [--n_seeds 100]
"""
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from wor_trial import trial_faq_wor


def compute_weights(n):
    """Compute exact weights w_s = Σ_{k=s}^{n-1} 1/k² for s = 1, ..., n-1.

    Returns (n-1,) array. w_s is a decreasing function: w_1 ≈ π²/6, w_{n-1} = 1/(n-1)².
    """
    ks = np.arange(1, n)  # k = 1, ..., n-1
    inv_k_sq = 1.0 / (ks ** 2)
    # w_s = cumulative sum from the right: w_s = Σ_{k=s}^{n-1} 1/k²
    w = np.cumsum(inv_k_sq[::-1])[::-1]
    return w


def compute_lambda(v_profile):
    """Compute non-stationarity index Λ from per-step variance profile.

    Uses exact weights: Λ = (1/σ̄²) Σ_{s=1}^{n-1} v_s · w_s
    where w_s = Σ_{k=s}^{n-1} 1/k² and σ̄² = (1/n) Σ v_s.

    Also computes harmonic approximation Λ_H = Σ g(s)/s for comparison.
    """
    n = len(v_profile)
    sigma_bar_sq = np.mean(v_profile)
    if sigma_bar_sq < 1e-15:
        return 0.0, 0.0, sigma_bar_sq

    # Exact weights
    w = compute_weights(n)
    Lambda = np.sum(v_profile[:n - 1] * w) / sigma_bar_sq

    # Harmonic approximation for comparison
    g = v_profile / sigma_bar_sq
    indices = np.arange(1, n)
    Lambda_harmonic = np.sum(g[:n - 1] / indices)

    return Lambda, Lambda_harmonic, sigma_bar_sq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mmlu-pro",
                        choices=["mmlu-pro", "bbh+gpqa+ifeval+math+musr"])
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--n_seeds", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_full_obs, mcar_obs_prob = None, 1.0

    # Load data
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

    # Load best hyperparameters
    best_settings = pd.read_csv("logs/val/wor_best_settings.csv")
    row = best_settings.query(
        f"dataset == '{args.dataset}'"
        f" and mcar_obs_prob == {mcar_obs_prob}"
        f" and prop_budget == {args.budget}"
    )
    beta0, rho, gamma, tau = row[["beta0", "rho", "gamma", "tau"]].values.flatten()

    print(f"Dataset: {args.dataset}, budget: {args.budget} (n={N_B}, N={N_QUESTIONS})")
    print(f"Hyperparameters: beta0={beta0}, rho={rho}, gamma={gamma}, tau={tau}")
    print(f"Running {args.n_seeds} seeds with log_profile=True...")

    # Run trials
    lambdas = []
    coverages = []
    all_profiles = []

    for seed in range(args.n_seeds):
        mw, cov, v_profile = trial_faq_wor(
            M2, V, MU0, SIGMA0,
            N_NEW, N_QUESTIONS, N_B,
            beta0, rho, gamma, tau, seed, device,
            ALPHA=0.05, counter=seed, disable_tqdm=True,
            log_profile=True)

        Lambda, Lambda_H, sigma_bar_sq = compute_lambda(v_profile)
        lambdas.append(Lambda)
        coverages.append(cov)
        all_profiles.append(v_profile)

        if (seed + 1) % 10 == 0:
            print(f"  seed {seed + 1}/{args.n_seeds} done")

    # Aggregate
    Lambda_mean = np.mean(lambdas)
    coverage_actual = np.mean(coverages)
    coverage_se = np.std(coverages) / np.sqrt(args.n_seeds)

    # Compute Λ₃ = (θ − ψ̄₁)² / σ̄²
    theta = M2.mean().item()
    # Initial model prediction: ψ̄₁ = (1/N) Σ sigmoid(μ₀ · v_j)
    from torch.nn.functional import sigmoid
    p_hat_init = sigmoid(MU0 @ V.T).mean().item()
    mean_profile = np.mean(all_profiles, axis=0)
    sigma_bar_sq = np.mean(mean_profile)
    Lambda3 = (theta - p_hat_init) ** 2 / sigma_bar_sq if sigma_bar_sq > 1e-15 else 0.0

    # Also compute Λ from the mean profile (more stable than averaging per-seed Λ)
    Lambda_from_mean, Lambda_H_from_mean, _ = compute_lambda(mean_profile)

    # Predicted coverage
    z = norm.ppf(0.975)
    phi_z = norm.pdf(z)
    prefactor = z * phi_z  # ≈ 0.114
    coverage_predicted = 0.95 - prefactor * (Lambda_mean - Lambda3) / N_B
    coverage_predicted_mean = 0.95 - prefactor * (Lambda_from_mean - Lambda3) / N_B

    # Stationary benchmark
    H_n = np.sum(1.0 / np.arange(1, N_B))  # H_{n-1}
    Lambda_excess = Lambda_mean - H_n

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"n (budget)            = {N_B}")
    print(f"N (questions)         = {N_QUESTIONS}")
    print(f"θ (true mean)         = {theta:.4f}")
    print(f"ψ̄₁ (initial model)    = {p_hat_init:.4f}")
    print(f"σ̄² (avg cond var)     = {sigma_bar_sq:.6f}")
    print()
    print(f"Λ (exact weights)     = {Lambda_mean:.3f}")
    print(f"Λ (from mean profile) = {Lambda_from_mean:.3f}")
    print(f"Λ (harmonic approx)   = {Lambda_H_from_mean:.3f}")
    print(f"H_{{n-1}} (stationary)  = {H_n:.3f}")
    print(f"Λ / H_{{n-1}}           = {Lambda_mean / H_n:.2f}x")
    print(f"Λ_excess              = {Lambda_excess:.3f}")
    print(f"Λ₃ (initial model)    = {Lambda3:.3f}")
    print(f"Λ − Λ₃               = {Lambda_mean - Lambda3:.3f}")
    print()
    print(f"Λ/n (relative bias)   = {Lambda_mean / N_B:.4f}")
    print(f"H_{{n-1}}/n (stationary)= {H_n / N_B:.4f}")
    print()
    print(f"z·φ(z) prefactor      = {prefactor:.4f}")
    print(f"Predicted ε           = {(Lambda_mean - Lambda3) / (2 * N_B):.4f}")
    print(f"Predicted coverage    = {coverage_predicted:.4f}")
    print(f"  (from mean profile) = {coverage_predicted_mean:.4f}")
    print(f"Actual coverage       = {coverage_actual:.4f} +/- {coverage_se:.4f}")
    print(f"Difference            = {abs(coverage_predicted - coverage_actual):.4f}")
    print()

    if abs(coverage_predicted - coverage_actual) < 0.01:
        print("PASS: Prediction matches actual coverage within +/-0.01")
    else:
        print("FAIL: Prediction does NOT match actual coverage within +/-0.01")

    # Variance profile summary
    print("\nVariance profile v_s (averaged over seeds):")
    print(f"  v_1 (first step)    = {mean_profile[0]:.6f}")
    print(f"  v_{{n/2}}              = {mean_profile[N_B // 2]:.6f}")
    print(f"  v_n (last step)     = {mean_profile[-1]:.6f}")
    print(f"  v_1 / v_n ratio     = {mean_profile[0] / max(mean_profile[-1], 1e-15):.2f}")
    print(f"  w_1 (first weight)  = {compute_weights(N_B)[0]:.4f}")
    print(f"  w_{{n/2}}              = {compute_weights(N_B)[N_B // 2 - 1]:.6f}")
    print(f"  H_{{n-1}} (= sum w)   = {H_n:.4f}")


if __name__ == "__main__":
    main()
