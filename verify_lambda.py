"""
Verify the non-stationarity index Lambda predicts WOR FAQ coverage.

Runs trial_faq_wor with log_profile=True on one (dataset, budget) setting,
computes Lambda from the per-step variance profile v_s, and checks:

    predicted_coverage = 0.95 - z*phi(z) * (Lambda - Lambda_3) / n
    ~ actual_coverage

Also computes exact-Phi prediction (no Taylor linearization) for comparison.

Results are printed AND appended to a CSV for easy aggregation across budgets.

See notes/wor_coverage_analysis.md Section 4.3-4.5 for the derivation.

Usage:
    python verify_lambda.py --dataset mmlu-pro --budget 0.25 [--n_seeds 100]
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from wor_trial import trial_faq_wor


def compute_weights(n):
    """Compute exact weights w_s = sum_{k=s}^{n-1} 1/k^2 for s = 1, ..., n-1.

    Returns (n-1,) array. w_s is a decreasing function: w_1 ~ pi^2/6, w_{n-1} = 1/(n-1)^2.
    """
    ks = np.arange(1, n)
    inv_k_sq = 1.0 / (ks ** 2)
    w = np.cumsum(inv_k_sq[::-1])[::-1]
    return w


def compute_lambda(v_profile):
    """Compute non-stationarity index Lambda from per-step variance profile.

    Uses exact weights: Lambda = (1/sigma_bar^2) sum_{s=1}^{n-1} v_s * w_s
    where w_s = sum_{k=s}^{n-1} 1/k^2 and sigma_bar^2 = (1/n) sum v_s.

    Also computes harmonic approximation Lambda_H = sum g(s)/s for comparison.

    Returns: (Lambda_exact, Lambda_harmonic, sigma_bar_sq)
    """
    n = len(v_profile)
    sigma_bar_sq = np.mean(v_profile)
    if sigma_bar_sq < 1e-15:
        return 0.0, 0.0, sigma_bar_sq

    w = compute_weights(n)
    Lambda = np.sum(v_profile[:n - 1] * w) / sigma_bar_sq

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
    parser.add_argument("--output_csv", default="logs/verify_lambda_results.csv",
                        help="CSV file to append results to")
    parser.add_argument("--save_profile", action="store_true",
                        help="Save mean variance profile to .npy file")
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

    # Load best hyperparameters (same file used by wor_faq_final.py)
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
    print(f"Running {args.n_seeds} seeds with log_profile=True...")

    # Run trials
    lambdas = []
    coverages = []
    mean_widths = []
    all_profiles = []

    for seed in range(args.n_seeds):
        result = trial_faq_wor(
            M2, V, MU0, SIGMA0,
            N_NEW, N_QUESTIONS, N_B,
            beta0, rho, gamma, tau, seed, device,
            ALPHA=0.05, counter=seed, disable_tqdm=True,
            log_profile=True)

        mw, cov, v_profile = result
        Lambda, Lambda_H, sigma_bar_sq = compute_lambda(v_profile)
        lambdas.append(Lambda)
        coverages.append(cov)
        mean_widths.append(mw)
        all_profiles.append(v_profile)

        if (seed + 1) % 10 == 0:
            print(f"  seed {seed + 1}/{args.n_seeds} done")

    # Aggregate
    Lambda_mean = np.mean(lambdas)
    Lambda_se = np.std(lambdas) / np.sqrt(args.n_seeds)
    coverage_actual = np.mean(coverages)
    coverage_se = np.std(coverages) / np.sqrt(args.n_seeds)
    mean_width_avg = np.mean(mean_widths)

    # Compute Lambda_3 = (theta - psi_bar_1)^2 / sigma_bar^2
    theta = M2.mean().item()
    from torch.nn.functional import sigmoid
    p_hat_init = sigmoid(MU0 @ V.T).mean().item()
    mean_profile = np.mean(all_profiles, axis=0)
    sigma_bar_sq = np.mean(mean_profile)
    Lambda3 = (theta - p_hat_init) ** 2 / sigma_bar_sq if sigma_bar_sq > 1e-15 else 0.0

    # Also compute Lambda from the mean profile (more stable)
    Lambda_from_mean, Lambda_H_from_mean, _ = compute_lambda(mean_profile)

    # --- Predictions ---
    z = norm.ppf(0.975)
    phi_z = norm.pdf(z)
    prefactor = z * phi_z  # ~ 0.1146

    # Linear prediction (first-order Taylor)
    bias_ratio = (Lambda_from_mean - Lambda3) / N_B
    cov_pred_linear = 0.95 - prefactor * bias_ratio

    # Exact-Phi prediction (no Taylor linearization)
    sigma_hat_over_sigma = np.sqrt(max(1.0 - bias_ratio, 0.0))
    cov_pred_exact_phi = 2.0 * norm.cdf(z * sigma_hat_over_sigma) - 1.0

    # Stationary benchmark
    H_n = np.sum(1.0 / np.arange(1, N_B))

    # --- Print results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"n (budget)            = {N_B}")
    print(f"N (questions)         = {N_QUESTIONS}")
    print(f"theta (true mean)     = {theta:.4f}")
    print(f"psi_1 (initial model) = {p_hat_init:.4f}")
    print(f"sigma_bar^2           = {sigma_bar_sq:.6f}")
    print()
    print(f"Lambda (exact)        = {Lambda_from_mean:.3f} +/- {Lambda_se:.3f}")
    print(f"Lambda (harmonic)     = {Lambda_H_from_mean:.3f}")
    print(f"H_{{n-1}} (stationary)  = {H_n:.3f}")
    print(f"Lambda / H_{{n-1}}      = {Lambda_from_mean / H_n:.2f}x")
    print(f"Lambda_3              = {Lambda3:.4f}")
    print(f"Lambda - Lambda_3     = {Lambda_from_mean - Lambda3:.3f}")
    print()
    print(f"(Lambda-Lambda_3)/n   = {bias_ratio:.6f}")
    print(f"H_{{n-1}}/n             = {H_n / N_B:.6f}")
    print()
    print(f"Hyperparameters: beta0={beta0}, rho={rho}, gamma={gamma}, tau={tau}")
    print()
    print(f"--- Coverage predictions ---")
    print(f"Predicted (linear)    = {cov_pred_linear:.5f}")
    print(f"Predicted (exact Phi) = {cov_pred_exact_phi:.5f}")
    print(f"Actual coverage       = {coverage_actual:.5f} +/- {coverage_se:.5f}")
    print(f"Error (linear)        = {cov_pred_linear - coverage_actual:+.5f}")
    print(f"Error (exact Phi)     = {cov_pred_exact_phi - coverage_actual:+.5f}")
    print(f"Mean width            = {mean_width_avg:.8f}")
    print()

    # Variance profile summary
    v1 = mean_profile[0]
    vn = mean_profile[-1]
    print(f"Variance profile:")
    print(f"  v_1 = {v1:.6f},  v_n = {vn:.6f},  v_1/v_n = {v1/max(vn, 1e-15):.2f}")
    print(f"  v_{{n/4}} = {mean_profile[N_B//4]:.6f},  v_{{n/2}} = {mean_profile[N_B//2]:.6f}")

    # --- Save to CSV ---
    csv_row = {
        "dataset": args.dataset,
        "prop_budget": args.budget,
        "n": N_B,
        "N": N_QUESTIONS,
        "n_seeds": args.n_seeds,
        "beta0": beta0, "rho": rho, "gamma": gamma, "tau": tau,
        "theta": theta,
        "psi_1": p_hat_init,
        "sigma_bar_sq": sigma_bar_sq,
        "Lambda": Lambda_from_mean,
        "Lambda_se": Lambda_se,
        "Lambda_harmonic": Lambda_H_from_mean,
        "H_n": H_n,
        "Lambda_over_Hn": Lambda_from_mean / H_n,
        "Lambda3": Lambda3,
        "bias_ratio": bias_ratio,
        "cov_pred_linear": cov_pred_linear,
        "cov_pred_exact_phi": cov_pred_exact_phi,
        "cov_actual": coverage_actual,
        "cov_se": coverage_se,
        "error_linear": cov_pred_linear - coverage_actual,
        "error_exact_phi": cov_pred_exact_phi - coverage_actual,
        "mean_width": mean_width_avg,
        "v1": v1,
        "vn": vn,
        "v1_over_vn": v1 / max(vn, 1e-15),
    }

    csv_path = args.output_csv
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write(",".join(csv_row.keys()) + "\n")
        f.write(",".join(str(v) for v in csv_row.values()) + "\n")
    print(f"\nResults appended to {csv_path}")

    # --- Save variance profile ---
    if args.save_profile:
        profile_path = f"logs/variance_profile_{args.dataset}_budget={args.budget}.npy"
        np.save(profile_path, mean_profile)
        print(f"Variance profile saved to {profile_path}")


if __name__ == "__main__":
    main()
