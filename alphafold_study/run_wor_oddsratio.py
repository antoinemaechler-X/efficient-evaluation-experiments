"""
WOR-AIPW ODDS RATIO estimator for AlphaFold dataset.

Extends WOR-AIPW (our method) to estimate the odds ratio:
  OR = (mu1/(1-mu1)) / (mu0/(1-mu0))
using delta method CI on log-odds-ratio scale.

Runs WOR independently per group, then combines estimates
via delta method (same as Zrnic's approach for Bernoulli).

Usage: python alphafold_study/run_wor_oddsratio.py <seed_chunk>
  seed_chunk in {0, 1, 2}: splits 1000 seeds into 3 chunks.
"""
import numpy as np
import pandas as pd
import torch
import sys
import os
from scipy.stats import norm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

seed_chunk = int(sys.argv[1])

# --- Parameters ---
BUDGET_PROPS = np.round(np.linspace(0.01, 0.2, 20), decimals=4)
ALPHA = 0.05
TAU = 0.5
N_SEEDS = 1000

if seed_chunk == 0:
    SEED_LIST = np.arange(0, 333)
elif seed_chunk == 1:
    SEED_LIST = np.arange(333, 666)
else:
    SEED_LIST = np.arange(666, 1000)

N_BATCH = len(SEED_LIST)
print(f"Seed chunk {seed_chunk}: seeds {SEED_LIST[0]}-{SEED_LIST[-1]} ({N_BATCH} seeds)")

# --- Load data ---
data_dir = "alphafold_study/data"
Y = np.load(f"{data_dir}/Y.npy").flatten().astype(np.float32)
Yhat = np.load(f"{data_dir}/Yhat.npy").flatten().astype(np.float32)
Z = np.load(f"{data_dir}/Z.npy").flatten().astype(np.float32)

mask0 = Z == 0
mask1 = Z == 1
Y0, Yhat0 = Y[mask0], Yhat[mask0]
Y1, Yhat1 = Y[mask1], Yhat[mask1]
n0, n1 = len(Y0), len(Y1)

mu0_true = Y0.mean()
mu1_true = Y1.mean()
true_odds_ratio = (mu1_true / (1 - mu1_true)) / (mu0_true / (1 - mu0_true))
print(f"Group 0 (non-phos): n={n0}, mu={mu0_true:.4f}")
print(f"Group 1 (phos):     n={n1}, mu={mu1_true:.4f}")
print(f"True odds ratio:    {true_odds_ratio:.4f}")

p0_frac = n0 / (n0 + n1)
p1_frac = n1 / (n0 + n1)

# --- Output setup ---
os.makedirs("alphafold_study/logs", exist_ok=True)
columns = ["prop_budget", "method", "seed", "width", "coverage"]
logs_fname = f"alphafold_study/logs/wor_oddsratio_sl={seed_chunk}.csv"

if not os.path.exists(logs_fname):
    with open(logs_fname, "w") as f:
        f.write(",".join(columns) + "\n")

existing = pd.read_csv(logs_fname)
checkpoint_counter = len(existing) // (2 * N_BATCH)
counter = 0

z_score = norm.ppf(1 - ALPHA / 2)


def run_wor_group(Y_g, Yhat_g, N_B, active, rng_seed):
    """
    Run WOR on one group. Returns per-seed point estimates and variances
    (not widths/coverages — those are computed after combining groups).

    Returns:
        theta_hat: (N_BATCH,) — WOR point estimate per seed
        var_hat: (N_BATCH,) — WOR variance estimate per seed (Theorem 4)
    """
    Y_t = torch.tensor(Y_g, device=device)
    Yhat_t = torch.tensor(Yhat_g, device=device)
    N = len(Y_g)

    Y_batch = Y_t.unsqueeze(0).expand(N_BATCH, -1)
    Yhat_batch = Yhat_t.unsqueeze(0).expand(N_BATCH, -1)

    if active:
        scores = torch.minimum(Yhat_t, 1.0 - Yhat_t).clamp(min=1e-12)
    else:
        scores = torch.ones(N, device=device)

    observed = torch.zeros(N_BATCH, N, dtype=torch.bool, device=device)
    thetahats = torch.zeros(N_BATCH, 1, dtype=torch.float32, device=device)
    varhats_main = torch.zeros(N_BATCH, 1, dtype=torch.float32, device=device)
    varhats_b = torch.zeros(N_BATCH, 1, dtype=torch.float32, device=device)

    torch.manual_seed(rng_seed)

    for t in range(N_B):
        n_unobs = N - t

        q_scores = scores.unsqueeze(0).expand(N_BATCH, -1).clone()
        q_scores[observed] = 0.0
        q_sum = q_scores.sum(dim=1, keepdim=True).clamp(min=1e-12)

        if active:
            q_js = ((q_scores / q_sum) * (1.0 - TAU)) + (TAU / n_unobs)
            q_js[observed] = 0.0
        else:
            q_js = q_scores / q_sum

        I_t = torch.multinomial(q_js, num_samples=1)

        z_It = torch.gather(Y_batch, 1, I_t)
        f_It = torch.gather(Yhat_batch, 1, I_t)
        q_It = torch.gather(q_js, 1, I_t)

        imputed_sum = (
            (observed.float() * Y_batch).sum(dim=1, keepdim=True)
            + ((~observed).float() * Yhat_batch).sum(dim=1, keepdim=True)
        )

        aipw_t = (z_It - f_It) / q_It
        phi_t = imputed_sum + aipw_t

        if t >= 1:
            ntheta_prev = thetahats / t
            varhats_b += (ntheta_prev - imputed_sum) ** 2

        thetahats += phi_t
        varhats_main += aipw_t ** 2
        observed.scatter_(1, I_t, True)

    # Theorem 4 variance
    theta_T = (thetahats / (N_B * N)).squeeze(1)  # (N_BATCH,)
    v_simp = varhats_main / (N_B * N ** 2)
    v_minus = varhats_b / (N_B * N ** 2)
    v_full = ((v_simp - v_minus).clamp(min=0) / N_B).squeeze(1)  # (N_BATCH,)

    return theta_T, v_full


# --- Main loop ---
total = len(BUDGET_PROPS)
print(f"Total budget props: {total} (x 2 methods x {N_BATCH} seeds each)")

for budget_idx, budget_prop in enumerate(BUDGET_PROPS):

    if counter >= checkpoint_counter:
        base_seed = seed_chunk * 100000 + budget_idx * 100

        results = {}
        for method_idx, (method, active) in enumerate([
            ("wor-active", True),
            ("wor-uniform", False),
        ]):
            # Independent seeds per group
            seed0 = base_seed + method_idx * 1000
            seed1 = base_seed + method_idx * 1000 + 500

            N_B0 = max(1, int(n0 * budget_prop))
            N_B1 = max(1, int(n1 * budget_prop))

            theta0, var0 = run_wor_group(Y0, Yhat0, N_B0, active, seed0)
            theta1, var1 = run_wor_group(Y1, Yhat1, N_B1, active, seed1)

            # Clamp for numerical stability
            theta0 = theta0.clamp(min=1e-6, max=1 - 1e-6)
            theta1 = theta1.clamp(min=1e-6, max=1 - 1e-6)

            # Log-odds ratio + delta method (same as Bernoulli version)
            log_or = torch.log(theta1 / (1 - theta1)) - torch.log(theta0 / (1 - theta0))

            delta_var0 = var0 / (theta0 * (1 - theta0)) ** 2
            delta_var1 = var1 / (theta1 * (1 - theta1)) ** 2
            total_var = delta_var0 / p0_frac + delta_var1 / p1_frac
            se = torch.sqrt(total_var / (n0 + n1))

            lb = torch.exp(log_or - z_score * se)
            ub = torch.exp(log_or + z_score * se)

            widths = (ub - lb).cpu().numpy()
            coverages = ((lb <= true_odds_ratio) & (true_odds_ratio <= ub)).float().cpu().numpy()

            results[method] = (widths, coverages)

        with open(logs_fname, "a") as f:
            for method in ["wor-active", "wor-uniform"]:
                w, c = results[method]
                for i, seed in enumerate(SEED_LIST):
                    f.write(f"{budget_prop:.4f},{method},{seed},{w[i]},{c[i]}\n")

        print(f"  budget={budget_prop:.4f}: "
              f"active width={results['wor-active'][0].mean():.4f}, "
              f"cov={results['wor-active'][1].mean():.3f}, "
              f"uniform width={results['wor-uniform'][0].mean():.4f}")

    counter += 1

print(f"\nDone! Results saved to {logs_fname}")
