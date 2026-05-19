"""
Bernoulli-based ODDS RATIO estimator for AlphaFold dataset.

Reproduces Zrnic & Candès (2024) alphafold-ptm experiment:
- Estimand: odds ratio = (mu1/(1-mu1)) / (mu0/(1-mu0))
  where mu0 = disorder rate in non-phosphorylated, mu1 = phosphorylated
- Delta method CI on log-odds-ratio scale, then exponentiated
- No CI clipping (matches Zrnic exactly)
- ddof=0 for variance (matches Zrnic's np.var)

Usage: python alphafold_study/run_bernoulli_oddsratio.py <seed_chunk>
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
TAU = 0.5  # Hardcoded, matches Zrnic's alphafold-ptm.ipynb
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

# Split by phosphorylation status
mask0 = Z == 0  # non-phosphorylated
mask1 = Z == 1  # phosphorylated
Y0, Yhat0 = Y[mask0], Yhat[mask0]
Y1, Yhat1 = Y[mask1], Yhat[mask1]
n0, n1 = len(Y0), len(Y1)

# True odds ratio (population parameter)
mu0_true = Y0.mean()
mu1_true = Y1.mean()
true_odds_ratio = (mu1_true / (1 - mu1_true)) / (mu0_true / (1 - mu0_true))
print(f"Group 0 (non-phos): n={n0}, mu={mu0_true:.4f}")
print(f"Group 1 (phos):     n={n1}, mu={mu1_true:.4f}")
print(f"True odds ratio:    {true_odds_ratio:.4f}")

# Group proportions for delta method
p0_frac = n0 / (n0 + n1)
p1_frac = n1 / (n0 + n1)

# --- Output setup ---
os.makedirs("alphafold_study/logs", exist_ok=True)
columns = ["prop_budget", "method", "seed", "width", "coverage"]
logs_fname = f"alphafold_study/logs/bernoulli_oddsratio_sl={seed_chunk}.csv"

if not os.path.exists(logs_fname):
    with open(logs_fname, "w") as f:
        f.write(",".join(columns) + "\n")

existing = pd.read_csv(logs_fname)
checkpoint_counter = len(existing) // (3 * N_BATCH)
counter = 0

z_score = norm.ppf(1 - ALPHA / 2)


def compute_aipw_group(Y_g, Yhat_g, pi_g, lhat, rng_seed):
    """
    Compute AIPW mean estimate and population variance for one group.

    Returns:
        mu_hat: (N_BATCH,) — AIPW point estimate per seed
        var_hat: (N_BATCH,) — population variance of AIPW increments (ddof=0)
    """
    N_g = len(Y_g)
    Y_t = torch.tensor(Y_g, device=device).unsqueeze(0).expand(N_BATCH, -1)
    Yhat_t = torch.tensor(Yhat_g, device=device).unsqueeze(0).expand(N_BATCH, -1)
    pi_t = pi_g.unsqueeze(0).expand(N_BATCH, -1)

    torch.manual_seed(rng_seed)
    xi = torch.bernoulli(pi_t)  # (N_BATCH, N_g)

    # AIPW increments: lhat * Yhat + (Y - lhat * Yhat) * xi / pi
    increments = lhat * Yhat_t + (Y_t - lhat * Yhat_t) * xi / pi_t  # (N_BATCH, N_g)

    mu_hat = increments.mean(dim=1)              # (N_BATCH,)
    var_hat = increments.var(dim=1, correction=0)  # (N_BATCH,) — ddof=0, matches Zrnic

    return mu_hat, var_hat


def run_odds_ratio(budget_prop, rng_seed):
    """
    Run 3 Bernoulli methods, compute odds ratio CI via delta method.
    Exactly follows Zrnic's odds_ratio_ci function from alphafold-ptm.ipynb.

    Returns: dict of method -> (widths, coverages) arrays of shape (N_BATCH,)
    """
    # Sampling probabilities per group (Zrnic computes independently per group)
    uncertainty0 = torch.tensor(np.minimum(Yhat0, 1 - Yhat0), device=device).clamp(min=1e-12)
    uncertainty1 = torch.tensor(np.minimum(Yhat1, 1 - Yhat1), device=device).clamp(min=1e-12)

    eta0 = budget_prop / uncertainty0.mean()
    eta1 = budget_prop / uncertainty1.mean()

    pi_active0 = ((1 - TAU) * eta0 * uncertainty0 + TAU * budget_prop).clamp(0, 1)
    pi_active1 = ((1 - TAU) * eta1 * uncertainty1 + TAU * budget_prop).clamp(0, 1)
    pi_uniform0 = torch.full((n0,), budget_prop, device=device)
    pi_uniform1 = torch.full((n1,), budget_prop, device=device)

    results = {}

    for method_idx, (method, pi0, pi1, lhat) in enumerate([
        ("bernoulli-active", pi_active0, pi_active1, 1.0),
        ("bernoulli-uniform", pi_uniform0, pi_uniform1, 1.0),
        ("classical", pi_uniform0, pi_uniform1, 0.0),
    ]):
        # Independent seeds for each group (avoid correlation)
        seed0 = rng_seed + method_idx * 1000
        seed1 = rng_seed + method_idx * 1000 + 500

        mu0_hat, var0 = compute_aipw_group(Y0, Yhat0, pi0, lhat, seed0)
        mu1_hat, var1 = compute_aipw_group(Y1, Yhat1, pi1, lhat, seed1)

        # Clamp mu away from 0 and 1 for numerical stability
        mu0_hat = mu0_hat.clamp(min=1e-6, max=1 - 1e-6)
        mu1_hat = mu1_hat.clamp(min=1e-6, max=1 - 1e-6)

        # Log-odds ratio (Zrnic's pointest_log)
        log_or = torch.log(mu1_hat / (1 - mu1_hat)) - torch.log(mu0_hat / (1 - mu0_hat))

        # Delta method variance (exactly as Zrnic)
        delta_var0 = var0 / (mu0_hat * (1 - mu0_hat)) ** 2
        delta_var1 = var1 / (mu1_hat * (1 - mu1_hat)) ** 2
        total_var = delta_var0 / p0_frac + delta_var1 / p1_frac
        se = torch.sqrt(total_var / (n0 + n1))

        # CI on log scale, then exponentiate (no clipping — matches Zrnic)
        lb = torch.exp(log_or - z_score * se)
        ub = torch.exp(log_or + z_score * se)

        widths = (ub - lb).cpu().numpy()
        coverages = ((lb <= true_odds_ratio) & (true_odds_ratio <= ub)).float().cpu().numpy()

        results[method] = (widths, coverages)

    return results


# --- Main loop ---
total = len(BUDGET_PROPS)
print(f"Total budget props: {total} (x 3 methods x {N_BATCH} seeds each)")

for budget_idx, budget_prop in enumerate(BUDGET_PROPS):

    if counter >= checkpoint_counter:
        rng_seed = seed_chunk * 100000 + budget_idx * 100
        res = run_odds_ratio(budget_prop, rng_seed)

        with open(logs_fname, "a") as f:
            for method in ["bernoulli-active", "bernoulli-uniform", "classical"]:
                w, c = res[method]
                for i, seed in enumerate(SEED_LIST):
                    f.write(f"{budget_prop:.4f},{method},{seed},{w[i]},{c[i]}\n")

        print(f"  budget={budget_prop:.4f}: "
              f"active width={res['bernoulli-active'][0].mean():.4f}, "
              f"cov={res['bernoulli-active'][1].mean():.3f}, "
              f"uniform width={res['bernoulli-uniform'][0].mean():.4f}, "
              f"classical width={res['classical'][0].mean():.4f}")

    counter += 1

print(f"\nDone! Results saved to {logs_fname}")
