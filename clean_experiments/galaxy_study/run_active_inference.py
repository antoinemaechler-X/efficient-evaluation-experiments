"""
Active Inference estimator for Galaxy Zoo 2 dataset.

Usage: python run_active_inference.py <seed_chunk>
  seed_chunk in {0, 1, 2}: splits 1000 seeds into 3 chunks.

Reproduces the Active Inference method from Zrnic & Candes (2024):
  https://github.com/tijana-zrnic/active-inference

  - tau tuned per budget by minimizing the variance proxy
    mean((Y - Yhat)^2 / pi(tau)) over a 100-point grid
  - Sampling: pi_i = clip((1-tau)*eta*uncertainty_i + tau*budget, 0, 1)
  - Estimator: mean(Yhat + (Y - Yhat)*xi/pi) with xi ~ Bernoulli(pi)
  - CI: point +/- z_{1-alpha/2} * std(increments) / sqrt(n)

Unlike the Pew study, Galaxy has no separate tuning split, so tau is tuned
on the same (Y, Yhat) population. Coverage is checked against theta* = mean(Y).
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
N_SEEDS = 1000

# Tau tuning grid: 100 values as in Zrnic's notebook
TAU_GRID = np.linspace(0.01, 0.99, 100)

if seed_chunk == 0:
    SEED_LIST = np.arange(0, 333)
elif seed_chunk == 1:
    SEED_LIST = np.arange(333, 666)
else:
    SEED_LIST = np.arange(666, 1000)

N_BATCH = len(SEED_LIST)
print(f"Seed chunk {seed_chunk}: seeds {SEED_LIST[0]}-{SEED_LIST[-1]} ({N_BATCH} seeds)")

# --- Load data ---
data_dir = "data"
Y = np.load(f"{data_dir}/Y.npy").flatten().astype(np.float32)
Yhat = np.load(f"{data_dir}/Yhat.npy").flatten().astype(np.float32)

theta_true = float(Y.mean())
N_total = len(Y)
print(f"N={N_total}, theta_true={theta_true:.6f}")

# No groups — single population
groups = {
    "all": np.ones(len(Y), dtype=bool),
}

# --- Output setup ---
os.makedirs("results", exist_ok=True)
columns = ["group", "prop_budget", "method", "seed", "width", "coverage"]
out_fname = f"results/active_inference_sl={seed_chunk}.csv"

if not os.path.exists(out_fname):
    with open(out_fname, "w") as f:
        f.write(",".join(columns) + "\n")

# Checkpoint per (group, budget): each produces 1 * N_BATCH rows
existing = pd.read_csv(out_fname)
checkpoint_counter = len(existing) // N_BATCH
counter = 0

z_score = norm.ppf(1 - ALPHA / 2)


def tune_tau(Y_t, Yhat_t, budget_prop):
    """
    Tune tau by minimizing the variance proxy mean((Y - Yhat)^2 / pi(tau)),
    exactly as in Zrnic's notebook.
    """
    uncertainty = np.minimum(Yhat_t, 1.0 - Yhat_t)
    residuals_sq = (Y_t - Yhat_t) ** 2
    eta = budget_prop / np.mean(uncertainty)

    best_var = np.inf
    best_tau = 0.5
    for tau in TAU_GRID:
        probs = np.clip((1.0 - tau) * eta * uncertainty + budget_prop * tau, 0, 1)
        var_tau = np.mean(residuals_sq / probs)
        if var_tau < best_var:
            best_var = var_tau
            best_tau = tau
    return best_tau


def run_active_inference(Y_group, Yhat_group, budget_prop, tuned_tau, rng_seed):
    """
    Run Active Inference for all seeds in this chunk.

    Returns: (widths, coverages) arrays of shape (N_BATCH,)
    """
    N = len(Y_group)

    uncertainty = np.minimum(Yhat_group, 1.0 - Yhat_group)
    eta = budget_prop / np.mean(uncertainty)
    pi = np.clip((1.0 - tuned_tau) * eta * uncertainty + tuned_tau * budget_prop, 0, 1)

    Y_t = torch.tensor(Y_group, device=device).unsqueeze(0).expand(N_BATCH, -1)
    Yhat_t = torch.tensor(Yhat_group, device=device).unsqueeze(0).expand(N_BATCH, -1)
    pi_t = torch.tensor(pi, device=device, dtype=torch.float32).unsqueeze(0).expand(N_BATCH, -1)

    torch.manual_seed(rng_seed)
    xi = torch.bernoulli(pi_t)  # (N_BATCH, N)

    increments = Yhat_t + (Y_t - Yhat_t) * xi / pi_t  # (N_BATCH, N)
    theta_hat = increments.mean(dim=1)  # (N_BATCH,)
    se = increments.std(dim=1, correction=0) / np.sqrt(N)  # ddof=0, matches Zrnic

    halfwidth = z_score * se
    ub = theta_hat + halfwidth
    lb = theta_hat - halfwidth

    widths = (2 * halfwidth).cpu().numpy()
    coverages = ((lb <= theta_true) & (theta_true <= ub)).float().cpu().numpy()

    return widths, coverages


# --- Main loop ---
total = len(groups) * len(BUDGET_PROPS)
print(f"Total (group, budget) pairs: {total} (x 1 method x {N_BATCH} seeds each)")

for group_idx, (group_name, group_mask) in enumerate(groups.items()):
    Y_g = Y[group_mask]
    Yhat_g = Yhat[group_mask]
    N_g = len(Y_g)
    print(f"\nGroup: {group_name} (N={N_g})")

    for budget_idx, budget_prop in enumerate(BUDGET_PROPS):

        if counter >= checkpoint_counter:
            tuned_tau = tune_tau(Y_g, Yhat_g, budget_prop)

            rng_seed = seed_chunk * 100000 + group_idx * 10000 + budget_idx * 100
            w, c = run_active_inference(Y_g, Yhat_g, budget_prop, tuned_tau, rng_seed)

            with open(out_fname, "a") as f:
                for i, seed in enumerate(SEED_LIST):
                    f.write(f"{group_name},{budget_prop:.4f},active-inference,{seed},{w[i]},{c[i]}\n")

            print(f"  budget={budget_prop:.4f} (n={max(1, int(N_g * budget_prop))}): "
                  f"tau={tuned_tau:.3f}, width={w.mean():.6f}, coverage={c.mean():.3f}")

        counter += 1

print(f"\nDone! Results saved to {out_fname}")
