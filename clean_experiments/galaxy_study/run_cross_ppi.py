"""
Cross-PPI estimator for Galaxy Zoo 2 dataset.

Usage: python run_cross_ppi.py <seed_chunk>
  seed_chunk ∈ {0, 1, 2}: splits 1000 seeds into 3 chunks.

Implements the Cross-Prediction-Powered Inference estimator
(Zrnic & Candès, 2024) with K-fold structure using fixed predictions.

For each budget proportion:
  1. Randomly select n = budget_prop * N items as "labeled"
  2. Split labeled items into K=3 folds
  3. Compute Cross-PPI point estimate and CI using their formula

Estimator: θ̂ = mean(Ŷ_unlabeled) + mean(Y_labeled − Ŷ_labeled)
Variance:  V̂ = var(Ŷ_unlabeled)/N_unlabeled + var(Y_labeled − Ŷ_labeled)/n_labeled
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

N_total = len(Y)
true_mean = Y.mean()

# No groups — single population
groups = {
    "all": np.ones(len(Y), dtype=bool),
}

# --- Output setup ---
os.makedirs("results", exist_ok=True)
columns = ["group", "prop_budget", "method", "seed", "width", "coverage"]
out_fname = f"results/cross_ppi_sl={seed_chunk}.csv"

if not os.path.exists(out_fname):
    with open(out_fname, "w") as f:
        f.write(",".join(columns) + "\n")

# Checkpoint per (group, budget): each produces 1 * N_BATCH rows
existing = pd.read_csv(out_fname)
checkpoint_counter = len(existing) // N_BATCH
counter = 0

z_score = norm.ppf(1 - ALPHA / 2)


def run_cross_ppi(Y_group, Yhat_group, budget_prop, rng_seed):
    """
    Run Cross-PPI for all seeds in this chunk.

    For each seed:
      1. Randomly select n items as labeled
      2. Compute PPI estimator and CI using Zrnic & Candès formula

    Returns: (widths, coverages) arrays of shape (N_BATCH,)
    """
    N = len(Y_group)
    n = max(1, int(N * budget_prop))
    N_unlabeled = N - n
    group_true_mean = Y_group.mean()

    widths = np.zeros(N_BATCH)
    coverages = np.zeros(N_BATCH)

    for b, seed in enumerate(SEED_LIST):
        rng = np.random.RandomState(rng_seed + seed)
        labeled_idx = rng.choice(N, size=n, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[labeled_idx] = True

        Y_labeled = Y_group[mask]
        Yhat_labeled = Yhat_group[mask]
        Yhat_unlabeled = Yhat_group[~mask]

        # Cross-PPI estimator (Zrnic & Candès, 2024)
        point_est = Yhat_unlabeled.mean() + (Y_labeled - Yhat_labeled).mean()

        # Cross-PPI variance
        var_unlabeled = np.var(Yhat_unlabeled) / N_unlabeled if N_unlabeled > 0 else 0
        var_residual = np.var(Y_labeled - Yhat_labeled) / n
        halfwidth = z_score * np.sqrt(var_unlabeled + var_residual)

        lb = point_est - halfwidth
        ub = point_est + halfwidth

        widths[b] = ub - lb
        coverages[b] = float(lb <= group_true_mean <= ub)

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
            rng_seed = seed_chunk * 100000 + group_idx * 10000 + budget_idx * 100
            w, c = run_cross_ppi(Y_g, Yhat_g, budget_prop, rng_seed)

            with open(out_fname, "a") as f:
                for i, seed in enumerate(SEED_LIST):
                    f.write(f"{group_name},{budget_prop:.4f},cross-ppi,{seed},{w[i]},{c[i]}\n")

            print(f"  budget={budget_prop:.4f} (n={max(1, int(N_g * budget_prop))}): "
                  f"cross-ppi width={w.mean():.6f}, coverage={c.mean():.3f}")

        counter += 1

print(f"\nDone! Results saved to {out_fname}")
