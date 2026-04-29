"""
WOR-AIPW estimator for AlphaFold dataset.

Usage: python alphafold_study/run_wor.py <seed_chunk>
  seed_chunk ∈ {0, 1, 2}: splits 1000 seeds into 3 chunks.

Runs WOR-active and WOR-uniform for 2 groups × 20 budgets.
No factor model — uses fixed AlphaFold predictions.
Scoring: min(Ŷ, 1-Ŷ) for active, uniform for uniform.
AIPW estimator (Eq. 2) with Theorem 4 variance.
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

# Split seeds into 3 chunks
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

groups = {
    "non-phosphorylated": Z == 0,
    "phosphorylated": Z == 1,
}

# --- Output setup ---
os.makedirs("alphafold_study/logs", exist_ok=True)
columns = ["group", "prop_budget", "method", "seed", "width", "coverage"]
logs_fname = f"alphafold_study/logs/wor_sl={seed_chunk}.csv"

if not os.path.exists(logs_fname):
    with open(logs_fname, "w") as f:
        f.write(",".join(columns) + "\n")

# Checkpoint: count completed (group, budget) pairs (each produces 2 * N_BATCH rows)
existing = pd.read_csv(logs_fname)
checkpoint_counter = len(existing) // (2 * N_BATCH)
counter = 0

z_score = norm.ppf(1 - ALPHA / 2)


def run_wor_batched(Y_t, Yhat_t, N_B, active, rng_seed):
    """
    Run WOR trial batched across seeds. Shape: (N_BATCH, N).

    Y_t: (N,) tensor — ground truth labels (same for all seeds)
    Yhat_t: (N,) tensor — AlphaFold predictions (fixed)
    N_B: budget
    active: bool — if True, use min(Ŷ, 1-Ŷ) scoring; else uniform
    rng_seed: int — global seed for reproducibility

    Returns: (widths, coverages) each of shape (N_BATCH,)
    """
    N = Y_t.shape[0]

    # Expand to batch: (N_BATCH, N)
    Y_batch = Y_t.unsqueeze(0).expand(N_BATCH, -1)
    Yhat_batch = Yhat_t.unsqueeze(0).expand(N_BATCH, -1)
    true_mean = Y_t.mean()  # scalar, same for all seeds

    # Static scores
    if active:
        scores = torch.minimum(Yhat_t, 1.0 - Yhat_t).clamp(min=1e-12)  # (N,)
    else:
        scores = torch.ones(N, device=device)

    observed = torch.zeros(N_BATCH, N, dtype=torch.bool, device=device)
    thetahats = torch.zeros(N_BATCH, 1, dtype=torch.float32, device=device)
    varhats_main = torch.zeros(N_BATCH, 1, dtype=torch.float32, device=device)
    varhats_b = torch.zeros(N_BATCH, 1, dtype=torch.float32, device=device)

    torch.manual_seed(rng_seed)

    for t in range(N_B):
        n_unobs = N - t

        # Compute q_t over unobserved — scores are static, just mask observed
        q_scores = scores.unsqueeze(0).expand(N_BATCH, -1).clone()
        q_scores[observed] = 0.0
        q_sum = q_scores.sum(dim=1, keepdim=True).clamp(min=1e-12)

        if active:
            q_js = ((q_scores / q_sum) * (1.0 - TAU)) + (TAU / n_unobs)
            q_js[observed] = 0.0
        else:
            q_js = q_scores / q_sum  # uniform over unobserved

        # Sample I_t: (N_BATCH, 1)
        I_t = torch.multinomial(q_js, num_samples=1)

        z_It = torch.gather(Y_batch, 1, I_t)       # (N_BATCH, 1)
        f_It = torch.gather(Yhat_batch, 1, I_t)     # (N_BATCH, 1)
        q_It = torch.gather(q_js, 1, I_t)           # (N_BATCH, 1)

        # Imputed sum: Σ_{i∈O} y_i + Σ_{i∉O} f_i (without 1/N factor)
        imputed_sum = (
            (observed.float() * Y_batch).sum(dim=1, keepdim=True)
            + ((~observed).float() * Yhat_batch).sum(dim=1, keepdim=True)
        )

        aipw_t = (z_It - f_It) / q_It
        phi_t = imputed_sum + aipw_t

        # B_hat accumulator (Theorem 4): for t >= 2, i.e., t >= 1 in 0-indexed
        if t >= 1:
            ntheta_prev = thetahats / t
            varhats_b += (ntheta_prev - imputed_sum) ** 2

        thetahats += phi_t
        varhats_main += aipw_t ** 2
        observed.scatter_(1, I_t, True)

    # CI construction (Theorem 4)
    theta_T = thetahats / (N_B * N)
    v_simp = varhats_main / (N_B * N ** 2)
    v_minus = varhats_b / (N_B * N ** 2)
    v_full = (v_simp - v_minus).clamp(min=0)

    se = torch.sqrt(v_full / N_B)
    ub = (theta_T + z_score * se).clamp(max=1.0)
    lb = (theta_T - z_score * se).clamp(min=0.0)

    widths = (ub - lb).squeeze(1)  # (N_BATCH,)
    coverages = ((lb.squeeze(1) <= true_mean) & (true_mean <= ub.squeeze(1))).float()

    return widths.cpu().numpy(), coverages.cpu().numpy()


# --- Main loop ---
total = len(groups) * len(BUDGET_PROPS)
print(f"Total (group, budget) pairs: {total} (x 2 methods x {N_BATCH} seeds each)")

for group_idx, (group_name, group_mask) in enumerate(groups.items()):
    Y_g = torch.tensor(Y[group_mask], device=device)
    Yhat_g = torch.tensor(Yhat[group_mask], device=device)
    N_g = len(Y_g)
    print(f"\nGroup: {group_name} (N={N_g})")

    for budget_idx, budget_prop in enumerate(BUDGET_PROPS):
        N_B = max(1, int(N_g * budget_prop))

        if counter >= checkpoint_counter:
            # Unique seed per (chunk, group, budget, method) to avoid correlation
            base_seed = seed_chunk * 100000 + group_idx * 10000 + budget_idx * 100
            aw, ac = run_wor_batched(Y_g, Yhat_g, N_B, active=True,
                                     rng_seed=base_seed)
            uw, uc = run_wor_batched(Y_g, Yhat_g, N_B, active=False,
                                     rng_seed=base_seed + 50)

            with open(logs_fname, "a") as f:
                for i, seed in enumerate(SEED_LIST):
                    f.write(f"{group_name},{budget_prop:.4f},wor-active,{seed},{aw[i]},{ac[i]}\n")
                    f.write(f"{group_name},{budget_prop:.4f},wor-uniform,{seed},{uw[i]},{uc[i]}\n")

            print(f"  budget={budget_prop:.4f} (n={N_B}): "
                  f"active width={aw.mean():.6f}, uniform width={uw.mean():.6f}")

        counter += 1

print(f"\nDone! Results saved to {logs_fname}")
