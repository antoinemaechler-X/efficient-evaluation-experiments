import numpy as np
import pandas as pd
import torch
import sys
import os
from scipy.stats import norm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

seed_chunk = int(sys.argv[1])

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

# Load data
Y = np.load("data/Y.npy").flatten().astype(np.float32)
Yhat = np.load("data/Yhat.npy").flatten().astype(np.float32)
Z = np.load("data/Z.npy").flatten().astype(np.float32)

groups = {
    "non-phosphorylated": Z == 0,
    "phosphorylated": Z == 1,
}

os.makedirs("logs", exist_ok=True)
columns = ["group", "prop_budget", "method", "seed", "width", "coverage"]
logs_fname = f"logs/bernoulli_sl={seed_chunk}.csv"

if not os.path.exists(logs_fname):
    with open(logs_fname, "w") as f:
        f.write(",".join(columns) + "\n")

existing = pd.read_csv(logs_fname)
checkpoint_counter = len(existing) // (3 * N_BATCH)
counter = 0

z_score = norm.ppf(1 - ALPHA / 2)


def run_bernoulli_vectorized(Y_group, Yhat_group, budget_prop, rng_seed):
    # Run all 3 Bernoulli methods vectorized across N_BATCH seeds.
    # Returns dict: method -> (widths, coverages) of shape (N_BATCH,).
    N = len(Y_group)

    Y_t = torch.tensor(Y_group, device=device).unsqueeze(0).expand(N_BATCH, -1)
    Yhat_t = torch.tensor(Yhat_group, device=device).unsqueeze(0).expand(N_BATCH, -1)
    true_mean = Y_t.float().mean(dim=1, keepdim=True)

    active_scores = torch.minimum(Yhat_t, 1.0 - Yhat_t).clamp(min=1e-12)
    active_scores_normed = active_scores / active_scores.sum(dim=1, keepdim=True)

    # Per-item inclusion probabilities
    pi_active = ((1.0 - TAU) * N * budget_prop * active_scores_normed
                 + TAU * budget_prop).clamp(min=1e-6, max=1.0)
    pi_uniform = torch.full((N_BATCH, N), budget_prop, device=device).clamp(min=1e-6, max=1.0)

    results = {}

    for method_idx, (method, pi, f_pred) in enumerate([
        ("bernoulli-active",  pi_active,  Yhat_t),
        ("bernoulli-uniform", pi_uniform, Yhat_t),
        ("classical",         pi_uniform, torch.zeros_like(Yhat_t)),
    ]):
        torch.manual_seed(rng_seed + method_idx * 10)
        xi = torch.bernoulli(pi)  # Bernoulli draws: (N_BATCH, N)

        # AIPW per item: f_i + (Y_i - f_i) * ξ_i / π_i
        aipw_items = f_pred + (Y_t - f_pred) * xi / pi

        mu_hat = aipw_items.mean(dim=1, keepdim=True)
        var_hat = aipw_items.var(dim=1, keepdim=True) / N

        se = torch.sqrt(var_hat.clamp(min=0))
        ub = (mu_hat + z_score * se).clamp(max=1.0)
        lb = (mu_hat - z_score * se).clamp(min=0.0)

        widths = (ub - lb).squeeze(1)
        coverages = ((lb <= true_mean) & (true_mean <= ub)).squeeze(1).float()

        results[method] = (widths.cpu().numpy(), coverages.cpu().numpy())

    return results


total = len(groups) * len(BUDGET_PROPS)
print(f"Total (group, budget) pairs: {total}")

for group_idx, (group_name, group_mask) in enumerate(groups.items()):
    Y_g = Y[group_mask]
    Yhat_g = Yhat[group_mask]
    N_g = len(Y_g)
    print(f"\nGroup: {group_name} (N={N_g})")

    for budget_idx, budget_prop in enumerate(BUDGET_PROPS):

        if counter >= checkpoint_counter:
            rng_seed = seed_chunk * 100000 + group_idx * 10000 + budget_idx * 100
            res = run_bernoulli_vectorized(Y_g, Yhat_g, budget_prop, rng_seed)

            with open(logs_fname, "a") as f:
                for method in ["bernoulli-active", "bernoulli-uniform", "classical"]:
                    w, c = res[method]
                    for i, seed in enumerate(SEED_LIST):
                        f.write(f"{group_name},{budget_prop:.4f},{method},{seed},{w[i]},{c[i]}\n")

            print(f"  budget={budget_prop:.4f}: "
                  f"active={res['bernoulli-active'][0].mean():.6f}, "
                  f"uniform={res['bernoulli-uniform'][0].mean():.6f}, "
                  f"classical={res['classical'][0].mean():.6f}")

        counter += 1

print(f"\nDone! Saved to {logs_fname}")
