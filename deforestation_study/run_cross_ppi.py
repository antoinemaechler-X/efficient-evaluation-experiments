"""
Cross-PPI estimator for Amazon deforestation dataset.

Usage: python deforestation_study/run_cross_ppi.py <seed_chunk>
  seed_chunk in {0, 1, 2}: splits 1000 seeds into 3 chunks.

EXACTLY reproduces Zrnic & Candes (2024) Cross-PPI from:
  https://github.com/tijana-zrnic/cross-ppi/blob/main/deforestation/deforestation.ipynb

For each trial at each budget:
  1. Randomly split ALL 3,192 items into n labeled + rest unlabeled
     (using train_test_split, same as Zrnic)
  2. K=10 fold cross-fitting on labeled data:
     - Train HistGradientBoostingClassifier(max_iter=100, max_depth=2) on K-1 folds
     - Predict on held-out fold → Yhat_labeled
     - Predict on all unlabeled, accumulate average → Yhat_unlabeled
  3. Point estimate: mean(Yhat_unlabeled) + mean(Y_labeled - Yhat_labeled)
  4. Bootstrap variance (B=30): resample labeled, train model, collect residuals
  5. CI: theta_hat +/- z * sqrt(var_hat)

CPU-only (scikit-learn, no GPU needed).
"""
import numpy as np
import pandas as pd
import sys
import os
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

seed_chunk = int(sys.argv[1])

# --- Parameters ---
BUDGET_PROPS = np.round(np.linspace(0.01, 0.2, 20), decimals=4)
ALPHA = 0.05
K = 10   # Cross-fitting folds (same as Zrnic)
B = 30   # Bootstrap resamples (same as Zrnic)
N_SEEDS = 1000

if seed_chunk == 0:
    SEED_LIST = np.arange(0, 333)
elif seed_chunk == 1:
    SEED_LIST = np.arange(333, 666)
else:
    SEED_LIST = np.arange(666, 1000)

N_BATCH = len(SEED_LIST)
print(f"Seed chunk {seed_chunk}: seeds {SEED_LIST[0]}-{SEED_LIST[-1]} ({N_BATCH} seeds)")

# --- Load FULL dataset (Cross-PPI operates on all data) ---
data_dir = "deforestation_study/data"
X_all = np.load(f"{data_dir}/X_all.npy").astype(np.float64)
Y_all = np.load(f"{data_dir}/Y_all.npy").astype(np.float64)

theta_true = float(np.load(f"{data_dir}/theta_true.npy")[0])
N_total = len(Y_all)
print(f"N={N_total}, theta_true={theta_true:.6f}")

# No groups — single population
groups = {
    "all": np.ones(N_total, dtype=bool),
}

# --- Output setup ---
os.makedirs("deforestation_study/logs", exist_ok=True)
columns = ["group", "prop_budget", "method", "seed", "width", "coverage"]
logs_fname = f"deforestation_study/logs/cross_ppi_sl={seed_chunk}.csv"

if not os.path.exists(logs_fname):
    with open(logs_fname, "w") as f:
        f.write(",".join(columns) + "\n")

# Checkpoint per (group, budget): each produces 1 * N_BATCH rows
existing = pd.read_csv(logs_fname)
checkpoint_counter = len(existing) // N_BATCH
counter = 0

z_score = norm.ppf(1 - ALPHA / 2)


def bootstrap_variance(X_labeled, X_unlabeled, Y_labeled, train_n, thetaPP, B=30, rng=None):
    """
    Bootstrap variance estimation — EXACTLY as Zrnic's code.

    For b in 1..B:
        Subsample train_n items (with replacement) from labeled
        Train model, predict on unlabeled → accumulate Yhat_unlabeled_avg
        Predict on out-of-sample labeled → collect residuals
    var_hat = var(Yhat_unlabeled_avg)/N + var(residuals)/n
    """
    n = X_labeled.shape[0]
    N = X_unlabeled.shape[0]

    Yhat_unlabeled = np.zeros(N)

    grad_diff = np.zeros(int((n - train_n) * B))

    for j in range(B):
        train_ind = rng.choice(range(n), train_n)
        X_train = X_labeled[train_ind, :]
        Y_train = Y_labeled[train_ind]

        cls = HistGradientBoostingClassifier(max_iter=100, max_depth=2).fit(
            X_train, Y_train
        )

        Yhat_unlabeled += cls.predict_proba(X_unlabeled)[:, 1] / B

        other_inds = np.delete(range(n), train_ind)[:n - train_n]
        Yhat_labeled_oos = cls.predict_proba(X_labeled[other_inds, :])[:, 1]

        grad_diff[j * (n - train_n):(j + 1) * (n - train_n)] = Yhat_labeled_oos - Y_labeled[other_inds]

    var_unlabeled = np.var(Yhat_unlabeled)
    var_labeled = np.var(grad_diff)

    var_hat = var_unlabeled / N + var_labeled / n

    return var_hat


def cross_prediction_mean_ci(X_labeled, X_unlabeled, Y_labeled, rng):
    """
    Cross-PPI estimator with K-fold cross-fitting — EXACTLY as Zrnic's code.

    1. K-fold cross-fit on labeled: train on K-1 folds, predict on held-out + unlabeled
    2. Point estimate: mean(Yhat_unlabeled) + mean(Y_labeled - Yhat_labeled)
    3. Bootstrap variance (B=30)
    4. CI: point +/- z * sqrt(var_hat)
    """
    n = X_labeled.shape[0]
    N = X_unlabeled.shape[0]

    fold_n = int(n / K)

    Yhat_labeled = np.zeros(n)
    Yhat_unlabeled = np.zeros(N)

    for j in range(K):
        X_val = X_labeled[j * fold_n:(j + 1) * fold_n, :]
        Y_val = Y_labeled[j * fold_n:(j + 1) * fold_n]
        train_ind = np.delete(range(n), range(j * fold_n, (j + 1) * fold_n))
        X_train = X_labeled[train_ind, :]
        Y_train = Y_labeled[train_ind]

        cls = HistGradientBoostingClassifier(max_iter=100, max_depth=2).fit(
            X_train, Y_train
        )

        Yhat_unlabeled += cls.predict_proba(X_unlabeled)[:, 1] / K
        Yhat_labeled[j * fold_n:(j + 1) * fold_n] = cls.predict_proba(X_val)[:, 1]

    thetaPP = np.mean(Yhat_unlabeled) + np.mean(Y_labeled - Yhat_labeled)

    var_hat = bootstrap_variance(
        X_labeled, X_unlabeled, Y_labeled, n - fold_n, thetaPP, B=B, rng=rng
    )

    halfwidth = z_score * np.sqrt(var_hat)

    return thetaPP - halfwidth, thetaPP + halfwidth


def run_cross_ppi_trials(X_group, Y_group, budget_prop, rng_seed):
    """
    Run Cross-PPI for all seeds in this chunk.

    For each seed:
      1. Randomly split into n labeled + rest unlabeled (using train_test_split like Zrnic)
      2. Run cross_prediction_mean_ci with K-fold cross-fitting + bootstrap variance

    Returns: (widths, coverages) arrays of shape (N_BATCH,)
    """
    N = len(Y_group)
    n = max(1, int(N * budget_prop))
    group_true_mean = theta_true

    widths = np.zeros(N_BATCH)
    coverages = np.zeros(N_BATCH)

    for b, seed in enumerate(SEED_LIST):
        rng = np.random.RandomState(rng_seed + seed)

        # Split into labeled and unlabeled — same as Zrnic's train_test_split
        perm = rng.permutation(N)
        labeled_idx = perm[:n]
        unlabeled_idx = perm[n:]

        X_labeled = X_group[labeled_idx]
        Y_labeled = Y_group[labeled_idx]
        X_unlabeled = X_group[unlabeled_idx]

        lb, ub = cross_prediction_mean_ci(X_labeled, X_unlabeled, Y_labeled, rng)

        # Clip to [0, 1] for width computation
        lb_clip = max(0.0, lb)
        ub_clip = min(1.0, ub)

        widths[b] = ub_clip - lb_clip
        coverages[b] = float(lb <= group_true_mean <= ub)

        if (b + 1) % 50 == 0:
            print(f"    seed {seed}: width={widths[b]:.6f}, coverage={coverages[b]:.0f}")

    return widths, coverages


# --- Main loop ---
total = len(groups) * len(BUDGET_PROPS)
print(f"Total (group, budget) pairs: {total} (x 1 method x {N_BATCH} seeds each)")
print(f"Each trial: K={K} cross-fitting + B={B} bootstrap = {K + B} model fits")
print(f"Total model fits: {N_BATCH} seeds x {len(BUDGET_PROPS)} budgets x {K + B} = "
      f"{N_BATCH * len(BUDGET_PROPS) * (K + B):,}")

for group_idx, (group_name, group_mask) in enumerate(groups.items()):
    X_g = X_all[group_mask]
    Y_g = Y_all[group_mask]
    N_g = len(Y_g)
    print(f"\nGroup: {group_name} (N={N_g})")

    for budget_idx, budget_prop in enumerate(BUDGET_PROPS):

        if counter >= checkpoint_counter:
            rng_seed = seed_chunk * 100000 + group_idx * 10000 + budget_idx * 100
            n = max(1, int(N_g * budget_prop))
            print(f"  budget={budget_prop:.4f} (n={n})...")

            w, c = run_cross_ppi_trials(X_g, Y_g, budget_prop, rng_seed)

            with open(logs_fname, "a") as f:
                for i, seed in enumerate(SEED_LIST):
                    f.write(f"{group_name},{budget_prop:.4f},cross-ppi,{seed},{w[i]},{c[i]}\n")

            print(f"    cross-ppi width={w.mean():.6f}, coverage={c.mean():.3f}")

        counter += 1

print(f"\nDone! Results saved to {logs_fname}")
