"""
WOR FAQ Final: Run FAQ with best hyperparameters on test models M2.
Without-replacement version of faq_final.py.

Only fully-observed historical data (n_full_obs=None, mcar_obs_prob=1.0).
Budgets: 10 evenly-spaced values between 2.5% and 25%.
100 seeds, split into 3 chunks via sys.argv[1].

Usage:
    python wor_faq_final.py <seed_chunk>
    seed_chunk: 0 = seeds 0-32, 1 = seeds 33-65, 2 = seeds 66-99
"""
import numpy as np
import pandas as pd
import torch
import sys, os, gc
from scipy.stats import norm

from wor_trial import trial_faq_wor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load best hyperparameter settings from validation
best_settings = pd.read_csv("logs/val/wor_best_settings.csv")

DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
ALPHA, N_SEEDS = 0.05, 100

# Split seeds into 3 chunks
if int(sys.argv[1]) == 0:
    SEED_LIST = np.arange(33)
elif int(sys.argv[1]) == 1:
    SEED_LIST = np.arange(33, 66)
elif int(sys.argv[1]) == 2:
    SEED_LIST = np.arange(66, 100)

# --- Log file setup ---
columns = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "seed", "mean_width", "coverage"]

logs_fname = f"wor_faq_final_sl={int(sys.argv[1])}.csv"
if logs_fname not in os.listdir("logs/final"):
    with open(f"logs/final/{logs_fname}", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")

# --- Counter + checkpointing ---
NUM_SETTINGS = len(DATASETS) * len(BUDGET_PROPS) * len(SEED_LIST)
counter = 0
checkpoint_counter = len(pd.read_csv(f"logs/final/{logs_fname}").index)

# Only fully-observed
n_full_obs, mcar_obs_prob = None, 1.0

for dataset in DATASETS:

    # Load M2
    M2 = pd.read_csv(f"data/processed/{dataset}/M2.csv")
    M2 = torch.tensor(M2.iloc[:,3:].to_numpy().astype(np.float32)).to(device)
    N_NEW, N_QUESTIONS = M2.shape

    # Load factor models (trained on all of M1)
    U = torch.load(
        f"factor_models/final/{dataset}/U_nfobs={n_full_obs}_p={mcar_obs_prob}.pt").to(device)
    V = torch.load(
        f"factor_models/final/{dataset}/V_nfobs={n_full_obs}_p={mcar_obs_prob}.pt").to(device)
    D = U.shape[1]

    MU0, SIGMA0 = U.mean(axis=0), torch.cov(U.T)

    for budget_prop in BUDGET_PROPS:
        N_B = int(N_QUESTIONS * budget_prop)

        # Get best hyperparameters for this (dataset, budget)
        beta0, rho, gamma, tau = best_settings.query(
            f"dataset == '{dataset}'"
            f" and mcar_obs_prob == {mcar_obs_prob}"
            f" and prop_budget == {budget_prop}"
        )[["beta0", "rho", "gamma", "tau"]].values.flatten()

        for seed in SEED_LIST:

            if counter >= checkpoint_counter:
                outputs = trial_faq_wor(
                    M2, V, MU0, SIGMA0,
                    N_NEW, N_QUESTIONS, N_B,
                    beta0, rho, gamma, tau, seed, device,
                    ALPHA=ALPHA, counter=counter)

                row = [dataset, n_full_obs, mcar_obs_prob, budget_prop, seed] + outputs

                with open(f"logs/final/{logs_fname}", "a") as file:
                    file.write(",".join([str(entry) for entry in row]))
                    file.write("\n")

            counter += 1

            if counter % 100 == 0:
                os.system("clear")
                print(f"Finished {counter} of {NUM_SETTINGS} settings.")

    del M2, U, V, D, MU0, SIGMA0; gc.collect()
