"""
WOR FAQ Final (fixed tau): Run FAQ with best hyperparameters for a fixed tau value.

Same as wor_faq_final.py but instead of reading wor_best_settings.csv,
it loads the validation logs, filters to a specific tau, and picks the
best (narrowest mean_width) combination per (dataset, prop_budget).

Usage:
    python wor_faq_final_tau.py <seed_chunk> <tau>
    seed_chunk: 0 = seeds 0-32, 1 = seeds 33-65, 2 = seeds 66-99
    tau: e.g. 0.25 or 0.5
"""
import numpy as np
import pandas as pd
import torch
import sys, os, gc, glob
from scipy.stats import norm

from wor_trial import trial_faq_wor

device = "cuda" if torch.cuda.is_available() else "cpu"

seed_chunk = int(sys.argv[1])
tau_fixed = float(sys.argv[2])

# --- Build best settings filtered by tau ---
val_files = glob.glob("logs/val/wor_faq_val_*.csv")
logs = pd.concat([pd.read_csv(f) for f in val_files], ignore_index=True)

# Filter to the requested tau
logs_tau = logs[np.isclose(logs["tau"], tau_fixed)]
if len(logs_tau) == 0:
    raise ValueError(f"No validation rows found for tau={tau_fixed}. "
                     f"Available tau values: {sorted(logs['tau'].unique())}")

setting_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget",
                "beta0", "rho", "gamma", "tau"]
scenario_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"]

mean_logs = logs_tau.groupby(setting_cols, dropna=False).mean().reset_index()
mean_logs = mean_logs.sort_values(by="mean_width")
best_settings = mean_logs.groupby(scenario_cols, dropna=False).first().reset_index()

print(f"Best settings for tau={tau_fixed}:")
print(best_settings[["dataset", "prop_budget", "beta0", "rho", "gamma", "tau",
                      "mean_width", "coverage"]].to_string())

DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
ALPHA, N_SEEDS = 0.05, 100

# Split seeds into 3 chunks
if seed_chunk == 0:
    SEED_LIST = np.arange(33)
elif seed_chunk == 1:
    SEED_LIST = np.arange(33, 66)
elif seed_chunk == 2:
    SEED_LIST = np.arange(66, 100)

# --- Log file setup ---
tau_tag = str(tau_fixed).replace(".", "")
columns = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "seed", "mean_width", "coverage"]

logs_fname = f"wor_faq_final_tau{tau_tag}_sl={seed_chunk}.csv"
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

        # Get best hyperparameters for this (dataset, budget) with fixed tau
        row = best_settings.query(
            f"dataset == '{dataset}'"
            f" and mcar_obs_prob == {mcar_obs_prob}"
            f" and prop_budget == {budget_prop}"
        )
        if len(row) == 0:
            print(f"WARNING: no best setting found for dataset={dataset}, "
                  f"budget={budget_prop}, tau={tau_fixed}. Skipping.")
            counter += len(SEED_LIST)
            continue

        beta0, rho, gamma, tau = row[["beta0", "rho", "gamma", "tau"]].values.flatten()

        for seed in SEED_LIST:

            if counter >= checkpoint_counter:
                outputs = trial_faq_wor(
                    M2, V, MU0, SIGMA0,
                    N_NEW, N_QUESTIONS, N_B,
                    beta0, rho, gamma, tau, seed, device,
                    ALPHA=ALPHA, counter=counter)

                out_row = [dataset, n_full_obs, mcar_obs_prob, budget_prop, seed] + outputs

                with open(f"logs/final/{logs_fname}", "a") as file:
                    file.write(",".join([str(entry) for entry in out_row]))
                    file.write("\n")

            counter += 1

            if counter % 100 == 0:
                os.system("clear")
                print(f"Finished {counter} of {NUM_SETTINGS} settings.")

    del M2, U, V, D, MU0, SIGMA0; gc.collect()
