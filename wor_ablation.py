"""
WOR Active Inference Ablation: Factor model + coverage-based scoring, without replacement.
Without-replacement version of active_inference_factor_ablation.py.

Uses the factor model for predictions (like FAQ) but with simpler Z&C scoring:
u(j) = 2*min(p̂_j, 1-p̂_j), mixed with uniform via tau.

Only fully-observed historical data. Post-hoc oracle tau selection.
100 seeds, split into 3 chunks via sys.argv[2].

Usage:
    python wor_ablation.py <dataset_idx> <seed_chunk>
    dataset_idx: 0 = mmlu-pro, 1 = bbh+gpqa+ifeval+math+musr
    seed_chunk: 0 = seeds 0-32, 1 = seeds 33-65, 2 = seeds 66-99
"""
import numpy as np
import pandas as pd
import torch
import sys, os, gc
from scipy.stats import norm

from wor_trial import trial_ablation_wor

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"][int(sys.argv[1])]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
TAUS = [0.05, 0.25, 0.5, 0.75]
ALPHA, N_SEEDS = 0.05, 100

# Split seeds
if int(sys.argv[2]) == 0:
    SEED_LIST = np.arange(33)
elif int(sys.argv[2]) == 1:
    SEED_LIST = np.arange(33, 66)
elif int(sys.argv[2]) == 2:
    SEED_LIST = np.arange(66, 100)

# --- Log file setup ---
columns = [
    "dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau", "seed",
    "mean_width", "coverage"
]

logs_fname = f"wor_ablation_dataset={dataset}_sl={int(sys.argv[2])}.csv"
if logs_fname not in os.listdir("logs/final"):
    with open(f"logs/final/{logs_fname}", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")

# --- Counter + checkpointing ---
NUM_SETTINGS = len(BUDGET_PROPS) * len(TAUS) * len(SEED_LIST)
counter = 0
checkpoint_counter = len(pd.read_csv(f"logs/final/{logs_fname}").index)

# Only fully-observed
n_full_obs, mcar_obs_prob = None, 1.0

# Load M2
M2 = pd.read_csv(f"data/processed/{dataset}/M2.csv")
M2 = torch.tensor(M2.iloc[:,3:].to_numpy().astype(np.float32)).to(device)
N_NEW, N_QUESTIONS = M2.shape

# Load factor models
U = torch.load(
    f"factor_models/final/{dataset}/U_nfobs={n_full_obs}_p={mcar_obs_prob}.pt", map_location=device)
V = torch.load(
    f"factor_models/final/{dataset}/V_nfobs={n_full_obs}_p={mcar_obs_prob}.pt", map_location=device)
D = U.shape[1]

MU0, SIGMA0 = U.mean(axis=0), torch.cov(U.T)

for budget_prop in BUDGET_PROPS:
    N_B = int(N_QUESTIONS * budget_prop)

    for tau in TAUS:
        for seed in SEED_LIST:

            if counter >= checkpoint_counter:
                outputs = trial_ablation_wor(
                    M2, V, MU0, SIGMA0,
                    N_NEW, N_QUESTIONS, N_B,
                    tau, seed, device,
                    ALPHA=ALPHA, counter=counter)

                row = [dataset, n_full_obs, mcar_obs_prob, budget_prop, tau, seed] + outputs

                with open(f"logs/final/{logs_fname}", "a") as file:
                    file.write(",".join([str(entry) for entry in row]))
                    file.write("\n")

            counter += 1

            if counter % 100 == 0:
                os.system("clear")
                print(f"Finished {counter} of {NUM_SETTINGS} settings.")

del M2, U, V, D, MU0, SIGMA0; gc.collect()
