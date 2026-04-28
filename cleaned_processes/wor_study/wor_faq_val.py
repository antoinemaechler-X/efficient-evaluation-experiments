import numpy as np
import pandas as pd
import torch
import sys, os, gc
from scipy.stats import norm

from wor_trial import trial_faq_wor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameter grid
DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
BETA0_VALS = [0.25, 0.5, 0.75, 1.0]
RHO_VALS = [0.0, 0.05, 0.25, 0.5, 0.75]
GAMMA_VALS = [0.0, 0.05, 0.25, 0.5, 0.75]
TAU_VALS = [0.05, 0.25, 0.5, 0.75]

ALPHA, N_SEEDS = 0.05, 5

# Command-line args: dataset_idx, gamma_idx
dataset = DATASETS[int(sys.argv[1])]
gamma = GAMMA_VALS[int(sys.argv[2])]

# --- Log file setup ---
columns = [
    "dataset", "n_full_obs", "mcar_obs_prob", "prop_budget",
    "beta0", "rho", "gamma", "tau", "seed",
    "mean_width", "coverage"
]

logs_fname = f"wor_faq_val_dataset={dataset}_gamma={gamma}.csv"
if logs_fname not in os.listdir("logs/val"):
    with open(f"logs/val/{logs_fname}", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")

counter = 0
checkpoint_counter = len(pd.read_csv(f"logs/val/{logs_fname}").index)
NUM_SETTINGS = len(BUDGET_PROPS) * len(BETA0_VALS) * len(RHO_VALS) * len(TAU_VALS) * N_SEEDS

# Only fully-observed
n_full_obs, mcar_obs_prob = None, 1.0

# Load M1 fully-observed, split 80/20
M1_full = pd.read_csv(f"data/processed/{dataset}/M1_nfobs={n_full_obs}_p={mcar_obs_prob}.csv")
M1_full = torch.tensor(M1_full.iloc[:,3:].to_numpy().astype(np.float32))
M1_full_val = M1_full[int(M1_full.shape[0] * 0.8):].to(device)

N_NEW, N_QUESTIONS = M1_full_val.shape

# Load factor models (trained on M1 train split)
U = torch.load(f"factor_models/val/{dataset}/U_nfobs={n_full_obs}_p={mcar_obs_prob}.pt").to(device)
V = torch.load(f"factor_models/val/{dataset}/V_nfobs={n_full_obs}_p={mcar_obs_prob}.pt").to(device)
D = U.shape[1]

MU0, SIGMA0 = U.mean(axis=0), torch.cov(U.T)

for budget_prop in BUDGET_PROPS:
    N_B = int(N_QUESTIONS * budget_prop)

    for beta0 in BETA0_VALS:
        for rho in RHO_VALS:
            for tau in TAU_VALS:
                for seed in range(N_SEEDS):

                    if counter >= checkpoint_counter:
                        outputs = trial_faq_wor(
                            M1_full_val, V, MU0, SIGMA0,
                            N_NEW, N_QUESTIONS, N_B,
                            beta0, rho, gamma, tau, seed, device,
                            ALPHA=ALPHA, counter=counter,
                            disable_tqdm=not sys.stderr.isatty())

                        row = [
                            dataset, n_full_obs, mcar_obs_prob, budget_prop,
                            beta0, rho, gamma, tau, seed] + outputs

                        with open(f"logs/val/{logs_fname}", "a") as file:
                            file.write(",".join([str(entry) for entry in row]))
                            file.write("\n")

                    counter += 1

                    if counter % 100 == 0:
                        os.system("clear")
                        print(f"Finished {counter} of {NUM_SETTINGS} settings.")

del M1_full, M1_full_val, U, V, D, MU0, SIGMA0
gc.collect()
