"""
WOR Baselines: Uniform, sqrt, 2min scoring WITHOUT replacement.
Without-replacement version of baselines_all.py.

Uses static predictions (PHATS from M1, no factor model updates).
Tests two estimators: f=0 (zero) and f=PHATS (mean).
Post-hoc oracle selection across (policy, tau, f).

Only fully-observed historical data.
100 seeds, split into 3 chunks via sys.argv[2].

Usage:
    python wor_baselines.py <dataset_idx> <seed_chunk>
    dataset_idx: 0 = mmlu-pro, 1 = bbh+gpqa+ifeval+math+musr
    seed_chunk: 0 = seeds 0-32, 1 = seeds 33-65, 2 = seeds 66-99
"""
import numpy as np
import pandas as pd
import torch
import sys, os, gc
from scipy.stats import norm
from tqdm.autonotebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"][int(sys.argv[1])]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
POLICIES = ["unif", "sqrt", "2min"]
TAUS = [0.05, 0.25, 0.5, 0.75]
ALPHA, N_SEEDS = 0.05, 100

# Split seeds
if int(sys.argv[2]) == 0:
    SEED_LIST = np.arange(33)
elif int(sys.argv[2]) == 1:
    SEED_LIST = np.arange(33, 66)
elif int(sys.argv[2]) == 2:
    SEED_LIST = np.arange(66, 100)


def trial_baseline_wor(M2, PHATS, N_NEW, N_QUESTIONS, N_B, policy, tau, seed, device, counter):
    """
    WOR baseline trial with static predictions (no factor model).
    Runs two estimators simultaneously: f=0 (zero) and f=PHATS (mean).
    Uses equation (2) PAI estimator with without-replacement sampling.
    """
    # --- Initialization ---
    observed = torch.zeros(N_NEW, N_QUESTIONS, dtype=torch.bool, device=device)

    # PHATS broadcast to (N_NEW, N_QUESTIONS) for imputed_sum computation
    PHATS_expanded = PHATS.unsqueeze(0).expand(N_NEW, -1)  # (N_NEW, N_QUESTIONS)
    ZEROS = torch.zeros_like(PHATS_expanded)  # for f=zero estimator

    # Accumulators for f=zero estimator
    thetahats_f1 = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_main_f1 = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_inner1_f1 = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_inner2_f1 = torch.zeros(N_NEW, N_B, dtype=torch.float32, device=device)

    # Accumulators for f=mean estimator
    thetahats_f2 = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_main_f2 = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_inner1_f2 = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_inner2_f2 = torch.zeros(N_NEW, N_B, dtype=torch.float32, device=device)

    # Precompute scoring
    if policy == "unif":
        scores = torch.ones(N_QUESTIONS, device=device)
    elif policy == "sqrt":
        scores = torch.sqrt(PHATS * (1.0 - PHATS)).clamp(min=1e-12)
    elif policy == "2min":
        scores = 2.0 * torch.minimum(PHATS, 1.0 - PHATS).clamp(min=1e-12)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    torch.random.manual_seed(seed)

    for t in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=not sys.stderr.isatty()):

        # --- Scoring: static scores, mask observed ---
        q_scores = scores.unsqueeze(0).expand(N_NEW, -1).clone()  # (N_NEW, N_QUESTIONS)
        q_scores[observed] = 0.0

        # Normalize + mix with uniform over unobserved
        n_unobs = N_QUESTIONS - t
        q_sum = q_scores.sum(dim=1, keepdim=True).clamp(min=1e-12)

        if policy == "unif":
            # Pure uniform over unobserved (no tau needed)
            q_js = q_scores / q_sum
        else:
            q_js = ((q_scores / q_sum) * (1.0 - tau)) + (tau / n_unobs)
            q_js[observed] = 0.0

        # Sample
        I_t = torch.multinomial(input=q_js, num_samples=1)

        # --- Gather labels and probabilities ---
        z_It = torch.gather(M2, dim=1, index=I_t)
        q_It = torch.gather(q_js, dim=1, index=I_t)

        # === f=zero estimator (f1) ===
        # imputed_sum_f1 = Σ_{j∈O} y_j + Σ_{j∉O} 0 = Σ_{j∈O} y_j
        imputed_sum_f1 = (observed.float() * M2).sum(dim=1, keepdim=True)
        phat_It_f1 = torch.tensor(0.0, device=device)
        aipw_f1 = (z_It - phat_It_f1) / q_It
        thetahats_f1 += imputed_sum_f1 + aipw_f1
        varhats_main_f1 += (aipw_f1 ** 2)
        varhats_inner1_f1 += (z_It / q_It)
        varhats_inner2_f1[:, t] = imputed_sum_f1.flatten()

        # === f=mean estimator (f2) ===
        # imputed_sum_f2 = Σ_{j∈O} y_j + Σ_{j∉O} PHATS_j
        imputed_sum_f2 = (
            (observed.float() * M2).sum(dim=1, keepdim=True)
            + ((~observed).float() * PHATS_expanded).sum(dim=1, keepdim=True)
        )
        phat_It_f2 = PHATS[I_t.flatten()].unsqueeze(1)  # (N_NEW, 1)
        aipw_f2 = (z_It - phat_It_f2) / q_It
        thetahats_f2 += imputed_sum_f2 + aipw_f2
        varhats_main_f2 += (aipw_f2 ** 2)
        varhats_inner1_f2 += (z_It / q_It)
        varhats_inner2_f2[:, t] = imputed_sum_f2.flatten()

        # Update observed set
        observed.scatter_(dim=1, index=I_t, value=True)

    # --- CI construction for both estimators ---
    z_score = norm.ppf(1 - (ALPHA / 2))
    mus_M2 = M2.mean(dim=1, keepdim=True)

    results = []
    for thetahats, varhats_main, varhats_inner1, varhats_inner2 in [
        (thetahats_f1, varhats_main_f1, varhats_inner1_f1, varhats_inner2_f1),
        (thetahats_f2, varhats_main_f2, varhats_inner1_f2, varhats_inner2_f2),
    ]:
        thetahats_T = thetahats / (N_B * N_QUESTIONS)
        v_T_sq_simp = varhats_main / (N_B * (N_QUESTIONS ** 2))
        v_T_sq_minus = (((varhats_inner1 / N_B) - varhats_inner2) ** 2).sum(dim=1, keepdim=True)
        v_T_sq_minus /= (N_B * (N_QUESTIONS ** 2))
        v_T_sq_full = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

        ub = torch.maximum(
            thetahats_T + z_score * torch.sqrt(v_T_sq_full / N_B),
            torch.tensor(0.0, device=device))
        lb = torch.minimum(
            thetahats_T - z_score * torch.sqrt(v_T_sq_full / N_B),
            torch.tensor(1.0, device=device))

        mean_width = (ub - lb).mean().item()
        coverage = ((lb <= mus_M2) & (mus_M2 <= ub)).mean(dtype=float).item()
        results.append([mean_width, coverage])

    return results[0], results[1]  # (f1_outputs, f2_outputs)


# --- Log file setup ---
columns = [
    "dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "policy", "tau", "f", "seed",
    "mean_width", "coverage"
]

logs_fname = f"wor_baselines_dataset={dataset}_sl={int(sys.argv[2])}.csv"
if logs_fname not in os.listdir("logs/final"):
    with open(f"logs/final/{logs_fname}", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")

# --- Counter + checkpointing ---
NUM_SETTINGS = len(BUDGET_PROPS) * ((len(TAUS) * (len(POLICIES) - 1)) + 1) * len(SEED_LIST)
counter = 0
checkpoint_counter = len(pd.read_csv(f"logs/final/{logs_fname}").index) // 2

# Only fully-observed
n_full_obs, mcar_obs_prob = None, 1.0

# Load M2
M2 = pd.read_csv(f"data/processed/{dataset}/M2.csv")
M2 = torch.tensor(M2.iloc[:,3:].to_numpy().astype(np.float32)).to(device)
N_NEW, N_QUESTIONS = M2.shape

# Load M1 to compute PHATS
M1 = pd.read_csv(f"data/processed/{dataset}/M1_nfobs={n_full_obs}_p={mcar_obs_prob}.csv")
M1 = torch.tensor(M1.iloc[:,3:].to_numpy().astype(np.float32))
PHATS = M1.nanmean(dim=0).to(device)
PHATS[PHATS.isnan()] = PHATS.nanmean()

for budget_prop in BUDGET_PROPS:
    N_B = int(N_QUESTIONS * budget_prop)

    for policy in POLICIES:
        WORKING_TAUS = [np.nan] if policy == "unif" else TAUS

        for tau in WORKING_TAUS:
            for seed in SEED_LIST:

                if counter >= checkpoint_counter:
                    outputs_f1, outputs_f2 = trial_baseline_wor(
                        M2, PHATS, N_NEW, N_QUESTIONS, N_B,
                        policy, tau, seed, device, counter)

                    row_f1 = [dataset, n_full_obs, mcar_obs_prob, budget_prop,
                              policy, tau, "zero", seed] + outputs_f1
                    row_f2 = [dataset, n_full_obs, mcar_obs_prob, budget_prop,
                              policy, tau, "mean", seed] + outputs_f2

                    with open(f"logs/final/{logs_fname}", "a") as file:
                        file.write(",".join([str(entry) for entry in row_f1]) + "\n")
                        file.write(",".join([str(entry) for entry in row_f2]) + "\n")

                counter += 1

                if counter % 100 == 0:
                    os.system("clear")
                    print(f"Finished {counter} of {NUM_SETTINGS} settings.")

del M2, M1, PHATS; gc.collect()
