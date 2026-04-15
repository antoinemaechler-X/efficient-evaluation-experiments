'''
Run FAQ on M2 (test models) at 25%-50% budgets using best hyperparameters
from the high-budget validation. Fully-observed only.

Usage: python faq_final_high_budget.py <seed_chunk>
  seed_chunk: 0 = seeds 0-32, 1 = seeds 33-65, 2 = seeds 66-99
'''
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import sigmoid
import sys, os, gc
from tqdm.autonotebook import tqdm
from scipy.stats import norm

device = "cuda" if torch.cuda.is_available() else "cpu"

# load best settings from high-budget validation
best_settings = pd.read_csv("logs/val/best_settings_high_budget.csv")

# settings
DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
BUDGET_PROPS = np.round(np.linspace(0.25, 0.5, 11)[1:], decimals=3)
ALPHA = 0.05

# seed splitting
if int(sys.argv[1]) == 0:
    SEED_LIST = np.arange(33)
elif int(sys.argv[1]) == 1:
    SEED_LIST = np.arange(33, 66)
elif int(sys.argv[1]) == 2:
    SEED_LIST = np.arange(66, 100)

# log file
os.makedirs("logs/final", exist_ok=True)
columns = ["dataset", "prop_budget", "seed", "mean_width", "coverage"]
logs_fname = f"logs/final/faq_final_high_budget_sl={int(sys.argv[1])}.csv"
if not os.path.exists(logs_fname):
    with open(logs_fname, "w") as f:
        f.write(",".join(columns) + "\n")


def trial(
    M2, V, MU0, SIGMA0,
    N_NEW, N_QUESTIONS, N_B,
    beta0, rho, gamma, tau, seed, device, counter):

    Uhats = torch.tile(MU0, dims=(N_NEW, 1)).to(device)
    Sigmahats = torch.tile(SIGMA0, dims=(N_NEW, 1, 1)).to(device)
    thetahats = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_main = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_inner1 = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    varhats_inner2 = torch.zeros(size=(N_NEW, N_B), dtype=torch.float32, device=device)

    torch.random.manual_seed(seed)

    for s in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=not sys.stderr.isatty()):

        p_hat_js = sigmoid(Uhats @ V.T).clamp(min=1e-12, max=1.0 - 1e-12)
        p1mp_hat_js = p_hat_js * (1.0 - p_hat_js)
        sqrt_p1mp_hat_js = torch.sqrt(p1mp_hat_js)

        ho_js = sqrt_p1mp_hat_js / sqrt_p1mp_hat_js.sum(dim=1, keepdim=True)

        Sigmahats = (Sigmahats + Sigmahats.mT) / 2.0

        vtSigmav_js = torch.einsum("ni,mij,nj->mn", V, Sigmahats, V)
        log_denominator = torch.log1p(p1mp_hat_js * vtSigmav_js)
        sq_term = torch.bmm(Sigmahats, ((p1mp_hat_js @ V) / N_QUESTIONS).unsqueeze(-1)).squeeze(-1) @ V.T
        log_numerator = torch.log(p1mp_hat_js.clamp_min(1e-12)) + 2 * torch.log(sq_term.abs().clamp_min(1e-12))
        log_d_js = log_numerator - log_denominator
        ha_js = torch.softmax(log_d_js, dim=1)

        alpha_s = torch.maximum(torch.tensor(0.0), torch.tensor(1.0 - ((s+1.0) / (rho * N_B)))) if rho != 0.0 else 0.0
        beta_s = beta0 * torch.minimum(torch.tensor((s+1.0) / (gamma * N_B)), torch.tensor(1.0)) if gamma != 0.0 else beta0

        hcat_js = (((1.0 - alpha_s) * ho_js) + (alpha_s * ha_js)) ** beta_s
        q_js = ((hcat_js / hcat_js.sum(dim=1, keepdim=True)) * (1.0 - tau)) + (tau / N_QUESTIONS)
        I_s = torch.multinomial(input=q_js, num_samples=1)

        z_Is = torch.gather(M2, dim=1, index=I_s)
        phat_Is = torch.gather(p_hat_js, dim=1, index=I_s)
        q_Is = torch.gather(q_js, dim=1, index=I_s)

        prob_sums = p_hat_js.sum(dim=1, keepdim=True)
        aipw_s = (z_Is - phat_Is) / q_Is
        phi_s = prob_sums + aipw_s
        thetahats += phi_s

        varhats_main += (aipw_s ** 2)
        varhats_inner1 += (z_Is / q_Is)
        varhats_inner2[:,s] = prob_sums.flatten()

        w_s = torch.gather(input=p1mp_hat_js, dim=1, index=I_s)
        v_Is = V[I_s.flatten()]
        Sigma_v_Is = torch.bmm(Sigmahats, v_Is.unsqueeze(-1))
        vT_Sigma_v_Is = torch.bmm(v_Is.unsqueeze(-2), Sigma_v_Is).squeeze(-1)
        denominator = 1.0 + (w_s * vT_Sigma_v_Is)
        numerator = w_s.unsqueeze(dim=-1) * (Sigma_v_Is @ Sigma_v_Is.mT)
        Sigmahats -= (numerator / denominator.unsqueeze(dim=-1))
        Uhats += torch.bmm(Sigmahats, ((z_Is - phat_Is) * v_Is).unsqueeze(dim=-1)).squeeze()

    thetahats_T = thetahats / (N_B * N_QUESTIONS)
    v_T_sq_simp = varhats_main / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_minus = (((varhats_inner1 / N_B) - varhats_inner2) ** 2).sum(dim=1, keepdim=True)
    v_T_sq_minus /= (N_B * (N_QUESTIONS ** 2))
    v_T_sq_full = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

    z_score = norm.ppf(1 - (ALPHA / 2))
    ub = torch.maximum(thetahats_T + z_score * torch.sqrt(v_T_sq_full / N_B), torch.tensor(0.0, device=device))
    lb = torch.minimum(thetahats_T - z_score * torch.sqrt(v_T_sq_full / N_B), torch.tensor(1.0, device=device))

    mus_M2 = M2.mean(dim=1, keepdim=True)
    mean_width = (ub - lb).mean().item()
    coverage = ((lb <= mus_M2) & (mus_M2 <= ub)).mean(dtype=float).item()

    return mean_width, coverage


# checkpointing
checkpoint_counter = len(pd.read_csv(logs_fname).index)
counter = 0

for dataset in DATASETS:

    M2 = pd.read_csv(f"data/processed/{dataset}/M2.csv")
    M2 = torch.tensor(M2.iloc[:,3:].to_numpy().astype(np.float32)).to(device)
    N_NEW, N_QUESTIONS = M2.shape

    U = torch.load(f"factor_models/final/{dataset}/U_nfobs=None_p=1.0.pt", map_location=device)
    V = torch.load(f"factor_models/final/{dataset}/V_nfobs=None_p=1.0.pt", map_location=device)
    MU0, SIGMA0 = U.mean(axis=0), torch.cov(U.T)

    for budget_prop in BUDGET_PROPS:
        N_B = int(N_QUESTIONS * budget_prop)

        # look up best hyperparameters
        row = best_settings.query(
            f"dataset == '{dataset}' and mcar_obs_prob == 1.0 and prop_budget == {budget_prop}")
        if len(row) == 0:
            print(f"WARNING: no best settings for {dataset}, budget={budget_prop}")
            counter += len(SEED_LIST)
            continue
        beta0, rho, gamma, tau = row[["beta0", "rho", "gamma", "tau"]].values.flatten()

        for seed in SEED_LIST:
            if counter >= checkpoint_counter:
                mw, cov = trial(
                    M2, V, MU0, SIGMA0,
                    N_NEW, N_QUESTIONS, N_B,
                    beta0, rho, gamma, tau, seed, device, counter)

                log_row = [dataset, budget_prop, seed, mw, cov]
                with open(logs_fname, "a") as f:
                    f.write(",".join([str(x) for x in log_row]) + "\n")

            counter += 1
            if counter % 50 == 0:
                print(f"Finished {counter} trials")

    del M2, U, V, MU0, SIGMA0; gc.collect()

print(f"Done! Results in {logs_fname}")
