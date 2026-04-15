'''
FAQ with tau=0.75 fixed, fully-observed historical data only.
Runs validation (5 seeds) to pick best (beta0, rho, gamma) per budget,
then runs final testing (100 seeds) on M2 with those settings.

Usage: python faq_tau075.py <dataset_idx>
  dataset_idx: 0 = mmlu-pro, 1 = bbh+gpqa+ifeval+math+musr
'''
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import sigmoid
import sys, os, gc
from tqdm.autonotebook import tqdm
from scipy.stats import norm

device = "cuda" if torch.cuda.is_available() else "cpu"

# settings
dataset = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"][int(sys.argv[1])]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
BETA0_VALS = [0.25, 0.5, 0.75, 1.0]
RHO_VALS = [0.0, 0.05, 0.25, 0.5, 0.75]
GAMMA_VALS = [0.0, 0.05, 0.25, 0.5, 0.75]
TAU = 0.75  # fixed
ALPHA = 0.05

os.makedirs("logs/tau075", exist_ok=True)


def run_faq_trial(
    data, V, MU0, SIGMA0,
    N_NEW, N_QUESTIONS, N_B,
    beta0, rho, gamma, tau, seed, device, counter):
    """Run one FAQ trial. Returns per-model widths and coverage."""

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

        z_Is = torch.gather(data, dim=1, index=I_s)
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

    # compute per-model CI widths
    thetahats_T = thetahats / (N_B * N_QUESTIONS)
    v_T_sq_simp = varhats_main / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_minus = (((varhats_inner1 / N_B) - varhats_inner2) ** 2).sum(dim=1, keepdim=True)
    v_T_sq_minus /= (N_B * (N_QUESTIONS ** 2))
    v_T_sq_full = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

    z_score = norm.ppf(1 - (ALPHA / 2))
    ub = torch.maximum(thetahats_T + z_score * torch.sqrt(v_T_sq_full / N_B), torch.tensor(0.0, device=device))
    lb = torch.minimum(thetahats_T - z_score * torch.sqrt(v_T_sq_full / N_B), torch.tensor(1.0, device=device))
    widths = (ub - lb).squeeze()

    mus = data.mean(dim=1, keepdim=True)
    mean_width = widths.mean().item()
    coverage = ((lb <= mus) & (mus <= ub)).mean(dtype=float).item()

    return mean_width, coverage


# ============================================================
# PHASE 1: VALIDATION — tune (beta0, rho, gamma) on M1 val split
# ============================================================
best_settings_fname = f"logs/tau075/best_settings_{dataset}.csv"

if os.path.exists(best_settings_fname):
    best_settings = pd.read_csv(best_settings_fname)
    if len(best_settings) == len(BUDGET_PROPS):
        print(f"=== PHASE 1: SKIPPED — {best_settings_fname} already has {len(best_settings)} budgets ===")
    else:
        print(f"WARNING: {best_settings_fname} has {len(best_settings)} budgets, expected {len(BUDGET_PROPS)}. Re-running validation.")
        os.remove(best_settings_fname)

if not os.path.exists(best_settings_fname):
    print(f"=== PHASE 1: Validation for {dataset} ===")

    M1_full = pd.read_csv(f"data/processed/{dataset}/M1_nfobs=None_p=1.0.csv")
    M1_full = torch.tensor(M1_full.iloc[:,3:].to_numpy().astype(np.float32))
    M1_val = M1_full[int(M1_full.shape[0] * 0.8):].to(device)

    N_NEW_VAL, N_QUESTIONS = M1_val.shape

    U = torch.load(f"factor_models/val/{dataset}/U_nfobs=None_p=1.0.pt", map_location=device)
    V_val = torch.load(f"factor_models/val/{dataset}/V_nfobs=None_p=1.0.pt", map_location=device)
    MU0_val, SIGMA0_val = U.mean(axis=0), torch.cov(U.T)

    val_results = []
    counter = 0
    N_VAL_SEEDS = 5

    for budget_prop in BUDGET_PROPS:
        N_B = int(N_QUESTIONS * budget_prop)
        for beta0 in BETA0_VALS:
            for rho in RHO_VALS:
                for gamma in GAMMA_VALS:
                    widths_across_seeds = []
                    for seed in range(N_VAL_SEEDS):
                        mw, cov = run_faq_trial(
                            M1_val, V_val, MU0_val, SIGMA0_val,
                            N_NEW_VAL, N_QUESTIONS, N_B,
                            beta0, rho, gamma, TAU, seed, device, counter)
                        widths_across_seeds.append(mw)
                        counter += 1
                    avg_width = np.mean(widths_across_seeds)
                    val_results.append({
                        "prop_budget": budget_prop,
                        "beta0": beta0, "rho": rho, "gamma": gamma,
                        "mean_width": avg_width
                    })

        print(f"  Budget {budget_prop}: validated {len(BETA0_VALS)*len(RHO_VALS)*len(GAMMA_VALS)} combos")

    val_df = pd.DataFrame(val_results)

    # pick best (beta0, rho, gamma) per budget
    best_settings = val_df.sort_values("mean_width").groupby("prop_budget").first().reset_index()
    best_settings.to_csv(best_settings_fname, index=False)
    print(f"Best settings saved to {best_settings_fname}")

    del M1_full, M1_val, U, V_val, MU0_val, SIGMA0_val
    gc.collect()

# ============================================================
# PHASE 2: FINAL TESTING — run on M2 with 100 seeds
# ============================================================
print(f"\n=== PHASE 2: Final testing for {dataset} ===")

M2 = pd.read_csv(f"data/processed/{dataset}/M2.csv")
M2 = torch.tensor(M2.iloc[:,3:].to_numpy().astype(np.float32)).to(device)
N_NEW, N_QUESTIONS = M2.shape

U = torch.load(f"factor_models/final/{dataset}/U_nfobs=None_p=1.0.pt", map_location=device)
V_final = torch.load(f"factor_models/final/{dataset}/V_nfobs=None_p=1.0.pt", map_location=device)
MU0_final, SIGMA0_final = U.mean(axis=0), torch.cov(U.T)

# log file with checkpointing
logs_fname = f"logs/tau075/final_{dataset}.csv"
columns = ["dataset", "prop_budget", "beta0", "rho", "gamma", "tau", "seed", "mean_width", "coverage"]
if not os.path.exists(logs_fname):
    with open(logs_fname, "w") as f:
        f.write(",".join(columns) + "\n")

checkpoint_counter = len(pd.read_csv(logs_fname).index)
counter = 0

N_FINAL_SEEDS = 100

for budget_prop in BUDGET_PROPS:
    N_B = int(N_QUESTIONS * budget_prop)

    # look up best hyperparameters for this budget
    row = best_settings.query(f"prop_budget == {budget_prop}").iloc[0]
    beta0, rho, gamma = row["beta0"], row["rho"], row["gamma"]

    for seed in range(N_FINAL_SEEDS):
        if counter >= checkpoint_counter:
            mw, cov = run_faq_trial(
                M2, V_final, MU0_final, SIGMA0_final,
                N_NEW, N_QUESTIONS, N_B,
                beta0, rho, gamma, TAU, seed, device, counter)

            log_row = [dataset, budget_prop, beta0, rho, gamma, TAU, seed, mw, cov]
            with open(logs_fname, "a") as f:
                f.write(",".join([str(x) for x in log_row]) + "\n")

        counter += 1

        if counter % 50 == 0:
            print(f"  Finished {counter} / {len(BUDGET_PROPS) * N_FINAL_SEEDS}")

del M2, U, V_final, MU0_final, SIGMA0_final
gc.collect()

print(f"\nDone! Results in {logs_fname}")
