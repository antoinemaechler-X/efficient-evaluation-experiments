"""
Without-Replacement Pro-Active Inference (WOR-PAI) Trial Function

Implements equation (2) from the paper:

    θ̂_n = (1/n) Σ_{t=1}^{n} φ_t

    φ_t = (1/N) [ Σ_{i ∈ O_{t-1}} y_i  +  Σ_{i ∉ O_{t-1}} f^{(t-1)}(x_i) ]
         + (1/N) · (y_{I_t} - f^{(t-1)}(x_{I_t})) / q_t(I_t)

where:
    - N = total number of questions (items)
    - n = budget (number of labels to collect)
    - O_{t-1} = set of question indices observed up to round t-1
    - q_t(·) is a probability distribution over I \ O_{t-1} (unobserved questions only)
    - f^{(t-1)} is the factor model prediction, updated after each observation
    - I_t is the question sampled at round t (without replacement)

Key differences vs. the with-replacement version (faq_final.py):
    1. Sampling: multinomial over unobserved questions only (mask observed, renormalize)
    2. φ_t uses actual y_i for already-observed questions, predictions only for unobserved
    3. q_t sums to 1 over unobserved questions only
"""

import torch
from torch.nn.functional import sigmoid
from scipy.stats import norm


def trial_faq_wor(
    M2, V, MU0, SIGMA0,
    N_NEW, N_QUESTIONS, N_B,
    beta0, rho, gamma, tau, seed, device,
    ALPHA=0.05, counter=0, disable_tqdm=True):
    """
    Run one FAQ trial with WITHOUT-REPLACEMENT sampling.

    Args:
        M2: (N_NEW, N_QUESTIONS) tensor of true labels (binary: 0/1)
        V: (N_QUESTIONS, D) factor loadings for questions (fixed, pre-trained)
        MU0: (D,) prior mean for new model factors
        SIGMA0: (D, D) prior covariance for new model factors
        N_NEW: number of test models
        N_QUESTIONS: total number of questions (= N in the formula)
        N_B: labeling budget (= n in the formula)
        beta0, rho, gamma, tau: FAQ hyperparameters
        seed: random seed
        device: torch device
        ALPHA: significance level for CIs (default 0.05)

    Returns:
        [mean_width, coverage]
    """
    from tqdm.autonotebook import tqdm

    # --- Initialization ---
    Uhats = torch.tile(MU0, dims=(N_NEW, 1)).to(device)           # (N_NEW, D)
    Sigmahats = torch.tile(SIGMA0, dims=(N_NEW, 1, 1)).to(device) # (N_NEW, D, D)

    # Observed mask: which questions have been observed for each model
    # O_{t-1} in the formula. Initialized to empty set.
    observed = torch.zeros(N_NEW, N_QUESTIONS, dtype=torch.bool, device=device)

    # Unnormalized accumulator: Σ_t φ_t (without the 1/N factor, divided at the end)
    thetahats = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)

    # Variance accumulators — correct WOR formula (Theorem 4 of PAI paper)
    # A_hat: Σ_t ((y_{I_t} - f^{(t-1)}(x_{I_t})) / q_t(I_t))^2
    varhats_main = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    # B_hat: Σ_{t=2}^n (N*theta_hat_{t-1} - imputed_sum_t)^2
    # where N*theta_hat_{t-1} = (1/(t-1)) * Σ_{s<t} phi_s  (running average of past phis)
    varhats_b = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)

    # Set seed
    torch.random.manual_seed(seed)

    # --- Main loop: t = 0, 1, ..., N_B - 1 ---
    for t in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=disable_tqdm):

        # ============================================================
        # PART 1: Compute sampling probabilities q_t over UNOBSERVED questions
        # ============================================================

        # Current predictions from factor model: f^{(t-1)}(x_j) = σ(u_i · v_j)
        p_hat_js = sigmoid(Uhats @ V.T).clamp(min=1e-12, max=1.0 - 1e-12)  # (N_NEW, N_QUESTIONS)
        p1mp_hat_js = p_hat_js * (1.0 - p_hat_js)
        sqrt_p1mp_hat_js = torch.sqrt(p1mp_hat_js)

        # --- h_o: oracle-inspired score ∝ √(p(1-p)) ---
        # Zero out observed questions before normalizing
        sqrt_p1mp_masked = sqrt_p1mp_hat_js.clone()
        sqrt_p1mp_masked[observed] = 0.0
        ho_js = sqrt_p1mp_masked / sqrt_p1mp_masked.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # Symmetrize Sigmahats for numerical stability
        Sigmahats = (Sigmahats + Sigmahats.mT) / 2.0

        # --- h_a: active-learning score (Fisher information-based) ---
        vtSigmav_js = torch.einsum("ni,mij,nj->mn", V, Sigmahats, V)
        log_denominator = torch.log1p(p1mp_hat_js * vtSigmav_js)
        sq_term = torch.bmm(
            Sigmahats, ((p1mp_hat_js @ V) / N_QUESTIONS).unsqueeze(-1)
        ).squeeze(-1) @ V.T
        log_numerator = (torch.log(p1mp_hat_js.clamp_min(1e-12))
                         + 2 * torch.log(sq_term.abs().clamp_min(1e-12)))
        log_d_js = log_numerator - log_denominator
        # Mask observed questions to -inf before softmax so they get ~0 probability
        log_d_js[observed] = -float('inf')
        ha_js = torch.softmax(log_d_js, dim=1)

        # --- Blend h_o and h_a with exploration/tempering schedule ---
        alpha_s = (torch.maximum(
            torch.tensor(0.0),
            torch.tensor(1.0 - (t + 1.0) / (rho * N_B))
        ) if rho != 0.0 else 0.0)

        beta_s = (beta0 * torch.minimum(
            torch.tensor((t + 1.0) / (gamma * N_B)),
            torch.tensor(1.0)
        ) if gamma != 0.0 else beta0)

        hcat_js = (((1.0 - alpha_s) * ho_js) + (alpha_s * ha_js)) ** beta_s

        # Zero out observed questions (should already be ~0 from masking above)
        hcat_js[observed] = 0.0

        # --- Final q_t: normalize + mix with uniform over UNOBSERVED questions ---
        n_unobs = N_QUESTIONS - t  # same for all models at round t
        hcat_sum = hcat_js.sum(dim=1, keepdim=True).clamp(min=1e-12)
        q_js = ((hcat_js / hcat_sum) * (1.0 - tau)) + (tau / n_unobs)
        # Zero out observed questions (uniform part gave them nonzero mass)
        q_js[observed] = 0.0
        # Note: q_js now sums to 1 over unobserved questions for each model

        # --- Sample I_t without replacement ---
        I_t = torch.multinomial(input=q_js, num_samples=1)  # (N_NEW, 1)

        # ============================================================
        # PART 2: Compute φ_t and update estimator (equation 2)
        # ============================================================

        # Gather values for sampled questions
        z_It = torch.gather(M2, dim=1, index=I_t)          # y_{I_t}
        phat_It = torch.gather(p_hat_js, dim=1, index=I_t)  # f^{(t-1)}(x_{I_t})
        q_It = torch.gather(q_js, dim=1, index=I_t)         # q_t(I_t)

        # --- Equation (2), term 1: imputed sum ---
        # (1/N) [ Σ_{i ∈ O_{t-1}} y_i  +  Σ_{i ∉ O_{t-1}} f^{(t-1)}(x_i) ]
        # We store unnormalized (without 1/N), divide at the end by N_B * N_QUESTIONS
        imputed_sum = (
            (observed.float() * M2).sum(dim=1, keepdim=True)            # Σ_{i ∈ O_{t-1}} y_i
            + ((~observed).float() * p_hat_js).sum(dim=1, keepdim=True)  # Σ_{i ∉ O_{t-1}} f^{(t-1)}(x_i)
        )

        # --- Equation (2), term 2: AIPW correction ---
        # (1/N) · (y_{I_t} - f^{(t-1)}(x_{I_t})) / q_t(I_t)
        aipw_t = (z_It - phat_It) / q_It

        # --- φ_t (unnormalized: without 1/N factor) ---
        phi_t = imputed_sum + aipw_t

        # --- B_hat accumulator: (N*theta_hat_{t-1} - imputed_sum_t)^2 for t >= 2 ---
        # N*theta_hat_{t-1} = thetahats / t  (thetahats holds sum of phi_s for s < t)
        # Paper sums from t=2; in 0-indexed loop this means t >= 1.
        if t >= 1:
            ntheta_prev = thetahats / t  # = (1/(t)) * Σ_{s=0}^{t-1} phi_s  [unnorm, = N*theta_hat_{t-1}]
            varhats_b += (ntheta_prev - imputed_sum) ** 2

        thetahats += phi_t

        # --- A_hat accumulator ---
        varhats_main += (aipw_t ** 2)

        # ============================================================
        # PART 3: Update observed set O_t = O_{t-1} ∪ {I_t}
        # ============================================================
        observed.scatter_(dim=1, index=I_t, value=True)

        # ============================================================
        # PART 4: Update factor model (Sherman-Morrison, same as original)
        # ============================================================
        w_t = torch.gather(p1mp_hat_js, dim=1, index=I_t)  # p(1-p) for sampled question
        v_It = V[I_t.flatten()]                              # (N_NEW, D)
        Sigma_v_It = torch.bmm(Sigmahats, v_It.unsqueeze(-1))
        vT_Sigma_v_It = torch.bmm(v_It.unsqueeze(-2), Sigma_v_It).squeeze(-1)
        denominator = 1.0 + (w_t * vT_Sigma_v_It)
        numerator = w_t.unsqueeze(-1) * (Sigma_v_It @ Sigma_v_It.mT)
        Sigmahats -= (numerator / denominator.unsqueeze(-1))

        # Mean update: u^{(t)} = u^{(t-1)} + Σ · v · (y - p̂)
        Uhats += torch.bmm(
            Sigmahats, ((z_It - phat_It) * v_It).unsqueeze(-1)
        ).squeeze()

    # ============================================================
    # PART 5: Compute point estimates + confidence intervals
    # ============================================================

    # θ̂_n = (1/n) · (1/N) · Σ_t φ_t  (φ_t was accumulated without 1/N)
    thetahats_T = thetahats / (N_B * N_QUESTIONS)

    # Variance estimation — correct WOR formula (Theorem 4):
    # sigma_hat^2 = A_hat - B_hat
    # A_hat = (1/(n*N^2)) * Σ_t (y_{I_t} - f_{I_t})^2 / q_t^2
    # B_hat = (1/(n*N^2)) * Σ_{t=2}^n (N*theta_hat_{t-1} - imputed_sum_t)^2
    v_T_sq_simp = varhats_main / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_minus = varhats_b / (N_B * (N_QUESTIONS ** 2))

    v_T_sq_full = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

    # 95% CI
    z_score = norm.ppf(1 - (ALPHA / 2))
    mus_M2 = M2.mean(dim=1, keepdim=True)

    ub = torch.maximum(
        thetahats_T + z_score * torch.sqrt(v_T_sq_full / N_B),
        torch.tensor(0.0, device=device))
    lb = torch.minimum(
        thetahats_T - z_score * torch.sqrt(v_T_sq_full / N_B),
        torch.tensor(1.0, device=device))

    mean_width = (ub - lb).mean().item()
    coverage = ((lb <= mus_M2) & (mus_M2 <= ub)).mean(dtype=float).item()

    return [mean_width, coverage]


def trial_ablation_wor(
    M2, V, MU0, SIGMA0,
    N_NEW, N_QUESTIONS, N_B,
    tau, seed, device,
    ALPHA=0.05, counter=0, disable_tqdm=True):
    """
    Active inference baseline (Zrnic & Candes '24) adapted for WITHOUT-REPLACEMENT.

    Instead of sequential Bernoulli, we sample exactly N_B questions without replacement
    using the coverage-based scoring u_t = 2·min(p, 1-p), mixed with uniform.
    Uses the new PAI estimator (equation 2).
    """
    from tqdm.autonotebook import tqdm

    # --- Initialization ---
    Uhats = torch.tile(MU0, dims=(N_NEW, 1)).to(device)
    Sigmahats = torch.tile(SIGMA0, dims=(N_NEW, 1, 1)).to(device)

    observed = torch.zeros(N_NEW, N_QUESTIONS, dtype=torch.bool, device=device)
    thetahats = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_main = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_b = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)

    torch.random.manual_seed(seed)

    for t in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=disable_tqdm):

        # --- Scoring: u(j) = 2·min(p̂_j, 1-p̂_j) (coverage measure) ---
        p_hat_js = sigmoid(Uhats @ V.T).clamp(min=1e-12, max=1.0 - 1e-12)
        p1mp_hat_js = p_hat_js * (1.0 - p_hat_js)
        u_js = 2.0 * torch.minimum(p_hat_js, 1.0 - p_hat_js)  # (N_NEW, N_QUESTIONS)

        # Mask observed questions
        u_js[observed] = 0.0

        # Normalize + mix with uniform over unobserved
        n_unobs = N_QUESTIONS - t
        u_sum = u_js.sum(dim=1, keepdim=True).clamp(min=1e-12)
        q_js = ((u_js / u_sum) * (1.0 - tau)) + (tau / n_unobs)
        q_js[observed] = 0.0

        # Sample
        I_t = torch.multinomial(input=q_js, num_samples=1)

        # --- PAI estimator (equation 2) ---
        z_It = torch.gather(M2, dim=1, index=I_t)
        phat_It = torch.gather(p_hat_js, dim=1, index=I_t)
        q_It = torch.gather(q_js, dim=1, index=I_t)

        imputed_sum = (
            (observed.float() * M2).sum(dim=1, keepdim=True)
            + ((~observed).float() * p_hat_js).sum(dim=1, keepdim=True)
        )

        aipw_t = (z_It - phat_It) / q_It
        phi_t = imputed_sum + aipw_t

        if t >= 1:
            ntheta_prev = thetahats / t
            varhats_b += (ntheta_prev - imputed_sum) ** 2

        thetahats += phi_t
        varhats_main += (aipw_t ** 2)

        # Update observed set
        observed.scatter_(dim=1, index=I_t, value=True)

        # --- Factor model update ---
        w_t = torch.gather(p1mp_hat_js, dim=1, index=I_t)
        v_It = V[I_t.flatten()]
        Sigma_v_It = torch.bmm(Sigmahats, v_It.unsqueeze(-1))
        vT_Sigma_v_It = torch.bmm(v_It.unsqueeze(-2), Sigma_v_It).squeeze(-1)
        denominator = 1.0 + (w_t * vT_Sigma_v_It)
        numerator = w_t.unsqueeze(-1) * (Sigma_v_It @ Sigma_v_It.mT)
        Sigmahats -= (numerator / denominator.unsqueeze(-1))
        Uhats += torch.bmm(
            Sigmahats, ((z_It - phat_It) * v_It).unsqueeze(-1)
        ).squeeze()

    # --- CI construction ---
    thetahats_T = thetahats / (N_B * N_QUESTIONS)
    v_T_sq_simp = varhats_main / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_minus = varhats_b / (N_B * (N_QUESTIONS ** 2))
    v_T_sq_full = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

    z_score = norm.ppf(1 - (ALPHA / 2))
    mus_M2 = M2.mean(dim=1, keepdim=True)
    ub = torch.maximum(
        thetahats_T + z_score * torch.sqrt(v_T_sq_full / N_B),
        torch.tensor(0.0, device=device))
    lb = torch.minimum(
        thetahats_T - z_score * torch.sqrt(v_T_sq_full / N_B),
        torch.tensor(1.0, device=device))

    mean_width = (ub - lb).mean().item()
    coverage = ((lb <= mus_M2) & (mus_M2 <= ub)).mean(dtype=float).item()

    return [mean_width, coverage]
