import torch
from torch.nn.functional import sigmoid
from scipy.stats import norm


def trial_faq_wor(
    M2, V, MU0, SIGMA0,
    N_NEW, N_QUESTIONS, N_B,
    beta0, rho, gamma, tau, seed, device,
    ALPHA=0.05, counter=0, disable_tqdm=True):
    # Run one FAQ trial with WITHOUT-REPLACEMENT sampling (equation 2).
    from tqdm.autonotebook import tqdm

    # PART 1: Initialization
    Uhats = torch.tile(MU0, dims=(N_NEW, 1)).to(device)           # (N_NEW, D)
    Sigmahats = torch.tile(SIGMA0, dims=(N_NEW, 1, 1)).to(device) # (N_NEW, D, D)

    # O_{t-1}: which questions have been observed for each model
    observed = torch.zeros(N_NEW, N_QUESTIONS, dtype=torch.bool, device=device)

    # Unnormalized accumulator for Σ_t φ_t
    thetahats = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)

    # Variance accumulators (WOR formula, Theorem 4 of PAI paper)
    # A_hat: Σ_t ((y_{I_t} - f^{(t-1)}(x_{I_t})) / q_t(I_t))^2
    varhats_main = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    # B_hat: Σ_{t=2}^n (N*theta_hat_{t-1} - imputed_sum_t)^2
    varhats_b = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)

    torch.random.manual_seed(seed)

    for t in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=disable_tqdm):

        # ============================================================
        # PART 2: Compute sampling probabilities q_t over UNOBSERVED questions
        # ============================================================

        # f^{(t-1)}(x_j) = σ(u_i · v_j)
        p_hat_js = sigmoid(Uhats @ V.T).clamp(min=1e-12, max=1.0 - 1e-12)  # (N_NEW, N_QUESTIONS)
        p1mp_hat_js = p_hat_js * (1.0 - p_hat_js)
        sqrt_p1mp_hat_js = torch.sqrt(p1mp_hat_js)

        # h_o: oracle-inspired score ∝ √(p(1-p)), masked to unobserved questions
        sqrt_p1mp_masked = sqrt_p1mp_hat_js.clone()
        sqrt_p1mp_masked[observed] = 0.0
        ho_js = sqrt_p1mp_masked / sqrt_p1mp_masked.sum(dim=1, keepdim=True).clamp(min=1e-12)

        Sigmahats = (Sigmahats + Sigmahats.mT) / 2.0

        # h_a: active-learning score (Fisher information-based)
        vtSigmav_js = torch.einsum("ni,mij,nj->mn", V, Sigmahats, V)
        log_denominator = torch.log1p(p1mp_hat_js * vtSigmav_js)
        sq_term = torch.bmm(
            Sigmahats, ((p1mp_hat_js @ V) / N_QUESTIONS).unsqueeze(-1)
        ).squeeze(-1) @ V.T
        log_numerator = (torch.log(p1mp_hat_js.clamp_min(1e-12))
                         + 2 * torch.log(sq_term.abs().clamp_min(1e-12)))
        log_d_js = log_numerator - log_denominator
        log_d_js[observed] = -float('inf')  # mask observed to zero probability
        ha_js = torch.softmax(log_d_js, dim=1)

        # Blend h_o and h_a with exploration/tempering schedule
        alpha_s = (torch.maximum(
            torch.tensor(0.0),
            torch.tensor(1.0 - (t + 1.0) / (rho * N_B))
        ) if rho != 0.0 else 0.0)

        beta_s = (beta0 * torch.minimum(
            torch.tensor((t + 1.0) / (gamma * N_B)),
            torch.tensor(1.0)
        ) if gamma != 0.0 else beta0)

        hcat_js = (((1.0 - alpha_s) * ho_js) + (alpha_s * ha_js)) ** beta_s
        hcat_js[observed] = 0.0

        # Final q_t: normalize + mix with uniform over UNOBSERVED questions
        n_unobs = N_QUESTIONS - t
        hcat_sum = hcat_js.sum(dim=1, keepdim=True).clamp(min=1e-12)
        q_js = ((hcat_js / hcat_sum) * (1.0 - tau)) + (tau / n_unobs)
        q_js[observed] = 0.0  # q_js sums to 1 over unobserved questions

        # Sample I_t without replacement
        I_t = torch.multinomial(input=q_js, num_samples=1)  # (N_NEW, 1)

        # ============================================================
        # PART 3: Compute φ_t and update estimator (equation 2)
        # ============================================================

        z_It = torch.gather(M2, dim=1, index=I_t)           # y_{I_t}
        phat_It = torch.gather(p_hat_js, dim=1, index=I_t)  # f^{(t-1)}(x_{I_t})
        q_It = torch.gather(q_js, dim=1, index=I_t)          # q_t(I_t)

        # Imputed sum: (1/N)[Σ_{i∈O} y_i + Σ_{i∉O} f^{(t-1)}(x_i)] (without 1/N factor)
        imputed_sum = (
            (observed.float() * M2).sum(dim=1, keepdim=True)
            + ((~observed).float() * p_hat_js).sum(dim=1, keepdim=True)
        )

        # AIPW correction: (1/N) · (y_{I_t} - f^{(t-1)}(x_{I_t})) / q_t(I_t)
        aipw_t = (z_It - phat_It) / q_It

        phi_t = imputed_sum + aipw_t

        # B_hat accumulator: (N*theta_hat_{t-1} - imputed_sum_t)^2 for t >= 2
        if t >= 1:
            ntheta_prev = thetahats / t
            varhats_b += (ntheta_prev - imputed_sum) ** 2

        thetahats += phi_t
        varhats_main += (aipw_t ** 2)

        # Update observed set O_t = O_{t-1} ∪ {I_t}
        observed.scatter_(dim=1, index=I_t, value=True)

        # ============================================================
        # PART 4: Update factor model (Sherman-Morrison)
        # ============================================================
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

    # ============================================================
    # PART 5: Compute point estimates + confidence intervals
    # ============================================================

    # θ̂_n = (1/n) · (1/N) · Σ_t φ_t
    thetahats_T = thetahats / (N_B * N_QUESTIONS)

    # Variance: sigma^2 = A_hat - B_hat (WOR Theorem 4)
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


def trial_ablation_wor(
    M2, V, MU0, SIGMA0,
    N_NEW, N_QUESTIONS, N_B,
    tau, seed, device,
    ALPHA=0.05, counter=0, disable_tqdm=True):
    # Active inference baseline (Zrnic & Candes '24) adapted for WITHOUT-REPLACEMENT.
    # Uses coverage-based scoring u_t = 2·min(p, 1-p), mixed with uniform.
    # Uses the WOR PAI estimator (equation 2).
    from tqdm.autonotebook import tqdm

    Uhats = torch.tile(MU0, dims=(N_NEW, 1)).to(device)
    Sigmahats = torch.tile(SIGMA0, dims=(N_NEW, 1, 1)).to(device)

    observed = torch.zeros(N_NEW, N_QUESTIONS, dtype=torch.bool, device=device)
    thetahats = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_main = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)
    varhats_b = torch.zeros(N_NEW, 1, dtype=torch.float32, device=device)

    torch.random.manual_seed(seed)

    for t in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=disable_tqdm):

        p_hat_js = sigmoid(Uhats @ V.T).clamp(min=1e-12, max=1.0 - 1e-12)
        p1mp_hat_js = p_hat_js * (1.0 - p_hat_js)

        # Scoring: u(j) = 2·min(p̂_j, 1-p̂_j), masked to unobserved
        u_js = 2.0 * torch.minimum(p_hat_js, 1.0 - p_hat_js)
        u_js[observed] = 0.0

        n_unobs = N_QUESTIONS - t
        u_sum = u_js.sum(dim=1, keepdim=True).clamp(min=1e-12)
        q_js = ((u_js / u_sum) * (1.0 - tau)) + (tau / n_unobs)
        q_js[observed] = 0.0

        I_t = torch.multinomial(input=q_js, num_samples=1)

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

        observed.scatter_(dim=1, index=I_t, value=True)

        # Factor model update (Sherman-Morrison)
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
