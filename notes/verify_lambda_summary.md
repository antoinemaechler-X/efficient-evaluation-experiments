# Verifying the Λ Coverage Formula: Summary

## 1. Context

The WOR FAQ estimator uses the Theorem 4 variance formula σ̂² = Â_n − B̂_n. The B̂_n correction (unique to WOR; absent in WR FAQ) is positively biased: E[B̂_n] > B_n, causing systematic undercoverage that grows with budget.

The mathematical analysis in `notes/wor_coverage_analysis.md` (§4.3–4.5) derives the exact coverage formula:

```
coverage = 0.95 − z·φ(z)·(Λ − Λ₃)/n + R
```

where:
- **Λ** = (1/σ̄²) Σ_{s=1}^{n-1} E[v_s]·w_s is the **non-stationarity index**
- **w_s** = Σ_{k=s}^{n-1} 1/k² are the exact weights (partial tails of Σ 1/k²)
- **v_s** = E[Δ_s² | F_{s-1}] is the per-step conditional variance
- **σ̄²** = (1/n) Σ E[v_s] is the average conditional variance
- **Λ₃** = (θ − ψ̄₁)²/σ̄² is the initial model correction (empirically small)
- **z·φ(z)** ≈ 0.1146 (with z = 1.96, φ(z) ≈ 0.0584)
- **R** is a remainder bounded by C₁/√n + C₂(Λ/n)² + C₃√(σ̄²)/N

The formula is exact up to computable remainders. The quantity Λ is **not analytically tractable** because v_s depends on the full adaptive sampling trajectory — but it can be estimated by simulation.

## 2. What `verify_lambda.py` Does

The script estimates Λ by simulation and compares the predicted coverage against actual coverage. Steps:

### 2.1 Inputs
- Reads the same `logs/val/wor_best_settings.csv` as `wor_faq_final.py` → ensures identical hyperparameters (beta0, rho, gamma, tau) per (dataset, budget).
- Loads M2 test data and factor model (U, V, MU0, SIGMA0).

### 2.2 Per-seed computation
For each seed (500 total), runs `trial_faq_wor(..., log_profile=True)` which returns:
- **mean_width**: CI half-width (= z·σ̂/√n)
- **coverage**: 1 if true θ falls in CI, 0 otherwise
- **v_profile**: array of length n, where v_profile[s] = Δ_s² (the squared martingale increment at step s)

Note: v_profile[s] is the *realized* Δ_s² for that seed, not E[Δ_s²]. The expectation is estimated by averaging over seeds.

### 2.3 Λ computation (`compute_lambda`)
From the seed-averaged variance profile v̄_s = (1/n_seeds) Σ_seed v_s^(seed):

1. **σ̄²** = (1/n) Σ_{s=1}^n v̄_s (average conditional variance)
2. **w_s** = Σ_{k=s}^{n-1} 1/k² for s = 1,...,n-1 (computed by reverse cumsum of 1/k²)
3. **Λ** = (1/σ̄²) Σ_{s=1}^{n-1} v̄_s · w_s (exact formula, no approximation)
4. **Λ_H** = Σ_{s=1}^{n-1} g(s)/s where g(s) = v̄_s/σ̄² (harmonic approximation, for comparison)

### 2.4 Coverage predictions
Two predictions are computed from Λ:

1. **Linear (first-order Taylor)**: `cov_pred = 0.95 − z·φ(z)·(Λ − Λ₃)/n`
   - Uses the linearization Φ(z(1−ε)) ≈ Φ(z) − z·φ(z)·ε

2. **Exact-Φ (no linearization)**: `cov_pred = 2·Φ(z·√(1 − (Λ−Λ₃)/n)) − 1`
   - Uses σ̂/σ ≈ √(1 − (Λ−Λ₃)/n) directly in the CDF
   - Better when (Λ−Λ₃)/n is large (e.g., > 0.1)

### 2.5 Outputs
- Prints detailed results (Λ, predictions, actual coverage, variance profile shape)
- Appends one row to `logs/verify_lambda_results.csv` per (dataset, budget)
- Optionally saves the mean variance profile to `logs/variance_profile_{dataset}_budget={budget}.npy`

## 3. Current Status

### What is complete
- **Validation** (`submit_wor_faq_val.sh`): 10 jobs finished → `logs/val/wor_faq_val_*.csv` (all 10 files present)
- **Best settings** (`wor_val_analyzer.py`): → `logs/val/wor_best_settings.csv` (20 rows: 2 datasets × 10 budgets)
- **Final run** (`submit_wor_faq_final.sh`): 3 jobs finished → `logs/final/wor_faq_final_sl={0,1,2}.csv` (100 seeds per setting)

### What is running
- **verify_lambda** (`submit_verify_lambda.sh`): 20 jobs (2 datasets × 10 budgets), 500 seeds each. Currently running on Marlowe.

### Final run coverage (from M2 test set, 100 seeds)

| Budget | MMLU-Pro coverage | BBH coverage |
|:------:|:-----------------:|:------------:|
| 2.5%   | 0.9408            | 0.9449       |
| 5.0%   | 0.9405            | 0.9466       |
| 7.5%   | 0.9423            | 0.9470       |
| 10%    | 0.9432            | 0.9467       |
| 12.5%  | 0.9424            | 0.9473       |
| 15%    | 0.9406            | 0.9460       |
| 17.5%  | 0.9378            | 0.9450       |
| 20%    | 0.9324            | 0.9436       |
| 22.5%  | 0.9262            | 0.9424       |
| 25%    | **0.9168**        | **0.9403**   |

Key patterns:
- MMLU-Pro: clear degradation from ~0.943 (10%) to 0.917 (25%)
- BBH: mild degradation from ~0.947 (12.5%) to 0.940 (25%)
- Both datasets: slight improvement at low budgets (CLT effect), then degradation (B̂ bias effect)

### Best hyperparameters selected by validation

All settings have **rho=0.0, gamma=0.0** (most conservative — no exploration blending). Only tau varies:
- MMLU-Pro: tau=0.25 at 2.5%, tau=0.05 at 5%–25%
- BBH: tau=0.25 at all budgets

This means the coverage degradation is **not caused by aggressive hyperparameters**. It is inherent to the WOR FAQ method.

## 4. What We Will Learn from verify_lambda Results

When the 20 jobs finish, `logs/verify_lambda_results.csv` will contain for each (dataset, budget):
- **Λ** and **Λ₃**: the non-stationarity index and initial model correction
- **cov_pred_linear** and **cov_pred_exact_phi**: the two coverage predictions
- **cov_actual**: actual coverage from the same 500 seeds (should match final run within SE)
- **error_linear** and **error_exact_phi**: prediction − actual

This will answer:
1. **Does Λ predict the coverage curve?** If `cov_pred ≈ cov_actual` at all 20 settings (error < 0.005), the theory fully explains the degradation.
2. **How large is Λ/H_{n-1}?** This ratio measures the non-stationarity amplification. The §4.1 back-computation suggests 4–70× for MMLU-Pro.
3. **How does the variance profile v_s evolve with budget?** The saved .npy files will show whether v_1/v_n increases with budget (steeper profiles → larger Λ).

## 5. Pipeline Reference

```
# Already done:
sbatch submit_wor_faq_val.sh        # validation (10 jobs)
python wor_val_analyzer.py          # → logs/val/wor_best_settings.csv
sbatch submit_wor_faq_final.sh      # final run (3 jobs)

# Currently running:
sbatch submit_verify_lambda.sh      # 20 jobs, 500 seeds each

# After verify_lambda finishes:
# Pull logs/verify_lambda_results.csv and logs/variance_profile_*.npy from cluster
# Analyze: does predicted coverage match actual?
```

## 6. Key Files

| File | Purpose |
|------|---------|
| `notes/wor_coverage_analysis.md` | Full mathematical derivation (§4.3–4.6) |
| `verify_lambda.py` | Estimates Λ by simulation, compares predicted vs actual coverage |
| `submit_verify_lambda.sh` | SLURM script: 20 array jobs (2 datasets × 10 budgets), 500 seeds |
| `wor_trial.py` | Core trial function; `log_profile=True` logs per-step v_s |
| `wor_faq_final.py` | Final coverage run (100 seeds, reads wor_best_settings.csv) |
| `wor_faq_val.py` | Hyperparameter validation sweep |
| `wor_val_analyzer.py` | Selects best hyperparameters (min mean_width) |
| `logs/val/wor_best_settings.csv` | Selected hyperparameters (shared by final + verify_lambda) |
| `logs/verify_lambda_results.csv` | Output: Λ, predictions, coverage for all 20 settings |
