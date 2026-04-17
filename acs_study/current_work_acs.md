# ACS Study — Current State

## Goal

Apply the FAQ algorithm (from the paper) to the ACS Census 2019 California dataset, producing
figures that look like Figure 2 of the paper but for a continuous regression task instead of
binary LLM evaluation.

**Estimand**: mean of z-scored PINCP (personal income) over the unlabeled split.
- Why mean, not OLS coefficient? Chosen for simplicity so the FAQ AIPW formulas apply verbatim.
- Why z-score? Makes CI widths scale-independent for comparison.
- theta_true ≈ 0 (close to 0 because z-score is relative to labeled mean/std).

---

## Files

### Data / preprocessing
| File | Purpose |
|------|---------|
| `utils.py` | `get_data`, `transform_features` (one-hot), `ols`. Reads from local CSV to avoid OOM. |
| `load_data.py`, `check_data.py` | Data verification utilities |
| `data/2019/1-Year/psam_p06.csv` | Raw ACS PUMS California (380k rows, on cluster + locally) |

### Experiments
| File | Purpose | Status |
|------|---------|--------|
| `train_model.py` | Trains two XGBoost regressors (PINCP + error magnitude). Saves `.npz`, `.tree.json`, `.tree_err.json`, `.features.npz` | Working. Used only by `run_baseline.py` now. |
| `run_baseline.py` | One-shot active inference (no fine-tuning). Three estimators: active, uniform, classical. Sandwich CI. | Verified working locally |
| `run_faq.py` | **Main experiment**. BLR-based FAQ. See design below. | Tested locally + on cluster (FAQ pass, baselines running) |

### Plotting
| File | Purpose |
|------|---------|
| `plot_paper.py` | **Paper Figure 2 style**: ESS vs budget (top) + coverage (bottom). Same RC params/colors/markers as `Main Text Figures.ipynb`. |
| `plot_faq.py` | Older seaborn-based plot. Less polished, kept for quick checks. |

### sbatch scripts (repo root)
| File | Purpose |
|------|---------|
| `submit_faq_acs_smoke.sh` | Smoke test: 5 trials, 3 budgets, n_max=10k. Runs + plots. |
| `submit_faq_acs.sh` | Full FAQ run: 100 trials, 11 budgets, D∈{8,16,32}. FAQ only. |
| `submit_faq_acs_baselines.sh` | Baselines only (classical + uniform+pai): 100 trials, 11 budgets, D∈{8,16,32}. Plots at end. |

---

## Design Choices

### Prediction model: BLR (not XGBoost)
- XGBoost has no clean Bayesian posterior → no analog for FAQ's active-learning score h_a.
- BLR (Bayesian linear regression) on D-dim SVD factors has a closed-form posterior → full FAQ structure preserved.
- Factor matrix V: truncated SVD (top-D singular vectors × singular values) of the stacked encoded feature matrix (one-hot categoricals + continuous). D=16 default.
- Prior: Normal(0, τ²I), τ=10. Posterior: closed-form on labeled split.
- sigma2: OLS residual variance on labeled split (homoscedastic assumption).

### Sampling and AIPW
- Exactly mirrors `faq_val.py`:
  - Oracle score h_o ∝ √(σ² + v_j^T Σ v_j) (predictive std; analog of √p(1-p))
  - Active score h_a ∝ |Σv̄·V_j|² / (σ² + v_j^T Σ v_j) (info gain on E[Y]; derived from Fisher info with w=1/σ²)
  - α_s (exploration), β_s (tempering), τ (uniform mix) — identical to paper
  - Multinomial WITH replacement, fixed budget N_B = int(budget × N_QUESTIONS)
  - Sherman-Morrison posterior update with w = 1/σ² (analog of p(1-p))
- AIPW: φ_s = Σ_j ŷ_j + (y_Is − ŷ_Is)/q_s(Is) — identical to paper
- Variance: v_T_sq_simp − v_T_sq_minus — identical to paper

### Three estimators
| Estimator | Role | How |
|-----------|------|-----|
| `classical` | "Uniform" in paper | Uniform w-replacement sampling + sample mean. No AIPW, no ML. |
| `uniform+pai` | "Best Baseline" in paper | trial() with τ=1.0 → uniform q_j=1/N. AIPW + online BLR. |
| `faq` | FAQ | Adaptive sampling with h_o + h_a. AIPW + online BLR. |

### Hyperparameters (defaults)
- D=16, prior_tau=10, tau=0.1, beta0=1.0, rho=0.05, gamma=0.25
- num_trials=100, num_budgets=11, budget_min=0.005, budget_max=0.10
- train_frac=0.5 (190k labeled, 190k unlabeled), seed=0

---

## Current Status

### Already computed (on cluster)
- `faq_acs_D8.csv`, `faq_acs_D16.csv`, `faq_acs_D32.csv` — FAQ only, 100 trials × 11 budgets
  - Schema: lb, ub, interval width, coverage, estimator, $n_b$ (OLD schema, no seed/prop_budget columns)
  - `plot_paper.py` handles the old schema via column renaming + cumcount for trial index

### Currently running (cluster)
- `submit_faq_acs_baselines.sh` — 3 array jobs (D=8,16,32) computing classical + uniform+pai
  - Will produce: `faq_acs_D8_baselines.csv`, `faq_acs_D16_baselines.csv`, `faq_acs_D32_baselines.csv`
  - Will produce: `faq_acs_D8_figure2.pdf`, `faq_acs_D16_figure2.pdf`, `faq_acs_D32_figure2.pdf`
  - Runtime estimate: ~1-2h per job on GPU

### Next steps when jobs finish
1. `git pull` on cluster → push results
2. `git pull` locally
3. Open `faq_acs_D16_figure2.pdf` (or re-plot locally with `plot_paper.py`)
4. Compare ESS multipliers across D=8/16/32 — if similar, D=8 is sufficient
5. Compare widths: classical > uniform+pai > faq at each budget?

---

## Mapping to Paper Figures

| Paper Figure | ACS equivalent | Notes |
|---|---|---|
| Figure 2 (ESS + coverage, fully-observed) | `plot_paper.py` output | Single dataset panel instead of 2 |
| Figure 3 (ESS under missingness) | Not yet done | Would require simulating missing data in V |
| Figure 5 (CI-width ratio ablation/FAQ) | Not yet done | Would require varying τ for uniform+pai |
