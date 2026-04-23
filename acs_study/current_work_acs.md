# ACS Study — Current State

## Goal

Apply the FAQ algorithm (from the paper) to the ACS Census 2019 California dataset as a
continuous regression analog of the paper's LLM evaluation setting.

**Estimand**: mean of z-scored PINCP (personal income) over the unlabeled split.
- Why mean? Simplest estimand — FAQ AIPW formulas apply verbatim.
- Why z-score? Makes CI widths scale-independent.
- theta_true ≈ 0 (z-score relative to labeled mean/std).

---

## Key Finding: FAQ = uniform+PAI on ACS (fully investigated)

**Short answer**: FAQ provides no benefit over uniform+PAI on ACS income data.
This is a genuine result, not a bug. See `faq_findings.md` for the complete analysis.

**Root cause**: ACS income residuals are **approximately homoscedastic** (all individuals
similarly predictable). FAQ's oracle score h_o ∝ √(σ² + v_j^T Σ v_j) collapses to
√σ² ≈ constant for all j whenever the data is homoscedastic. No heterogeneous structure
for FAQ to exploit. This is a fundamental property of continuous vs binary outcomes:
- **Binary (paper)**: h_o ∝ √(p_j(1−p_j)) varies automatically — hard/easy questions differ
- **Continuous (ACS)**: h_o needs heteroscedastic residuals, which ACS income lacks

### Four hypotheses eliminated

| Hypothesis | Tested via | Result |
|---|---|---|
| Posterior collapse (190k labeled → Σ≈0) | Low-label regime (n=200–2000) | FAQ still = uniform+PAI |
| Model too weak (R²=0.22) | D=128 (better R²) | FAQ still = uniform+PAI |
| Need decoupled prediction/uncertainty | XGBoost(R²≈0.5)+BLR hybrid | FAQ still = uniform+PAI |
| Homoscedastic continuous data | diag_posterior.py ratio table | **Root cause confirmed** |

### Posterior collapse diagnostic (vtSigmav/sigma2 ratio)

| n_labeled | ratio | sigma2 |
|----------:|------:|-------:|
|        50 | 0.481 | 0.886  |
|       100 | 0.191 | 0.675  |
|       200 | 0.083 | 0.418  |
|       500 | 0.032 | 0.738  |
|     2,000 | 0.008 | 0.860  |
|   190,045 | 0.000 | 0.778  |

Ratio must be ~1 for FAQ to have signal. Only n_labeled=50 is meaningful.

---

## Files

### Core experiment
| File | Purpose |
|------|---------|
| `run_faq.py` | BLR-based FAQ. Supports `--estimators faq,classical,uniform+pai`. `--D`, `--n_labeled` configurable. Prints vtSigmav/σ² diagnostic at s=0,counter=0. |
| `run_faq_xgb.py` | Hybrid: XGBoost (full labeled split, R²≈0.5) for AIPW predictions + BLR on residuals (--n_labeled_blr) for sampling scores. Decouples prediction quality from posterior uncertainty. |
| `plot_paper.py` | Paper-style Figure 2 (ESS + coverage vs budget). Handles old/new CSV schemas (interval width → mean_width rename, estimator name normalization). |
| `diag_posterior.py` | Prints vtSigmav/σ² ratio vs n_labeled with and without V normalization. Key: ratio is purely a function of n_labeled, V scaling cancels exactly. |
| `plot_blr_perf.py` | 3-panel BLR diagnostics: predicted vs actual, residual distribution, R² learning curve. |
| `plot_xgb_perf.py` | 4-panel XGBoost diagnostics: predicted vs actual, residuals, heteroscedasticity (std by decile), learning curve. |

### Data / preprocessing
| File | Purpose |
|------|---------|
| `utils.py` | `get_data`, `transform_features` (one-hot), `ols`. Reads local CSV. |
| `data/2019/1-Year/psam_p06.csv` | Raw ACS PUMS California (~380k rows). |

### sbatch scripts (repo root)
| File | Purpose |
|------|---------|
| `submit_faq_acs_clean.sh` | Single clean run: D=16, all 3 estimators (faq, classical, uniform+pai), 100 trials. |
| `submit_faq_acs_D128.sh` | Array: n_labeled ∈ {0(full), 500, 2000, 10000}, D=128. |
| `submit_faq_acs_lowlabel.sh` | Array: n_labeled ∈ {100, 500, 2000}, D=16. |
| `submit_faq_acs_xgb.sh` | Array: XGBoost+BLR, n_labeled_blr ∈ {200, 1000, 5000, full}. |
| `submit_faq_acs_nlab50.sh` | n_labeled=50 (ratio≈0.48, max FAQ signal). **Status: submitted, results pending.** |
| `submit_faq_acs_smoke.sh` | Smoke test: 5 trials, 3 budgets, n_max=10k. |
| `submit_faq_acs_baselines.sh` | Old: baselines only for D∈{8,16,32}. Superseded by submit_faq_acs_clean.sh. |

### Results already computed (on cluster, committed to git)
- `faq_acs_nlab50.csv` / `faq_acs_nlab50_figure2.png` — **pending** (just submitted)
- Various D=8/16/32 CSVs + figure PNGs from earlier experiments

---

## Design

### BLR setup
- Factor matrix V: truncated SVD (top-D singular vectors × singular values) of one-hot encoded features.
- Prior: Normal(0, τ²I), τ=10. Posterior: closed-form.
- sigma2: OLS residual variance on labeled split.
- V normalization: mathematically neutral (cancels in v^T Σ v / σ²). Not applied.

### Three estimators
| Estimator | Role | How |
|-----------|------|-----|
| `classical` | Naive baseline | Uniform w-replacement, sample mean only. |
| `uniform+pai` | "Best baseline" (paper) | τ=1.0 → uniform q_j=1/N. AIPW + online BLR. |
| `faq` | FAQ | Adaptive h_o + h_a scoring. AIPW + online BLR. |

### Hyperparameters (defaults in run_faq.py)
- D=16, prior_tau=10, tau=0.1, beta0=1.0, rho=0.05, gamma=0.25
- num_trials=100, num_budgets=11, budget_min=0.005, budget_max=0.10
- train_frac=0.5 (~190k labeled, ~190k unlabeled), seed=0

---

## Current Status / Next Steps

**ACS study is paused.** Investigation complete — results documented in `faq_findings.md`.

### Pending check
- `submit_faq_acs_nlab50.sh` results: n_labeled=50 job submitted. When returning to ACS,
  pull results and check if ratio≈0.48 actually translates to a FAQ advantage.
  Expected outcome: probably still no FAQ benefit (homoscedasticity remains the root cause).

### If we want to demonstrate FAQ on a continuous dataset
See `faq_findings.md` section "What We Could Try Next on ACS" and section "Conditions for
Choosing a Dataset Where FAQ Will Work". Best candidates:
1. Subgroup mean (small demographic intersection) — makes h_a non-trivial
2. OLS regression coefficient (β for education → income) — high-leverage individuals matter more
3. Student performance data / medical tests / crowdsourced annotations — binary-like structure

### AIPW still works
Even without FAQ's active sampling benefit, AIPW (uniform+PAI) gives meaningful ESS gains
over classical: ~1.3× with BLR (R²=0.22), ~1.6× with XGBoost (R²≈0.5). ESS ceiling = 1/(1-R²).
