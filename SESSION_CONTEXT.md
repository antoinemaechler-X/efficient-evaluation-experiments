# Session Context: Efficiently Evaluating LLMs

## What This Project Does

Paper: **"Efficiently Evaluating LLM Performance with Statistical Guarantees"**

**Problem**: Evaluating LLMs on benchmark suites requires running every model on every question — expensive at scale. Can we evaluate ~2,200 models using only a fraction of the questions while still getting valid confidence intervals on per-model accuracy?

**Solution**: FAQ (Factorized Active Querying) — an active learning algorithm that uses a pre-trained factor model (binary matrix factorization) to decide which (model, question) entries to observe next. It produces valid confidence intervals via AIPW (Augmented Inverse-Probability Weighting).

**Baseline comparison**: Traditional active inference (Zrnic & Candes 2024), which uses a simpler uniform-mixing sampling rule parameterized by `tau`. We compare FAQ to this baseline using post-hoc selection (picking the best `tau` after seeing test results — favorable to the baseline).

## Key Concepts

| Term | Meaning |
|------|---------|
| **M1** | Historical models (~400-700). Split 80/20 into train/val for hyperparameter tuning |
| **M2** | Test models (~2,200). These are what we evaluate |
| **Budget** | Fraction of questions labeled per model (e.g., 10% = observe 10% of all questions per model) |
| **Factor model** | Binary matrix factorization (U, V matrices) pre-trained on M1. Predicts which questions a model might get right |
| **Missingness** | Simulated missing data in M1. `(n_full_obs, mcar_obs_prob)` — how many fully-observed rows + elementwise obs probability |
| **tau** | Uniform mixing parameter in active inference. Low tau = more aggressive targeting |
| **beta0, rho, gamma** | FAQ hyperparameters: tempering, exploration ratio, tempering schedule |
| **AIPW** | Augmented Inverse-Probability Weighting — statistical estimator that corrects for non-uniform sampling |
| **Coverage** | Fraction of models whose true accuracy falls within the CI (target: 95%) |
| **Mean width** | Average CI width across models (lower = better, tighter intervals) |

## Two Benchmark Suites

1. **`mmlu-pro`** — ~2,141 questions, ~2,141 test models
2. **`bbh+gpqa+ifeval+math+musr`** — ~2,213 questions, aggregates 5 benchmarks from HuggingFace Open LLM Leaderboard

## Missingness Settings

8 total, split into two groups for parallelization:

| Group | Setting | `n_full_obs` | `mcar_obs_prob` | Meaning |
|-------|---------|-------------|-----------------|---------|
| ms=0 | 0 | 50 | 0.1 | 50 fully-observed rows, 10% of rest observed |
| ms=0 | 1 | 200 | 0.1 | 200 fully-observed rows, 10% of rest observed |
| ms=0 | 2 | 800 | 0.1 | 800 fully-observed rows, 10% of rest observed |
| ms=0 | 3 | None | 1.0 | **Fully observed** (no missingness) |
| ms=1 | 4 | 0 | 0.01 | No fully-observed rows, 1% observed |
| ms=1 | 5 | 0 | 0.001 | 0.1% observed |
| ms=1 | 6 | 0 | 0.0001 | 0.01% observed |
| ms=1 | 7 | 0 | 0.00001 | 0.001% observed |

## Data Layout

```
data/
  processed/
    mmlu-pro.zip          → unzip to mmlu-pro/
    bbh+gpqa+ifeval+math+musr.zip → unzip to bbh+gpqa+ifeval+math+musr/
    Each contains:
      M1.csv  (fully-observed historical data)
      M1_nfobs={n}_p={p}.csv  (historical data with simulated missingness)
      M2.csv  (test models — rows are models, columns are questions)
  data_helpers.py         (dataset name → size constants)

factor_models/
  val/{dataset}/U_nfobs=...pt, V_nfobs=...pt   (trained on M1 train split)
  final/{dataset}/U_nfobs=...pt, V_nfobs=...pt  (trained on all of M1)

logs/
  val/
    best_settings.csv       (best FAQ hyperparams per dataset/missingness/budget, low budgets)
    faq_val_logs.csv        (all validation runs)
  final/
    faq_final_logs.csv      (FAQ test results, low budgets 2.5%-25%, all missingness)
    active_inference_ablation_logs.csv  (ablation test results, low budgets, all missingness)
    baseline_logs.csv       (other baselines)
    cleaned/                (post-hoc best summaries from cleaning_results.py)
```

## File Map: Original Paper vs. New Extensions

### Original paper files (tracked in git)
These reproduce the paper exactly. **Do not modify.**

| File | Purpose |
|------|---------|
| `faq_val.py` | Tune FAQ hyperparams on M1 train/val split (low budgets 2.5%-25%) |
| `faq_final.py` | Run FAQ on test models M2 (low budgets) |
| `active_inference_factor_ablation.py` | Run ablation baseline on M2 (low budgets) |
| `baselines_all.py` | Run all other baselines |
| `factor_models_cv.py` | Cross-validate factor model K and lambda |
| `factor_models_val.py` | Train factor models for validation |
| `factor_models_final.py` | Train factor models for final test |
| `missingness_data_generator.py` | Generate M1 with simulated missingness |
| `cleaning_results.py` | Post-hoc select best baseline settings |
| `faq_val_analyzer.py` | Analyze validation results → `best_settings.csv` |
| `faq_coverage_analysis.py` | Per-date/accuracy coverage audit |
| `Main Text Figures.ipynb` | Generate paper figures |
| `Appendix B Figures.ipynb` / `Appendix D Figures.ipynb` | Appendix figures |

### New extension files (untracked, created during our sessions)
These extend the paper to higher budgets (25%-50%), additional experiments, and the WOR extension.

#### WOR (Without-Replacement) Extension — NEW

New PAI estimator (equation 2) that samples WITHOUT replacement. Full-stack reproduction of Figure 1 (ESS + coverage, fully-observed only, both datasets, budgets 2.5%-25%).

| File | Purpose | Notes |
|------|---------|-------|
| `wor_trial.py` | Core WOR-PAI trial functions: `trial_faq_wor`, `trial_ablation_wor` | Import from other WOR scripts |
| `wor_faq_val.py` | Tune FAQ hyperparams on M1 val (WOR) | `python wor_faq_val.py <dataset_idx> <gamma_idx>` |
| `wor_faq_final.py` | Run WOR-FAQ on M2 | `python wor_faq_final.py <seed_chunk 0-2>` |
| `wor_ablation.py` | Ablation (Z&C scoring + factor model + WOR PAI) — curiosity | `python wor_ablation.py <dataset_idx> <seed_chunk>` |
| `wor_baselines.py` | Baselines (unif/sqrt/2min × taus, static PHATS) — curiosity | `python wor_baselines.py <dataset_idx> <seed_chunk>` |
| `wor_val_analyzer.py` | Extract best FAQ settings → `logs/val/wor_best_settings.csv` | Run after validation |
| `wor_cleaning_results.py` | Combine WOR-FAQ with ORIGINAL baseline results; process curiosity results if present | Run after FAQ final |
| `wor_plot_figure.py` | Produce `figures/wor_ess+coverage_fully-observed.pdf` | Run after cleaning |
| `submit_wor_faq_val.sh` | 10 SLURM jobs (2 datasets × 5 gammas) | |
| `submit_wor_faq_final.sh` | 3 SLURM jobs (seed chunks) | Needs `wor_best_settings.csv` |
| `submit_wor_ablation.sh` | 6 SLURM jobs (2 datasets × 3 seed chunks) | Curiosity, independent |
| `submit_wor_baselines.sh` | 6 SLURM jobs (2 datasets × 3 seed chunks) | Curiosity, independent |

**Comparison strategy**: WOR FAQ vs **original (with-replacement) baselines** — we only improved our method, not theirs. ESS multiplier computed vs original uniform. WOR baselines/ablation are curiosity-only.

**Pipeline**:
```bash
# Step 1 — all in parallel:
sbatch submit_wor_faq_val.sh      # FAQ validation (needed)
sbatch submit_wor_baselines.sh    # curiosity, independent
sbatch submit_wor_ablation.sh     # curiosity, independent

# After validation:
python wor_val_analyzer.py        # → logs/val/wor_best_settings.csv
sbatch submit_wor_faq_final.sh

# After FAQ final:
python wor_cleaning_results.py    # uses EXISTING original baseline CSVs
python wor_plot_figure.py
```

**Status**: Validation + curiosity jobs pushed to cluster and results returned locally ✅. Next: `python wor_val_analyzer.py` then `sbatch submit_wor_faq_final.sh`.

#### High-budget / other extensions

| File | Purpose | Notes |
|------|---------|-------|
| **Experiment scripts** | | |
| `faq_val.py` + `submit_faq_val.sh` | Extended: also tunes at high budgets (25%-50%), all ms | Produces `logs/val/faq_val_hb_*.csv` |
| `faq_final_high_budget.py` + `submit_faq_final_high_budget.sh` | FAQ final at high budgets, fully-obs only | Produces `faq_final_high_budget_sl=*.csv` |
| `faq_final_high_budget_mmlu_fix.py` | One-off fix for missing mmlu-pro budgets 0.45-0.5 | Appends to same files |
| `faq_final_ci_saved.py` + `submit_ci_widths.sh` | FAQ final saving per-model CI widths | Produces `logs/ci_widths/*.npy` |
| `faq_tau075.py` + `submit_tau075.sh` | FAQ with tau=0.75 fixed, tune other params | Produces `logs/final/faq_tau075_*.csv` |
| `submit_ablation.sh` | Ablation at high budgets | Produces `active_inference_ablation_dataset=*.csv` |
| **Analysis scripts** | | |
| `analyze_faq_val_high_budget.py` | Task 1: Find best FAQ settings at high budgets | Produces `best_settings_high_budget.csv` |
| `analyze_ablation_posthoc.py` | Task 2: Post-hoc best ablation, all missingness | Produces `ablation_posthoc_best.csv` + 3 plots |
| `analyze_figure5.py` | Task 3: Full Figure 5 (2.5%-50%), all missingness | Produces 2 ratio plots |
| `analyze_ci_widths.py` | Task 4: CI width distribution, outlier model analysis | Produces histograms + metadata plots |
| `analyze_tau075.py` | Tau=0.75 experiment analysis | Produces ratio/width/coverage plots |

### Why there are "redundant" files
The original scripts (`faq_val.py`, `faq_final.py`) use `BUDGET_PROPS = np.linspace(0, 0.25, 11)[1:]` (low budgets only). Rather than modifying them (to preserve paper reproducibility), we created new scripts for high budgets and additional experiments. Each `analyze_*.py` then stitches together low-budget (original) + high-budget (new) results.

## Pipeline: How to Run Everything From Scratch

### Prerequisites
1. **GPU required** — PyTorch + CUDA
2. **Unzip data**: `cd data/processed && unzip mmlu-pro.zip -d mmlu-pro/ && unzip bbh+gpqa+ifeval+math+musr.zip -d bbh+gpqa+ifeval+math+musr/`
3. **Factor models** must exist in `factor_models/final/` (already in repo)
4. **Python env**: torch, numpy, pandas, scipy, tqdm, matplotlib

### Step-by-step (on the Marlowe cluster)

```bash
# 1. Validate FAQ hyperparameters at high budgets (25%-50%)
#    Reads: factor_models/val/, data/processed/, logs/val/best_settings.csv
#    Writes: logs/val/faq_val_hb_dataset=*_gamma=*_ms=*.csv
sbatch submit_faq_val.sh
# Then analyze:
python analyze_faq_val_high_budget.py
# Produces: logs/val/best_settings_high_budget.csv

# 2. Run ablation baseline at high budgets
#    Reads: factor_models/final/, data/processed/
#    Writes: logs/final/active_inference_ablation_dataset=*.csv
sbatch submit_ablation.sh
# Then analyze:
python analyze_ablation_posthoc.py

# 3. Run FAQ final at high budgets (fully-observed)
#    Reads: best_settings_high_budget.csv, factor_models/final/, data/processed/
#    Writes: logs/final/faq_final_high_budget_sl=*.csv
sbatch submit_faq_final_high_budget.sh
# Then analyze:
python analyze_figure5.py

# 4. CI width analysis (fully-observed, low budgets)
#    Reads: logs/val/best_settings.csv, factor_models/final/, data/processed/
#    Writes: logs/ci_widths/*.npy
sbatch submit_ci_widths.sh
# Then analyze:
python analyze_ci_widths.py

# 5. Tau=0.75 experiment
#    Reads: factor_models/val/, data/processed/
#    Writes: logs/final/faq_tau075_*.csv, logs/val/best_settings_tau075_*.csv
sbatch submit_tau075.sh
# Then analyze:
python analyze_tau075.py
```

### Running analysis scripts
Analysis scripts (`analyze_*.py`) can run on the cluster or locally — they only need CSV/NPY files from `logs/`, no GPU. If running locally, `scp` the log files first:
```bash
scp -r maechler@login.marlowe.stanford.edu:~/efficiently-evaluating-llms/logs/final/ logs/final/
scp -r maechler@login.marlowe.stanford.edu:~/efficiently-evaluating-llms/logs/val/ logs/val/
scp -r maechler@login.marlowe.stanford.edu:~/efficiently-evaluating-llms/logs/ci_widths/ logs/ci_widths/
```

## Cluster Reference

See `marlowe_cluster.md` for full Marlowe SLURM details (SSH, conda, sbatch template, common pitfalls).

Quick reference:
```bash
ssh maechler@login.marlowe.stanford.edu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env
cd ~/efficiently-evaluating-llms
```

## Key Design Patterns in the Codebase

1. **Seed parallelization**: Scripts split 100 seeds into chunks (sl=0: seeds 0-32, sl=1: 33-65, sl=2: 66-99) via `sys.argv[1]`. SLURM array jobs run chunks in parallel.

2. **Checkpointing**: Scripts count existing CSV rows and skip completed trials on restart. Watch out for counter-based checkpointing when appending new budget values — the counter may think the job is already done.

3. **Budget rounding**: Always use `np.round(..., decimals=3)` when comparing budget values. `np.linspace` produces float imprecision that breaks `isin()` and merge operations.

4. **Post-hoc selection**: For the ablation baseline, we pick the best `tau` per (dataset, missingness, budget) AFTER seeing test results. This is intentionally favorable to the baseline — if FAQ still wins, it's a strong result.

5. **CSV output columns**: Original paper files have `dataset, n_full_obs, mcar_obs_prob, prop_budget, seed, mean_width, coverage`. Some early high-budget files omit missingness columns (fully-observed only).

## Adapting to New Datasets

To apply this pipeline to a different dataset:

1. **Prepare data**: Create `M1.csv` (historical models × questions, binary 0/1 correctness) and `M2.csv` (test models × questions). Include `created_date` column in M2 if you want temporal analysis.

2. **Simulate missingness**: Adapt `missingness_data_generator.py` to produce `M1_nfobs=*_p=*.csv` variants.

3. **Train factor models**: Run `factor_models_cv.py` → `factor_models_val.py` → `factor_models_final.py` with your dataset name.

4. **Update dataset constants**: Add your dataset to `data/data_helpers.py` and to the `DATASETS` list in experiment scripts.

5. **Run the pipeline**: Follow the steps above, substituting your dataset name.

The key files to modify are the `DATASETS` list and the data loading paths. The core algorithm code (FAQ scoring, AIPW estimation, CI construction) is self-contained within `faq_val.py` and `faq_final.py`.

---

## ACS Extension: Continuous Regression

We extended FAQ to a **continuous regression** task using the ACS Census 2019 California PUMS dataset.
See `acs_study/current_work_acs.md` for full design details and `acs_study/faq_findings.md` for findings.

### Goal
Reproduce Figure 2 of the paper (ESS + coverage vs. budget) but for ACS instead of MMLU-Pro.
Estimand: mean of z-scored PINCP (personal income) over the unlabeled split.

### Key Design Choices vs. Paper

| Paper | ACS adaptation |
|-------|---------------|
| Binary matrix factorization (U,V) | Truncated SVD (top-D singular vectors) of one-hot encoded features → factor matrix V |
| Factor model for LLM responses (binary) | BLR (Bayesian linear regression) on D-dim SVD factors for continuous income |
| h_o ∝ √(p(1-p)) | h_o ∝ √(σ² + v_j^T Σ v_j) (predictive std) |
| h_a via Fisher info with w=p(1-p) | h_a via Fisher info with w=1/σ² (Gaussian noise) |
| Sherman-Morrison update with w=p(1-p) | Sherman-Morrison update with w=1/σ² |
| M1/M2 train-test split | 50/50 labeled/unlabeled random split |

### Key Finding: FAQ = uniform+PAI on ACS (investigation complete)

**FAQ provides no benefit over uniform+PAI on ACS.** This is a genuine result, not a bug.

**Root cause**: ACS income residuals are approximately homoscedastic. FAQ's h_o ∝ √(σ² + v_j^T Σ v_j)
collapses to √σ² ≈ constant for all j. Binary outcomes (paper) provide intrinsic heterogeneity via
p_j(1−p_j); continuous outcomes require heteroscedastic residuals, which ACS income lacks.

Four hypotheses were systematically eliminated:
1. Posterior collapse (190k labels) → tested low-label regime → still equal
2. Weak model (R²=0.22) → tested D=128, XGBoost (R²≈0.5) → still equal
3. Coupled prediction/uncertainty → XGBoost+BLR hybrid → still equal
4. **Root cause**: homoscedastic continuous data — confirmed via vtSigmav/σ² diagnostic

AIPW (uniform+PAI) still beats classical: ~1.3× with BLR, ~1.6× with XGBoost. ESS ceiling = 1/(1-R²).

### Files

| File | Purpose |
|------|---------|
| `acs_study/run_faq.py` | BLR-based FAQ, `--estimators`, `--D`, `--n_labeled` configurable |
| `acs_study/run_faq_xgb.py` | Hybrid XGBoost+BLR experiment |
| `acs_study/plot_paper.py` | Paper-style Figure 2 (handles old/new CSV schemas) |
| `acs_study/diag_posterior.py` | Prints vtSigmav/σ² ratio vs n_labeled |
| `acs_study/faq_findings.md` | Full investigation: when FAQ works/fails, all hypotheses, next steps |
| `acs_study/current_work_acs.md` | File list, design, current status |

### Current Status
**Paused.** Investigation complete and documented.
- `submit_faq_acs_nlab50.sh` submitted (n_labeled=50, max possible signal). Results pending.
- When returning: pull results, check if FAQ wins at n_labeled=50 (ratio≈0.48).
- If not, find a heteroscedastic dataset (see `faq_findings.md` for criteria).
