# Pew ATP Wave 79 Study — Detailed Description

## 1. Context and Motivation

This study applies our WOR-AIPW (Without-Replacement Augmented Inverse Propensity Weighting) method to the **Pew Research Center American Trends Panel (ATP) Wave 79** survey data. The goal is to benchmark WOR-AIPW against **Active Inference** (Zrnic & Candes, 2024), the Bernoulli-based AIPW method with adaptive tau tuning, reproducing results from https://github.com/tijana-zrnic/active-inference.

The setup mirrors the existing `galaxy_study/` in structure, code patterns, and output format.

## 2. The Dataset

### Source
The data comes from the **Pew Research Center ATP Wave 79** survey (November 2020, post-election). The raw SPSS file (`ATPW79.sav`) must be obtained from Pew Research Center (requires free registration at https://www.pewresearch.org/datasets/).

### Contents
- **N ~ 5,000–6,000 respondents** (after filtering missing values)
- **Y** in {0, 1}: binary label indicating whether the respondent **approves of Biden's post-election messaging** (binarized from Likert scale: values 1-2 = approve).
- **Yhat** in [0, 1]: predicted approval probability from XGBoost trained on 50% of data (10 survey covariates).

### Feature variables (10 covariates)
| Variable | Description |
|----------|-------------|
| `F_PARTYSUM_FINAL` | Party affiliation |
| `COVIDFOL_W79` | How closely following COVID news |
| `COVIDTHREAT_a_W79` | COVID threat to personal health |
| `COVIDTHREAT_b_W79` | COVID threat to US economy |
| `COVIDTHREAT_c_W79` | COVID threat to personal finances |
| `COVIDTHREAT_d_W79` | COVID threat to day-to-day life |
| `COVIDMASK1_W79` | Mask-wearing frequency |
| `COVID_SCI6E_W79` | Trust in scientists on COVID |
| `F_EDUCCAT` | Education category |
| `F_AGECAT` | Age category |

### Outcome variable
`ELECTBIDENMSSG_W79`: Approval of Biden's post-election messaging.
- Original values: 1 (strongly approve), 2 (somewhat approve), 3 (somewhat disapprove), 4 (strongly disapprove), 99 (refused)
- Binarized: Y = 1 if value < 2.5 (i.e., approve or strongly approve), Y = 0 otherwise
- Rows with value 99 are excluded

### Comparison with Galaxy Zoo 2 dataset
| | Galaxy Zoo 2 | Pew ATP Wave 79 |
|---|---|---|
| N (test set) | 16,743 | ~2,500–3,000 |
| Y type | Binary (spiral) | Binary (approve) |
| Yhat source | Pre-trained ResNet50 (ppi_py) | XGBoost (trained on 50% split) |
| Groups | 1 (all galaxies) | 1 (all respondents) |
| Estimand | Spiral fraction | Biden approval rate |
| Covariates | Image pixels (implicit) | 10 survey variables |

### Why this dataset is a good fit
1. **Binary outcome** — matches our AIPW framework (binary Y gives natural heterogeneity via p(1-p) for active scoring).
2. **Moderate prediction quality** — XGBoost on 10 covariates provides useful but imperfect predictions, showing the value of AIPW correction.
3. **Established benchmark** — Used in Zrnic & Candes (2024) Active Inference paper, making our comparison directly credible.
4. **Real survey data** — Demonstrates practical applicability to survey sampling.

## 3. The Estimand

We estimate **theta* = E[Y] = approval rate of Biden's post-election messaging** in the test population. In each trial, we observe labels Y for only a budget-constrained subset and must construct a confidence interval for theta*.

### No groups
Like the Galaxy study, we estimate a single population mean without stratification.

## 4. Methods Compared (6 total)

### Our methods (WOR-AIPW)
These use **sequential sampling without replacement**, observing one item at a time:

1. **WOR-Active**: Samples items using uncertainty-based scores `s_j = min(Yhat_j, 1-Yhat_j)`, smoothed with tau=0.5. Uses the AIPW estimator (Eq. 2) with Theorem 4 variance.

2. **WOR-Uniform**: Same WOR framework but samples uniformly at random (no active scoring).

### Bernoulli baselines (our implementation)
These use **independent Bernoulli sampling** — each item is included with probability pi_i:

3. **Bernoulli-Active**: pi_i = (1-tau)*N*b*(s_i/sum(s)) + tau*b, with fixed tau=0.5. Uses AIPW.

4. **Bernoulli-Uniform**: pi_i = b (constant). Uses AIPW.

5. **Classical**: pi_i = b (constant). No imputation — just sample mean.

### Active Inference baseline (new for this study)
6. **Active Inference** (Zrnic & Candes, 2024): Bernoulli-AIPW with **adaptive tau tuning** per budget. For each budget proportion b:
   - Sampling probability: pi_i = (1-tau) * eta * s_i + tau * b, where eta = b / mean(s)
   - Tau is tuned per budget by minimizing empirical variance on held-out training data:
     `V(tau) = mean((Y_train2 - Yhat_train2)^2 / pi_train2(tau))`
   - Point estimate: theta_hat = (1/n) * sum(Yhat_i + (Y_i - Yhat_i) * xi_i / pi_i)
   - CI: theta_hat +/- z_{0.975} * std(increments) / sqrt(n)

## 5. Design Choices and Rationale

### 5.1 Train/test split for predictions
**Choice**: 50/50 train/test split, TWO separate XGBoost models.

**Rationale**: Matches Zrnic's experimental setup exactly:
- **Main model**: trained on FULL training data (50%), predicts on test set → `Yhat_test` (used by all methods)
- **Tuning model**: trained on 80% of training data, predicts on remaining 20% → `Yhat_train2` (used only for tau tuning in Active Inference)

This two-model approach matches Zrnic's notebook exactly (Cell 5 trains main model, Cell 6 trains tuning model).

### 5.2 XGBoost hyperparameters
**Choice**: eta=0.001, max_depth=5, 3000 rounds, objective=reg:logistic.

**Rationale**: Identical to Zrnic's notebook to ensure we reproduce their prediction quality.

### 5.3 theta_true definition
**Choice**: theta_true = mean(Y_all) — the mean across ALL data (before train/test split).

**Rationale**: Matches Zrnic exactly. Coverage is checked against this population parameter for all methods.

### 5.4 Active Inference with tau tuning
**Choice**: Implement Active Inference with per-budget tau tuning in a separate script (`run_active_inference.py`).

**Rationale**: The adaptive tau tuning is the key distinguishing feature of Zrnic's method. A separate script keeps the comparison clean and makes it clear what their method does vs. ours.

Tau tuning procedure (exactly as Zrnic):
- Grid: 100 values in linspace(0.01, 0.99, 100)
- For each tau: compute `V(tau) = mean((Y_train2 - Yhat_train2)^2 / pi(tau))`
- pi(tau) = clip((1-tau)*eta*uncertainty + tau*budget, 0, 1)
- eta = budget / mean(uncertainty_train2)
- Select tau minimizing V(tau)

### 5.5 Budget range and parameters
- **Budget proportions**: 1% to 20% in 20 linearly-spaced steps
- **alpha = 0.05** (95% confidence intervals)
- **tau = 0.5** (for our methods; Active Inference tunes tau per budget)
- **1,000 seeds** (split into 3 chunks of 333/333/334 for parallel SLURM jobs)

Note: Zrnic uses alpha=0.1 (90% CI) and budgets linspace(0.005, 0.2, 20). We use alpha=0.05 for consistency with our paper, and start at 1% budget.

### 5.6 Active scoring function
**Choice**: `s_j = min(Yhat_j, 1-Yhat_j)` — same as galaxy study.

**Rationale**: For binary outcomes, items near the decision boundary (Yhat ~ 0.5) have highest conditional variance. This is also the scoring used by Zrnic in Active Inference, making the comparison of tau-fixed vs tau-tuned clean.

## 6. Experimental Pipeline

### Step-by-step workflow
```
1. download_data.py    → loads ATPW79.sav, trains XGBoost, saves Y/Yhat .npy files
2. submit.sh           → launches 9 SLURM jobs (3 scripts x 3 seed chunks)
   ├── run_wor.py 0,1,2                → pew_study/logs/wor_sl={0,1,2}.csv
   ├── run_bernoulli.py 0,1,2          → pew_study/logs/bernoulli_sl={0,1,2}.csv
   └── run_active_inference.py 0,1,2   → pew_study/logs/active_inference_sl={0,1,2}.csv
3. analyze_results.py  → merges CSVs, computes ESS multipliers → logs/cleaned/summary.csv
4. plot_results.py     → generates figures → pew_study/figures/pew_main.pdf, etc.
```

### Output CSV format (same as galaxy study)
Each row: `group, prop_budget, method, seed, width, coverage`

### Figures produced
1. **pew_main.pdf** — Main: WOR-Active vs Active Inference vs Classical (ESS + coverage)
2. **pew_main_sparse.pdf** — Same with fewer budget points
3. **pew_ess+coverage_vs_classical.pdf** — All 6 methods, ESS relative to Classical
4. **pew_ess+coverage_vs_wor_uniform.pdf** — All 6 methods, ESS relative to WOR-Uniform

## 7. Relationship to the Active Inference Paper

The original Active Inference paper (Zrnic & Candes, 2024) compared Active vs Uniform vs Classical on Pew ATP Wave 79 with:
- alpha = 0.1 (90% CI)
- Budgets: linspace(0.005, 0.2, 20)
- 1000 trials
- XGBoost trained on 50% split

Our study differs in:
1. **alpha = 0.05** (95% CI) — consistent with our other studies
2. **Budget range** (1-20%) — slightly shifted from their 0.5-20%
3. **Additional methods** — we add WOR-Active, WOR-Uniform, and Bernoulli variants
4. **Fixed tau baseline** — our Bernoulli-Active uses tau=0.5 vs their tuned tau

The core comparison (our WOR-Active vs their Active Inference vs Classical) is valid because all methods operate on the same (Y, Yhat) test data and estimate the same quantity.
