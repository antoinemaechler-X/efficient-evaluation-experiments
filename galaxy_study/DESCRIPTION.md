# Galaxy Zoo 2 Study — Detailed Description

## 1. Context and Motivation

This study applies our WOR-AIPW (Without-Replacement Augmented Inverse Propensity Weighting) method to a new dataset: **Galaxy Zoo 2**, a large-scale citizen science project that classified galaxy morphologies from Sloan Digital Sky Survey images. The goal is to demonstrate that our method generalizes beyond the AlphaFold proteomics setting and to benchmark it against **Cross-Prediction-Powered Inference (Cross-PPI)**, a recent method by Zrnic & Candès (2024, PNAS).

The setup mirrors the existing `alphafold_study/` as closely as possible in structure, code patterns, and output format.

## 2. The Dataset

### Source
The data comes from the **`ppi_py` Python package** (by Angelopoulos et al.), which provides a pre-processed version of the Galaxy Zoo 2 dataset as a downloadable `.npz` file (Google Drive ID: `1pDLQesPhbH5fSZW1m4aWC-wnJWnp1rGV`). This is the same data ecosystem used by both the original PPI paper and the Cross-PPI paper.

### Contents
- **N = 16,743 galaxies**
- **Y** ∈ {0, 1}: binary label indicating whether the galaxy is **spiral** (1) or not (0). Derived from Galaxy Zoo 2 volunteer vote fractions (thresholded).
- **Ŷ** ∈ [0, 1]: predicted probability of being spiral, from a pre-trained model (likely a fine-tuned ResNet50, as used in the Cross-PPI paper).

### Key statistics
| | Value |
|---|---|
| Total galaxies | 16,743 |
| Spiral (Y=1) | 4,341 (25.93%) |
| Not spiral (Y=0) | 12,402 (74.07%) |
| True spiral fraction θ* | 0.2593 |
| Mean prediction Ŷ | 0.2603 |

### Comparison with AlphaFold dataset
| | AlphaFold | Galaxy Zoo 2 |
|---|---|---|
| N | 10,802 | 16,743 |
| Y type | Binary (disordered) | Binary (spiral) |
| Ŷ source | AlphaFold model | ResNet50 (ppi_py) |
| Groups | 2 (phosphorylated / not) | 1 (all galaxies) |
| Estimand | Group mean disorder rate | Overall spiral fraction |

### Why this dataset is a good fit
1. **Binary outcome** — matches our AIPW framework perfectly (binary Y gives natural heterogeneity via p(1−p) for active scoring).
2. **Good but imperfect predictions** — Ŷ has mean 0.2603 vs true 0.2593, so there's a small bias to correct. The prediction quality determines how much efficiency gain AIPW methods achieve over classical.
3. **Established benchmark** — Galaxy Zoo 2 is used in both PPI and Cross-PPI papers, making our comparison credible.

## 3. The Estimand

We estimate **θ* = E[Y] = fraction of spiral galaxies** in the population. This is a simple mean estimation problem. In each trial, we observe labels Y for only a budget-constrained subset and must construct a confidence interval for θ*.

### No groups
Unlike the AlphaFold study which stratified by phosphorylation status (2 groups), the Galaxy study has **no groups** — we estimate a single population mean. This matches the Cross-PPI paper's setup, which also estimates the overall spiral fraction without stratification.

## 4. Methods Compared (6 total)

### Our methods (WOR-AIPW)
These use **sequential sampling without replacement**, observing one item at a time:

1. **WOR-Active**: Samples items using uncertainty-based scores `s_j = min(Ŷ_j, 1−Ŷ_j)`, smoothed with τ=0.5. Items with predictions near 0.5 (most uncertain) are sampled more frequently. Uses the AIPW estimator (Eq. 2 of our paper) with Theorem 4 variance.

2. **WOR-Uniform**: Same WOR framework but samples uniformly at random (no active scoring). This isolates the value of the WOR+AIPW machinery from the active scoring.

### Bernoulli baselines (from AlphaFold study)
These use **independent Bernoulli sampling** — each item is included with probability π_i:

3. **Bernoulli-Active**: π_i = (1−τ)·N·b·(s_i/Σs) + τ·b, where b is the budget proportion and s_i = min(Ŷ_i, 1−Ŷ_i). Uses AIPW with Ŷ as imputation.

4. **Bernoulli-Uniform**: π_i = b (constant). Uses AIPW with Ŷ as imputation. This is essentially equivalent to the PPI estimator.

5. **Classical**: π_i = b (constant). Uses f=0 (no imputation) — just the sample mean of labeled items. This is the naive baseline.

### Cross-PPI baseline (new for this study)
6. **Cross-PPI** (Zrnic & Candès, 2024): For each trial, randomly selects n = b·N items as "labeled," then computes:
   - Point estimate: θ̂ = mean(Ŷ_unlabeled) + mean(Y_labeled − Ŷ_labeled)
   - Variance: V̂ = var(Ŷ_unlabeled)/N_unlabeled + var(Y_labeled − Ŷ_labeled)/n_labeled
   - CI: θ̂ ± z_{0.975} · √V̂

   This is the estimator from Eq. (7) of the Cross-PPI paper. In the original paper they use cross-fitting (train K=3 models on folds, predict with out-of-fold model), but since we use fixed pre-computed predictions Ŷ from ppi_py, the cross-fitting step is implicit — the predictions were already generated independently of the labels.

## 5. Design Choices and Rationale

### 5.1 Using fixed predictions (no factor model)
**Choice**: Use the pre-computed Ŷ from ppi_py as-is, rather than training our own model.

**Rationale**: This is the same approach as the AlphaFold study, where we used AlphaFold's pre-computed disorder predictions. The key point is that our method's value comes from the *sampling and estimation machinery*, not from the prediction model. Using fixed, high-quality predictions isolates this contribution. Training a custom model would introduce confounding factors (model quality, hyperparameter choices) that would obscure the comparison.

### 5.2 No groups
**Choice**: Estimate a single population mean (no stratification).

**Rationale**: The Cross-PPI paper estimates the overall spiral fraction without groups, so we match their setup for a fair comparison. The AlphaFold study used groups because the underlying biology (phosphorylated vs non-phosphorylated proteins) provided a natural and scientifically meaningful stratification. Galaxy Zoo 2 doesn't have an analogously natural binary covariate in the ppi_py data.

### 5.3 Cross-PPI as separate script
**Choice**: Implement Cross-PPI in its own script (`run_cross_ppi.py`) rather than folding it into `run_bernoulli.py`.

**Rationale**: The Cross-PPI estimator uses a fundamentally different variance formula (separating unlabeled prediction variance from labeled residual variance) compared to the Bernoulli-AIPW estimator (which uses sample variance of AIPW pseudo-outcomes). Even though the point estimates are closely related, the CI widths differ. A separate script keeps the comparison clean and the code auditable.

### 5.4 Budget range and parameters
All parameters match the AlphaFold study for consistency:
- **Budget proportions**: 1% to 20% in 20 linearly-spaced steps
- **α = 0.05** (95% confidence intervals)
- **τ = 0.5** (mixing parameter for active scoring — balances exploration vs exploitation)
- **1,000 seeds** (split into 3 chunks of 333/333/334 for parallel SLURM jobs)

### 5.5 Active scoring function
**Choice**: `s_j = min(Ŷ_j, 1−Ŷ_j)` — highest score for items near the decision boundary (Ŷ ≈ 0.5).

**Rationale**: Same as AlphaFold. For binary outcomes, items near the boundary have the highest conditional variance p(1−p) and contribute most to estimation uncertainty. Querying these items first yields the largest variance reduction per label. The τ-smoothing ensures every item has nonzero sampling probability (exploration guarantee).

## 6. Experimental Pipeline

### Step-by-step workflow
```
1. download_data.py    → downloads galaxies.npz from ppi_py, saves Y.npy + Yhat.npy
2. submit.sh           → launches 9 SLURM jobs (3 scripts × 3 seed chunks)
   ├── run_wor.py 0,1,2        → galaxy_study/logs/wor_sl={0,1,2}.csv
   ├── run_bernoulli.py 0,1,2  → galaxy_study/logs/bernoulli_sl={0,1,2}.csv
   └── run_cross_ppi.py 0,1,2  → galaxy_study/logs/cross_ppi_sl={0,1,2}.csv
3. analyze_results.py  → merges CSVs, computes ESS multipliers → logs/cleaned/summary.csv
4. plot_results.py     → generates figures → figures/galaxy_main.pdf, etc.
```

### Output CSV format (same as AlphaFold)
Each row: `group, prop_budget, method, seed, width, coverage`

### ESS multiplier computation
For each (budget, seed), we compute:
- **ESS multiplier vs Classical** = (width_classical / width_method)²
  - Interpretation: "how many classical samples is one active sample worth?"
  - ESS multiplier of 2.0 means the method achieves the same precision with half the labels

### Figures produced
1. **galaxy_main.pdf** — Main comparison: WOR-Active vs Cross-PPI vs Classical (ESS + coverage)
2. **galaxy_main_sparse.pdf** — Same, with fewer budget points for cleaner presentation
3. **galaxy_ess+coverage_vs_classical.pdf** — All 6 methods, ESS relative to Classical
4. **galaxy_ess+coverage_vs_wor_uniform.pdf** — All 6 methods, ESS relative to WOR-Uniform

## 7. Expected Results

From the smoke test (budget = 10%, N = 16,743):

| Method | CI Width | Relative to Classical |
|---|---|---|
| WOR-Active | 0.024 | 1.76× narrower |
| Cross-PPI | 0.029 | 1.46× narrower |
| Classical | 0.042 | 1.00× (baseline) |

**Interpretation**: WOR-Active achieves ~3.1× ESS multiplier vs Classical (1.76² ≈ 3.1), while Cross-PPI achieves ~2.1× (1.46² ≈ 2.1). The gap comes from two sources:
1. **Active scoring** — WOR-Active preferentially queries uncertain items, while Cross-PPI samples uniformly
2. **WOR variance reduction** — sampling without replacement reduces variance compared to independent sampling

## 8. Relationship to the Cross-PPI Paper

The original Cross-PPI paper (Zrnic & Candès, 2024) compared Cross-PPI against Classical and PPI on Galaxy Zoo 2 with N=167,434 galaxies and n ∈ {10K, 20K, 30K}. They trained a ResNet50 from scratch using cross-fitting (K=3 folds).

Our study differs in:
1. **Smaller subset** (16,743 vs 167,434) — we use the ppi_py version, which is a curated subset with pre-computed predictions
2. **Fixed predictions** — we don't retrain models; predictions are given
3. **Additional methods** — we add WOR-Active, WOR-Uniform, and Bernoulli variants
4. **Different budget framing** — we vary budget as a proportion (1–20%) rather than fixed n

The core comparison (our WOR-Active vs their Cross-PPI vs Classical) is valid because all methods operate on the same (Y, Ŷ) data and estimate the same quantity.
