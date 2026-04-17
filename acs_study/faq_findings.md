# FAQ Algorithm — Empirical Findings & Insights

## Overview

We applied FAQ (Factorized Active Querying) to the ACS Census 2019 California
dataset as a continuous regression analog of the paper's LLM evaluation setting.
This document records what we learned about when FAQ works, when it fails, and why.

---

## What FAQ Needs to Win

FAQ improves over uniform+PAI (AIPW with uniform sampling) only when **two
conditions hold simultaneously**:

### Condition 1 — Heteroscedastic residuals
The prediction error |y_j − ŷ_j| must vary substantially across items.
Some items must be much harder to predict than others. Formally, h_o ∝
√(σ² + v_j^T Σ v_j) must be non-uniform across j.

- In LLM evaluation: question difficulty p_j varies from ~0 to ~1. Items
  with p_j ≈ 0.5 are high-variance (worth sampling), items with p_j ≈ 0 or
  p_j ≈ 1 are low-variance (not worth sampling). This creates a strong,
  systematic non-uniformity in h_o.
- In ACS income: all individuals have similar residual variance (roughly
  Gaussian, homoscedastic). h_o ≈ uniform regardless of model or D.

### Condition 2 — Non-trivial informational structure for the estimand
The target quantity θ must weight items unequally, or some items must be
more informative about θ than others. This makes h_a non-uniform.

- For mean(y) over all items: every item contributes equally → h_a ≈ uniform.
- For a subgroup mean, a regression coefficient, or a conditional expectation:
  some items matter more → h_a is non-trivial.

**If either condition fails, FAQ collapses to uniform sampling (= uniform+PAI).**

---

## Posterior Collapse: Verified Diagnosis

We added a diagnostic that measures `mean(v_j^T Σ v_j) / σ²` at the first sampling
step. This ratio must be ~1 for FAQ's h_o to be non-uniform. When it is <<1,
`predictive_var ≈ σ²` (constant across all j) and h_o collapses to uniform.

Running `diag_posterior.py` on the full ACS dataset produced:

| n_labeled | ratio  | sigma2 |
|----------:|-------:|-------:|
|        50 | 0.4812 | 0.8864 |
|       100 | 0.1906 | 0.6748 |
|       200 | 0.0828 | 0.4179 |
|       500 | 0.0321 | 0.7378 |
|     2,000 | 0.0079 | 0.8604 |
|   190,045 | 0.0001 | 0.7776 |

Key findings from this table:

1. **The ratio is purely a function of n_labeled** — it does not depend on V scaling.
   Normalising V to unit-variance columns makes zero difference (mathematically,
   scaling V by a constant cancels exactly in v^T Σ v / σ²).

2. **Only n_labeled=50 gives meaningful signal** (ratio~0.48). All our previous
   low-label experiments used n_labeled ≥ 200 (ratio ≤ 0.08) which is still largely
   collapsed.

3. **sigma2 is unreliable at very small n_labeled** — with only 50 points and D=16
   parameters, the OLS residual variance jumps erratically (0.89 → 0.67 → 0.42
   → 0.74 as n goes 50→100→200→500). The BLR is not well-calibrated in this regime.

4. A job with n_labeled=50 has been submitted (`submit_faq_acs_nlab50.sh`) to check
   whether ratio~0.48 actually translates to a FAQ advantage in practice.

---

## The ACS Experiment: A Systematic Elimination

We ran four experiments to isolate the root cause of FAQ's failure on ACS.

### Experiment 1 — Baseline BLR (D=16, full labels, n=190k)

**Setup**: BLR on top-16 SVD factors of one-hot encoded features.
Fitted on 190k labeled points.

**Results**:
- R² = 0.22, σ² = 0.78
- FAQ = uniform+PAI (identical ESS curves)
- uniform+PAI beats classical (ESS ~1.29×) — AIPW helps

**Hypothesis from this**: the BLR posterior Σ_post ≈ 0 after 190k points
(posterior collapse). h_o and h_a both become uniform because Σ v̄ ≈ 0.

---

### Experiment 2 — Low Label Regime (n_labeled ∈ {200, 500, 2000})

**Setup**: Same BLR, but fit on only 200–2000 points → genuine posterior
uncertainty → Σ is far from zero → h_o and h_a are non-trivial.

**Results**:
- FAQ still = uniform+PAI at all label counts tested
- Posterior collapse was NOT the root cause

**Lesson**: even with full posterior uncertainty, if the underlying data is
homoscedastic, the non-uniform sampling distribution that FAQ computes does
not reduce AIPW variance. The sampling scores are non-trivial but not
*useful* — they point toward SVD-high-norm individuals, not high-residual
individuals.

---

### Experiment 3 — Higher Dimensionality (D=128)

**Setup**: Increase SVD factors from D=16 to D=128. More expressive model,
better R².

**Results**:
- R² improves (fill in exact number from your run)
- ESS gap between classical and uniform+PAI grows (better predictions → better AIPW)
- FAQ still = uniform+PAI

**Lesson**: model quality (R²) determines the *level* of AIPW gain (ESS
ceiling = 1/(1−R²)), but does not create heteroscedasticity. Higher R² lifts
all estimators equally. FAQ needs heterogeneous residuals, not just smaller
average residuals.

---

### Experiment 4 — Hybrid XGBoost + BLR

**Setup**: XGBoost (R²~0.45) for AIPW predictions (full 190k training).
BLR on residuals (n_labeled_blr ∈ {200, 1000, 5000}) for sampling scores.
Decouples prediction quality from posterior uncertainty.

**Results**:
- uniform+PAI beats classical more strongly (bigger ESS gap — XGBoost helps)
- FAQ still = uniform+PAI regardless of n_labeled_blr

**Lesson**: decoupling prediction quality from sampling guidance does not help
if the residuals after XGBoost are still homoscedastic. XGBoost explains
~45% of income variance, leaving ~55% as essentially structureless noise.
The remaining residuals have no systematic pattern by individual that FAQ
could exploit.

---

## The Root Cause: Structural Mismatch

| Property | LLM evaluation (FAQ works) | ACS income (FAQ fails) |
|---|---|---|
| Per-item difficulty | Bimodal: easy (p≈0/1) and hard (p≈0.5) questions | All individuals similarly predictable |
| Residual distribution | Highly heterogeneous across items | Approximately homoscedastic Gaussian |
| h_o distribution | Strongly non-uniform | Uniform |
| Estimand structure | Per-model accuracy (each model has unique factor U_i) | Population mean (all items equal weight) |
| Active sampling signal | Strong: avoid easy/hard questions, focus on discriminative ones | None: no item is more informative than another |
| ESS multiplier (paper) | 2–5× | ~1.0× (FAQ vs uniform+PAI) |
| AIPW gain (vs classical) | Strong | Moderate (~1.3× with BLR, ~1.6× with XGBoost) |

The paper's setting is special: LLM evaluation matrices have a **U-shaped
distribution of per-question difficulty** (many easy questions, many hard
questions, fewer in the middle). This is structurally analogous to a
bimodal variance distribution, which is exactly what FAQ's h_o exploits.
Income prediction from demographics does not have this structure.

---

## ESS Ceiling Formula

A useful diagnostic before running FAQ on any new dataset:

```
ESS ceiling (uniform+PAI vs classical) = 1 / (1 − R²)
```

This is the maximum ESS multiplier achievable by AIPW with a perfect model.
With R²=0.22 → ceiling = 1.28×. With R²=0.45 → ceiling = 1.82×.

FAQ can theoretically exceed this ceiling only if it samples
non-uniformly and the sampling concentrates on high-residual items. In
practice it can get close to the ceiling when heteroscedasticity is strong.

---

## Conditions for Choosing a Dataset Where FAQ Will Work

Based on our experiments, a dataset is likely to show FAQ's advantage if:

1. **Per-item difficulty is heterogeneous and systematic**: some items
   are near-certain outcomes (low variance), others are genuinely uncertain
   (high variance). The heterogeneity should be predictable from covariates
   (not pure noise).

2. **Residual distribution is heavy-tailed or multimodal**, not Gaussian.
   A U-shaped or skewed distribution of |y_j − ŷ_j| is ideal.

3. **The estimand weights items unequally**: subgroup means, regression
   coefficients, or conditional expectations are better than global means.

4. **Model R² is in the range 0.2–0.7**: too low → AIPW corrections are
   noisy for everyone; too high → uniform sampling is already near-optimal.

5. **Structural analog to LLM evaluation**: any setting where "some items
   are universally easy, some universally hard" in a continuous sense.
   Examples: student performance data (some questions all students get
   right/wrong), medical tests (some cases are near-certain diagnoses),
   crowdsourced annotation (some items have high inter-annotator agreement,
   others are ambiguous).

---

## What We Could Try Next on ACS

If we still want to demonstrate FAQ on ACS specifically:

1. **Different estimand**: estimate an OLS regression coefficient (e.g., β
   for education → income) instead of the global mean. Some individuals
   contribute more to the coefficient estimate than others (those with
   unusual education levels, high leverage points).

2. **Subgroup mean**: estimate mean income for a small demographic subgroup
   (e.g., a specific occupation × race × sex intersection). Now in-group
   individuals are much more informative than out-group ones — h_a becomes
   strongly non-uniform.

3. **Income variance (not mean)**: estimate Var(income) or the Gini
   coefficient. High-income individuals contribute more to variance →
   heterogeneous item weights → non-trivial h_a.

4. **Heteroscedastic subgroup stratification**: split by occupation. Within
   professions like finance or medicine, income variance is huge; within
   retail or food service, it is small. Run FAQ within the high-variance
   subgroup.

5. **Truly different dataset**: see the ChatGPT Deep Research prompt in
   this directory for a detailed specification of what to look for.

---

## Key Code Files

| File | What it tests |
|---|---|
| `run_faq.py` | BLR-based FAQ, all estimators, configurable D and n_labeled |
| `run_faq_xgb.py` | Hybrid XGBoost + BLR, decoupled prediction / uncertainty |
| `plot_blr_perf.py` | BLR model diagnostics: R², residuals, learning curve |
| `plot_paper.py` | Paper-style Figure 2 (ESS + coverage) |
| `submit_faq_acs_clean.sh` | Single clean run, all estimators, D=16 |
| `submit_faq_acs_D128.sh` | D=128 experiment across label regimes |
| `submit_faq_acs_lowlabel.sh` | Low-label BLR experiment |
| `submit_faq_acs_xgb.sh` | XGBoost+BLR hybrid experiment |
