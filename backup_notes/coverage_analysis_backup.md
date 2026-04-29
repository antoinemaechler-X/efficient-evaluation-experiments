# WOR PAI Coverage Analysis: Why CIs Are Anti-Conservative

## 1. Summary of Empirical Findings

Running the corrected Theorem 4 variance formula (replacing the old WR B̂_n with the correct WOR version) still produces **systematic undercoverage** that grows with budget:

| Dataset | Budget | WOR FAQ Coverage | WOR FAQ ESS | WR FAQ Coverage | WR FAQ ESS |
|---------|--------|:---------------:|:-----------:|:--------------:|:----------:|
| MMLU-Pro | 2.5% | 0.941 | 3.67× | 0.940 | 3.57× |
| MMLU-Pro | 10% | 0.943 | 5.48× | 0.944 | 4.89× |
| MMLU-Pro | 17.5% | 0.938 | 6.44× | 0.947 | 5.07× |
| MMLU-Pro | 25% | **0.917** | **7.31×** | 0.948 | 4.91× |
| BBH suite | 2.5% | 0.945 | 3.10× | 0.945 | 3.05× |
| BBH suite | 10% | 0.947 | 4.53× | 0.947 | 4.10× |
| BBH suite | 17.5% | 0.945 | 5.21× | 0.948 | 4.26× |
| BBH suite | 25% | **0.940** | **5.73×** | 0.950 | 4.13× |

Key patterns:
- **WR FAQ** holds ~95% coverage at all budgets; its coverage **improves** with budget.
- **WOR FAQ** starts near 95% at low budgets but degrades monotonically from ~10% budget onward.
- The undercoverage is strongly correlated with the ESS multiplier.
- **WOR ESS grows monotonically** (3.7→7.3× on MMLU-Pro); WR ESS **saturates and decreases** (3.6→5.1→4.9×).
- This is not a coding bug. It is a structural finite-sample phenomenon.

---

## 2. The Estimator (Equation 2)

$$\hat{\theta}_n := \frac{1}{n} \sum_{t=1}^{n} \phi_t, \qquad \phi_t := \underbrace{\frac{1}{N}\left(\sum_{i \in O_{t-1}} y_i + \sum_{i \notin O_{t-1}} \hat{f}^{(t-1)}(x_i)\right)}_{\psi_t \text{ (imputed mean)}} + \underbrace{\frac{1}{N} \cdot \frac{y_{I_t} - \hat{f}^{(t-1)}(x_{I_t})}{q_t(I_t)}}_{\text{AIPW correction}}$$

**Unbiasedness** (finite $n$): $\mathbb{E}[\hat{\theta}_n] = \theta$ exactly at every finite $n$.

---

## 3. The Variance Formula (Theorem 4)

$$\hat{\sigma}^2_n := \hat{A}_n - \hat{B}_n$$

$$\hat{A}_n = \frac{1}{nN^2} \sum_{t=1}^{n} \frac{\left(y_{I_t} - \hat{f}^{(t-1)}(x_{I_t})\right)^2}{q_t(I_t)^2}, \qquad \hat{B}_n = \frac{1}{nN^2} \sum_{t=2}^{n} \left(N\hat{\theta}_{t-1} - N\psi_t\right)^2$$

The CI is: $\hat{\theta}_n \pm z_{0.975} \cdot \hat{\sigma}_n / \sqrt{n}$.

The population quantities are:

$$A_n = \frac{1}{nN^2}\sum_{t=1}^{n} \sum_{i \notin O_{t-1}} \frac{(y_i - \hat{f}_i)^2}{q_t(i)}, \qquad B_n = \frac{1}{nN^2}\sum_{t=1}^{n} \left(\sum_{i \notin O_{t-1}} (y_i - \hat{f}_i)\right)^2$$

The true conditional variance satisfies $\sigma^2 = A_n - B_n$, and $\hat{\sigma}^2_n \xrightarrow{p} \sigma^2$.

---

## 4. Rigorous Analysis of Undercoverage

### 4.1 Three Components of Undercoverage

The fractional underestimation of $\sigma$ is $\varepsilon = 1 - \hat{\sigma}/\sigma_{\text{true}}$, where $\sigma_{\text{true}}$ is the actual standard deviation of $\hat{\theta}_n$. Coverage satisfies:

$$\text{coverage} \approx 2\Phi(z(1-\varepsilon)) - 1 \approx 0.95 - 2z\varphi(z)\varepsilon \approx 0.95 - 0.229\varepsilon$$

where $\varphi(z) = 0.0584$ is the standard normal PDF at $z = 1.96$.

We can back $\varepsilon$ out from observed coverage via $\varepsilon = 1 - \Phi^{-1}((1+\text{cov})/2) / z$:

| Budget | WOR $\varepsilon$ (MMLU) | WR $\varepsilon$ (MMLU) | Excess $\varepsilon$ | WOR $\varepsilon$ (BBH) | WR $\varepsilon$ (BBH) | Excess $\varepsilon$ |
|:------:|:-------:|:------:|:-------:|:-------:|:------:|:-------:|
| 2.5% | 0.037 | 0.038 | −0.001 | 0.022 | 0.021 | +0.001 |
| 10% | 0.028 | 0.023 | +0.005 | 0.014 | 0.011 | +0.003 |
| 17.5% | 0.048 | 0.013 | +0.035 | 0.021 | 0.007 | +0.014 |
| 25% | **0.116** | 0.010 | **+0.106** | **0.039** | 0.002 | **+0.037** |

Three sources contribute to $\varepsilon$:

**(A) Baseline CLT error**: Finite-sample deviation of $\sqrt{n}(\hat{\theta}_n - \theta)/\sigma$ from $\mathcal{N}(0,1)$. Affects both WOR and WR equally. Decreases as $O(1/\sqrt{n})$.

**(B) $\hat{A}_n$ noise**: $\hat{A}_n$ is unbiased for $A_n$ (martingale increments), so its noise contributes to the *variance* of $\hat{\sigma}$, not the bias. Verified to be a small effect (< 1% of coverage deficit at all budgets, via analysis of mean_width_serr from the data).

**(C) $\hat{B}_n$ positive bias**: $\mathbb{E}[\hat{B}_n] > B_n$ because $\hat{B}_n$ uses $\hat{\theta}_{t-1}$ (noisy estimate) where $B_n$ uses $\theta$ (true value). This causes $\hat{\sigma}^2 = \hat{A} - \hat{B}$ to systematically underestimate $\sigma^2 = A - B$. **This is unique to WOR** (WR FAQ uses $\hat{\sigma}^2 = \hat{A}$ with no $B$ correction).

The "excess $\varepsilon$" column isolates component (C): it is near zero at low budgets (where WOR ≈ WR) and grows to 0.106 (MMLU-Pro) / 0.037 (BBH) at 25% budget.

### 4.2 The $\hat{B}_n$ Bias Decomposition

The paper proves (page 5–6):

$$\hat{B}_n - B_n = T_1 + T_2 - T_3$$

where:
- $T_1 = \frac{1}{n}\sum_{t=2}^{n}(\hat{\theta}_{t-1} - \theta)^2 \geq 0$ — MSE of running average
- $T_2 = \frac{2}{n}\sum_{t=2}^{n}(\theta - \psi_t)(\hat{\theta}_{t-1} - \theta)$ — cross-correlation
- $T_3 = \frac{1}{n}(\theta - \frac{1}{N}\sum_i \hat{f}^{(0)}(x_i))^2 \geq 0$ — initial model error correction

The paper bounds: $\mathbb{E}[T_1] = O(\log n / n)$, $\mathbb{E}[T_2] = O(1/\sqrt{n})$, $\mathbb{E}[T_3] = O(1/n)$, so all vanish asymptotically.

### 4.3 Why the *Relative* Bias Grows: The Core Mechanism

The bias that matters for coverage is the *relative* bias:

$$\frac{\mathbb{E}[\hat{B}_n - B_n]}{\sigma^2} = \frac{\mathbb{E}[T_1 + T_2 - T_3]}{A_n - B_n}$$

We can compute this from the data. Defining $\delta^2 = \sigma_{\text{true}}^2 - \hat{\sigma}^2$ (= empirical average of $\hat{B} - B$, since $\hat{A}$ is unbiased):

| Budget (MMLU) | $\hat{\sigma}^2$ | $\sigma_{\text{true}}^2$ | $\delta^2$ | $\delta^2/\sigma_{\text{true}}^2$ | ESS |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2.5% | 0.0858 | 0.0926 | 0.0068 | 7.3% | 3.67 |
| 10% | 0.0529 | 0.0559 | 0.0031 | 5.5% | 5.48 |
| 17.5% | 0.0413 | 0.0456 | 0.0043 | 9.4% | 6.44 |
| 25% | 0.0331 | 0.0423 | **0.0093** | **21.9%** | 7.31 |

The relative bias $\delta^2/\sigma_{\text{true}}^2$ has a **U-shape**: it decreases from 7.3% to 5.5% (budget 2.5%→10%), then increases sharply to 21.9% (budget 25%).

**Two simultaneous effects produce this:**

1. **$\sigma_{\text{true}}^2$ shrinks monotonically** (the estimator becomes more efficient with budget). $\sigma_{\text{true}}^2$ falls from 0.093 to 0.042, a 2.2× decrease.

2. **$\delta^2$ first decreases then increases**. It falls from 0.0068 to 0.0031 (budget 2.5%→10%) as more data improves the variance estimate, then rises back to 0.0093 at 25% budget. The absolute bias *increases* at high budgets.

The relative bias is their ratio, and since the denominator shrinks faster than the numerator, the ratio grows.

### 4.4 Why the Absolute Bias $\delta^2$ Increases at High Budgets

This is the most surprising empirical finding. The absolute bias $\delta^2 = \sigma_{\text{true}}^2 - \hat{\sigma}^2$ does not monotonically decrease. From the data:

| Budget (MMLU) | $\delta^2$ | $\delta^2 \times n$ |
|:---:|:---:|:---:|
| 2.5% | 0.00680 | 2.05 |
| 5% | 0.00557 | 3.35 |
| 10% | 0.00309 | 3.71 |
| 15% | 0.00361 | 6.51 |
| 20% | 0.00574 | 13.81 |
| 25% | 0.00925 | 27.83 |

$\delta^2 \cdot n$ grows dramatically (2.0 → 27.8), showing that the bias per observation is *increasing*, not decreasing.

The explanation involves three interacting effects:

**(i) Model non-stationarity.** The factor model $\hat{f}^{(t)}$ is retrained via Bayesian updates at each round. At early rounds, the model is poor and predictions have large residuals. At later rounds, predictions improve. The running average $\hat{\theta}_{t-1}$ carries the "memory" of early, noisy $\phi_t$ values, while $\psi_t$ (the imputed mean) reflects the *current* model. This mismatch inflates $\hat{B}_n$: the squared difference $(N\hat{\theta}_{t-1} - N\psi_t)^2$ includes a component from historical estimation noise that does not appear in $B_n = (1/(nN^2))\sum(\sum_{i \notin O} e_i)^2$.

**(ii) The WOR "self-improvement" amplification.** For WOR, sampling removes items from the pool. Items with large $|y_i - \hat{f}_i|$ are preferentially sampled (PAI scoring), leaving behind items with progressively smaller residuals. This causes:
- $A_n$ to decrease (fewer and smaller residuals in the unobserved pool)
- $B_n$ to decrease even faster (the mean residual over remaining items shrinks rapidly)
- $\sigma^2 = A_n - B_n$ to shrink, becoming a smaller and smaller difference of two quantities

**(iii) The cancellation amplification.** When $\sigma^2 = A - B$ is a small difference of two moderately-sized quantities, any error in estimating $B$ gets amplified. The estimated $B/A$ ratio grows from ~3% at 2.5% budget to ~49% at 25% budget (computed as $(\hat{\sigma}^2_{\text{WR}} - \hat{\sigma}^2_{\text{WOR}})/\hat{\sigma}^2_{\text{WOR}}$ under the approximation $\hat{A}_{\text{WOR}} \approx \hat{A}_{\text{WR}}$). When $B/A \approx 0.49$, even a 10% relative error in $\hat{B}$ produces a $\sim 10\%$ relative error in $\hat{\sigma}^2$.

### 4.5 Why WR FAQ Does Not Suffer

For WR FAQ, the variance formula is simply $\hat{\sigma}^2 = \hat{A}$ (no $B$ correction), because with-replacement samples are independent and no cross-round anti-correlation correction is needed. This eliminates the entire cancellation problem.

Additionally, WR FAQ's ESS **saturates and then decreases** at high budgets (5.1×→4.9× for MMLU-Pro), because re-sampling previously-seen items provides diminishing returns. This means $\sigma^2_{\text{WR}}$ stops shrinking, so the baseline CLT error (component A) continues to decrease, giving **improving** coverage with budget.

For WOR FAQ, ESS keeps growing (3.67→7.31×) because every sample is new information. This makes $\sigma^2$ keep shrinking, amplifying the $\hat{B}$ bias.

### 4.6 The U-Shape Explained

The coverage deficit $d(b) = 0.95 - \text{cov}(b) \approx 0.229\varepsilon(b)$ has a minimum (best coverage) at budget $b^* \approx 8\text{--}10\%$:

**Below $b^*$**: The CLT improvement (component A, decreasing as $1/\sqrt{n}$) dominates. More data → better normal approximation → better coverage.

**Above $b^*$**: The $\hat{B}$ relative bias amplification (component C) dominates. More data → higher ESS → smaller $\sigma^2$ → larger relative bias → worse coverage.

Fitting $\varepsilon$ as a quadratic in ESS:

| Dataset | Model | $R^2$ | ESS at minimum $\varepsilon$ | Budget at minimum |
|---------|-------|:-----:|:---:|:---:|
| MMLU-Pro | $\varepsilon = 0.383 - 0.146\cdot\text{ESS} + 0.0148\cdot\text{ESS}^2$ | 0.930 | 4.93 | ~7% |
| BBH suite | $\varepsilon = 0.199 - 0.091\cdot\text{ESS} + 0.0108\cdot\text{ESS}^2$ | 0.954 | 4.17 | ~7% |

### 4.7 Quantitative Verification

The formula $\text{coverage} \approx 0.95 - 0.229\varepsilon$ matches the data:

| | $\varepsilon$ | Predicted coverage | Actual coverage |
|---|:---:|:---:|:---:|
| MMLU-Pro, 25% | 0.116 | 0.923 | 0.917 |
| BBH, 25% | 0.039 | 0.941 | 0.940 |
| MMLU-Pro, 10% | 0.028 | 0.944 | 0.943 |
| BBH, 10% | 0.014 | 0.947 | 0.947 |

The small discrepancy at MMLU-Pro 25% (0.923 vs 0.917) is from the nonlinearity of $\Phi$ at large $\varepsilon$.

---

## 5. Summary

The WOR FAQ coverage degradation has a **single root cause**: the $\hat{B}_n$ variance correction is positively biased ($\mathbb{E}[\hat{B}_n] > B_n$), causing $\hat{\sigma}^2 = \hat{A} - \hat{B}$ to underestimate $\sigma^2 = A - B$.

This bias is asymptotically negligible ($\hat{B}_n - B_n = T_1 + T_2 - T_3 \to 0$), but at finite sample sizes its **relative** impact grows with budget because:

1. **$\sigma^2$ shrinks** as ESS grows (WOR keeps improving, unlike WR which saturates).
2. **$\delta^2$ (absolute bias) also increases** at high budgets due to model non-stationarity and the increasing importance of the $B$ correction (higher $B/A$ ratio means more cancellation).
3. The two effects multiply: the relative bias $\delta^2/\sigma^2$ grows from ~5% to ~22% across the budget range.

The resulting coverage deficit is well-described by $\varepsilon \approx c_0 + c_1 \cdot \text{ESS} + c_2 \cdot \text{ESS}^2$ with $R^2 > 0.93$ for both datasets.

---

## 6. Implications and Options

### Where coverage is acceptable
- **BBH suite** at all budgets: worst coverage 0.940 (at 25%). Borderline but usable.
- **MMLU-Pro at ≤ 10% budget**: coverage 0.940–0.943. Acceptable for most purposes.
- **MMLU-Pro at > 15% budget**: coverage degrades to 0.917. Should be noted as a limitation.

### The estimator itself is fine
The estimator $\hat{\theta}_n$ is **exactly unbiased** at every finite $n$. The issue is purely with **CI width estimation**, not the point estimate or the ESS gains.

### Potential fixes (not currently implemented)
1. **Partial B correction**: $\hat{\sigma}^2 = \hat{A} - \alpha\hat{B}$ with $\alpha < 1$ (e.g., 0.5). No theoretical justification for choice of $\alpha$.
2. **Bootstrap CI**: Resample the trial trajectory. Computationally expensive.
3. **Finite-population correction**: Inflate $\hat{\sigma}^2$ by $(1 + c \cdot \hat{B}/\hat{A})$ for some factor $c$.
4. **Accept and report**: Note that CIs are asymptotically valid, report observed coverage alongside ESS gains. The enormous efficiency improvement (7×) is the main contribution.

---

## 7. Code Location

The variance formula is in `wor_trial.py`, lines 162–172 (FAQ) and 285–290 (ablation). The implementation correctly follows Theorem 4.
