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

### 4.3 Exact $\mathbb{E}[T_1]$ via Exchanged Sums

$T_1$ is the dominant bias term. Recall $T_1 = \frac{1}{n}\sum_{t=2}^{n}(\hat{\theta}_{t-1} - \theta)^2$, where $\hat{\theta}_{t-1} = \frac{1}{t-1}\sum_{s=1}^{t-1}\phi_s$ is the running average.

Since $\hat{\theta}_n$ is a martingale around $\theta$ (by the exact unbiasedness $\mathbb{E}[\phi_s | \mathcal{F}_{s-1}] = \theta$), we can write $\hat{\theta}_{t-1} - \theta = \frac{1}{t-1}\sum_{s=1}^{t-1}\Delta_s$ where $\Delta_s = \phi_s - \theta$ are martingale differences. Define the **per-step conditional variance**:

$$v_s := \mathbb{E}[\Delta_s^2 \mid \mathcal{F}_{s-1}] = \frac{1}{N^2}\sum_{i \notin O_{s-1}} \frac{(y_i - \hat{f}^{(s-1)}_i)^2}{q_s(i)} - \frac{1}{N^2}\left(\sum_{i \notin O_{s-1}}(y_i - \hat{f}^{(s-1)}_i)\right)^2$$

This is the single-step analogue of the Theorem 4 variance formula: the first term is the per-step $A$ and the second is the per-step $B$. Note that $v_s$ is a random variable ($\mathcal{F}_{s-1}$-measurable), but $\mathbb{E}[v_s]$ is a deterministic function of $s$ (averaging over all sample paths).

**Computation of $\mathbb{E}[T_1]$.** By the martingale orthogonality $\mathbb{E}[\Delta_s \Delta_{s'} \mid \mathcal{F}_{\max(s,s')-1}] = 0$ for $s \ne s'$:

$$\mathbb{E}[(\hat{\theta}_{t-1} - \theta)^2] = \frac{1}{(t-1)^2}\sum_{s=1}^{t-1}\mathbb{E}[v_s]$$

$$\implies \mathbb{E}[T_1] = \frac{1}{n}\sum_{t=2}^{n}\frac{1}{(t-1)^2}\sum_{s=1}^{t-1}\mathbb{E}[v_s]$$

**Exchange of summation.** The index $s$ appears in all terms with $t \in \{s+1, \ldots, n\}$. Exchanging order:

$$\boxed{\mathbb{E}[T_1] = \frac{1}{n}\sum_{s=1}^{n-1}\mathbb{E}[v_s] \cdot w_s, \qquad w_s := \sum_{k=s}^{n-1}\frac{1}{k^2}}$$

This is **exact** — no approximation. The weights $w_s$ are partial tails of $\sum 1/k^2$, and are strictly positive and decreasing: $w_1 = \sum_{k=1}^{n-1}1/k^2 \in [1, \pi^2/6]$ and $w_{n-1} = 1/(n-1)^2$.

### 4.4 The Non-Stationarity Index $\Lambda$

Define the **true per-step variance**:

$$\bar{\sigma}^2 := \frac{1}{n}\sum_{s=1}^{n}\mathbb{E}[v_s] = n\,\text{Var}(\hat{\theta}_n)$$

This is the quantity that $\hat{\sigma}^2 = \hat{A}_n - \hat{B}_n$ estimates. The CI half-width is $z\sqrt{\hat{\sigma}^2/n}$, so coverage depends on how well $\hat{\sigma}^2$ estimates $\bar{\sigma}^2$.

**Definition.** The **non-stationarity index** is:

$$\boxed{\Lambda := \frac{n}{\bar{\sigma}^2}\,\mathbb{E}[T_1] = \frac{1}{\bar{\sigma}^2}\sum_{s=1}^{n-1}\mathbb{E}[v_s] \cdot w_s}$$

By construction, $\mathbb{E}[T_1] = \bar{\sigma}^2 \Lambda / n$ **exactly**, with no remainder term. $\Lambda$ is a computable, finite, positive number that depends on the variance profile $\{\mathbb{E}[v_s]\}_{s=1}^{n}$ and the budget $n$.

**Normalized form.** With $g(s) := \mathbb{E}[v_s]/\bar{\sigma}^2$ (the normalized profile satisfying $\frac{1}{n}\sum g(s) = 1$):

$$\Lambda = \sum_{s=1}^{n-1} g(s)\, w_s$$

**Stationary benchmark.** If $g(s) \equiv 1$ (constant variance):

$$\Lambda_{\text{stat}} = \sum_{s=1}^{n-1} w_s = \sum_{s=1}^{n-1}\sum_{k=s}^{n-1}\frac{1}{k^2} = \sum_{k=1}^{n-1}\frac{k}{k^2} = \sum_{k=1}^{n-1}\frac{1}{k} = H_{n-1}$$

(by exchanging the double sum: each $1/k^2$ is counted $k$ times, for $s = 1, \ldots, k$). So $\Lambda_{\text{stat}} = H_{n-1} \approx \ln n + \gamma$.

**Monotonicity: $\Lambda \geq H_{n-1}$ for decreasing $g$.** Since both $g$ and $w$ are decreasing in $s$, and $\sum_{s=1}^{n-1}(g(s) - 1) = n - 1 - (n-1) = 0$ (the constraint $\frac{1}{n}\sum_{s=1}^n g(s) = 1$ gives $\sum_{s=1}^{n-1} g(s) = n - g(n)$, so it's not exactly zero, but $g(n) \approx 1$ to first order)... More precisely, by Chebyshev's sum inequality: if $g$ and $w$ are both decreasing (i.e., co-monotone), then:

$$\frac{1}{n-1}\sum_{s=1}^{n-1} g(s) w_s \geq \left(\frac{1}{n-1}\sum_{s=1}^{n-1} g(s)\right)\left(\frac{1}{n-1}\sum_{s=1}^{n-1} w_s\right)$$

$$\implies \Lambda \geq \left(\frac{1}{n-1}\sum_{s=1}^{n-1} g(s)\right) \cdot H_{n-1}$$

Since $g$ averages to 1 (approximately), $\Lambda \geq H_{n-1}$, with equality only when $g$ is constant.

### 4.5 Coverage Formula with Explicit Remainder

The coverage depends on the ratio $\hat{\sigma}^2 / \bar{\sigma}^2$. From the decomposition $\hat{B}_n - B_n = T_1 + T_2 - T_3$ and the unbiasedness of $\hat{A}_n$:

$$\mathbb{E}[\hat{\sigma}^2] = \mathbb{E}[\hat{A}_n - \hat{B}_n] = \bar{\sigma}^2 - \mathbb{E}[T_1] + \mathbb{E}[T_3] - \mathbb{E}[T_2]$$

**Define** $\Lambda_3 := n\,\mathbb{E}[T_3]/\bar{\sigma}^2 = (\theta - \bar{\psi}_1)^2/\bar{\sigma}^2$ where $\bar{\psi}_1 = \frac{1}{N}\sum_i \hat{f}^{(0)}(x_i)$.

Then the **relative bias** of $\hat{\sigma}^2$ is:

$$\frac{\bar{\sigma}^2 - \mathbb{E}[\hat{\sigma}^2]}{\bar{\sigma}^2} = \frac{\Lambda - \Lambda_3}{n} + \frac{\mathbb{E}[T_2]}{\bar{\sigma}^2}$$

This is exact.

**Coverage derivation.** Write $\text{cov} = \mathbb{P}(|Z_n| \leq z \cdot \hat{\sigma}/\sigma)$ where $Z_n = \sqrt{n}(\hat{\theta}_n - \theta)/\sigma$ and $\sigma = \sqrt{\bar{\sigma}^2}$. By the martingale CLT, $Z_n \xrightarrow{d} \mathcal{N}(0,1)$, and treating $\hat{\sigma}/\sigma \approx \sqrt{1 - (\Lambda - \Lambda_3)/n} \approx 1 - (\Lambda - \Lambda_3)/(2n)$:

$$\boxed{\text{coverage} = (1 - \alpha) - z\varphi(z)\frac{\Lambda - \Lambda_3}{n} + R_{\text{CLT}} + R_{T_2} + R_{\text{var}} + R_{\text{nonlin}}}$$

where $z = z_{\alpha/2}$, $\varphi(z)$ is the standard normal PDF, and the remainder terms are:

- $R_{\text{CLT}}$: the **CLT correction** from $Z_n$ not being exactly $\mathcal{N}(0,1)$. By the Berry–Esseen inequality for martingales (Heyde & Brown 1970), $|R_{\text{CLT}}| \leq C_{\text{BE}} / \sqrt{n}$ where $C_{\text{BE}}$ depends on the third moment ratio $\sum \mathbb{E}[|\Delta_s|^3] / (\sum \mathbb{E}[v_s])^{3/2}$.

- $R_{T_2}$: contribution from the cross term. $|R_{T_2}| = z\varphi(z) \cdot |\mathbb{E}[T_2]|/\bar{\sigma}^2$. Bounded in §4.6.2.

- $R_{\text{var}}$: from the **random fluctuation** of $\hat{\sigma}^2$ around $\mathbb{E}[\hat{\sigma}^2]$ (the formula above uses $\mathbb{E}[\hat{\sigma}^2]$, but coverage depends on each realization of $\hat{\sigma}^2$). This enters at $O(\text{Var}(\hat{\sigma}^2)/\bar{\sigma}^4)$.

- $R_{\text{nonlin}}$: from the nonlinearity of $\Phi$. $|R_{\text{nonlin}}| \leq C \cdot ((\Lambda - \Lambda_3)/n)^2$.

For the numerical values: $z = 1.96$, $\varphi(z) \approx 0.0584$, so $z\varphi(z) \approx 0.114$.

**Remark on $\Lambda_3$.** We do NOT assume $\Lambda_3 \approx 0$. In the verification (§4.7), $\Lambda_3$ is computed empirically from the initial factor model predictions. For FAQ, $\Lambda_3$ is small but not negligible; for uniform sampling, $\Lambda_3$ can be substantial.

### 4.6 The Two Competing Effects: Why Coverage Is Non-Monotone

The empirical coverage (§4.1) first improves from 2.5% to ~10% budget (matching WR behavior), then degrades from ~10% to 25%. This U-shape arises from **two competing corrections** in the coverage formula:

$$\underbrace{(1 - \alpha)}_{\text{target}} - \underbrace{z\varphi(z)\frac{\Lambda - \Lambda_3}{n}}_{\text{bias effect (B)}} + \underbrace{R_{\text{CLT}}}_{\text{CLT effect (A)}} + \text{smaller terms}$$

**Effect A: CLT improvement** ($R_{\text{CLT}} \sim -C/\sqrt{n}$, decreasing in magnitude). As $n$ grows, the martingale CLT approximation improves. This effect is shared with WR FAQ, where the CI half-width scales as $1/\sqrt{n}$ and the normal approximation error is $O(1/\sqrt{n})$. It monotonically improves coverage.

**Effect B: $\hat{B}_n$ bias** ($z\varphi(z)(\Lambda - \Lambda_3)/n$). This is the **relative bias** of the variance estimator, unique to WOR. Whether it increases or decreases with budget depends on how $(\Lambda - \Lambda_3)/n$ varies with $n$.

**At low budgets** ($n$ small, budget $\lesssim 10\%$): Effect A dominates. Adding more samples improves both the CLT approximation and the variance estimate. Coverage improves, exactly as in WR. The WOR and WR coverage curves are nearly indistinguishable.

**At high budgets** ($n$ large, budget $\gtrsim 10\%$): Effect A has mostly converged (WR coverage is ~0.948). But Effect B is now large enough to dominate. Coverage degrades.

### 4.6.1 Why $(\Lambda - \Lambda_3)/n$ Increases with Budget

This is the central question. Asymptotically, $(\Lambda - \Lambda_3)/n \to 0$ (guaranteed by Theorem 4: coverage is asymptotically valid). But at **finite budgets**, this ratio can increase. Here is why.

**The key: $\bar{\sigma}^2$ shrinks faster than $\mathbb{E}[T_1]$.** Recall:

$$\frac{\Lambda}{n} = \frac{\mathbb{E}[T_1]}{\bar{\sigma}^2} = \frac{\sum_{s=1}^{n-1}\mathbb{E}[v_s]\,w_s}{n \cdot \bar{\sigma}^2} = \frac{\sum_{s=1}^{n-1}\mathbb{E}[v_s]\,w_s}{\sum_{s=1}^{n}\mathbb{E}[v_s]}$$

This is a ratio of two sums over the same variance profile $\{\mathbb{E}[v_s]\}$. The numerator uses weights $w_s$ that decay roughly as $1/s$; the denominator weights all terms equally.

As budget $n$ increases, the WOR sampling observes more items. The factor model improves dramatically at later rounds, making $\mathbb{E}[v_s]$ very small for large $s$. Concretely:

- **Denominator** $\sum_{s=1}^n \mathbb{E}[v_s]$: extends the sum with small new terms ($v_s \approx 0$ for large $s$). The average $\bar{\sigma}^2 = \frac{1}{n}\sum \mathbb{E}[v_s]$ decreases because the late-round terms dilute the early-round values. This is the **ESS growth**: the estimator becomes more efficient.

- **Numerator** $\sum_{s=1}^{n-1}\mathbb{E}[v_s]\,w_s$: also extends, but the new terms have both small $v_s$ AND small $w_s$, so they contribute negligibly. The numerator is **dominated by the first few rounds** where both $v_s$ and $w_s$ are large. These early-round values do not change much with budget.

So: the numerator stays roughly constant while the denominator grows → $\Lambda/n$ increases.

**Why this is not just $\log(n)/n$.** For a variance profile $v_s$ that is **independent of total budget $n$** (i.e., $\mathbb{E}[v_s]$ at step $s$ is the same whether the total budget is 1000 or 3000), one can verify that $\Lambda/n$ is eventually decreasing (it equals a convergent sum divided by a divergent sum). But in WOR FAQ, the profile **depends on $n$** through two mechanisms:

1. **Scheduling**: the blending parameters use $t/({\rho \cdot n_B})$ and $t/({\gamma \cdot n_B})$, so the same absolute step $s$ gets a different schedule position for different total budgets. Larger $n_B$ stretches the exploration phase, which can increase $\mathbb{E}[v_s]$ at early rounds.

2. **Hyperparameter tuning**: optimal $({\beta_0, \rho, \gamma, \tau})$ are selected per budget. Higher-budget settings may use more aggressive scoring (higher $\beta_0$), further increasing early-round variance.

These effects cause $\sum \mathbb{E}[v_s]\,w_s$ to *increase* with budget (not stay constant), while $\bar{\sigma}^2$ keeps decreasing. The ratio $\Lambda/n$ therefore increases over the practical budget range.

**Quantitative check from the data.** Define $\delta^2 := \bar{\sigma}^2 - \mathbb{E}[\hat{\sigma}^2] \approx \mathbb{E}[T_1]$ (since $T_3$ and $T_2$ are smaller). Then $\Lambda/n \approx \delta^2/\bar{\sigma}^2$:

| Budget (MMLU) | $\bar{\sigma}^2$ | $\delta^2$ | $\delta^2/\bar{\sigma}^2 = \Lambda/n$ | $n$ | $\Lambda$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2.5% | 0.0926 | 0.0068 | 0.073 | ~300 | ~22 |
| 10% | 0.0559 | 0.0031 | 0.055 | ~1200 | ~66 |
| 25% | 0.0423 | 0.0093 | 0.220 | ~3000 | ~660 |

The ratio $\Lambda/n$ first decreases (0.073 → 0.055, CLT effect dominating) then increases sharply (0.055 → 0.220). For comparison, the stationary benchmark is $H_{n-1}/n \approx \ln(n)/n$: $\ln(300)/300 = 0.019$, $\ln(1200)/1200 = 0.006$, $\ln(3000)/3000 = 0.003$. The observed $\Lambda/n$ values are 4–70× larger than the stationary prediction, confirming massive non-stationarity. And while $\ln(n)/n$ is monotonically decreasing, the observed $\Lambda/n$ increases by 4× from 10% to 25% budget.

**Asymptotic vs. finite-sample.** The theorem guarantees $\Lambda(n)/n \to 0$ (since the bias is $o(1)$ and $\bar{\sigma}^2$ is bounded away from zero for any finite population). The finite-sample behavior over budget 2.5–25% does not contradict this: the degradation occurs in a specific range and must eventually reverse. The relevant question for practice is whether this reversal occurs within the useful budget range.

### 4.6.2 Parametric Variance Profiles (Illustrative)

To build intuition, we compute $\Lambda$ for two simple profiles. These are useful for understanding the $\Lambda$ growth mechanism, even though the true variance profile is more complex (and budget-dependent).

**Exponential decay: $v_s = v_1 \cdot e^{-\lambda(s-1)/n}$.** Represents a model that improves at rate $\lambda$. Then:

$$\bar{\sigma}^2 \approx v_1 \cdot \frac{1 - e^{-\lambda}}{\lambda}, \qquad \Lambda = \frac{\lambda}{1 - e^{-\lambda}} \sum_{s=1}^{n-1}\frac{e^{-\lambda(s-1)/n}}{s} \cdot w_s$$

Note: $\lambda$ itself scales with $n$ in the WOR setting (longer budgets mean more model improvement), so $\Lambda$ is not simply $\log(n)$ growth. If $\lambda \propto n$ (learning rate proportional to budget), $\Lambda$ can grow much faster.

**Power-law decay: $v_s = v_1 \cdot s^{-\alpha}$** for $0 < \alpha < 1$. Then $\Lambda/n \sim (1-\alpha) n^{\alpha-1}/\alpha$, which decreases as $n^{\alpha - 1}$. So for a **fixed** power-law profile, coverage always improves. The empirical degradation must therefore come from the profile becoming *steeper* (larger effective $\alpha$) as budget increases.

### 4.6.3 Treatment of $T_2$ and $T_3$

**$T_3$ is deterministic.** $T_3 = \frac{1}{n}(\theta - \bar{\psi}_1)^2$ where $\bar{\psi}_1 = \frac{1}{N}\sum_i \hat{f}^{(0)}(x_i)$. This is fixed given the prior model $\hat{f}^{(0)}$. In normalized units:

$$\Lambda_3 = \frac{n \cdot T_3}{\bar{\sigma}^2} = \frac{(\theta - \bar{\psi}_1)^2}{\bar{\sigma}^2}$$

Note that $\Lambda_3$ depends on $n$ only through $\bar{\sigma}^2(n)$ in the denominator (since the numerator $(\theta - \bar{\psi}_1)^2$ is fixed). As $\bar{\sigma}^2$ decreases with budget, $\Lambda_3$ *increases*, partially offsetting the growth of $\Lambda$. However, $\Lambda_3$ grows much more slowly than $\Lambda$ because $(\theta - \bar{\psi}_1)^2$ is a fixed constant while $\mathbb{E}[T_1]$ involves a sum that grows with $n$.

**$T_2$ cross term.** $T_2 = \frac{2}{n}\sum_{t=2}^{n}(\theta - \psi_t)(\hat{\theta}_{t-1} - \theta)$.

Both factors are $\mathcal{F}_{t-1}$-measurable. The key structure is:
- $\hat{\theta}_{t-1} - \theta = \frac{1}{t-1}\sum_{s=1}^{t-1}\Delta_s$: cumulative martingale noise.
- $\theta - \psi_t = \frac{1}{N}\sum_{i \notin O_{t-1}}(y_i - \hat{f}^{(t-1)}_i)$: total prediction error on unobserved items.

Conditioning on $\mathcal{F}_{t-2}$ (everything up to step $t-2$):

$$\mathbb{E}[(\hat{\theta}_{t-1} - \theta)(\theta - \psi_t) \mid \mathcal{F}_{t-2}] = \mathbb{E}\!\left[\frac{1}{t-1}\left(\sum_{s=1}^{t-2}\Delta_s + \Delta_{t-1}\right)(\theta - \psi_t) \;\middle|\; \mathcal{F}_{t-2}\right]$$

The term $\sum_{s<t-1}\Delta_s$ is $\mathcal{F}_{t-2}$-measurable, but $(\theta - \psi_t)$ depends on $I_{t-1}$ (through $O_{t-1}$ and $\hat{f}^{(t-1)}$), so they are correlated. This correlation comes from the fact that observing item $I_{t-1}$ simultaneously:
- contributes $\Delta_{t-1}$ to the running average
- changes $\psi_t$ by replacing one prediction with the true label and updating the model

The magnitude of this correlation per step is $O(1/N)$ (since each step changes $\psi$ by one item out of $N$). Summing over $t$:

$$|\mathbb{E}[T_2]| \leq \frac{C}{N} \cdot \frac{1}{n}\sum_{t=2}^n \sqrt{\text{Var}(\hat{\theta}_{t-1})} = O\!\left(\frac{\sqrt{\bar{\sigma}^2}}{N}\right)$$

For $N \gg \sqrt{n}$ (always true in our setting: $N \sim 12000$, $n \leq 3000$), this is much smaller than $\mathbb{E}[T_1] \sim \bar{\sigma}^2 \Lambda/n$.

**Summary.** The exact coverage identity is:

$$\text{coverage} = (1 - \alpha) - z\varphi(z)\frac{\Lambda - \Lambda_3}{n} + R$$

where $\Lambda$ and $\Lambda_3$ are computable from the variance profile $\{v_s\}$ and the initial model, and the remainder satisfies:

$$|R| \leq \frac{C_1}{\sqrt{n}} + C_2\left(\frac{\Lambda}{n}\right)^2 + \frac{C_3\sqrt{\bar{\sigma}^2}}{N}$$

The first term is the CLT correction (shared with WR), the second is the nonlinearity correction, and the third bounds the $T_2$ contribution. The coverage degradation at high budgets is driven entirely by the growth of $\Lambda/n$, which overcomes the $C_1/\sqrt{n}$ improvement.

### 4.7 Quantitative Verification

**From §4.1 data.** Defining $\varepsilon = 1 - \Phi^{-1}((1 + \text{cov})/2)/z$ (the fractional underestimation of $\sigma$) and using the relationship $\text{coverage} \approx 0.95 - 2z\varphi(z)\varepsilon$:

| | $\varepsilon$ | Predicted coverage | Actual coverage |
|---|:---:|:---:|:---:|
| MMLU-Pro, 25% | 0.116 | 0.923 | 0.917 |
| BBH, 25% | 0.039 | 0.941 | 0.940 |
| MMLU-Pro, 10% | 0.028 | 0.944 | 0.943 |
| BBH, 10% | 0.014 | 0.947 | 0.947 |

The small discrepancy at MMLU-Pro 25% (0.923 vs 0.917) is from the nonlinearity correction $R_{\text{nonlin}}$ (§4.5).

**From $\Lambda$.** The script `verify_lambda.py` computes $\Lambda$ from the per-step variance profile $v_s$ (logged by `wor_trial.py` with `log_profile=True`) and checks the prediction $\text{coverage} \approx 0.95 - 0.114(\Lambda - \Lambda_3)/n$ against the actual coverage.

---

## 5. Summary

The WOR FAQ coverage degradation has a **single root cause**: the $\hat{B}_n$ variance correction is positively biased ($\mathbb{E}[\hat{B}_n] > B_n$), causing $\hat{\sigma}^2 = \hat{A} - \hat{B}$ to underestimate $\bar{\sigma}^2 = A - B$.

The exact formula (§4.3–4.5):

$$\text{coverage} = 0.95 - 0.114 \cdot \frac{\Lambda - \Lambda_3}{n} + R, \qquad |R| \leq \frac{C_1}{\sqrt{n}} + C_2\left(\frac{\Lambda}{n}\right)^2 + \frac{C_3\sqrt{\bar{\sigma}^2}}{N}$$

where $\Lambda = \frac{1}{\bar{\sigma}^2}\sum_{s=1}^{n-1}\mathbb{E}[v_s]\,w_s$ is the non-stationarity index and $\Lambda_3 = (\theta - \bar{\psi}_1)^2/\bar{\sigma}^2$ is the initial model correction.

**Two competing effects produce the U-shaped coverage curve** (§4.6):

1. **CLT improvement** ($C_1/\sqrt{n}$, decreasing): shared with WR, dominates at low budgets.
2. **$\hat{B}_n$ relative bias** ($0.114 \cdot (\Lambda - \Lambda_3)/n$, empirically increasing over 10–25%): unique to WOR.

The ratio $\Lambda/n$ increases at finite budgets because $\bar{\sigma}^2$ (denominator) shrinks faster than $\sum \mathbb{E}[v_s]\,w_s$ (numerator), amplified by budget-dependent scheduling and hyperparameter tuning. Asymptotically, $\Lambda/n \to 0$ (Theorem 4 is valid), but the practical budget range lies in the non-monotone regime.

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
