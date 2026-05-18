# Why Higher $\tau$ Improves Coverage (and Why the Problem Only Appears at High Budget)

## The intuition in one paragraph

At each round $s$, the estimator takes a "step" with some variance $v_s$. Our variance estimator $\hat{\sigma}^2 = \hat{A} - \hat{B}$ tries to estimate the average step-variance. It gets $\hat{A}$ right, but $\hat{B}$ is biased high because it squares the running average $\hat{\theta}_{t-1}$, adding a noise floor $\text{Var}(\hat{\theta}_{t-1})$. This noise floor is dominated by the early noisy steps — once those have happened, their noise is baked into the running average forever. So $\hat{B}$ overshoots by an amount anchored to the early steps. Meanwhile the true average variance $\bar{\sigma}^2$ reflects all steps equally. When early steps are much noisier than late steps, the bias (set by early steps) is a large fraction of $\bar{\sigma}^2$ (diluted by late steps). That fraction — the relative bias — determines undercoverage. $\tau$ prevents this by putting a floor on sampling probabilities, which prevents any step from being too noisy relative to the others.

---

## 1. Setup: the estimator and the variance formula

The WOR estimator averages $n$ AIPW-corrected terms:

$$\hat{\theta}_n = \frac{1}{n}\sum_{t=1}^n \phi_t, \qquad \phi_t = \underbrace{\frac{1}{N}\left(\sum_{i \in O_{t-1}} y_i + \sum_{i \notin O_{t-1}} \hat{f}^{(t-1)}(x_i)\right)}_{\psi_t} + \underbrace{\frac{1}{N} \cdot \frac{y_{I_t} - \hat{f}^{(t-1)}(x_{I_t})}{q_t(I_t)}}_{\text{AIPW correction}}$$

This is exactly unbiased: $\mathbb{E}[\hat{\theta}_n] = \theta$ for every $n$. The issue is with the variance estimator used for CIs. Theorem 4 gives:

$$\hat{\sigma}^2 = \hat{A}_n - \hat{B}_n$$

$$\hat{A}_n = \frac{1}{nN^2}\sum_{t=1}^n \frac{(y_{I_t} - \hat{f}^{(t-1)}(x_{I_t}))^2}{q_t(I_t)^2}, \qquad \hat{B}_n = \frac{1}{nN^2}\sum_{t=2}^n (N\hat{\theta}_{t-1} - N\psi_t)^2$$

and the CI is $\hat{\theta}_n \pm 1.96 \cdot \hat{\sigma}_n / \sqrt{n}$.

The population analogues (conditional on the sample path) are:

$$A_n = \frac{1}{nN^2}\sum_{t=1}^n \sum_{i \notin O_{t-1}} \frac{(y_i - \hat{f}_i)^2}{q_t(i)}, \qquad B_n = \frac{1}{nN^2}\sum_{t=1}^n \left(\sum_{i \notin O_{t-1}}(y_i - \hat{f}_i)\right)^2$$

The true conditional variance of the estimator is $\bar{\sigma}^2 = A_n - B_n$.

---

## 2. The problem: $\hat{B}_n$ is positively biased

$\hat{A}_n$ is unbiased for $A_n$ (its increments are martingale differences). But $\hat{B}_n$ overshoots $B_n$. To see why, expand the $t$-th term of $\hat{B}_n$:

$$\hat{B}_n \text{ uses } (\hat{\theta}_{t-1} - \psi_t)^2, \qquad B_n \text{ uses } (\theta - \psi_t)^2$$

Since $\hat{\theta}_{t-1}$ is a noisy estimate of $\theta$, squaring the noisy version inflates the result:

$$\mathbb{E}[(\hat{\theta}_{t-1} - \psi_t)^2] = (\theta - \psi_t)^2 + \text{Var}(\hat{\theta}_{t-1}) + 2(\theta - \psi_t)\underbrace{\mathbb{E}[\hat{\theta}_{t-1} - \theta]}_{= 0}$$

The last term vanishes by unbiasedness. So each term of $\hat{B}_n$ overshoots the corresponding term of $B_n$ by $\text{Var}(\hat{\theta}_{t-1})$ in expectation. The full decomposition from the paper is:

$$\hat{B}_n - B_n = T_1 + T_2 - T_3$$

where $T_1 = \frac{1}{n}\sum_{t=2}^n (\hat{\theta}_{t-1} - \theta)^2$ is a **random variable** (the MSE of the running average along the sample path). Its expectation is exactly the sum of the noise floors:

$$\mathbb{E}[T_1] = \frac{1}{n}\sum_{t=2}^n \text{Var}(\hat{\theta}_{t-1})$$

The other two terms $T_2$ (cross-correlation) and $T_3$ (initial model error) are much smaller; see §8 for their treatment. So $\mathbb{E}[\hat{B}_n - B_n] \approx \mathbb{E}[T_1]$.

Since $\hat{\sigma}^2 = \hat{A} - \hat{B}$ and $\hat{A}$ is unbiased:

$$\mathbb{E}[\hat{\sigma}^2] \approx \bar{\sigma}^2 - \mathbb{E}[T_1]$$

So $\hat{\sigma}^2$ **underestimates** the true variance by $\mathbb{E}[T_1]$ on average. This makes CIs too narrow and causes undercoverage.

---

## 3. Computing $T_1$: why the bias is anchored to early rounds

### 3.1 Per-step variance

Define the martingale differences $\Delta_s = \phi_s - \theta$ and the per-step conditional variance:

$$v_s := \mathbb{E}[\Delta_s^2 \mid \mathcal{F}_{s-1}] = \underbrace{\frac{1}{N^2}\sum_{i \notin O_{s-1}} \frac{(y_i - \hat{f}_i^{(s-1)})^2}{q_s(i)}}_{A_s} - \underbrace{\frac{1}{N^2}\left(\sum_{i \notin O_{s-1}}(y_i - \hat{f}_i^{(s-1)})\right)^2}_{B_s}$$

This is the single-step analogue of the Theorem 4 variance formula. Note that $v_s$ is $\mathcal{F}_{s-1}$-measurable (a random variable), but $\mathbb{E}[v_s]$ is a deterministic function of $s$.

### 3.2 Variance of the running average

The running average $\hat{\theta}_{t-1} = \frac{1}{t-1}\sum_{s=1}^{t-1}\phi_s$ satisfies $\hat{\theta}_{t-1} - \theta = \frac{1}{t-1}\sum_{s=1}^{t-1}\Delta_s$. By the martingale orthogonality ($\mathbb{E}[\Delta_s \Delta_{s'}] = 0$ for $s \neq s'$):

$$\text{Var}(\hat{\theta}_{t-1}) = \mathbb{E}[(\hat{\theta}_{t-1} - \theta)^2] = \frac{1}{(t-1)^2}\sum_{s=1}^{t-1}\mathbb{E}[v_s]$$

Note: $v_1$ contributes to $\text{Var}(\hat{\theta}_{t})$ for **every** $t \geq 2$. But $v_{t-1}$ contributes only to $\text{Var}(\hat{\theta}_{t})$ for that single $t$. So $\text{Var}(\hat{\theta}_{t-1})$ is dominated by the early-round variances.

### 3.3 Summing to get $T_1$

$$\mathbb{E}[T_1] = \frac{1}{n}\sum_{t=2}^n \text{Var}(\hat{\theta}_{t-1}) = \frac{1}{n}\sum_{t=2}^n \frac{1}{(t-1)^2}\sum_{s=1}^{t-1}\mathbb{E}[v_s]$$

This is a double sum over the triangle $1 \leq s \leq t-1$, $2 \leq t \leq n$, i.e., $1 \leq s < t \leq n$. We exchange the order of summation. For a fixed $s$, the index $t$ ranges from $s+1$ to $n$, and the $t$-dependent factor is $1/(t-1)^2$. Substituting $k = t-1$:

$$\mathbb{E}[T_1] = \frac{1}{n}\sum_{s=1}^{n-1}\mathbb{E}[v_s] \sum_{k=s}^{n-1}\frac{1}{k^2}$$

Define $w_s := \sum_{k=s}^{n-1} \frac{1}{k^2}$. Then:

$$\boxed{\mathbb{E}[T_1] = \frac{1}{n}\sum_{s=1}^{n-1}\mathbb{E}[v_s] \cdot w_s}$$

This is **exact**. The weights $w_s$ are:

- $w_1 = \sum_{k=1}^{n-1} 1/k^2 \approx \pi^2/6 \approx 1.645$ (for large $n$)
- $w_s \approx 1/s$ for moderate $s$ (since $\sum_{k=s}^{\infty} 1/k^2 \approx 1/s$)
- $w_{n-1} = 1/(n-1)^2 \approx 0$

So $\mathbb{E}[T_1]$ is a weighted average of the per-step variances, with weights $\sim 1.6$ for early rounds decaying to $\sim 0$ for late rounds. The bias is **anchored to the early rounds**.

---

## 4. Coverage depends on the relative bias

### 4.1 Why relative, not absolute

The CI is $\hat{\theta}_n \pm 1.96 \cdot \hat{\sigma}/\sqrt{n}$. Coverage is 95% when $\hat{\sigma} = \bar{\sigma}$. What matters is the fractional error:

$$\delta := \frac{\bar{\sigma}^2 - \mathbb{E}[\hat{\sigma}^2]}{\bar{\sigma}^2} = \frac{\mathbb{E}[T_1]}{\bar{\sigma}^2}$$

If $\hat{\sigma}^2$ is too small by fraction $\delta$, then $\hat{\sigma} \approx \bar{\sigma}\sqrt{1-\delta} \approx \bar{\sigma}(1-\delta/2)$, and the CI is a fraction $\delta/2$ too narrow. Coverage drops by:

$$\text{coverage} \approx 0.95 - 2 \cdot 1.96 \cdot \varphi(1.96) \cdot \frac{\delta}{2} = 0.95 - 0.114 \cdot \delta$$

where $\varphi(1.96) \approx 0.0584$ is the standard normal PDF. So a relative bias of $\delta = 0.22$ (MMLU-Pro at 25%) gives coverage $\approx 0.95 - 0.114 \times 0.22 = 0.925$, close to the observed 0.917 (the gap is a higher-order correction).

### 4.2 The relative bias as a ratio of two sums

The true variance of the estimator is also a sum over per-step variances, but with **uniform** weights:

$$\bar{\sigma}^2 = \frac{1}{n}\sum_{s=1}^n \mathbb{E}[v_s]$$

So the relative bias is:

$$\delta = \frac{\mathbb{E}[T_1]}{\bar{\sigma}^2} = \frac{\sum_{s=1}^{n-1}\mathbb{E}[v_s] \cdot w_s}{\sum_{s=1}^{n}\mathbb{E}[v_s]}$$

(the $1/n$ cancels). This is a ratio of two sums over the same per-step variances $\mathbb{E}[v_s]$, but with different weighting: the numerator uses early-heavy weights $w_s$, the denominator uses uniform weights (all equal to 1).

### 4.3 Stationary benchmark

If $\mathbb{E}[v_s] = c$ for all $s$ (constant variance profile), both sums simplify:

$$\delta_{\text{stat}} = \frac{c \sum_{s=1}^{n-1} w_s}{c \cdot n} = \frac{1}{n}\sum_{s=1}^{n-1} w_s$$

The inner double sum can be evaluated by exchanging: $\sum_{s=1}^{n-1} w_s = \sum_{s=1}^{n-1}\sum_{k=s}^{n-1} 1/k^2 = \sum_{k=1}^{n-1} k/k^2 = \sum_{k=1}^{n-1} 1/k = H_{n-1}$, where the step uses the fact that each $1/k^2$ is counted $k$ times (for $s = 1, \ldots, k$).

So $\delta_{\text{stat}} = H_{n-1}/n \approx \ln(n)/n$. For $n = 3000$: $\delta_{\text{stat}} \approx 8/3000 \approx 0.003$. This is tiny — a stationary martingale has negligible $\hat{B}_n$ bias.

---

## 5. How $\tau$ controls the relative bias: rigorous analysis

We now prove that lower $\tau$ leads to larger $\delta$, and that higher budget amplifies this effect.

### 5.1 $\delta$ exceeds the stationary baseline when the profile is decreasing

**Proposition 1 (Chebyshev's sum inequality).** If $\mathbb{E}[v_1] \geq \mathbb{E}[v_2] \geq \cdots \geq \mathbb{E}[v_{n-1}]$ (decreasing profile), then:

$$\delta \;\geq\; \frac{H_{n-1}}{n} \cdot \frac{\sum_{s=1}^{n-1}\mathbb{E}[v_s]}{\sum_{s=1}^{n}\mathbb{E}[v_s]}$$

with equality if and only if all $\mathbb{E}[v_s]$ are equal (stationary profile).

*Proof.* Since $\mathbb{E}[v_s]$ and $w_s$ are both decreasing sequences (co-monotone), Chebyshev's sum inequality gives:

$$(n-1)\sum_{s=1}^{n-1}\mathbb{E}[v_s]\,w_s \;\geq\; \left(\sum_{s=1}^{n-1}\mathbb{E}[v_s]\right)\left(\sum_{s=1}^{n-1}w_s\right)$$

The right-hand side involves $\sum_{s=1}^{n-1} w_s = \sum_{s=1}^{n-1}\sum_{k=s}^{n-1}\frac{1}{k^2} = \sum_{k=1}^{n-1}\frac{k}{k^2} = H_{n-1}$, where we exchanged the summation order (each $1/k^2$ is counted $k$ times, for $s = 1, \ldots, k$). So:

$$\sum_{s=1}^{n-1}\mathbb{E}[v_s]\,w_s \;\geq\; \frac{H_{n-1}}{n-1}\sum_{s=1}^{n-1}\mathbb{E}[v_s]$$

Dividing by $\sum_{s=1}^{n}\mathbb{E}[v_s]$:

$$\delta = \frac{\sum \mathbb{E}[v_s]\,w_s}{\sum_{s=1}^{n}\mathbb{E}[v_s]} \;\geq\; \frac{H_{n-1}}{n-1}\cdot\frac{\sum_{s=1}^{n-1}\mathbb{E}[v_s]}{\sum_{s=1}^{n}\mathbb{E}[v_s]} \;\approx\; \frac{H_{n-1}}{n} \qquad \square$$

The last approximation uses $\sum_{s=1}^{n-1}\mathbb{E}[v_s] / \sum_{s=1}^n \mathbb{E}[v_s] \approx (n-1)/n$ (the last term contributes little to the sum).

**Interpretation.** The stationary baseline $H_{n-1}/n \approx \ln(n)/n$ is a *lower bound* on $\delta$ for any decreasing profile. Every departure from stationarity increases $\delta$ above this baseline. The question is: how much above?

### 5.2 Decomposition: stationary part + covariance excess

To quantify the gap, decompose $\delta$ using the covariance. Define $\bar{v} = \frac{1}{n-1}\sum_{s=1}^{n-1}\mathbb{E}[v_s]$ and $\bar{w} = \frac{1}{n-1}\sum_{s=1}^{n-1} w_s = \frac{H_{n-1}}{n-1}$. Then:

$$\sum_{s=1}^{n-1}\mathbb{E}[v_s]\,w_s = (n-1)\left(\bar{v}\,\bar{w} + \text{Cov}(\mathbb{E}[v], w)\right)$$

where $\text{Cov}(\mathbb{E}[v], w) = \frac{1}{n-1}\sum_{s=1}^{n-1}\mathbb{E}[v_s]\,w_s - \bar{v}\,\bar{w}$ is the empirical covariance over $s \in \{1, \ldots, n-1\}$. Dividing by $\sum_{s=1}^n \mathbb{E}[v_s] \approx n\bar{v}$:

$$\boxed{\delta \;=\; \underbrace{\frac{H_{n-1}}{n}}_{\text{stationary baseline}} \;+\; \underbrace{\frac{(n-1)\,\text{Cov}(\mathbb{E}[v],\, w)}{n\,\bar{v}}}_{\text{non-stationarity excess}}}$$

By Chebyshev's sum inequality, the covariance is non-negative when $\mathbb{E}[v_s]$ is decreasing, so the excess is $\geq 0$.

The covariance satisfies $\text{Cov}(\mathbb{E}[v], w) \leq \sigma_v \cdot \sigma_w$, where $\sigma_v$ and $\sigma_w$ are the standard deviations of $\mathbb{E}[v_s]$ and $w_s$ over $s \in \{1, \ldots, n-1\}$. Since $\sigma_w$ depends only on $n$ (the weights $w_s$ are fixed), the excess is controlled by:

$$\text{excess} \;\leq\; \frac{(n-1)\,\sigma_v\,\sigma_w}{n\,\bar{v}} \;=\; \frac{\sigma_v}{\bar{v}} \cdot \frac{(n-1)\sigma_w}{n}$$

The factor $\sigma_v / \bar{v}$ is the **coefficient of variation** of the profile — a dimensionless measure of how much $\mathbb{E}[v_s]$ varies across rounds. **This is the quantity that $\tau$ controls.**

### 5.3 The floor bound: $\tau$ caps the per-step variance

**Proposition 2.** The sampling distribution $q_s(i) = (1-\tau)\,h_s(i)/\|h_s\|_1 + \tau/(N-s+1)$ satisfies:

$$q_s(i) \;\geq\; \frac{\tau}{N - s + 1}$$

and consequently:

$$v_s \;\leq\; A_s \;\leq\; \frac{1}{\tau}\,A_s^{\,\text{unif}}$$

where $A_s^{\,\text{unif}} = \frac{N-s+1}{N^2}\sum_{i \notin O_{s-1}}(y_i - \hat{f}_i)^2$ is the per-step $A$ under uniform sampling.

*Proof.* The floor follows from $h_s(i) \geq 0$. Then $1/q_s(i) \leq (N-s+1)/\tau$, so:

$$A_s = \frac{1}{N^2}\sum_{i \notin O_{s-1}}\frac{(y_i - \hat{f}_i)^2}{q_s(i)} \;\leq\; \frac{N-s+1}{\tau N^2}\sum_{i \notin O_{s-1}}(y_i - \hat{f}_i)^2 = \frac{1}{\tau}\,A_s^{\,\text{unif}}$$

and $v_s = A_s - B_s \leq A_s$ since $B_s \geq 0$. $\square$

**Consequence for $\sigma_v/\bar{v}$.** The bound says each $v_s$ lies in $[0,\; A_s^{\,\text{unif}}/\tau]$. The range of $v_s$ — and hence $\sigma_v$ — scales as $1/\tau$. At $\tau = 1$, $v_s = v_s^{\,\text{unif}}$ exactly, and the profile varies only through model improvement. At $\tau = 0.05$, the upper bound is $20\times$ larger, allowing much more variation and hence larger $\sigma_v/\bar{v}$.

### 5.4 The bound is achieved at early rounds with poor scoring

The bound $v_s \leq A_s^{\,\text{unif}}/\tau$ is not just a loose ceiling — it is approximately achieved whenever items with large residuals $(y_i - \hat{f}_i)^2$ receive scores near the floor.

**Proposition 3.** Suppose at step $s$ there exists a set $S \subset \{i \notin O_{s-1}\}$ of items with $h_s(i) = 0$ (the scoring function assigns them zero score). Then $q_s(i) = \tau/(N-s+1)$ for all $i \in S$, and:

$$A_s \;\geq\; \frac{N - s + 1}{\tau N^2}\sum_{i \in S}(y_i - \hat{f}_i)^2$$

If the set $S$ contains a fraction $\alpha$ of the total squared residuals ($\sum_{i \in S} r_i^2 = \alpha \sum_{i \notin O_{s-1}} r_i^2$), then:

$$A_s \;\geq\; \frac{\alpha}{\tau}\,A_s^{\,\text{unif}}$$

*Proof.* Direct substitution of $q_s(i) = \tau/(N-s+1)$ for $i \in S$. $\square$

At **early rounds** (small $s$), the model $\hat{f}$ is poor and the scoring function $h_s$ (which is based on $\hat{f}$) is unreliable. Items where the model is confidently wrong — large $(y_i - \hat{f}_i)^2$ but the model predicts well, so $h_s$ assigns a low score — land near the floor. With many such items, $\alpha$ is substantial and $A_s \approx A_s^{\,\text{unif}}/\tau$.

At **late rounds** (large $s$), the model is accurate, the scoring function reliably identifies high-residual items and assigns them high scores ($q_s(i) \gg \tau/(N-s+1)$), so these items do not hit the floor. The bound is loose, and $A_s \ll A_s^{\,\text{unif}}/\tau$.

**This is the mechanism:** low $\tau$ inflates $v_s$ at early rounds (where scoring is poor and the floor is binding) but not at late rounds (where scoring is good and the floor is irrelevant). The resulting contrast is the steep profile that drives up $\sigma_v/\bar{v}$ and hence $\delta$.

### 5.5 Sensitivity of $A_s$ to $\tau$: the derivative

To see precisely how $\tau$ affects each step, differentiate $A_s$ with respect to $\tau$. Write $r_i = y_i - \hat{f}_i$, $p_i = h_s(i)/\|h_s\|_1$ (the scoring probability), and $u = 1/(N-s+1)$ (the uniform probability). Then $q_s(i) = (1-\tau)p_i + \tau u$, and:

$$\frac{\partial A_s}{\partial \tau} = -\frac{1}{N^2}\sum_{i \notin O_{s-1}} \frac{r_i^2\,(u - p_i)}{q_s(i)^2}$$

This sum has **mixed signs**:

- Items with $p_i < u$ (under-scored relative to uniform): contribute $\frac{\partial A_s}{\partial \tau} < 0$, i.e., increasing $\tau$ **decreases** their contribution to $A_s$. These are items the scoring misses — exactly the items that get inflated when $\tau$ is low.

- Items with $p_i > u$ (over-scored): contribute $\frac{\partial A_s}{\partial \tau} > 0$, i.e., increasing $\tau$ **increases** their contribution. These are items the scoring correctly targets.

**At early rounds** (poor scoring): many high-residual items have $p_i < u$ (the scoring misses them). The negative terms dominate, so $\partial A_s / \partial \tau < 0$: **increasing $\tau$ decreases $A_s$**. Equivalently, **decreasing $\tau$ inflates $A_s$** at early rounds.

**At late rounds** (good scoring): high-residual items have $p_i > u$ (the scoring finds them). The positive terms dominate, so $\partial A_s / \partial \tau > 0$: **increasing $\tau$ increases $A_s$**. Equivalently, **decreasing $\tau$ reduces $A_s$** at late rounds — this is the ESS benefit of adaptive sampling.

So decreasing $\tau$ simultaneously inflates early $v_s$ and deflates late $v_s$, steepening the profile in both directions. This increases $\sigma_v/\bar{v}$, increases the covariance excess, and increases $\delta$.

### 5.6 Putting it together: $\delta$ as a function of $\tau$

Combining Proposition 2 and the derivative analysis:

1. At $\tau = 1$: $q_s(i) = u$ for all $i$, so $A_s = A_s^{\,\text{unif}}$ at every step. The profile is $v_s = v_s^{\,\text{unif}}$, which varies only through model improvement. The coefficient of variation $\sigma_v/\bar{v}$ is small, and $\delta \approx H_{n-1}/n$.

2. As $\tau$ decreases below 1: at each step, $A_s$ moves toward $\frac{1}{N^2}\sum r_i^2/p_i$. By §5.5, early-round $A_s$ increases (poor scoring) and late-round $A_s$ decreases (good scoring). Both effects steepen the profile: $\sigma_v/\bar{v}$ grows, and by §5.2, $\delta$ increases.

3. At $\tau \to 0$: the floor vanishes, $q_s(i) \to p_i$, and items with small $p_i$ and large $r_i$ produce $r_i^2/p_i \to \infty$. The early-round $v_s$ diverge, $\sigma_v/\bar{v} \to \infty$, and $\delta \to \infty$. (In practice, $\tau > 0$ always, but this shows the limiting behavior.)

---

## 6. Why higher budget amplifies the effect when $\tau$ is small

The analysis above holds at any fixed budget. We now explain why the problem **worsens with budget** when $\tau$ is small, but **improves with budget** when $\tau$ is large.

### 6.1 For a fixed profile, $\delta$ does not grow

**Proposition 4.** If $\mathbb{E}[v_s]$ at step $s$ does not depend on the total budget $n$ (the per-step variance is determined solely by the history up to step $s$), then $\delta$ converges as $n \to \infty$ and is eventually non-increasing.

*Proof sketch.* Both $\sum_{s=1}^{n-1} \mathbb{E}[v_s]\,w_s$ and $\sum_{s=1}^n \mathbb{E}[v_s]$ are partial sums of convergent series (since $\mathbb{E}[v_s] \to 0$ and $w_s \to 0$). Their ratio converges to a finite limit. For a power-law profile $v_s \sim s^{-\alpha}$ with $\alpha \in (0,1)$, the ratio scales as $n^{\alpha-1} \to 0$. $\square$

So with a fixed profile, **more budget always helps**: $\delta$ decreases, coverage improves. This is the regime $\tau = 1$ lives in — the profile doesn't change with budget, and adding more rounds only improves things.

### 6.2 The profile changes with budget when $\tau < 1$

When $\tau < 1$, the variance profile at step $s$ depends on the total budget $n$ through the **scheduling**. The scoring function uses blending parameters that depend on the normalized progress $s/(\rho \cdot n_B)$. At fixed step $s$:

$$\frac{s}{\rho \cdot n_B^{(1)}} > \frac{s}{\rho \cdot n_B^{(2)}} \quad \text{when } n_B^{(2)} > n_B^{(1)}$$

With a larger budget, the same step $s$ is at an earlier stage of the schedule. The scoring is more aggressive (more concentrated $p_i$), more items hit the floor $q_s(i) \approx \tau/(N-s+1)$, and by Proposition 3, $A_s$ is closer to $A_s^{\,\text{unif}}/\tau$.

Meanwhile, the late-round $v_n$ gets **smaller** at higher budgets: by step $n$, the model has seen $n-1$ items, so predictions improve with budget, residuals shrink, and $v_n$ decreases.

Both effects steepen the profile: early $v_s$ increases, late $v_s$ decreases. By the decomposition in §5.2, this increases $\sigma_v/\bar{v}$ and hence $\delta$.

### 6.3 With $\tau$ near 1, the budget effect vanishes

When $\tau$ is large, the floor $\tau/(N-s+1)$ dominates the sampling distribution at all steps. By Proposition 2, $v_s \leq A_s^{\,\text{unif}}/\tau$, and this bound is tight: with large $\tau$, $q_s(i)$ is close to uniform for all items regardless of the scoring. The profile is approximately $v_s \approx v_s^{\,\text{unif}}$, which changes only through model improvement — the same at all budgets.

So the profile does not steepen with budget, $\sigma_v/\bar{v}$ stays small, and $\delta \approx H_{n-1}/n$ **decreases** with budget. Coverage improves monotonically, just as with WR FAQ.

### 6.4 Summary: the $\tau \times$ budget interaction

| | Low budget | High budget |
|:---|:---:|:---:|
| **High $\tau$** | CLT error dominates, moderate coverage | CLT improves, $\delta$ small and decreasing → good coverage |
| **Low $\tau$** | CLT error dominates, same as high $\tau$ | Profile steepens via scheduling, $\delta$ grows → undercoverage |

The $\tau$ effect is invisible at low budget (where the CLT term $C/\sqrt{n}$ dominates regardless) and decisive at high budget (where the CLT term has converged and $\delta$ determines coverage).

---

## 7. Empirical confirmation

### 7.1 Coverage across $\tau$ and budget

**MMLU-Pro:**

| Budget | $\tau = 0.05$ (tuned) | $\tau = 0.25$ | $\tau = 0.50$ | WR FAQ |
|:---:|:---:|:---:|:---:|:---:|
| 2.5% | 0.941 | 0.941 | 0.944 | 0.940 |
| 10% | 0.943 | 0.947 | 0.947 | 0.944 |
| 17.5% | 0.938 | 0.948 | 0.949 | 0.947 |
| 25% | **0.917** | **0.948** | **0.950** | 0.948 |

**ESS at 25% budget:**

| Dataset | $\tau = 0.05$ | $\tau = 0.25$ | $\tau = 0.50$ | WR FAQ |
|:---:|:---:|:---:|:---:|:---:|
| MMLU-Pro | 7.31x | 6.34x | 5.66x | 4.91x |
| BBH suite | 5.73x | 5.42x | 5.01x | 4.13x |

### 7.2 Relative bias across $\tau$ (MMLU-Pro 25%)

| $\tau$ | $\delta$ | $\delta_{\text{stat}} = H_{n-1}/n$ | $\delta / \delta_{\text{stat}}$ |
|:---:|:---:|:---:|:---:|
| 0.05 | 0.220 | 0.003 | 73x |
| 0.25 | ~0.012 | 0.003 | 4x |
| 0.50 | ~0.003 | 0.003 | 1x |

At $\tau = 0.50$, $\delta$ matches the stationary baseline exactly. The 70x reduction from $\tau = 0.05$ to $\tau = 0.50$ confirms the theory.

### 7.3 Key observations

These match the predictions of §5–6:
- At low budget (2.5%), coverage is ~0.941 regardless of $\tau$: the CLT term dominates (§6.4).
- At high budget with high $\tau$: coverage reaches 0.950, matching the prediction $0.95 - 0.114 \times H_{n-1}/n \approx 0.9497$ (§6.3).
- At high budget with low $\tau$: the profile has steepened (§6.2), $\delta = 0.22$, and coverage drops to $0.95 - 0.114 \times 0.22 \approx 0.925$, close to the observed 0.917.

---

## 8. The full bias decomposition (for completeness)

The exact decomposition from the paper is $\hat{B}_n - B_n = T_1 + T_2 - T_3$, where:

$$T_1 = \frac{1}{n}\sum_{t=2}^n (\hat{\theta}_{t-1} - \theta)^2 \geq 0 \qquad \text{(MSE of running average)}$$

$$T_2 = \frac{2}{n}\sum_{t=2}^n (\theta - \psi_t)(\hat{\theta}_{t-1} - \theta) \qquad \text{(cross-correlation)}$$

$$T_3 = \frac{1}{n}\left(\theta - \frac{1}{N}\sum_i \hat{f}^{(0)}(x_i)\right)^2 \geq 0 \qquad \text{(initial model error)}$$

**$T_3$ is deterministic** (depends only on the initial model) and acts as a partial offset: it makes $\hat{B}_n$ overshoot $B_n$ less (since it enters with a minus sign). In normalized units, $T_3/\bar{\sigma}^2 = (\theta - \bar{\psi}_1)^2 / (n\bar{\sigma}^2)$, which is small for FAQ (the initial model is decent).

**$T_2$ is a cross-correlation** between the estimation error $\hat{\theta}_{t-1} - \theta$ and the prediction error $\theta - \psi_t$. Each step changes $\psi_t$ by one item out of $N$, so the per-step correlation is $O(1/N)$. Summing over $t$: $|\mathbb{E}[T_2]| \leq C\sqrt{\bar{\sigma}^2}/N$. Since $N \sim 12000 \gg \sqrt{n} \leq 55$, this is negligible compared to $\mathbb{E}[T_1]$.

**$T_1$ dominates.** The full relative bias is:

$$\frac{\bar{\sigma}^2 - \mathbb{E}[\hat{\sigma}^2]}{\bar{\sigma}^2} = \frac{\mathbb{E}[T_1] - \mathbb{E}[T_3] + \mathbb{E}[T_2]}{\bar{\sigma}^2} \approx \frac{\mathbb{E}[T_1]}{\bar{\sigma}^2}$$

with the approximation being accurate to within a few percent of coverage.

---

## 9. The trade-off and the resolution

$\tau$ mediates a trade-off:

$$\tau \;\uparrow \quad\implies\quad \text{less amplification} \quad\implies\quad \text{flatter profile} \quad\implies\quad \text{smaller } \delta \quad\implies\quad \text{better coverage, but lower ESS}$$

The variance estimator $\hat{\sigma}^2 = \hat{A} - \hat{B}$ is designed for approximately stationary martingales. When $\tau$ is low, the martingale is highly non-stationary, and $\hat{B}$ systematically overshoots.

**The bias-corrected estimator** ($\hat{\sigma}^2_{\text{cor}} = \hat{A} - \hat{B} + \hat{C}$) breaks this trade-off. It estimates the noise floor $\text{Var}(\hat{\theta}_{t-1})$ at each step and subtracts it from $\hat{B}$, using:

$$\hat{C}_n = \frac{1}{nN^2}\sum_{s=1}^{n-1} a_s^2 \cdot w_s, \qquad a_s = \frac{y_{I_s} - \hat{f}^{(s-1)}(x_{I_s})}{q_s(I_s)}$$

This uses the same weights $w_s$ as in $\mathbb{E}[T_1]$. By construction, $\mathbb{E}[\hat{C}_n] \approx \mathbb{E}[T_1]$ (with a small conservative residual), so the bias cancels regardless of profile steepness. This allows low $\tau$ (high ESS) with valid coverage: the best of both worlds.
