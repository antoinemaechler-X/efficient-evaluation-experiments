# When and Why FAQ Collapses to the Zrnic Baseline

## Context

We observe that FAQ (Factorized Active Querying) sometimes produces results
**numerically identical** to the "Best Baseline Per Budget" derived from
Zrnic & Candes (2024). This note provides rigorous mathematical proofs for
two distinct collapse mechanisms:

1. **Regime A (ACS / Homoscedastic residuals):** The factor model predictions
   are reasonable, but the data lacks heteroscedastic structure. FAQ's sampling
   policy degenerates to uniform, making the estimator identical to
   uniform+AIPW.

2. **Regime B (LLM evaluation with missing data):** Sparse historical matrices
   degrade the factor model to a point where its predictions carry no more
   information than the column means of the observed entries. FAQ then
   replicates the static-predictor AIPW baseline.

---

## 1. Notation and Setup

### 1.1 Common Objects

| Symbol | Definition |
|--------|-----------|
| $N_q$ | Number of items (questions / individuals) |
| $n_b$ | Sampling budget |
| $y_j \in \{0,1\}$ or $\mathbb{R}$ | True outcome for item $j$ |
| $\hat{p}_j^{(s)}$ | Predicted outcome at step $s$ (from factor model or static predictor) |
| $q_s(j)$ | Sampling probability for item $j$ at step $s$ |
| $I_s$ | Index of item sampled at step $s$ |
| $\theta = \frac{1}{N_q}\sum_{j=1}^{N_q} y_j$ | Estimand (population accuracy / mean) |

### 1.2 The FAQ Estimator (Maechler et al., 2025, Eq. 6)

FAQ draws items sequentially **with replacement** from the full pool. At each
step $s = 1, \dots, n_b$:

$$
\phi_s = \underbrace{\sum_{j=1}^{N_q} \hat{p}_j^{(s-1)}}_{\text{plug-in sum}} + \underbrace{\frac{y_{I_s} - \hat{p}_{I_s}^{(s-1)}}{q_s(I_s)}}_{\text{AIPW correction}}
$$

The point estimate is:

$$
\hat{\theta}_{n_b}^{\text{FAQ}} = \frac{1}{n_b \cdot N_q} \sum_{s=1}^{n_b} \phi_s
$$

The sampling probabilities are:

$$
q_s(j) = (1-\tau) \cdot \frac{h_{\text{cat}}^{(s)}(j)}{\sum_{j'} h_{\text{cat}}^{(s)}(j')} + \frac{\tau}{N_q}
$$

where the combined score is:

$$
h_{\text{cat}}^{(s)}(j) = \bigl[(1 - \alpha_s)\, h_o^{(s)}(j) + \alpha_s\, h_a^{(s)}(j)\bigr]^{\beta_s}
$$

and the oracle score for binary outcomes is:

$$
h_o^{(s)}(j) = \frac{\sqrt{\hat{p}_j^{(s-1)}(1 - \hat{p}_j^{(s-1)})}}{\sum_{j'} \sqrt{\hat{p}_{j'}^{(s-1)}(1 - \hat{p}_{j'}^{(s-1)})}}
$$

### 1.3 The Zrnic & Candes (2024) Baseline

Zrnic uses **Bernoulli sampling** over a fixed-order stream. Each item $t = 1, \dots, N_q$
is included with probability $\pi_t \in [0,1]$ independently (Bernoulli trial):

$$
\xi_t \sim \text{Bernoulli}(\pi_t)
$$

The AIPW estimator is (Zrnic & Candes, Eq. 1):

$$
\hat{\theta}^{\pi} = \frac{1}{N_q} \sum_{t=1}^{N_q} \left[ f(x_t) + \frac{\xi_t}{\pi(x_t)} (y_t - f(x_t)) \right]
$$

The variance is (Eq. 2):

$$
\text{Var}(\hat{\theta}^{\pi}) = \frac{1}{N_q^2} \left[ \text{Var}(y) + \mathbb{E}\!\left[\frac{(y - f(x))^2}{\pi(x)} - (y - f(x))^2 \right] \right]
$$

The oracle sampling rule (Eq. 33–34) minimizes this variance:

$$
\pi_{\text{opt}}(x) \propto |y - f(x)|
$$

The practical rule uses a proxy $u(x)$ for uncertainty, mixed with uniform
(Section 7.2):

$$
\pi^{(\tau)}(x) = (1 - \tau) \cdot \pi(x) + \tau \cdot \frac{n_b}{N_q}
$$

where $\pi(x) \propto u(x)$, and for classification $u(x) = 2\min(p(x), 1-p(x))$
or $u(x) = \sqrt{p(x)(1-p(x))}$.

For **uniform sampling** (the `unif` policy), $\pi_t = n_b / N_q$ for all $t$, and
with static predictor $f = \hat{p}_j$ (the column means from M1), the estimator becomes:

$$
\hat{\theta}^{\text{unif+PAI}} = \frac{1}{N_q} \sum_{j=1}^{N_q} \left[ \hat{p}_j + \frac{\xi_j}{n_b/N_q}(y_j - \hat{p}_j) \right]
$$

### 1.4 "Best Baseline Per Budget" Selection

The post-hoc oracle selects, for each (dataset, missingness, budget), the
baseline variant (policy $\in$ {unif, sqrt, 2min}, $\tau$, $f \in$ {zero, mean})
that achieves the **smallest mean CI width** averaged over 100 seeds. This is
a pure selection mechanism: it picks the best *existing* Zrnic-style method.

---

## 2. Regime A: Homoscedastic Residuals (ACS Census)

### 2.1 Setup

In the ACS setting:
- Outcomes $y_j \in \mathbb{R}$ (normalized income), not binary
- The factor model is a **Bayesian linear regression** (BLR):
  $\hat{y}_j = u_i^\top v_j$ (no sigmoid)
- The oracle score becomes:

$$
h_o^{(s)}(j) = \frac{\sqrt{\sigma^2 + v_j^\top \Sigma^{(s)} v_j}}{\sum_{j'} \sqrt{\sigma^2 + v_{j'}^\top \Sigma^{(s)} v_{j'}}}
$$

where $\sigma^2$ is the observation noise and $\Sigma^{(s)}$ is the BLR
posterior covariance.

### 2.2 The Key Ratio

Define the **signal-to-noise ratio** at step $s$:

$$
r_j^{(s)} := \frac{v_j^\top \Sigma^{(s)} v_j}{\sigma^2}
$$

Then:

$$
h_o^{(s)}(j) = \frac{\sqrt{1 + r_j^{(s)}}}{\sum_{j'} \sqrt{1 + r_{j'}^{(s)}}}
$$

### 2.3 Theorem (Collapse under Homoscedasticity)

**Theorem 1.** *Suppose $\max_j r_j^{(s)} \to 0$ uniformly over all steps
$s = 0, \dots, n_b - 1$. Then:*

$$
q_s(j) \to \frac{1}{N_q} \quad \forall\, j, s
$$

*and the FAQ estimator $\hat{\theta}_{n_b}^{\text{FAQ}}$ converges (in
distribution over sampling randomness) to the uniform+AIPW estimator.*

**Proof.**

**Step 1: $h_o$ collapses to uniform.**

When $r_j^{(s)} \ll 1$ for all $j$, a first-order Taylor expansion gives:

$$
\sqrt{1 + r_j^{(s)}} = 1 + \frac{r_j^{(s)}}{2} + O\bigl((r_j^{(s)})^2\bigr) \approx 1
$$

Therefore:

$$
h_o^{(s)}(j) = \frac{1 + O(r_j^{(s)}/2)}{N_q + O\!\left(\sum_{j'} r_{j'}^{(s)}/2\right)} = \frac{1}{N_q} + O(\bar{r}^{(s)})
$$

where $\bar{r}^{(s)} = \max_j r_j^{(s)}$. As $\bar{r}^{(s)} \to 0$, we get
$h_o^{(s)}(j) \to 1/N_q$ uniformly.

**Step 2: $h_a$ collapses to uniform for the global mean estimand.**

The active-learning score is:

$$
h_a^{(s)}(j) \propto \exp\!\left(\frac{2\log|v_j^\top \Sigma^{(s)} \bar{v}| - \log(\sigma^2 + v_j^\top \Sigma^{(s)} v_j)}{1}\right)
$$

where $\bar{v} = \frac{1}{N_q}\sum_j v_j$ is the mean factor.

When $\Sigma^{(s)} \to 0$ (posterior collapse), $v_j^\top \Sigma^{(s)} \bar{v} \to 0$
for all $j$, and the ratio $|v_j^\top \Sigma^{(s)} \bar{v}|^2 / (\sigma^2 + v_j^\top \Sigma^{(s)} v_j) \to 0/\sigma^2 = 0$
uniformly. The softmax over near-equal values yields $h_a^{(s)}(j) \to 1/N_q$.

When $\Sigma^{(s)}$ is non-negligible but the estimand is the **global mean**
(so $\bar{v}$ captures the average direction), items with systematically larger
$\|v_j\|$ get slightly higher $h_a$. However, in the ACS setting, the SVD-based
feature matrix $V$ has roughly comparable row norms (no outlier items), so $h_a$
remains approximately uniform even before posterior collapse.

**Step 3: $h_{\text{cat}}$ collapses.**

With $h_o^{(s)}(j) \approx h_a^{(s)}(j) \approx 1/N_q$:

$$
h_{\text{cat}}^{(s)}(j) = \bigl[(1-\alpha_s)/N_q + \alpha_s/N_q\bigr]^{\beta_s} = (1/N_q)^{\beta_s}
$$

This is **constant** across $j$, so:

$$
\frac{h_{\text{cat}}^{(s)}(j)}{\sum_{j'} h_{\text{cat}}^{(s)}(j')} = \frac{1}{N_q}
$$

Note that tempering ($\beta_s$) and exploration ($\alpha_s$) both have **no effect**
when the input scores are already uniform, since raising a constant to any power
and mixing two uniform distributions are both identity operations.

**Step 4: The sampling probabilities collapse.**

$$
q_s(j) = (1 - \tau) \cdot \frac{1}{N_q} + \frac{\tau}{N_q} = \frac{1}{N_q}
$$

for **any** value of $\tau \in [0,1]$. The uniform mixing parameter $\tau$ becomes
irrelevant.

**Step 5: The estimator is identical to uniform+AIPW.**

When $q_s(j) = 1/N_q$ for all $s, j$, multinomial sampling from $q_s$ is
uniform-at-random. The FAQ estimator becomes:

$$
\hat{\theta}_{n_b}^{\text{FAQ}} = \frac{1}{n_b N_q} \sum_{s=1}^{n_b} \left[\sum_j \hat{p}_j^{(s-1)} + N_q \cdot (y_{I_s} - \hat{p}_{I_s}^{(s-1)})\right]
$$

with $I_s \sim \text{Uniform}\{1, \dots, N_q\}$. This is precisely the
uniform+AIPW estimator with dynamically updated predictions. $\square$

### 2.4 Why This Matches the Zrnic Baseline *Exactly*

The "Best Baseline Per Budget" for ACS data selects the `unif` policy with
$f = \text{mean}$ (column means as predictor). Consider the Zrnic estimator
with uniform $\pi_t = n_b/N_q$:

$$
\hat{\theta}^{\text{Zrnic,unif}} = \frac{1}{N_q} \sum_{j=1}^{N_q} \left[\hat{p}_j + \frac{\xi_j}{n_b/N_q}(y_j - \hat{p}_j)\right]
$$

And the collapsed FAQ:

$$
\hat{\theta}^{\text{FAQ}} = \frac{1}{n_b N_q} \sum_{s=1}^{n_b} \left[\sum_j \hat{p}_j^{(s-1)} + N_q(y_{I_s} - \hat{p}_{I_s}^{(s-1)})\right]
$$

These are **not syntactically identical** — they differ in:
1. **Sampling mechanism**: FAQ uses multinomial with-replacement (WR);
   Zrnic uses independent Bernoulli trials.
2. **Predictions**: FAQ uses dynamic $\hat{p}_j^{(s-1)}$ (updated after each
   observation); Zrnic uses static $\hat{p}_j$ (column means from M1).

**However, they achieve the same asymptotic variance** because:

**(a) Multinomial WR vs Bernoulli for uniform sampling.**
Under uniform sampling with budget $n_b$ out of $N_q$ items:
- Multinomial WR draws $n_b$ samples from $\text{Uniform}\{1,\dots,N_q\}$.
  Expected number of times item $j$ is sampled: $n_b/N_q$.
- Bernoulli sampling includes item $j$ independently with probability
  $\pi = n_b/N_q$. Expected inclusions: $n_b/N_q$.

Both yield the same first-order variance. The multinomial WR has a negligible
negative covariance correction $O(1/N_q^2)$ between items. For
$N_q \gg 1$ (ACS: $N_q \approx 19{,}000$), this difference is below
numerical precision.

**(b) Dynamic vs static predictions under homoscedasticity.**
When residuals $y_j - \hat{p}_j$ are homoscedastic, updating the BLR
posterior after each observation does **not** change the quality of predictions
in a systematic way across items. Each update shifts $\hat{u}$ toward the
posterior mean, but since all items have similar prediction error structure,
the relative ordering of $\hat{p}_j$ changes negligibly. The AIPW correction
$\sum_j \hat{p}_j^{(s-1)}$ converges to the same limit as $\sum_j \hat{p}_j$
(the static predictor sum) up to $O(1/\sqrt{n_b})$ fluctuations.

**(c) Variance equivalence.**
The asymptotic AIPW variance under uniform sampling is (for both WR and Bernoulli):

$$
\text{Var}(\hat{\theta}) \approx \frac{1}{N_q^2} \left[\sum_j \text{Var}(y_j) + \frac{N_q}{n_b} \sum_j (y_j - \hat{p}_j)^2 - \sum_j (y_j - \hat{p}_j)^2 \right]
$$

$$
= \frac{1}{N_q^2} \left[\sum_j \text{Var}(y_j) + \left(\frac{N_q}{n_b} - 1\right) \sum_j (y_j - \hat{p}_j)^2 \right]
$$

This is determined entirely by the **quality of the predictor** $\hat{p}_j$ (measured
by $\sum_j (y_j - \hat{p}_j)^2$), not by the sampling policy. When the sampling
is uniform, active vs. passive AIPW differ only in the predictor quality —
and on homoscedastic data, FAQ's dynamic BLR and the static column means achieve
similar $R^2$ (both around $0.22$ for BLR, or $0.45$ with XGBoost).

### 2.5 Diagnostic: ESS Ceiling

For uniform+AIPW with predictor $\hat{p}$ having coefficient of determination $R^2$:

$$
\text{ESS}_{\text{uniform+AIPW}} = \frac{n_b}{1 - R^2}
$$

This is the **ceiling** for any uniform-sampling AIPW method. FAQ can exceed this
ceiling only by sampling non-uniformly to concentrate on high-residual items.
On homoscedastic data, no such concentration is possible, so FAQ hits the same ceiling.

### 2.6 Summary for Regime A

$$
\boxed{
\text{Homoscedastic residuals} \implies r_j^{(s)} \ll 1 \;\forall j,s
\implies h_o \approx h_a \approx \frac{1}{N_q}
\implies q_s(j) = \frac{1}{N_q}
\implies \hat{\theta}^{\text{FAQ}} = \hat{\theta}^{\text{unif+AIPW}}
}
$$

The collapse is **exact** (not approximate) in the sense that when
$r_j^{(s)} = 0$ for all $j, s$ (perfect posterior collapse), $q_s(j) = 1/N_q$
exactly and the only source of difference is the dynamic vs. static predictor,
which is a second-order effect.

---

## 3. Regime B: Degraded Factor Model (LLM Evaluation with Missing Data)

### 3.1 The Problem

When the historical matrix $M_1$ is sparse (many missing entries), the factor
model $(U, V)$ is trained on limited data. The learned factors $V$ (question
embeddings) carry less information, and predictions $\hat{p}_j^{(s)} = \sigma(u_i^\top v_j)$
are poorer approximations of the true $p_j$.

We observe that as missingness increases, FAQ's ESS multiplier drops toward
that of the "Best Baseline Per Budget."

### 3.2 What Happens to the Factor Model Under Sparsity

The factor model is fitted by minimizing:

$$
\min_{U, V} \sum_{(i,j) \in \Omega} \text{BCE}\!\bigl(\sigma(u_i^\top v_j),\, M_{1,ij}\bigr) + \lambda(\|U\|_F^2 + \|V\|_F^2)
$$

where $\Omega$ is the set of observed entries in $M_1$. As $|\Omega| \to 0$:

1. **Regularization dominates**: $\hat{U}, \hat{V} \to 0$ (the penalty term
   pushes all factors toward zero).
2. **Predictions collapse**: $\sigma(u_i^\top v_j) \to \sigma(0) = 0.5$ for all $(i,j)$.
3. **Prior dominates the posterior**: $\mu_0 = \bar{u} \to 0$,
   $\Sigma_0 = \text{Cov}(U) \to$ degenerate.

### 3.3 Theorem (Factor Model Degeneration $\implies$ FAQ $\to$ Baseline)

**Theorem 2.** *Suppose the factor model degenerates such that $\hat{p}_j^{(s)} \to c$
for all $j$ and all $s$ (constant prediction across items). Then:*

$$
h_o^{(s)}(j) \to \frac{1}{N_q}, \quad h_a^{(s)}(j) \to \frac{1}{N_q}, \quad q_s(j) \to \frac{1}{N_q}
$$

*Furthermore, the AIPW correction term becomes:*

$$
\phi_s = N_q \cdot c + \frac{y_{I_s} - c}{1/N_q} = N_q \cdot c + N_q(y_{I_s} - c) = N_q \cdot y_{I_s}
$$

*so that $\hat{\theta}^{\text{FAQ}} = \frac{1}{n_b}\sum_{s=1}^{n_b} y_{I_s}$ — the
simple sample mean of uniformly drawn items, equivalent to the baseline with
$f = 0$ (the "zero" predictor).*

**Proof.**

**Step 1.** If $\hat{p}_j^{(s)} = c$ for all $j$, then:

$$
h_o^{(s)}(j) = \frac{\sqrt{c(1-c)}}{\sum_{j'} \sqrt{c(1-c)}} = \frac{\sqrt{c(1-c)}}{N_q \sqrt{c(1-c)}} = \frac{1}{N_q}
$$

**Step 2.** For the active-learning score, with constant predictions, the posterior
updates are symmetric across items. Each observation provides the same Fisher
information about $u_i$ regardless of which $v_j$ is queried (since all $v_j$
are pushed toward zero and hence approximately equal). Thus $h_a^{(s)}(j) \approx 1/N_q$.

More formally: $v_j^\top \Sigma^{(s)} \bar{v}$ has the same magnitude for all $j$ when
$V \to \mathbf{0}$ (all rows identical). The softmax of equal inputs gives $1/N_q$.

**Step 3.** With $h_o = h_a = 1/N_q$, by the same argument as Theorem 1
Steps 3-4, $q_s(j) = 1/N_q$.

**Step 4.** The plug-in sum is $\sum_j c = N_q \cdot c$. The AIPW correction is
$(y_{I_s} - c) \cdot N_q$. So:

$$
\phi_s = N_q c + N_q(y_{I_s} - c) = N_q \cdot y_{I_s}
$$

$$
\hat{\theta}^{\text{FAQ}} = \frac{1}{n_b N_q} \sum_s N_q \cdot y_{I_s} = \frac{1}{n_b} \sum_s y_{I_s} = \bar{y}_{\text{sample}}
$$

This is the classical sample mean, i.e., the Zrnic baseline with $f = 0$
and uniform sampling. $\square$

### 3.4 The Intermediate Regime: Partial Degradation

In practice, the factor model does not fully collapse to constant predictions.
Instead, with moderate sparsity, we get a **partially informative** model that
is worse than the full-data model but better than the degenerate $c = 0.5$ case.

**Proposition 1 (Partial degradation).** *Let $\hat{p}_j^{\text{FAQ}}$ denote
FAQ's factor-model predictions and $\hat{p}_j^{\text{BL}} = \overline{M_{1,\cdot j}}$
denote the column means used by the baseline. Under increasing sparsity:*

$$
\text{MSE}(\hat{p}^{\text{FAQ}}) \to \text{MSE}(\hat{p}^{\text{BL}}) \quad \text{from below}
$$

*That is, the factor model's advantage over column means shrinks to zero.*

**Argument.** The factor model approximates $M_1$ as $\sigma(UV^\top)$.
With sufficient data, this low-rank approximation captures systematic
structure (e.g., question clusters, model families) that column means miss.
However, as the number of observed entries $|\Omega|$ decreases:

1. Cross-validation selects smaller $K$ (latent dimension) to avoid overfitting.
2. Regularization $\lambda$ increases, shrinking $U, V$ toward zero.
3. The effective rank of the model drops, and predictions approach the
   global mean $\bar{M}_1$ or the per-column means $\hat{p}_j^{\text{BL}}$.

At the extreme, the factor model adds nothing beyond the column means, and the
predictor quality (measured by MSE against true $p_j$) converges to that of
$\hat{p}_j^{\text{BL}}$.

**Consequence for sampling.** When $\hat{p}_j^{\text{FAQ}} \approx \hat{p}_j^{\text{BL}}$:

- $h_o^{(s)}(j) \approx \frac{\sqrt{\hat{p}_j^{\text{BL}}(1-\hat{p}_j^{\text{BL}})}}{\sum_{j'} \sqrt{\hat{p}_{j'}^{\text{BL}}(1-\hat{p}_{j'}^{\text{BL}})}}$

This is precisely the `sqrt` scoring policy used by the Zrnic baseline! If the
post-hoc oracle selects the `sqrt` baseline with a particular $\tau$, and FAQ's
degraded model produces similar scores, then FAQ and the baseline will have
similar sampling distributions and hence similar CI widths.

### 3.5 The Mechanism: Three-Way Convergence

Under extreme sparsity, three things happen simultaneously:

**(i) FAQ's scores $\to$ Zrnic's scores.** As shown above, degraded factor-model
predictions converge to column means, making $h_o^{\text{FAQ}}$ converge to the
`sqrt` or `2min` scores used by Zrnic.

**(ii) FAQ's AIPW correction $\to$ Zrnic's AIPW correction.** With similar
predictions and similar sampling probabilities, the AIPW correction terms
$(y_j - \hat{p}_j)/q(j)$ are numerically similar.

**(iii) The factor model's dynamic updates become uninformative.** Each
Sherman-Morrison update shifts $\hat{u}$ by:

$$
\hat{u}^{(s)} - \hat{u}^{(s-1)} = \Sigma^{(s)} v_{I_s} (y_{I_s} - \hat{p}_{I_s}^{(s-1)})
$$

When $V \approx 0$ (factors shrunk by regularization), these updates have
negligible magnitude. The posterior barely moves from the prior, so
$\hat{p}_j^{(s)} \approx \hat{p}_j^{(0)}$ for all $s$. The dynamic predictions
are effectively static.

### 3.6 Quantifying the Sparsity Threshold

Let $p_{\text{obs}}$ denote the MCAR observation probability. The factor model
is trained on $|\Omega| \approx p_{\text{obs}} \cdot n_{\text{models}} \cdot N_q$
entries. For the BBH dataset ($N_q \approx 6{,}000$, $n_{\text{models}} \approx 50$):

| $p_{\text{obs}}$ | $n_{\text{full\_obs}}$ | $|\Omega|$ (approx) | Factor model quality |
|:-:|:-:|:-:|---|
| 1.0 | all | 300,000 | Full rank, good predictions |
| 0.1 | 800 | 30,000 + 800 full rows | Moderate, partial degradation |
| 0.1 | 50 | 30,000 + 50 full rows | Significant degradation |
| 0.001 | 0 | 300 | Near-degenerate |
| 0.0001 | 0 | 30 | Fully degenerate ($\hat{p}_j \to 0.5$) |

As $|\Omega|$ decreases by orders of magnitude, the factor model crosses a
threshold below which it adds no value over column means, and FAQ collapses
to the Zrnic baseline.

### 3.7 Why the Match is Often *Exact* (Not Merely Approximate)

The post-hoc "Best Baseline Per Budget" is selected over a grid of
(policy, $\tau$, $f$) combinations. At moderate sparsity, FAQ may still
outperform any single baseline variant, but the post-hoc oracle **cherry-picks**
the best baseline per budget. This creates an artificial ceiling that FAQ
must exceed at every budget level.

When FAQ's advantage is marginal (due to degraded factor model), the
cherry-picked baseline can match or exceed FAQ at each budget:

- At low budgets: `sqrt` with low $\tau$ may match FAQ's mildly non-uniform
  sampling.
- At high budgets: `unif` with $f = \text{mean}$ may match FAQ's AIPW with
  near-uniform sampling.

The "exact" match in figures often reflects FAQ and the best baseline producing
CIs of essentially equal width, with the differences being smaller than the
error bars (standard error across 100 seeds).

### 3.8 Summary for Regime B

$$
\boxed{
\text{Sparse } M_1 \implies V \to 0 \implies \hat{p}_j^{(s)} \to 0.5
\implies h_o, h_a \to \frac{1}{N_q}
\implies q_s(j) \to \frac{1}{N_q}
\implies \hat{\theta}^{\text{FAQ}} \to \bar{y}_{\text{sample}} = \hat{\theta}^{\text{baseline}(f=0)}
}
$$

At moderate sparsity: factor model $\to$ column means, FAQ $\to$ best Zrnic variant.

---

## 4. Unified View: The Variance Decomposition

Both collapse regimes can be understood through a single variance decomposition.
The asymptotic variance of the FAQ estimator (with replacement, one sample per step) is:

$$
V_{\text{FAQ}} = \frac{1}{n_b N_q^2} \sum_{s=1}^{n_b} \mathbb{E}\!\left[\frac{(y_{I_s} - \hat{p}_{I_s}^{(s-1)})^2}{q_s(I_s)}\right]
$$

$$
= \frac{1}{n_b N_q^2} \sum_{s=1}^{n_b} \sum_j q_s(j) \cdot \frac{(y_j - \hat{p}_j^{(s-1)})^2}{q_s(j)}
= \frac{1}{n_b N_q^2} \sum_{s=1}^{n_b} \sum_j \frac{(y_j - \hat{p}_j^{(s-1)})^2}{1}
$$

Wait — this collapses only if $q_s(j)$ cancels. Let us be more careful.

The conditional variance at step $s$ (given the history) is:

$$
\text{Var}(\phi_s \mid \mathcal{F}_{s-1}) = \sum_j q_s(j) \left(\frac{y_j - \hat{p}_j^{(s-1)}}{q_s(j)}\right)^2 - \left(\sum_j (y_j - \hat{p}_j^{(s-1)})\right)^2
$$

$$
= \sum_j \frac{(y_j - \hat{p}_j^{(s-1)})^2}{q_s(j)} - \left(\sum_j (y_j - \hat{p}_j^{(s-1)})\right)^2
$$

The **only term that depends on the sampling policy** is:

$$
A_s := \sum_j \frac{(y_j - \hat{p}_j^{(s-1)})^2}{q_s(j)}
$$

Minimizing $A_s$ over $q_s$ subject to $\sum_j q_s(j) = 1$ gives the
**Neyman allocation**:

$$
q_s^*(j) = \frac{|y_j - \hat{p}_j^{(s-1)}|}{\sum_{j'} |y_{j'} - \hat{p}_{j'}^{(s-1)}|}
$$

This is Zrnic's oracle rule (Eq. 34) adapted to the FAQ setting.

### 4.1 When Does Non-Uniform $q_s$ Help?

Non-uniform sampling helps if and only if:

$$
\sum_j \frac{e_j^2}{q_s^*(j)} < \sum_j \frac{e_j^2}{1/N_q} = N_q \sum_j e_j^2
$$

where $e_j := y_j - \hat{p}_j^{(s-1)}$. By the Cauchy-Schwarz inequality:

$$
\sum_j \frac{e_j^2}{q_s^*(j)} = \sum_j |e_j| \cdot \sum_{j'} |e_{j'}| = \left(\sum_j |e_j|\right)^2
$$

vs. $N_q \sum_j e_j^2$. The ratio is:

$$
\frac{\text{Var}(\text{optimal})}{\text{Var}(\text{uniform})} = \frac{(\sum_j |e_j|)^2}{N_q \sum_j e_j^2} = \frac{\bar{|e|}^2}{\overline{e^2}} = 1 - \frac{\text{Var}(|e_j|)}{\overline{e^2}}
$$

This ratio equals 1 (no benefit) when $\text{Var}(|e_j|) = 0$, i.e., when
$|e_j|$ is **constant across items** (homoscedastic residuals).

### 4.2 Connection to Both Regimes

**Regime A (ACS):** The residuals $e_j = y_j - \hat{p}_j$ have approximately
constant absolute value across items (homoscedastic Gaussian noise with
$|e_j| \approx \sigma$ for all $j$). Therefore $\text{Var}(|e_j|) \approx 0$
and the optimal allocation reduces to uniform.

**Regime B (Missing data):** When the factor model is degraded, the predictions
$\hat{p}_j^{(s)} \approx c$ are approximately constant. Then $e_j = y_j - c$
and $|e_j|$ varies only through the true $y_j$ values. However, FAQ does not
know $y_j$ — it can only proxy $|e_j|$ through $h_o^{(s)}(j) \propto \sqrt{\hat{p}_j(1-\hat{p}_j)}$.
With $\hat{p}_j \approx c$ for all $j$, this proxy is constant, and FAQ
cannot identify which items have high $|e_j|$. The method defaults to uniform
despite the fact that the oracle $q_s^*$ would be non-uniform.

### 4.3 Key Distinction Between Regimes

| | Regime A (ACS) | Regime B (Missing data) |
|---|---|---|
| Oracle optimal $q^*$ | Uniform (homoscedastic $|e_j|$) | Non-uniform (heteroscedastic $|e_j|$) |
| FAQ's $q_s$ | Uniform (correct!) | Uniform (incorrect, but unavoidable) |
| Gap vs. oracle | None | Positive, grows with sparsity |
| Root cause | Data structure | Information loss |
| Could more data fix it? | No (structural) | Yes (more historical obs.) |

In Regime A, FAQ is **optimally** uniform — no method can do better. In Regime B,
FAQ is **suboptimally** uniform — a method with access to the true $|e_j|$ could
do better, but FAQ lacks the information to compute non-trivial sampling weights.

---

## 5. The Binary Outcome Structural Advantage

A final remark explaining why FAQ works well on LLM benchmarks but not on ACS.

For **binary outcomes** $y_j \in \{0,1\}$ with $\mathbb{E}[y_j] = p_j$:

$$
|e_j| = |y_j - p_j| = \begin{cases} 1 - p_j & \text{w.p. } p_j \\ p_j & \text{w.p. } 1-p_j \end{cases}
$$

$$
\mathbb{E}[e_j^2] = p_j(1-p_j), \qquad \mathbb{E}[|e_j|] = 2p_j(1-p_j)
$$

So $\mathbb{E}[e_j^2]$ varies across items as a function of $p_j$. Items with
$p_j \approx 0$ or $p_j \approx 1$ have low variance; items with $p_j \approx 0.5$
have high variance.

Crucially, the oracle score $h_o(j) \propto \sqrt{p_j(1-p_j)}$ is a **direct
proxy for the optimal Neyman weights** $|e_j|$. FAQ's sampling policy is
*automatically* close to optimal just from knowing $\hat{p}_j$, without needing
to observe any $y_j$ values.

For **continuous outcomes** $y_j \in \mathbb{R}$ with $\text{Var}(y_j | x_j) = \sigma_j^2$:

$$
\mathbb{E}[e_j^2] = \sigma_j^2 + (\hat{p}_j - \mu_j)^2
$$

If $\sigma_j^2 \equiv \sigma^2$ (homoscedastic) and $\hat{p}_j \approx \mu_j$
(good predictions), then $\mathbb{E}[e_j^2] \approx \sigma^2$ uniformly — no
structure for FAQ to exploit.

$$
\boxed{
\text{Binary: } \text{Var}(|e_j|) \propto \text{Var}(p_j(1-p_j)) > 0 \text{ (automatic heteroscedasticity)}
}
$$
$$
\boxed{
\text{Continuous: } \text{Var}(|e_j|) \approx 0 \text{ when homoscedastic (no structure to exploit)}
}
$$

This is the fundamental asymmetry. Binary outcomes provide free
heteroscedasticity through the Bernoulli variance function $p(1-p)$.
Continuous outcomes require it to come from heteroscedastic noise $\sigma_j^2$,
which is a property of the data, not of the method.

---

## 6. Summary

| Collapse Regime | Cause | Mathematical Mechanism | Reversible? |
|---|---|---|---|
| **A: Homoscedastic residuals** | Data structure (e.g., ACS income) | $r_j^{(s)} \ll 1 \implies h_o, h_a \to 1/N_q \implies q_s \to 1/N_q$ | No (structural) |
| **B: Degraded factor model** | Sparse $M_1$ | $V \to 0 \implies \hat{p}_j \to 0.5 \implies h_o \to 1/N_q$ | Yes (more data) |

Both regimes share the same proximate mechanism — **FAQ's sampling probabilities
become uniform** — but for fundamentally different reasons:

- In Regime A, uniform sampling is actually *optimal* (no method can do better).
- In Regime B, uniform sampling is *suboptimal* but unavoidable given the
  available information.

The Cauchy-Schwarz analysis (Section 4) provides the unifying lens: non-uniform
sampling helps if and only if $\text{Var}(|y_j - \hat{p}_j|) > 0$, which
requires either heteroscedastic data (violated in Regime A) or informative
predictions $\hat{p}_j$ (violated in Regime B).
