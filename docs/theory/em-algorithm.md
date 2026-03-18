---
layout: default
title: EM Algorithm
parent: Theory
nav_order: 1
math: true
---

# The Expectation-Maximization Algorithm
{: .no_toc }

A complete mathematical treatment of EM for finite mixture models.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Finite Mixture Models

A **finite mixture model** assumes that the observed data $\mathbf{x} = \{x_1, \ldots, x_n\}$ are drawn i.i.d. from a distribution that is a convex combination of $k$ component distributions:

$$p(x \mid \boldsymbol{\theta}) = \sum_{j=1}^{k} \pi_j \, f(x \mid \theta_j)$$

where:
- $\pi_j \geq 0$ are the **mixing weights** with $\sum_{j=1}^k \pi_j = 1$
- $f(x \mid \theta_j)$ is the **component density** for the $j$-th component
- $\theta_j$ are the **component parameters** (e.g., $(\mu_j, \sigma_j)$ for Gaussian)
- $\boldsymbol{\theta} = (\pi_1, \ldots, \pi_k, \theta_1, \ldots, \theta_k)$ is the full parameter vector

### The Complete-Data View

The mixture model has a natural interpretation via **latent variables**. For each observation $x_i$, introduce a latent indicator $z_i \in \{1, \ldots, k\}$ indicating which component generated it:

$$P(z_i = j) = \pi_j$$

$$p(x_i \mid z_i = j, \boldsymbol{\theta}) = f(x_i \mid \theta_j)$$

The joint distribution is:

$$p(x_i, z_i \mid \boldsymbol{\theta}) = \pi_{z_i} \, f(x_i \mid \theta_{z_i})$$

The observed data likelihood marginalizes over the latent variables:

$$p(x_i \mid \boldsymbol{\theta}) = \sum_{j=1}^{k} \pi_j \, f(x_i \mid \theta_j)$$

---

## The Likelihood Function

The **observed-data log-likelihood** for $n$ independent observations is:

$$\ell(\boldsymbol{\theta} \mid \mathbf{x}) = \sum_{i=1}^{n} \log \left[ \sum_{j=1}^{k} \pi_j \, f(x_i \mid \theta_j) \right]$$

Direct maximization of this expression is intractable because the log of a sum cannot be simplified analytically when the components share parameters in a complex way. The logarithm and summation cannot be exchanged.

The **complete-data log-likelihood**, treating both $\mathbf{x}$ and $\mathbf{z}$ as observed, is:

$$\ell_c(\boldsymbol{\theta} \mid \mathbf{x}, \mathbf{z}) = \sum_{i=1}^{n} \sum_{j=1}^{k} \mathbf{1}[z_i = j] \left[ \log \pi_j + \log f(x_i \mid \theta_j) \right]$$

This is a sum of logs — much easier to maximize! The EM algorithm exploits this by alternating between estimating the latent variables and maximizing the complete-data likelihood.

---

## The EM Algorithm

EM is a general algorithm for maximum likelihood estimation in the presence of missing or latent data. It finds a local maximum of $\ell(\boldsymbol{\theta} \mid \mathbf{x})$ by iteratively:

1. **E-step**: Compute the expected complete-data log-likelihood under the current parameters
2. **M-step**: Maximize that expectation with respect to $\boldsymbol{\theta}$

### E-Step: Computing Responsibilities

Define the **responsibility** of component $j$ for observation $i$ as the posterior probability:

$$\gamma_{ij} = P(z_i = j \mid x_i, \boldsymbol{\theta}^{(t)})$$

By Bayes' theorem:

$$\gamma_{ij} = \frac{P(z_i = j \mid \boldsymbol{\theta}^{(t)}) \cdot p(x_i \mid z_i = j, \boldsymbol{\theta}^{(t)})}{p(x_i \mid \boldsymbol{\theta}^{(t)})}$$

$$\boxed{\gamma_{ij} = \frac{\pi_j^{(t)} \, f(x_i \mid \theta_j^{(t)})}{\sum_{\ell=1}^{k} \pi_\ell^{(t)} \, f(x_i \mid \theta_\ell^{(t)})}}$$

The E-step computes the **Q function** — the expected complete-data log-likelihood:

$$Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) = \mathbb{E}_{\mathbf{z} \mid \mathbf{x}, \boldsymbol{\theta}^{(t)}} \left[ \ell_c(\boldsymbol{\theta} \mid \mathbf{x}, \mathbf{z}) \right]$$

$$= \sum_{i=1}^{n} \sum_{j=1}^{k} \gamma_{ij} \left[ \log \pi_j + \log f(x_i \mid \theta_j) \right]$$

### M-Step: Maximizing Q

The M-step finds $\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})$.

**Updating mixing weights:** Maximizing $\sum_{i,j} \gamma_{ij} \log \pi_j$ subject to $\sum_j \pi_j = 1$ (via Lagrange multipliers) gives:

$$\boxed{\pi_j^{(t+1)} = \frac{1}{n} \sum_{i=1}^{n} \gamma_{ij} = \frac{N_j}{n}}$$

where $N_j = \sum_{i=1}^n \gamma_{ij}$ is the **effective count** of observations assigned to component $j$.

**Updating component parameters:** The Q function decomposes over components. For each $j$, maximize:

$$Q_j(\theta_j) = \sum_{i=1}^{n} \gamma_{ij} \log f(x_i \mid \theta_j)$$

This is a **weighted** maximum likelihood problem. The form of the solution depends on the distribution family; see [Distributions]({% link theory/distributions.md %}) for the closed-form M-step updates for all 35 families.

For the **Gaussian** case with $f(x \mid \mu_j, \sigma_j^2) = \mathcal{N}(x; \mu_j, \sigma_j^2)$:

$$\mu_j^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{ij} \, x_i}{N_j}$$

$$(\sigma_j^2)^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{ij} \, (x_i - \mu_j^{(t+1)})^2}{N_j}$$

---

## Convergence Theory

### Monotonic Increase of the Log-Likelihood

**Theorem (Dempster, Laird & Rubin, 1977):** Each EM iteration satisfies

$$\ell(\boldsymbol{\theta}^{(t+1)} \mid \mathbf{x}) \geq \ell(\boldsymbol{\theta}^{(t)} \mid \mathbf{x})$$

**Proof sketch:** Decompose the log-likelihood via the evidence lower bound (ELBO):

$$\ell(\boldsymbol{\theta} \mid \mathbf{x}) = Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) - H(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})$$

where $H(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) = \mathbb{E}_{\mathbf{z} \mid \mathbf{x}, \boldsymbol{\theta}^{(t)}}[\log p(\mathbf{z} \mid \mathbf{x}, \boldsymbol{\theta})]$ (note the sign convention).

By Jensen's inequality (applied to the concave log function):

$$H(\boldsymbol{\theta}^{(t+1)} \mid \boldsymbol{\theta}^{(t)}) \leq H(\boldsymbol{\theta}^{(t)} \mid \boldsymbol{\theta}^{(t)})$$

Since the M-step guarantees $Q(\boldsymbol{\theta}^{(t+1)} \mid \boldsymbol{\theta}^{(t)}) \geq Q(\boldsymbol{\theta}^{(t)} \mid \boldsymbol{\theta}^{(t)})$, combining these:

$$\ell(\boldsymbol{\theta}^{(t+1)} \mid \mathbf{x}) \geq \ell(\boldsymbol{\theta}^{(t)} \mid \mathbf{x}) \quad \blacksquare$$

### Rate of Convergence

EM converges at a **linear rate** near a stationary point. The rate constant is the **fraction of missing information**:

$$r = \left\| \frac{\partial^2 \ell_{\mathrm{obs}}}{\partial \boldsymbol{\theta}^2} \right\|^{-1} \left\| \frac{\partial^2 \ell_{\mathrm{miss}}}{\partial \boldsymbol{\theta}^2} \right\|$$

When $r$ is close to 1 (much missing information), EM is very slow. This motivates SQUAREM acceleration — see [SQUAREM]({% link theory/squarem.md %}).

### Convergence Criteria

Gemmulem uses the following stopping rule by default:

$$|\ell(\boldsymbol{\theta}^{(t+1)}) - \ell(\boldsymbol{\theta}^{(t)})| < \varepsilon \cdot (1 + |\ell(\boldsymbol{\theta}^{(t)})|)$$

with $\varepsilon = 10^{-6}$ (relative tolerance). An absolute tolerance $|ΔLL| < 10^{-8}$ is also checked. The maximum iteration count defaults to 1000.

{: .important }
> **Why relative tolerance?** The log-likelihood can have very different magnitudes for different dataset sizes and families. A fixed absolute tolerance that works for 100 observations may never converge for $10^6$ observations. Relative tolerance scales correctly.

---

## Numerical Stability: The Log-Sum-Exp Trick

The naive E-step computation:

```c
for (j = 0; j < k; j++)
    gamma[i][j] = pi[j] * pdf(x[i], theta[j]);
// normalize:
sum = 0; for (j=0; j<k; j++) sum += gamma[i][j];
for (j = 0; j < k; j++) gamma[i][j] /= sum;
```

suffers from **numerical underflow** when densities are very small (e.g., $f(x_i \mid \theta_j) \approx 10^{-300}$). The sum becomes exactly 0 in double precision, causing division by zero.

### The Fix: Log-Space Computation

Compute in log-space using the **log-sum-exp** trick:

$$\log \sum_{j=1}^{k} \pi_j f(x_i \mid \theta_j) = a_i + \log \sum_{j=1}^{k} e^{(\log \pi_j + \log f(x_i \mid \theta_j) - a_i)}$$

where $a_i = \max_j (\log \pi_j + \log f(x_i \mid \theta_j))$ is chosen to prevent underflow.

```c
// Log-sum-exp E-step (Gemmulem implementation)
double log_gamma[k];
double a = -INFINITY;

for (j = 0; j < k; j++) {
    log_gamma[j] = log(pi[j]) + log_pdf(x[i], theta[j]);
    if (log_gamma[j] > a) a = log_gamma[j];  // track max
}

double log_sum = 0.0;
for (j = 0; j < k; j++)
    log_sum += exp(log_gamma[j] - a);
log_sum = a + log(log_sum);  // log of normalizer

// Responsibilities and log-likelihood contribution
ll_i = log_sum;
for (j = 0; j < k; j++)
    gamma[i][j] = exp(log_gamma[j] - log_sum);
```

This is numerically stable for all reasonable inputs: the exponents are now in $(-\infty, 0]$, so no overflow occurs, and the maximum term contributes $e^0 = 1$, preventing underflow of the dominant term.

### Handling Degenerate Components

If a component's effective count $N_j$ falls below a threshold (default: 1.0), Gemmulem **reinitializes** that component by:
1. Randomly selecting an observation weighted by its assignment probability to the best component
2. Setting the component's parameters to a small perturbation of that observation
3. Rebalancing weights

This prevents the common failure mode where components "collapse" to a single point.

---

## Relationship to Jensen's Inequality

The EM algorithm can be viewed as coordinate ascent on the ELBO. Define an auxiliary distribution $q_i(j) = \gamma_{ij}$ over components for each observation. By Jensen's inequality (log is concave):

$$\log \sum_j \pi_j f(x_i \mid \theta_j) = \log \sum_j q_i(j) \frac{\pi_j f(x_i \mid \theta_j)}{q_i(j)} \geq \sum_j q_i(j) \log \frac{\pi_j f(x_i \mid \theta_j)}{q_i(j)}$$

The right side is the ELBO. EM maximizes it alternately:
- **E-step**: Choose $q_i(j) = \gamma_{ij}$ to make the inequality **tight** (the bound is exact when $q_i(j) \propto \pi_j f(x_i \mid \theta_j)$)
- **M-step**: Maximize the ELBO over $\boldsymbol{\theta}$

This perspective connects EM to variational inference and explains why SQUAREM works (see [SQUAREM]({% link theory/squarem.md %})).

---

## M-Step for the Exponential Family

Many distributions belong to the **exponential family**:

$$f(x \mid \boldsymbol{\eta}) = h(x) \exp\left( \boldsymbol{\eta}^\top T(x) - A(\boldsymbol{\eta}) \right)$$

where $\boldsymbol{\eta}$ are natural parameters, $T(x)$ are sufficient statistics, $A(\boldsymbol{\eta})$ is the log-partition function.

For exponential family distributions, the weighted M-step has a **closed form**: the weighted MLE sets the expected sufficient statistics equal to their weighted empirical averages:

$$\frac{\partial A(\boldsymbol{\eta}_j)}{\partial \boldsymbol{\eta}_j} = \frac{\sum_{i=1}^n \gamma_{ij} T(x_i)}{\sum_{i=1}^n \gamma_{ij}}$$

The left side is $\mathbb{E}[T(x) \mid \boldsymbol{\eta}_j]$ by the properties of the log-partition function. This gives a clean, general update rule. For non-exponential family distributions (Cauchy, stable distributions, etc.), numerical optimization is needed.

---

## References

- Dempster, A.P., Laird, N.M., & Rubin, D.B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1), 1–38.
- McLachlan, G.J. & Krishnan, T. (2008). *The EM Algorithm and Extensions* (2nd ed.). Wiley.
- Neal, R.M. & Hinton, G.E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In *Learning in Graphical Models*, 355–368.
