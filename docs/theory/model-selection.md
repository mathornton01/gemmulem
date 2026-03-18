---
layout: default
title: Model Selection
parent: Theory
nav_order: 4
math: true
---

# Model Selection Criteria
{: .no_toc }

Choosing the right number of components and distribution family.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## The Model Selection Problem

After fitting a mixture model, two questions remain:
1. **How many components** $k$ does the data support?
2. **Which distribution family** fits best?

Simply maximizing the log-likelihood fails: adding more components always increases (or maintains) the likelihood, leading to overfitting. Model selection criteria penalize complexity to find the **best-generalizing** model.

Gemmulem implements five criteria: **BIC**, **AIC**, **ICL**, **VBEM**, and **MML**.

---

## Bayesian Information Criterion (BIC)

The BIC (Schwarz, 1978) is defined as:

$$\text{BIC} = -2\ell(\hat{\boldsymbol{\theta}} \mid \mathbf{x}) + p \cdot \ln(n)$$

where:
- $\ell(\hat{\boldsymbol{\theta}} \mid \mathbf{x})$ is the maximized log-likelihood
- $p$ is the number of free parameters
- $n$ is the number of observations

**Lower BIC = better model.**

### Parameter Count

For a $k$-component mixture with a $d$-parameter family:

$$p = k \cdot d + (k - 1)$$

The $(k-1)$ term accounts for the mixing weights (they sum to 1, so the last weight is determined by the others).

| Family | Parameters per component $d$ | Total $p$ (k components) |
|---|---|---|
| Gaussian | 2 ($\mu$, $\sigma$) | $3k - 1$ |
| Gamma | 2 (shape, rate) | $3k - 1$ |
| StudentT | 3 ($\mu$, $\sigma$, $\nu$) | $4k - 1$ |
| Multivariate Gaussian (full, dim $D$) | $D + D(D+1)/2$ | $k(D + D(D+1)/2) + k - 1$ |

### Asymptotic Justification

BIC approximates twice the log of the **Bayes factor** between models. Under regularity conditions, BIC converges to the **Laplace approximation** of the log marginal likelihood:

$$\log p(\mathbf{x} \mid \mathcal{M}) \approx \ell(\hat{\boldsymbol{\theta}} \mid \mathbf{x}) - \frac{p}{2} \ln n + \text{const}$$

This makes BIC a consistent model selector: as $n \to \infty$, it selects the true model with probability 1 (assuming the true model is in the candidate set).

{: .note }
> BIC's $\ln(n)$ penalty grows with sample size. For large $n$, BIC is conservative — it prefers simpler models even when additional components are statistically detectable. This is appropriate for inference but may underfit for prediction tasks.

---

## Akaike Information Criterion (AIC)

$$\text{AIC} = -2\ell(\hat{\boldsymbol{\theta}} \mid \mathbf{x}) + 2p$$

AIC (Akaike, 1974) uses a fixed penalty of $2p$ regardless of sample size. This makes it less conservative than BIC and more likely to select models with more components.

### Theoretical Basis

AIC minimizes the **Kullback-Leibler divergence** between the fitted model and the true data-generating process, averaged over hypothetical repeated datasets. It is an asymptotically unbiased estimator of the expected log-predictive density.

AIC is **not consistent**: it does not converge to the true model as $n \to \infty$ if the true model has a finite number of components. However, for prediction (rather than inference), AIC often outperforms BIC.

### AIC vs BIC: When to Use Which

| Goal | Recommended criterion |
|---|---|
| Identify the true number of components | BIC |
| Minimize predictive error | AIC |
| Balance between inference and prediction | ICL or BIC |
| Robust to model misspecification | VBEM |

---

## Integrated Complete-data Likelihood (ICL)

The ICL (Biernacki, Celeux & Govaert, 2000) adds an **entropy penalty** to BIC:

$$\text{ICL} = \text{BIC} + 2\sum_{i=1}^{n} \sum_{j=1}^{k} \gamma_{ij} \log \gamma_{ij}$$

The entropy term $\sum_{i,j} \gamma_{ij} \log \gamma_{ij}$ is **non-positive** (entropy is non-negative). So ICL $\geq$ BIC, applying an even stronger penalty for models with **ambiguous assignments**.

### Intuition

When responsibilities $\gamma_{ij}$ are close to 0 or 1 (clean separation), the entropy term is near 0 and ICL ≈ BIC. When responsibilities are diffuse (many points assigned ~equally to multiple components), the entropy term is large and ICL strongly penalizes.

ICL selects models where components are **well-separated and identifiable**. If your goal is **clustering** (finding distinct groups), use ICL. If your goal is **density estimation** (modeling the distribution shape), use BIC.

---

## Variational Bayes EM (VBEM)

VBEM replaces point estimation with a **variational Bayesian** approach. Instead of maximizing the likelihood, it maximizes the ELBO with respect to a factorized variational distribution.

### Dirichlet Prior on Mixing Weights

Place a symmetric Dirichlet prior on the mixing weights:

$$\boldsymbol{\pi} \sim \text{Dirichlet}(\alpha_0, \ldots, \alpha_0)$$

with concentration parameter $\alpha_0$ (default: $\alpha_0 = 1/k$).

### Variational Update for Mixing Weights

The VBEM update for the mixing weight parameters is:

$$\tilde{\alpha}_j = \alpha_0 + N_j$$

where $N_j = \sum_i \gamma_{ij}$ is the effective count. The approximate posterior on $\pi_j$ is:

$$\pi_j \mid \mathbf{x} \approx \text{Beta}(\tilde{\alpha}_j, \sum_{\ell \neq j} \tilde{\alpha}_\ell)$$

### Automatic Relevance Determination

A key feature of VBEM: with the Dirichlet prior, components with insufficient support are **automatically pruned** — their effective mixing weight $\tilde{\pi}_j = \tilde{\alpha}_j / \sum_\ell \tilde{\alpha}_\ell$ shrinks toward 0.

This means you can start with a large $k$ (e.g., $k = 20$) and let VBEM determine the effective number of components. This is an elegant alternative to grid search over $k$.

### VBEM Bound

VBEM maximizes the **evidence lower bound** (ELBO):

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z}, \boldsymbol{\pi}, \boldsymbol{\theta})] - \mathbb{E}_q[\log q(\mathbf{z}, \boldsymbol{\pi}, \boldsymbol{\theta})]$$

The VBEM "criterion" for model comparison is:

$$\text{VBEM score} = -\mathcal{L}(q^*)$$

where $q^*$ is the converged variational posterior. Lower scores indicate better models.

---

## Minimum Message Length (MML)

MML (Wallace & Freeman, 1987) is an information-theoretic criterion based on the **minimum description length principle**: the best model is the one that allows the data to be transmitted with the fewest bits.

### Wallace-Freeman MML87

For a $k$-component mixture:

$$\text{MML} = -\ell(\hat{\boldsymbol{\theta}} \mid \mathbf{x}) + \frac{p}{2} \ln \frac{n}{12} + \frac{1}{2} \ln |I(\hat{\boldsymbol{\theta}})| + \frac{p}{2}(1 + \ln \kappa_p)$$

where:
- $I(\hat{\boldsymbol{\theta}})$ is the Fisher information matrix (estimated numerically)
- $\kappa_p$ is a lattice constant (approximated as $\kappa_p \approx 1$ for $p > 5$)

In practice, Gemmulem uses the simplified MML approximation (without the full Fisher information computation):

$$\text{MML} \approx -\ell(\hat{\boldsymbol{\theta}} \mid \mathbf{x}) + \frac{p}{2}\ln\frac{n}{2\pi e} + \frac{1}{2}\ln(k!) - \sum_{j=1}^k \ln \pi_j$$

The $\ln(k!)$ term penalizes label-switching (the $k!$ orderings of component labels), and the $\sum_j \ln \pi_j$ term prefers balanced mixtures.

---

## Practical Guide: Comparing Criteria

For a 5-component Gaussian mixture with $n = 1000$:

| Criterion | Selected $k$ (typical) | Sensitivity to overlap |
|---|---|---|
| AIC | 5–7 | Low (may overfit) |
| BIC | 4–5 | Medium |
| ICL | 3–5 | High (prefers separation) |
| VBEM | 3–6 | Medium (prior-dependent) |
| MML | 4–5 | Medium |

### Decision Flowchart

```
Is n > 10,000?
  Yes → BIC or ICL (AIC overfits on large data)
  No → AIC or BIC

Is your goal clustering (hard assignments)?
  Yes → ICL (penalizes ambiguous components)
  No → BIC or AIC

Is k unknown and you want to fit once?
  Yes → VBEM with large k_max (auto-pruning)
  No → Grid search over k with BIC

Is model misspecification a concern?
  Yes → VBEM (marginalizes over parameters)
  No → BIC (computationally cheaper)
```

### Command-Line Usage

```bash
# Use BIC (default)
gemmulem -g data.txt -k 3 --criterion bic -o results.csv

# Use ICL
gemmulem -g data.txt -k 3 --criterion icl -o results.csv

# Auto-k with BIC (tries k=1..10)
gemmulem -g data.txt --auto-k --k-max 10 --criterion bic -o results.csv

# VBEM with automatic k selection
gemmulem -g data.txt --vbem --k-max 20 --alpha0 0.1 -o results.csv
```

---

## References

- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461–464.
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.
- Biernacki, C., Celeux, G., & Govaert, G. (2000). Assessing a mixture model for clustering with the integrated completed likelihood. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(7), 719–725.
- Wallace, C.S. & Freeman, P.R. (1987). Estimation and inference by compact coding. *Journal of the Royal Statistical Society, Series B*, 49(3), 240–252.
- Attias, H. (1999). Inferring parameters and structure of latent variable models by variational Bayes. *Proceedings of UAI*, 21–30.
