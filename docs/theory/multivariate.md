---
layout: default
title: Multivariate EM
parent: Theory
nav_order: 6
math: true
---

# Multivariate Expectation-Maximization
{: .no_toc }

Fitting mixture models to vector-valued observations.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Setup

Let $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^D$ be $D$-dimensional observations. A multivariate finite mixture model has density:

$$p(\mathbf{x}) = \sum_{j=1}^{k} \pi_j \, f_j(\mathbf{x} \mid \boldsymbol{\theta}_j)$$

Gemmulem currently supports two multivariate families: **Multivariate Gaussian** and **Multivariate Student-t**.

---

## Multivariate Gaussian Mixture

### Density

$$f(\mathbf{x} \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}_j|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_j)^\top \boldsymbol{\Sigma}_j^{-1} (\mathbf{x} - \boldsymbol{\mu}_j)\right)$$

Parameters for each component:
- $\boldsymbol{\mu}_j \in \mathbb{R}^D$: mean vector
- $\boldsymbol{\Sigma}_j \in \mathbb{R}^{D \times D}$: positive definite covariance matrix

### E-Step

The responsibilities are identical in form to the univariate case:

$$\gamma_{ij} = \frac{\pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}{\sum_{\ell=1}^k \pi_\ell \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_\ell, \boldsymbol{\Sigma}_\ell)}$$

Computing $\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$ requires:
1. The Mahalanobis distance: $d_{ij}^2 = (\mathbf{x}_i - \boldsymbol{\mu}_j)^\top \boldsymbol{\Sigma}_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j)$
2. The log-determinant: $\ln |\boldsymbol{\Sigma}_j|$

Both are computed efficiently using the **Cholesky decomposition** $\boldsymbol{\Sigma}_j = \mathbf{L}_j \mathbf{L}_j^\top$.

### Cholesky Decomposition in the E-Step

Given $\boldsymbol{\Sigma}_j = \mathbf{L}_j \mathbf{L}_j^\top$:

1. Solve the lower triangular system: $\mathbf{L}_j \mathbf{y} = \mathbf{x}_i - \boldsymbol{\mu}_j$ (by forward substitution, $O(D^2)$)
2. Mahalanobis distance: $d_{ij}^2 = \|\mathbf{y}\|^2$
3. Log-determinant: $\ln|\boldsymbol{\Sigma}_j| = 2\sum_{d=1}^D \ln L_{j,dd}$ (sum of log diagonal elements)

This is numerically superior to computing $\boldsymbol{\Sigma}_j^{-1}$ explicitly, and is $O(D^2)$ per observation vs $O(D^3)$ for matrix inversion.

### M-Step

Let $N_j = \sum_{i=1}^n \gamma_{ij}$.

**Mean update:**

$$\hat{\boldsymbol{\mu}}_j = \frac{1}{N_j} \sum_{i=1}^n \gamma_{ij} \mathbf{x}_i$$

**Covariance update:**

$$\hat{\boldsymbol{\Sigma}}_j = \frac{1}{N_j} \sum_{i=1}^n \gamma_{ij} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j)^\top$$

This is a **weighted outer product sum** — $O(nD^2k)$ total.

After the M-step, Gemmulem re-Cholesky-factorizes each $\hat{\boldsymbol{\Sigma}}_j$ for use in the next E-step.

**Numerical regularization:** To prevent singular covariance matrices (which cause $|\boldsymbol{\Sigma}_j| = 0$ and $\boldsymbol{\Sigma}_j^{-1}$ to be undefined), Gemmulem adds a small ridge:

$$\hat{\boldsymbol{\Sigma}}_j \leftarrow \hat{\boldsymbol{\Sigma}}_j + \epsilon \mathbf{I}, \quad \epsilon = 10^{-6} \cdot \text{tr}(\hat{\boldsymbol{\Sigma}}_{\text{global}}) / D$$

---

## Covariance Structure Types

Constraining the covariance structure reduces the number of free parameters and improves conditioning.

| Type | Structure | Parameters per component | Use case |
|---|---|---|---|
| **Full** | Arbitrary PD matrix | $D(D+1)/2$ | Correlated features |
| **Diagonal** | $\text{diag}(\sigma_1^2, \ldots, \sigma_D^2)$ | $D$ | Independent features |
| **Spherical** | $\sigma^2 \mathbf{I}$ | $1$ | Isotropic clusters |
| **Tied** | Same $\boldsymbol{\Sigma}$ for all components | $D(D+1)/2$ (shared) | LDA-like discriminant |

### Diagonal M-step

$$\hat{\sigma}_{j,d}^2 = \frac{\sum_i \gamma_{ij}(x_{id} - \hat{\mu}_{jd})^2}{N_j}, \quad d = 1, \ldots, D$$

### Spherical M-step

$$\hat{\sigma}_j^2 = \frac{1}{D} \cdot \frac{\sum_i \gamma_{ij} \|\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j\|^2}{N_j}$$

### Tied M-step

$$\hat{\boldsymbol{\Sigma}} = \frac{1}{n} \sum_{j=1}^k \sum_{i=1}^n \gamma_{ij}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j)^\top$$

**CLI usage:**

```bash
gemmulem -g data.txt -k 3 --multivariate --cov-type full
gemmulem -g data.txt -k 3 --multivariate --cov-type diagonal
gemmulem -g data.txt -k 3 --multivariate --cov-type spherical
gemmulem -g data.txt -k 3 --multivariate --cov-type tied
```

---

## Multivariate Student-t Mixture

The multivariate Student-t distribution provides robustness to outliers in high-dimensional data.

### Density

$$f(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}, \nu) = \frac{\Gamma((\nu+D)/2)}{\Gamma(\nu/2)(\nu\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \left(1 + \frac{\delta(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\Sigma})}{\nu}\right)^{-(\nu+D)/2}$$

where $\delta(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) = (\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})$ is the squared Mahalanobis distance.

### ECM Algorithm

The multivariate Student-t admits a Gaussian scale mixture representation:

$$\mathbf{x} \mid u \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}/u), \qquad u \sim \text{Gamma}(\nu/2, \nu/2)$$

This leads to an **ECM (Expectation Conditional Maximization)** algorithm.

**E-step:** Compute responsibilities $\gamma_{ij}$ and **u-weights**:

$$u_{ij} = \frac{\nu_j + D}{\nu_j + \delta(\mathbf{x}_i, \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**CM-step 1 (update $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$):**

$$\hat{\boldsymbol{\mu}}_j = \frac{\sum_i \gamma_{ij} u_{ij} \mathbf{x}_i}{\sum_i \gamma_{ij} u_{ij}}$$

$$\hat{\boldsymbol{\Sigma}}_j = \frac{\sum_i \gamma_{ij} u_{ij} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j)^\top}{N_j}$$

**CM-step 2 (update $\nu$):** Solve by Newton-Raphson:

$$\ln\frac{\nu_j}{2} - \psi\!\left(\frac{\nu_j}{2}\right) + 1 + \frac{\sum_i \gamma_{ij}(\ln u_{ij} - u_{ij})}{N_j} + \psi\!\left(\frac{\nu_j + D}{2}\right) - \ln\frac{\nu_j + D}{2} = 0$$

where $\psi$ is the digamma function.

**CLI usage:**

```bash
gemmulem -g data.txt -k 3 --multivariate --family studentt
```

---

## Automatic k Selection (BIC-Driven)

For multivariate data, the parameter count grows with $D$, making the BIC penalty more severe:

$$\text{BIC} = -2\ell + p \ln n$$

where for full covariance multivariate Gaussian:

$$p = k\left(D + \frac{D(D+1)}{2}\right) + (k-1)$$

For $D = 10$, $k = 5$: $p = 5 \times 65 + 4 = 329$ parameters. This means BIC will require substantially more data ($n \gg 329$) to support 5 components.

The auto-k search uses the same interface as univariate:

```bash
gemmulem -g multivariate.txt --multivariate --auto-k --k-max 8 --criterion bic -o results.csv
```

**Recommendation:** For $D > 10$, start with diagonal covariance to reduce parameter count, then try full covariance if BIC improves.

---

## Computational Complexity

| Operation | Complexity |
|---|---|
| Cholesky factorization (per component) | $O(D^3)$ |
| E-step (all observations, all components) | $O(nkD^2)$ |
| M-step (covariance update) | $O(nkD^2)$ |
| Total per iteration | $O(nkD^2 + kD^3)$ |

For $n = 10^5$, $k = 5$, $D = 50$: $\approx 1.25 \times 10^9$ FLOPs per iteration. With AVX2 vectorization (see [SIMD]({% link theory/simd.md %})), this takes approximately 2–5 seconds per iteration.

For high-dimensional data ($D > 100$), consider:
- Diagonal covariance (reduces $O(D^2)$ to $O(D)$ in the M-step)
- PCA preprocessing to reduce $D$
- Factor analysis mixture models (structured covariance)

---

## References

- Peel, D. & McLachlan, G.J. (2000). Robust mixture modelling using the t distribution. *Statistics and Computing*, 10(4), 339–348.
- Celeux, G. & Govaert, G. (1995). Gaussian parsimonious clustering models. *Pattern Recognition*, 28(5), 781–793.
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*, Chapter 9. Springer.
