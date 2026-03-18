---
layout: default
title: Distribution Families
parent: Theory
nav_order: 5
math: true
---

# Distribution Families
{: .no_toc }

All 35 families supported by Gemmulem: PDF, parameters, M-step updates, and use cases.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Gemmulem supports 35 distribution families, organized into four groups:

| Group | Families |
|---|---|
| **Continuous Symmetric** | Gaussian, StudentT, Laplace, Cauchy, Logistic, GenGaussian, Uniform |
| **Continuous Skewed/Bounded** | Exponential, Gamma, LogNormal, Weibull, Beta, InvGaussian, Rayleigh, Pareto, Gumbel, SkewNormal, ChiSquared, F, LogLogistic, Nakagami, Levy, Gompertz, Burr, HalfNormal, Maxwell, Kumaraswamy, Triangular |
| **Discrete** | Poisson, Binomial, NegBinomial, Geometric, Zipf |
| **Nonparametric** | KDE |
| **Auto-classified** | Pearson (Types I–VII) |

For each family, this page provides: PDF formula, parameter constraints, M-step update equations (weighted MLE), and typical use cases.

---

## Continuous Symmetric Distributions

---

### Gaussian (Normal)

$$f(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(-\infty, \infty)$ | Mean (location) |
| $\sigma$ | $(0, \infty)$ | Standard deviation (scale) |

**M-step (weighted MLE):**

$$\hat{\mu}_j = \frac{\sum_{i=1}^n \gamma_{ij} x_i}{N_j}, \qquad \hat{\sigma}_j^2 = \frac{\sum_{i=1}^n \gamma_{ij}(x_i - \hat{\mu}_j)^2}{N_j}$$

**CLI flag:** `-f gaussian` (default)

**Use cases:** Data from additive noise processes, heights, weights, measurement errors, any data well-described by the central limit theorem.

---

### Student's t

$$f(x \mid \mu, \sigma, \nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\pi\nu}\,\sigma} \left(1 + \frac{(x-\mu)^2}{\nu\sigma^2}\right)^{-(\nu+1)/2}$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(-\infty, \infty)$ | Location |
| $\sigma$ | $(0, \infty)$ | Scale |
| $\nu$ | $(0, \infty)$ | Degrees of freedom (tail heaviness) |

**M-step:** Uses the ECM (Expectation Conditional Maximization) algorithm. Define u-weights:

$$u_{ij} = \frac{\nu_j + 1}{\nu_j + (x_i - \mu_j)^2/\sigma_j^2}$$

Then:

$$\hat{\mu}_j = \frac{\sum_i \gamma_{ij} u_{ij} x_i}{\sum_i \gamma_{ij} u_{ij}}, \qquad \hat{\sigma}_j^2 = \frac{\sum_i \gamma_{ij} u_{ij}(x_i - \hat{\mu}_j)^2}{N_j}$$

$\nu_j$ is updated by one-dimensional Newton-Raphson on the profile likelihood.

**CLI flag:** `-f studentt`

**Use cases:** Robust clustering with outliers, financial returns, any Gaussian-like data with heavier tails.

---

### Laplace (Double Exponential)

$$f(x \mid \mu, b) = \frac{1}{2b} \exp\!\left(-\frac{|x - \mu|}{b}\right)$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(-\infty, \infty)$ | Location (median) |
| $b$ | $(0, \infty)$ | Scale |

**M-step:**

$$\hat{\mu}_j = \text{weighted median of } \{x_i\} \text{ with weights } \gamma_{ij}$$

$$\hat{b}_j = \frac{\sum_i \gamma_{ij} |x_i - \hat{\mu}_j|}{N_j}$$

The weighted median is computed in $O(n \log n)$ by sorting.

**CLI flag:** `-f laplace`

**Use cases:** Sparse signal recovery (LASSO connection), image gradients, data with sharp peaks and exponential tails.

---

### Cauchy

$$f(x \mid \mu, \gamma) = \frac{1}{\pi\gamma\left[1 + \left(\frac{x-\mu}{\gamma}\right)^2\right]}$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(-\infty, \infty)$ | Location (median, mode) |
| $\gamma$ | $(0, \infty)$ | Scale (half-width at half-maximum) |

**M-step:** No closed form. Gemmulem uses iteratively reweighted least squares (IRLS):

$$\hat{\mu}_j = \frac{\sum_i \gamma_{ij} w_{ij} x_i}{\sum_i \gamma_{ij} w_{ij}}, \quad w_{ij} = \frac{1}{\gamma_j^2 + (x_i - \mu_j)^2}$$

$$\hat{\gamma}_j = \text{numerical root of } \sum_i \gamma_{ij}\frac{\gamma_j^2 - (x_i-\hat{\mu}_j)^2}{(\gamma_j^2 + (x_i-\hat{\mu}_j)^2)^2} = 0$$

**CLI flag:** `-f cauchy`

**Use cases:** Heavy-tailed data where mean and variance don't exist, ratio of normals, some physics applications.

{: .warning }
> **Cauchy has no finite moments.** Mean, variance, and all higher moments are undefined. Standard diagnostics based on sample mean/variance are meaningless for Cauchy mixtures.

---

### Logistic

$$f(x \mid \mu, s) = \frac{\exp(-(x-\mu)/s)}{s\left(1 + \exp(-(x-\mu)/s)\right)^2}$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(-\infty, \infty)$ | Location |
| $s$ | $(0, \infty)$ | Scale |

**Variance:** $\text{Var}(X) = \pi^2 s^2 / 3$

**M-step:** Numerical optimization (no closed form). Gemmulem uses Newton's method with the score function.

**CLI flag:** `-f logistic`

**Use cases:** Logistic regression, sigmoid-shaped CDFs, data slightly heavier-tailed than Gaussian.

---

### Generalized Gaussian (Exponential Power)

$$f(x \mid \mu, \alpha, \beta) = \frac{\beta}{2\alpha\,\Gamma(1/\beta)} \exp\!\left(-\left(\frac{|x-\mu|}{\alpha}\right)^\beta\right)$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(-\infty, \infty)$ | Location |
| $\alpha$ | $(0, \infty)$ | Scale |
| $\beta$ | $(0, \infty)$ | Shape (tail parameter) |

Special cases: $\beta = 1$ → Laplace, $\beta = 2$ → Gaussian, $\beta \to \infty$ → Uniform.

**M-step:** Numerical (Newton). $\mu$ is the weighted $\ell^\beta$ median.

**CLI flag:** `-f gengaussian`

---

### Uniform

$$f(x \mid a, b) = \frac{1}{b - a} \cdot \mathbf{1}[a \leq x \leq b]$$

| Parameter | Domain | Description |
|---|---|---|
| $a$ | $(-\infty, b)$ | Lower bound |
| $b$ | $(a, \infty)$ | Upper bound |

**M-step:**

$$\hat{a}_j = \min\{x_i : \gamma_{ij} > 0.5\}, \qquad \hat{b}_j = \max\{x_i : \gamma_{ij} > 0.5\}$$

(Soft boundary using high-responsibility observations.)

**CLI flag:** `-f uniform`

---

## Continuous Skewed / Bounded Distributions

---

### Exponential

$$f(x \mid \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0$$

| Parameter | Domain | Description |
|---|---|---|
| $\lambda$ | $(0, \infty)$ | Rate (inverse mean) |

**M-step:**

$$\hat{\lambda}_j = \frac{N_j}{\sum_i \gamma_{ij} x_i}$$

**CLI flag:** `-f exponential`

**Use cases:** Inter-arrival times, failure times, memoryless processes.

---

### Gamma

$$f(x \mid k, \beta) = \frac{\beta^k}{\Gamma(k)} x^{k-1} e^{-\beta x}, \quad x > 0$$

| Parameter | Domain | Description |
|---|---|---|
| $k$ | $(0, \infty)$ | Shape |
| $\beta$ | $(0, \infty)$ | Rate |

**M-step:**

$$\hat{\beta}_j = \frac{\hat{k}_j N_j}{\sum_i \gamma_{ij} x_i}$$

$\hat{k}_j$ satisfies the transcendental equation (solved by Newton-Raphson):

$$\ln \hat{k}_j - \psi(\hat{k}_j) = \frac{1}{N_j}\sum_i \gamma_{ij}\ln x_i - \ln\!\left(\frac{\sum_i \gamma_{ij} x_i}{N_j}\right)$$

where $\psi$ is the digamma function.

**CLI flag:** `-f gamma`

**Use cases:** Waiting times, insurance claims, precipitation amounts, positively skewed continuous data.

---

### Log-Normal

$$f(x \mid \mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(-\infty, \infty)$ | Log-scale mean |
| $\sigma$ | $(0, \infty)$ | Log-scale standard deviation |

**M-step:** Apply Gaussian M-step to log-transformed data:

$$\hat{\mu}_j = \frac{\sum_i \gamma_{ij} \ln x_i}{N_j}, \qquad \hat{\sigma}_j^2 = \frac{\sum_i \gamma_{ij}(\ln x_i - \hat{\mu}_j)^2}{N_j}$$

**CLI flag:** `-f lognormal`

**Use cases:** Income distributions, stock prices, particle sizes, reaction times, any multiplicative growth processes.

---

### Weibull

$$f(x \mid k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}, \quad x > 0$$

| Parameter | Domain | Description |
|---|---|---|
| $k$ | $(0, \infty)$ | Shape |
| $\lambda$ | $(0, \infty)$ | Scale |

**M-step:** Iterative (Newton-Raphson). $\hat{k}$ satisfies:

$$\frac{1}{\hat{k}} + \frac{\sum_i \gamma_{ij} x_i^{\hat{k}} \ln x_i}{\sum_i \gamma_{ij} x_i^{\hat{k}}} - \frac{\sum_i \gamma_{ij} \ln x_i}{N_j} = 0$$

Then: $\hat{\lambda}_j = \left(\sum_i \gamma_{ij} x_i^{\hat{k}} / N_j\right)^{1/\hat{k}}$

**CLI flag:** `-f weibull`

**Use cases:** Reliability analysis, wind speed, survival analysis. $k < 1$: decreasing hazard; $k = 1$: Exponential; $k > 1$: increasing hazard.

---

### Beta

$$f(x \mid \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 < x < 1$$

| Parameter | Domain | Description |
|---|---|---|
| $\alpha$ | $(0, \infty)$ | Shape 1 |
| $\beta$ | $(0, \infty)$ | Shape 2 |

**M-step:** Method of moments as initial estimate, then Newton-Raphson:

$$\psi(\hat{\alpha}) - \psi(\hat{\alpha}+\hat{\beta}) = \frac{\sum_i \gamma_{ij} \ln x_i}{N_j}$$
$$\psi(\hat{\beta}) - \psi(\hat{\alpha}+\hat{\beta}) = \frac{\sum_i \gamma_{ij} \ln(1-x_i)}{N_j}$$

**CLI flag:** `-f beta`

**Use cases:** Proportions, probabilities, bounded continuous data in [0,1].

---

### Inverse Gaussian (Wald)

$$f(x \mid \mu, \lambda) = \sqrt{\frac{\lambda}{2\pi x^3}} \exp\!\left(-\frac{\lambda(x-\mu)^2}{2\mu^2 x}\right), \quad x > 0$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(0, \infty)$ | Mean |
| $\lambda$ | $(0, \infty)$ | Shape (precision) |

**M-step:**

$$\hat{\mu}_j = \frac{\sum_i \gamma_{ij} x_i}{N_j}, \qquad \frac{1}{\hat{\lambda}_j} = \frac{1}{N_j}\sum_i \gamma_{ij}\left(\frac{1}{x_i} - \frac{1}{\hat{\mu}_j}\right)$$

**CLI flag:** `-f invgaussian`

**Use cases:** First-passage times of Brownian motion, reaction times in psychology, insurance claim sizes.

---

### Rayleigh

$$f(x \mid \sigma) = \frac{x}{\sigma^2} e^{-x^2/(2\sigma^2)}, \quad x \geq 0$$

| Parameter | Domain | Description |
|---|---|---|
| $\sigma$ | $(0, \infty)$ | Scale |

**M-step:**

$$\hat{\sigma}_j^2 = \frac{\sum_i \gamma_{ij} x_i^2}{2N_j}$$

**CLI flag:** `-f rayleigh`

**Use cases:** Wind speed, signal envelopes in Rayleigh fading channels, distances from origin in 2D Gaussian.

---

### Pareto

$$f(x \mid x_m, \alpha) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}, \quad x \geq x_m$$

| Parameter | Domain | Description |
|---|---|---|
| $x_m$ | $(0, \infty)$ | Minimum value (scale) |
| $\alpha$ | $(0, \infty)$ | Shape (tail index) |

**M-step:**

$$\hat{x}_{m,j} = \min\{x_i : \gamma_{ij} > 0.5\}, \qquad \hat{\alpha}_j = \frac{N_j}{\sum_i \gamma_{ij} \ln(x_i/\hat{x}_{m,j})}$$

**CLI flag:** `-f pareto`

**Use cases:** Power-law phenomena, wealth distribution, file sizes, earthquake magnitudes (Gutenberg-Richter).

---

### Gumbel (Type I Extreme Value)

$$f(x \mid \mu, \beta) = \frac{1}{\beta}\exp\!\left(-\frac{x-\mu}{\beta} - e^{-(x-\mu)/\beta}\right)$$

| Parameter | Domain | Description |
|---|---|---|
| $\mu$ | $(-\infty, \infty)$ | Location (mode) |
| $\beta$ | $(0, \infty)$ | Scale |

**M-step:** Iterative. The score equations are:

$$\hat{\mu}_j = -\hat{\beta}_j \ln\!\left(\frac{\sum_i \gamma_{ij} e^{-x_i/\hat{\beta}_j}}{N_j}\right)$$

$$\hat{\beta}_j: \text{Newton on } \bar{x}_j - \hat{\mu}_j - \hat{\beta}_j + \frac{\sum_i \gamma_{ij}(x_i-\hat{\mu}_j)e^{-(x_i-\hat{\mu}_j)/\hat{\beta}_j}}{\sum_i \gamma_{ij} e^{-(x_i-\hat{\mu}_j)/\hat{\beta}_j}} = 0$$

**CLI flag:** `-f gumbel`

**Use cases:** Extreme value analysis, maximum annual floods, maximum temperatures.

---

### Skew-Normal

$$f(x \mid \xi, \omega, \alpha) = \frac{2}{\omega}\phi\!\left(\frac{x-\xi}{\omega}\right)\Phi\!\left(\alpha\frac{x-\xi}{\omega}\right)$$

where $\phi$ and $\Phi$ are the standard normal PDF and CDF.

| Parameter | Domain | Description |
|---|---|---|
| $\xi$ | $(-\infty, \infty)$ | Location |
| $\omega$ | $(0, \infty)$ | Scale |
| $\alpha$ | $(-\infty, \infty)$ | Skewness ($\alpha = 0$ → Gaussian) |

**M-step:** ECM algorithm with latent truncated normals.

**CLI flag:** `-f skewnormal`

**Use cases:** Mildly asymmetric data where Gaussian fits poorly; alternative to LogNormal for data that can be negative.

---

### Chi-Squared

$$f(x \mid k) = \frac{x^{k/2-1} e^{-x/2}}{2^{k/2}\Gamma(k/2)}, \quad x > 0$$

Special case of Gamma: shape $= k/2$, rate $= 1/2$.

**M-step:**

$$\hat{k}_j = 2 \cdot \hat{\text{shape}} = 2 \cdot \text{[Gamma shape M-step with rate fixed at 1/2]}$$

**CLI flag:** `-f chisquared`

---

### F Distribution

$$f(x \mid d_1, d_2) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x \, B(d_1/2, d_2/2)}, \quad x > 0$$

**M-step:** Numerical (Newton-Raphson on the digamma equations for $d_1$, $d_2$).

**CLI flag:** `-f f`

---

### Log-Logistic (Fisk)

$$f(x \mid \alpha, \beta) = \frac{(\beta/\alpha)(x/\alpha)^{\beta-1}}{(1+(x/\alpha)^\beta)^2}, \quad x > 0$$

**M-step:** Iterative Newton-Raphson.

**CLI flag:** `-f loglogistic`

**Use cases:** Survival analysis with non-monotone hazard, hydrology.

---

### Nakagami

$$f(x \mid m, \Omega) = \frac{2m^m}{\Gamma(m)\Omega^m} x^{2m-1} \exp\!\left(-\frac{m}{\Omega}x^2\right), \quad x \geq 0$$

| Parameter | Domain | Description |
|---|---|---|
| $m$ | $[0.5, \infty)$ | Shape (fading parameter) |
| $\Omega$ | $(0, \infty)$ | Spread (mean power) |

**M-step:**

$$\hat{\Omega}_j = \frac{\sum_i \gamma_{ij} x_i^2}{N_j}, \qquad \hat{m}_j: \text{Newton on } \ln(\hat{m}_j) - \psi(\hat{m}_j) = \ln\hat{\Omega}_j - \frac{\sum_i\gamma_{ij}\ln x_i^2}{N_j}$$

**CLI flag:** `-f nakagami`

**Use cases:** Wireless channel fading modeling, radar scattering.

---

### Lévy

$$f(x \mid \mu, c) = \sqrt{\frac{c}{2\pi}} \frac{e^{-c/(2(x-\mu))}}{(x-\mu)^{3/2}}, \quad x > \mu$$

**M-step:**

$$\hat{\mu}_j = \min\{x_i\} - \epsilon, \qquad \hat{c}_j = \frac{N_j}{\sum_i \gamma_{ij}/(x_i - \hat{\mu}_j)}$$

**CLI flag:** `-f levy`

**Use cases:** Extreme heavy-tailed phenomena, Lévy flights, diffusion in disordered systems.

---

### Gompertz

$$f(x \mid b, \eta) = b\eta e^{\eta} e^{bx} \exp\!\left(-\eta e^{bx}\right), \quad x \geq 0$$

**M-step:** Numerical.

**CLI flag:** `-f gompertz`

**Use cases:** Human mortality, biological aging, tumor growth.

---

### Burr (Type XII)

$$f(x \mid c, k) = \frac{ck x^{c-1}}{(1+x^c)^{k+1}}, \quad x > 0$$

**CLI flag:** `-f burr`

**Use cases:** Actuarial science, income modeling, flood frequency.

---

### Half-Normal

$$f(x \mid \sigma) = \frac{\sqrt{2}}{\sigma\sqrt{\pi}} \exp\!\left(-\frac{x^2}{2\sigma^2}\right), \quad x \geq 0$$

**M-step:**

$$\hat{\sigma}_j^2 = \frac{\sum_i \gamma_{ij} x_i^2}{N_j}$$

**CLI flag:** `-f halfnormal`

---

### Maxwell-Boltzmann

$$f(x \mid a) = \sqrt{\frac{2}{\pi}} \frac{x^2}{a^3} \exp\!\left(-\frac{x^2}{2a^2}\right), \quad x \geq 0$$

**M-step:**

$$\hat{a}_j^2 = \frac{\sum_i \gamma_{ij} x_i^2}{3N_j}$$

**CLI flag:** `-f maxwell`

**Use cases:** Molecular speeds in ideal gases.

---

### Kumaraswamy

$$f(x \mid a, b) = abx^{a-1}(1-x^a)^{b-1}, \quad 0 < x < 1$$

Similar to Beta but with a simpler CDF: $F(x) = 1 - (1-x^a)^b$.

**M-step:** Numerical.

**CLI flag:** `-f kumaraswamy`

---

### Triangular

$$f(x \mid a, b, c) = \begin{cases} \frac{2(x-a)}{(b-a)(c-a)} & a \leq x \leq c \\ \frac{2(b-x)}{(b-a)(b-c)} & c < x \leq b \end{cases}$$

| Parameter | Domain | Description |
|---|---|---|
| $a$ | $(-\infty, b)$ | Lower bound |
| $b$ | $(a, \infty)$ | Upper bound |
| $c$ | $[a, b]$ | Mode |

**M-step:** Numerical (MLE for triangular is non-trivial, uses order statistics).

**CLI flag:** `-f triangular`

---

## Discrete Distributions

---

### Poisson

$$P(X = k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

| Parameter | Domain | Description |
|---|---|---|
| $\lambda$ | $(0, \infty)$ | Mean (= variance) |

**M-step:**

$$\hat{\lambda}_j = \frac{\sum_i \gamma_{ij} x_i}{N_j}$$

**CLI flag:** `-f poisson`

**Use cases:** Event counts over time/space, insurance claims, document word frequencies.

---

### Binomial

$$P(X = k \mid n, p) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

| Parameter | Domain | Description |
|---|---|---|
| $n$ | $\{1, 2, \ldots\}$ | Number of trials (fixed) |
| $p$ | $(0, 1)$ | Success probability |

**M-step ($n$ fixed):**

$$\hat{p}_j = \frac{\sum_i \gamma_{ij} x_i}{n \cdot N_j}$$

**CLI flag:** `-f binomial --n-trials N`

**Use cases:** Number of successes in $n$ trials, proportion data (when $n$ is known).

---

### Negative Binomial

$$P(X = k \mid r, p) = \binom{k+r-1}{k}(1-p)^r p^k, \quad k = 0, 1, 2, \ldots$$

| Parameter | Domain | Description |
|---|---|---|
| $r$ | $(0, \infty)$ | Number of failures (dispersion) |
| $p$ | $(0, 1)$ | Success probability |

**M-step:** $\hat{p}_j = \hat{\mu}_j / (\hat{r}_j + \hat{\mu}_j)$ where $\hat{\mu}_j = \sum_i \gamma_{ij} x_i / N_j$; $\hat{r}_j$ via Newton-Raphson.

**CLI flag:** `-f negbinomial`

**Use cases:** Overdispersed count data (variance > mean), RNA-seq read counts, accident frequencies.

---

### Geometric

$$P(X = k \mid p) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

**M-step:**

$$\hat{p}_j = \frac{N_j}{\sum_i \gamma_{ij} x_i}$$

**CLI flag:** `-f geometric`

**Use cases:** Number of trials until first success, run lengths.

---

### Zipf (Zeta)

$$P(X = k \mid s) = \frac{k^{-s}}{\zeta(s)}, \quad k = 1, 2, 3, \ldots$$

where $\zeta(s) = \sum_{k=1}^\infty k^{-s}$ is the Riemann zeta function.

**M-step:** Numerical (Newton on $-\zeta'(s)/\zeta(s) = \sum_i \gamma_{ij} \ln x_i / N_j$).

**CLI flag:** `-f zipf`

**Use cases:** Word frequencies, city sizes, web link counts, power-law rank distributions.

---

## Nonparametric

---

### Kernel Density Estimation (KDE)

KDE is not a parametric family — it represents each component as a sum of kernels centered at the data points. For component $j$ with bandwidth $h_j$:

$$\hat{f}_j(x) = \frac{1}{N_j h_j} \sum_{i=1}^n \gamma_{ij} K\!\left(\frac{x - x_i}{h_j}\right)$$

where $K$ is the Epanechnikov kernel by default:

$$K(u) = \frac{3}{4}(1 - u^2)\mathbf{1}[|u| \leq 1]$$

**Bandwidth selection:** Silverman's rule of thumb:

$$h_j = 1.06 \hat{\sigma}_j N_j^{-1/5}$$

**M-step:** Update responsibilities using the kernel evaluations; bandwidth updates via cross-validation or Silverman.

**CLI flag:** `-f kde`

**Use cases:** When no parametric family fits, exploratory data analysis, density estimation without distributional assumptions.

{: .note }
> KDE components do not have interpretable parameters. The output reports only weights and bandwidths. For interpretable components, use a parametric family.

---

## Pearson System
{: #pearson-system }

The **Pearson family** covers a wide range of distributions unified by the differential equation:

$$\frac{d}{dx}\ln f(x) = \frac{x - a}{b_0 + b_1 x + b_2 x^2}$$

Classification is based on the **κ criterion**:

$$\kappa = \frac{\beta_1(\beta_2 + 3)^2}{4(4\beta_2 - 3\beta_1)(2\beta_2 - 3\beta_1 - 6)}$$

where $\beta_1 = \mu_3^2/\mu_2^3$ (squared skewness) and $\beta_2 = \mu_4/\mu_2^2$ (kurtosis).

| $\kappa$ range | Pearson Type | Distribution |
|---|---|---|
| $\kappa < 0$ | Type I | Beta |
| $\kappa = 0$ | Type II | Symmetric Beta |
| $0 < \kappa < 1$ | Type IV | Generalized |
| $\kappa = 1$ | Type V | Inverse Gamma |
| $\kappa > 1$ | Type VI | Beta prime |
| $b_2 = 0$ | Type VII | Student-t |
| $\beta_1 = 0, b_2 = 0$ | Normal | Gaussian |

**CLI flag:** `-f pearson`

Gemmulem automatically classifies and fits the appropriate Pearson type based on the weighted moments.

---

## References

- Johnson, N.L., Kotz, S., & Balakrishnan, N. (1994). *Continuous Univariate Distributions*, Vols. 1–2. Wiley.
- Johnson, N.L., Kemp, A.W., & Kotz, S. (2005). *Univariate Discrete Distributions* (3rd ed.). Wiley.
- McLachlan, G.J. & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Pearson, K. (1895). Contributions to the mathematical theory of evolution, II: Skew variation in homogeneous material. *Philosophical Transactions of the Royal Society A*, 186, 343–414.
