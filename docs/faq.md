---
layout: default
title: FAQ
nav_order: 7
---

# Frequently Asked Questions
{: .no_toc }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## General

### What is a finite mixture model?

A finite mixture model assumes your data come from one of several "groups" or "components," each with its own distribution. You don't observe which component generated each data point — EM infers this automatically.

For example, if you measure heights of a mixed-sex population, you might model the data as a 2-component Gaussian mixture: one component for females (μ ≈ 163 cm), one for males (μ ≈ 175 cm). EM learns these parameters from the unlabeled height data.

### When should I use a mixture model vs. a single distribution?

Use a mixture model when:
- Your data appears **multimodal** (multiple peaks in a histogram)
- You suspect the data comes from **heterogeneous subpopulations**
- A single distribution gives a poor fit (high BIC, visually bad)
- You want to perform **soft clustering** (assign probabilities, not just labels)

Stick with a single distribution when:
- Data is unimodal and well-described by a known family
- You have very few observations ($n < 5k$; mixture models overfit)
- Interpretability is critical and a simple distribution suffices

### How many observations do I need?

A rough rule of thumb: you need at least **5–10 observations per parameter**. For a $k$-component Gaussian mixture, that's $3k-1$ parameters, so:

- $k = 2$: need $\geq 25$ observations
- $k = 5$: need $\geq 70$ observations
- $k = 10$: need $\geq 145$ observations

For non-Gaussian families with more parameters (e.g., Student-t: $4k-1$ parameters), increase these estimates by 30–50%.

If BIC selects a lower $k$ than you expected, your dataset may not have enough power to distinguish more components.

---

## Installation

### CMake fails with "Could not find OpenCL"

If you don't need GPU support:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENCL=OFF
```

If you do need GPU support, install the appropriate OpenCL SDK:

- **AMD**: Install ROCm: `sudo apt install rocm-opencl-dev`
- **NVIDIA**: Install CUDA toolkit (includes OpenCL headers)
- **Intel**: Install Intel OpenCL runtime: `sudo apt install intel-opencl-icd`

### gemmulem --version shows "(scalar)" — is my CPU too old?

On x86-64, SSE2 is always available. If you see `(scalar)`, Gemmulem couldn't confirm AVX2 support via CPUID. This can happen if:
- You're running in a VM that doesn't expose AVX2 flags
- The binary was compiled without `-march=native`

Force-check with:

```bash
grep avx2 /proc/cpuinfo
```

If AVX2 is present but Gemmulem doesn't detect it, rebuild with:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-march=native"
```

### Building on Windows fails with link errors

Make sure you're using the **Developer Command Prompt for Visual Studio**, not a regular PowerShell or CMD. The MSVC linker needs to be on PATH.

---

## Algorithm

### Why doesn't EM always find the best solution?

EM converges to a *local* maximum of the log-likelihood. The likelihood surface for mixture models typically has many local maxima due to:

1. **Label-switching symmetry**: Any permutation of component labels gives the same likelihood (there are $k!$ equivalent global maxima)
2. **Spurious local maxima**: Components can get "trapped" in poor configurations

**Mitigation strategies:**
- Multiple restarts (`--restarts 20` or more)
- Better initialization (K-means++ is already used by default)
- SQUAREM acceleration (converges faster from the same start, but doesn't escape bad local optima)

### My log-likelihood is NaN or -Inf. What happened?

Common causes:

1. **Data outside the family's support**: Exponential requires $x > 0$; Beta requires $0 < x < 1$. Check your data with `--verbose`.

2. **Component collapse**: A component shrinks to a single point, making its variance → 0 and log-likelihood → +∞ (a degenerate solution). Gemmulem normally handles this by reinitializing collapsed components. If it persists, try `--min-sigma 1e-3`.

3. **All data in one component**: If $k$ is too large for the data, some components get near-zero weight and their log-density becomes very negative. Use auto-k or reduce $k$.

4. **Numerical overflow in exotic families**: Pareto and Lévy can have very long tails; extreme data values cause overflow. Check for outliers first.

### How does SQUAREM work? Is it safe?

SQUAREM accelerates EM by taking a much larger step in parameter space than a single EM step would. The stabilizing M-step at the end guarantees that the log-likelihood never decreases.

In rare cases (very non-smooth likelihoods, numerical issues), SQUAREM may take a step that causes the stabilizing M-step to "undo" the acceleration — falling back to the standard EM step. This is safe (monotone ascent is maintained) but slightly wasteful (3 EM steps to make 1 EM step of progress).

Disable SQUAREM with `--no-squarem` if you're debugging or comparing against a reference implementation.

### How do I know if EM has converged to the global maximum?

You can't know for certain. Practical strategies:

1. **Run many restarts**: If all restarts converge to the same log-likelihood, it's likely the global maximum (for that $k$).
2. **Check agreement**: Do all restarts give similar component parameters (up to label permutation)?
3. **Increase restarts**: If adding more restarts changes the best solution, keep increasing.
4. **Simulated annealing / random perturbation**: Advanced: add noise to parameter estimates and re-run EM from the perturbed point.

For well-separated, moderate-k mixtures, 10–20 restarts are usually sufficient. For highly overlapping, large-k mixtures, consider 50–100 restarts.

### What's the difference between BIC and AIC for choosing k?

- **BIC** penalizes complexity proportional to $\ln(n)$. For large datasets, BIC is conservative: it prefers parsimonious models. BIC is **consistent** (selects the true $k$ as $n \to \infty$).

- **AIC** uses a fixed penalty of 2 per parameter. AIC tends to select more components (higher $k$) because its penalty doesn't grow with $n$. AIC minimizes expected prediction error (Kullback-Leibler divergence).

**Rule of thumb:**
- If you care about the "true" number of clusters → BIC or ICL
- If you want the best density estimate for prediction → AIC

---

## Distributions

### Which family should I use when I don't know the distribution?

Use `--auto-family --criterion bic`. This fits all applicable families and selects by BIC. It's slower but often worth it for exploratory analysis.

As a starting point without running auto-family:
- Positive, right-skewed → try Gamma, LogNormal, Weibull (in that order)
- Integer counts → try Poisson, then NegBinomial if overdispersed
- Bounded [0,1] → Beta or Kumaraswamy
- Heavy-tailed symmetric → Student-t (small ν) or Cauchy
- Unknown → KDE (nonparametric, always fits)

### The Beta distribution says my data is out of range. What do I do?

Beta requires data in the **open interval** $(0, 1)$. If your proportions include exact 0s or 1s:

1. Apply a small-sample correction: transform $x \to (x \cdot (n-1) + 0.5) / n$ (Smithson & Verkuilen, 2006)
2. Use a zero-inflated Beta mixture
3. Use Kumaraswamy (same support but sometimes more tolerant of boundary values numerically)

### Can I mix different distribution families in a single mixture?

Not directly — Gemmulem assumes all components share the same family. For mixed-family mixtures, you'd need to implement a custom EM loop using the C API, calling `gem_log_pdf()` with different families for different components.

This is a planned feature for a future release.

### What's the Pearson family and when should I use it?

The Pearson system is a unified family of distributions parameterized by skewness and kurtosis. Gemmulem automatically classifies your data into the appropriate Pearson type (I–VII) based on weighted moment estimates.

Use `-f pearson` when:
- You want the "right" distribution from a classical statistical perspective
- Your data doesn't obviously match any standard family
- You need a distribution that exactly matches sample skewness and kurtosis

---

## Output and Interpretation

### What do the responsibilities (γ) mean?

The responsibility $\gamma_{ij}$ is the **posterior probability** that observation $x_i$ was generated by component $j$. It satisfies:
- $0 \leq \gamma_{ij} \leq 1$ for all $i, j$
- $\sum_j \gamma_{ij} = 1$ for each observation $i$

High $\gamma_{ij}$ means observation $i$ is "probably from component $j$." The `assignment` column gives the hard assignment $\arg\max_j \gamma_{ij}$.

### Components are in different orders each run. Is that a bug?

No — this is **label-switching**. Mixture model components are exchangeable: any permutation of the labels gives the same likelihood. K-means++ initialization is random, so components can emerge in any order.

To get stable ordering, sort by a parameter after fitting:

```bash
# Sort by mean (for Gaussian) in Python:
import pandas as pd
df = pd.read_csv('results.csv', comment='#')
df.sort_values('mu', inplace=True)
```

Or use `--sort-components mu` (ascending order by mean):

```bash
gemmulem -g data.txt -k 3 --sort-components mu -o results.csv
```

### Why does AIC say k=4 but BIC says k=3?

AIC and BIC have different penalty magnitudes. AIC uses $2p$; BIC uses $p \ln(n)$. For $n = 1000$, $\ln(1000) \approx 6.9$, so BIC penalizes complexity ~3.5× more than AIC.

If your data genuinely has 3 distinguishable clusters (by ICL), BIC is probably right. If the data has 4 modes but one is weak, AIC may detect it while BIC dismisses it as noise.

This disagreement is informative: it suggests the evidence for $k=4$ is real but modest.

---

## Performance

### Gemmulem is slow on my large dataset. What should I do?

1. **Check backend**: Is AVX2 active? Run `gemmulem --version` and look for `(AVX2)`.
2. **Enable GPU**: If you have an OpenCL-compatible GPU, rebuild with `-DENABLE_OPENCL=ON`.
3. **Reduce restarts**: `--restarts 3` instead of the default.
4. **Use streaming**: For $n > 10^6$, `--streaming` uses $O(k)$ memory and is often faster.
5. **Reduce k**: Each EM iteration scales as $O(nk)$. Halving $k$ halves the time.
6. **Use a simpler family**: Gaussian and Exponential are 3–5× faster than Gamma or Weibull (which need Newton iterations in the M-step).

### Can Gemmulem use multiple CPU cores?

The E-step is parallelized across cores using OpenMP (if compiled with `-DENABLE_OPENMP=ON`, which is default). Set threads with `--threads N` or `OMP_NUM_THREADS=N`.

```bash
OMP_NUM_THREADS=8 gemmulem -g data.txt -k 5 -o results.csv
```

The M-step is not parallelized (it's much cheaper than the E-step and parallelization overhead isn't worth it for moderate $k$).

---

## Contributing and Support

### How do I report a bug?

Open an issue on [GitHub](https://github.com/mathornton01/gemmulem/issues) with:
1. Gemmulem version (`gemmulem --version`)
2. OS and CPU info
3. Minimal reproducing command (including data file if possible, or a script to generate it)
4. Expected vs. actual behavior

### Can I add a new distribution family?

Yes! See the [Contributing Guide](https://github.com/mathornton01/gemmulem/blob/main/CONTRIBUTING.md) for the process. Adding a new family requires implementing:
1. `log_pdf(x, params)` — log probability density
2. `estep_update(x, gamma, params, n)` — weighted M-step update
3. `param_count()` — number of free parameters (for BIC/AIC)
4. Test cases with known analytical M-step results
