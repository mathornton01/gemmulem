---
layout: default
title: Streaming EM
parent: Theory
nav_order: 8
math: true
---

# Streaming EM for Large Datasets
{: .no_toc }

Online EM for datasets that don't fit in memory.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## The Problem: Memory-Bounded EM

Standard EM requires storing the full dataset in RAM to compute expectations. For a dataset with $n = 10^8$ double-precision observations, this requires $10^8 \times 8 = 800$ MB — feasible on modern hardware. But for $n = 10^9$ or multivariate data with $D = 100$, even loading the data may be impossible.

**Standard EM memory:** $O(n \cdot k)$ for responsibilities + $O(n)$ for data = $O(nk)$

**Streaming EM memory:** $O(k)$ — stores only sufficient statistics, not the data or responsibilities

---

## Online EM: Cappé & Moulines (2009)

The **online EM** algorithm of Cappé & Moulines (2009) processes observations **one at a time** (or in mini-batches) and maintains running estimates of the sufficient statistics.

### Key Insight: Sufficient Statistics

For exponential family distributions, the M-step can be written entirely in terms of **sufficient statistics**:

$$\hat{\theta}_j = \arg\max_\theta \sum_{i} \gamma_{ij} \log f(x_i \mid \theta) = h\!\left(\frac{\sum_i \gamma_{ij} T(x_i)}{N_j}\right)$$

where $T(x)$ are sufficient statistics and $h$ is the inverse link function.

For the Gaussian mixture:
- $T(x) = (x, x^2)$ (first and second moments)
- Sufficient statistics: $S_{j,1} = \sum_i \gamma_{ij} x_i$ and $S_{j,2} = \sum_i \gamma_{ij} x_i^2$

The M-step only needs $S_{j,1}$, $S_{j,2}$, and $N_j$ — **not** the individual $\gamma_{ij}$ values!

### Algorithm

Maintain running sufficient statistics $\bar{S}_j^{(t)}$ for each component $j$. On receiving observation $x_t$:

**E-step (online):** Compute responsibilities for $x_t$:

$$\gamma_{tj} = \frac{\pi_j^{(t)} f(x_t \mid \theta_j^{(t)})}{\sum_\ell \pi_\ell^{(t)} f(x_t \mid \theta_\ell^{(t)})}$$

**S-step (sufficient statistic update):**

$$\bar{S}_{j}^{(t)} = (1 - \eta_t) \bar{S}_{j}^{(t-1)} + \eta_t \gamma_{tj} T(x_t)$$

$$\bar{N}_j^{(t)} = (1 - \eta_t) \bar{N}_j^{(t-1)} + \eta_t \gamma_{tj}$$

**M-step (online):**

$$\theta_j^{(t+1)} = h\!\left(\frac{\bar{S}_j^{(t)}}{\bar{N}_j^{(t)}}\right)$$

$$\pi_j^{(t+1)} = \bar{N}_j^{(t)} / \sum_\ell \bar{N}_\ell^{(t)}$$

### Step-Size Schedule

The step-size $\eta_t$ controls the trade-off between memory (large $\eta_t$: forget the past) and stability (small $\eta_t$: incorporate new data slowly).

Gemmulem uses:

$$\eta_t = (t + 2)^{-0.6}$$

This satisfies the Robbins-Monro conditions for convergence:
- $\sum_{t=1}^\infty \eta_t = \infty$ (steps sum to infinity — algorithm keeps moving)
- $\sum_{t=1}^\infty \eta_t^2 < \infty$ (squared steps converge — variance goes to 0)

With exponent $-0.6$: $\eta_1 = 3^{-0.6} \approx 0.644$, $\eta_{100} \approx 0.063$, $\eta_{10000} \approx 0.012$.

**Why 0.6?** Exponents in $(0.5, 1)$ work theoretically. Values near 0.5 converge slowly but to better solutions; values near 1 are faster but may oscillate. The value 0.6 is empirically robust across many mixture problems.

You can customize:

```bash
# Faster initial adaptation (larger exponent = slower decay)
gemmulem -g huge_data.txt --streaming --step-size-exp 0.75 -k 5 -o results.csv

# Constant step-size (non-decreasing adaptation)
gemmulem -g huge_data.txt --streaming --step-size 0.01 -k 5 -o results.csv
```

---

## Sufficient Statistics for All Families

### Gaussian

$$T(x) = \begin{pmatrix} x \\ x^2 \end{pmatrix}, \qquad h(s_1, s_2, N) = \left(\frac{s_1}{N},\ \sqrt{\frac{s_2}{N} - \left(\frac{s_1}{N}\right)^2}\right)$$

### Poisson

$$T(x) = x, \qquad h(s, N) = \frac{s}{N}$$

### Exponential

$$T(x) = x, \qquad h(s, N) = \frac{N}{s}$$

### Gamma

$T(x) = (x, \ln x)$; M-step requires Newton (no closed-form inverse).

### LogNormal

$$T(x) = (\ln x, (\ln x)^2), \qquad h = \text{Gaussian } h \text{ applied to } \ln x$$

### Bernoulli / Binomial

$$T(x) = x, \qquad h(s, N) = \frac{s}{n \cdot N}$$

---

## Mini-Batch Streaming

Processing one observation at a time is inefficient due to function call overhead and poor cache utilization. Gemmulem processes mini-batches of size $B$ (default: $B = 512$):

**Mini-batch E-step:** For observations $\{x_{tB}, \ldots, x_{(t+1)B - 1}\}$, compute all responsibilities using the SIMD E-step.

**Mini-batch S-step:**

$$\bar{S}_j^{(t)} = (1-\eta_t)\bar{S}_j^{(t-1)} + \eta_t \cdot \frac{1}{B}\sum_{b=0}^{B-1} \gamma_{tB+b,j} \cdot T(x_{tB+b})$$

The factor $1/B$ ensures the step-size semantics are preserved (step-size $\eta_t$ corresponds to fraction of weight given to the current batch, not the current sample).

**Mini-batch M-step:** Performed after each mini-batch, using the updated sufficient statistics.

---

## Memory Usage Comparison

| Mode | Memory (Gaussian, $k$ components) |
|---|---|
| Standard EM | $O(n \cdot k)$ for $\gamma$ + $O(n)$ for data |
| Streaming EM | $O(k)$ for sufficient statistics |
| Mini-batch streaming | $O(k + B)$ (B = batch size) |

**Example:** $n = 10^9$, $k = 10$, $B = 512$:

| Mode | Memory |
|---|---|
| Standard EM | $10^9 \times 10 \times 8$ B = **80 GB** |
| Streaming EM | $10 \times 3 \times 8$ B = **240 bytes** |
| Mini-batch | $240 + 512 \times 8$ = **4 KB** |

---

## Convergence Properties

Online EM converges to a **local maximum** of the observed-data log-likelihood, but the path is stochastic (depends on the order of observations). Key differences from batch EM:

| Property | Batch EM | Online EM |
|---|---|---|
| Monotone LL increase | Guaranteed | Not guaranteed |
| Convergence | Deterministic | Stochastic (a.s. convergence) |
| Data shuffling | Not needed | Recommended (each epoch) |
| Multiple passes | Same result | Improves with epochs |

### Multiple Epochs

For datasets that fit in RAM but are too slow for batch EM (e.g., $n = 10^7$), run multiple epochs of mini-batch streaming:

```bash
# 3 epochs over the data, shuffle between epochs
gemmulem -g data.txt --streaming --epochs 3 --shuffle -k 5 -o results.csv
```

After each epoch, Gemmulem resets $\eta_t$ to a smaller value (warmup-proportional decay) to refine the solution.

---

## Implementation Details

### File I/O in Streaming Mode

Gemmulem reads the file using a **sliding buffer** of 64 KB, parsing one mini-batch at a time. This is compatible with:
- Regular files (disk-backed)
- Named pipes: `data_generator | gemmulem --streaming -k 5 -o results.csv`
- Network streams (via `netcat` or similar)
- `/dev/stdin` for piped data

### Checkpointing

For very long streaming runs, Gemmulem can save sufficient statistics at intervals:

```bash
gemmulem -g huge_data.txt --streaming -k 5 --checkpoint 10000 \
         --checkpoint-file checkpoint.bin -o results.csv
```

This saves $\bar{S}_j$, $\bar{N}_j$, and current $\theta_j$ every 10,000 mini-batches. On restart:

```bash
gemmulem -g huge_data.txt --streaming -k 5 --resume checkpoint.bin -o results.csv
```

---

## References

- Cappé, O. & Moulines, E. (2009). On-line expectation-maximization algorithm for latent data models. *Journal of the Royal Statistical Society: Series B*, 71(3), 593–613.
- Sato, M.A. & Ishii, S. (2000). On-line EM algorithm for the normalized Gaussian network. *Neural Computation*, 12(2), 407–432.
- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. *Proceedings of COMPSTAT*, 177–186.
- Robbins, H. & Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3), 400–407.
