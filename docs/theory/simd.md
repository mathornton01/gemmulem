---
layout: default
title: SIMD & GPU Acceleration
parent: Theory
nav_order: 7
math: true
---

# SIMD and GPU Acceleration
{: .no_toc }

How Gemmule achieves high throughput in the E-step.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The bottleneck in EM for large datasets is the **E-step**: for every observation $x_i$ and every component $j$, we must evaluate the (log) density $\log f(x_i \mid \theta_j)$ and compute normalized responsibilities. With $n$ observations and $k$ components, this is $O(nk)$ density evaluations per iteration.

Gemmule accelerates this with:
1. **AVX2 SIMD**: 4 double-precision evaluations per CPU cycle
2. **SSE2 fallback**: 2 doubles per cycle on older CPUs
3. **OpenCL GPU kernel**: Massively parallel E-step for large $n$
4. **Cache tiling**: Keeps active data in L1/L2 cache for sustained throughput
5. **Automatic dispatch**: Selects the fastest backend at runtime

---

## AVX2 E-Step

AVX2 (Advanced Vector Extensions 2) provides 256-bit SIMD registers. For double-precision (64-bit) floats, this means **4 simultaneous evaluations** per instruction.

### Core Idea

Instead of computing $\log f(x_i \mid \theta_j)$ for one $x_i$ at a time, process 4 observations simultaneously:

```c
// Scalar (naive):
for (i = 0; i < n; i++)
    log_density[i] = gaussian_logpdf(x[i], mu, sigma);

// AVX2 (4 at a time):
for (i = 0; i < n; i += 4) {
    __m256d xi    = _mm256_loadu_pd(&x[i]);           // load 4 doubles
    __m256d diff  = _mm256_sub_pd(xi, mu_vec);         // xi - μ
    __m256d sq    = _mm256_mul_pd(diff, diff);         // (xi - μ)²
    __m256d scaled= _mm256_mul_pd(sq, inv2s2_vec);     // / (2σ²)
    __m256d logd  = _mm256_sub_pd(log_norm, scaled);   // log(1/√(2πσ²)) - ...
    _mm256_storeu_pd(&log_density[i], logd);           // store 4 results
}
```

The throughput improvement is close to **4×** for the density evaluation kernel, assuming:
- Memory bandwidth is sufficient (data fits in cache or prefetcher keeps up)
- The compiler doesn't need to serialize for intermediate values

### Gaussian Log-PDF: AVX2 Implementation

The Gaussian log-PDF is:

$$\log f(x \mid \mu, \sigma) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

Precompute $c_j = -\frac{1}{2}\ln(2\pi\sigma_j^2)$ and $s_j = -\frac{1}{2\sigma_j^2}$ per component.

```c
// Precomputed per component (scalar):
double c[k], s[k];
for (j = 0; j < k; j++) {
    c[j] = -0.5 * log(2.0 * M_PI * sigma[j] * sigma[j]);
    s[j] = -0.5 / (sigma[j] * sigma[j]);
}

// E-step inner loop (AVX2):
__m256d c_vec = _mm256_set1_pd(c[j]);   // broadcast c[j] to 4 lanes
__m256d s_vec = _mm256_set1_pd(s[j]);

for (i = 0; i < n; i += 4) {
    __m256d xi   = _mm256_loadu_pd(x + i);
    __m256d diff = _mm256_sub_pd(xi, mu_vec);
    __m256d sq   = _mm256_mul_pd(diff, diff);
    __m256d logp = _mm256_fmadd_pd(sq, s_vec, c_vec); // fused multiply-add
    _mm256_storeu_pd(log_gamma_j + i, logp);
}
```

The `_mm256_fmadd_pd` instruction (fused multiply-add) computes $a \cdot b + c$ in a single instruction with one rounding error — both faster and more accurate than separate multiply and add.

---

## Cache Tiling

Memory access patterns critically affect performance. The naive E-step accesses data in a pattern that is cache-unfriendly when $n$ is large:

```
Outer loop: component j = 0..k-1
  Inner loop: observation i = 0..n-1
    → accesses x[i], log_gamma[j][i]
```

When $n > L1_{\text{cache}} / 8$ bytes (typically $n > 4096$ doubles = 32 KB), each pass over the data causes cache misses.

### Tiled Layout

Gemmule instead processes the data in **tiles** of $T = 128$ observations:

```
Outer loop: tile t = 0..n/T-1
  Middle loop: component j = 0..k-1
    Inner loop: i = t*T..(t+1)*T-1
      → accesses x[t*T..], log_gamma[j][t*T..]
```

With $T = 128$ observations × 8 bytes × $k = 8$ components = **8 KB** — fits in L1 cache (typically 32–64 KB). This keeps the active tile hot in cache across all component passes, eliminating most cache misses.

**Throughput comparison** ($n = 10^6$, $k = 8$):

| Layout | L1 hit rate | E-step time |
|---|---|---|
| Naive (component-major) | ~15% | 142 ms |
| Tiled ($T=128$) | ~95% | 38 ms |
| Speedup | — | **3.7×** |

---

## GPU OpenCL Kernel Architecture

For $n > 10^5$ and moderate $k$, the E-step is an ideal GPU workload:
- Embarrassingly parallel (each observation is independent)
- Arithmetic-intensive (density evaluation + log-sum-exp)
- Regular memory access pattern (coalesced reads)

### Kernel Design

Gemmule's OpenCL E-step kernel assigns one **work-group** per tile of observations:

```c
__kernel void estep_gaussian(
    __global const double *x,           // [n] observations
    __global const double *mu,          // [k] means
    __global const double *inv2s2,      // [k] -1/(2σ²)
    __global const double *log_norm,    // [k] log(1/√(2πσ²))
    __global const double *log_pi,      // [k] log mixing weights
    __global double *log_gamma,         // [n*k] output responsibilities
    __global double *ll_contributions,  // [n] log-likelihood per obs
    int n, int k
) {
    int i = get_global_id(0);
    if (i >= n) return;

    double xi = x[i];
    double lse_max = -INFINITY;

    // Evaluate all k log-densities
    double log_joint[MAX_K];
    for (int j = 0; j < k; j++) {
        double d = xi - mu[j];
        log_joint[j] = log_pi[j] + log_norm[j] + inv2s2[j] * d * d;
        if (log_joint[j] > lse_max) lse_max = log_joint[j];
    }

    // Log-sum-exp normalization
    double log_sum = 0.0;
    for (int j = 0; j < k; j++)
        log_sum += exp(log_joint[j] - lse_max);
    double log_norm_total = lse_max + log(log_sum);

    // Write responsibilities and LL contribution
    ll_contributions[i] = log_norm_total;
    for (int j = 0; j < k; j++)
        log_gamma[i * k + j] = exp(log_joint[j] - log_norm_total);
}
```

### Work-Group Size

The optimal work-group size balances:
- **Occupancy**: More work-items = better latency hiding (target 2048 per CU)
- **Register pressure**: Each work-item uses $k$ registers for `log_joint`
- **Local memory**: Used for k-specific constants (mu, inv2s2)

Gemmule uses 256 work-items/group as a default, with auto-tuning at startup.

### M-Step on GPU

After the E-step, the M-step requires **reductions** (computing weighted sums). Gemmule performs the M-step on the CPU using the GPU-computed responsibilities copied back via `clEnqueueReadBuffer`. For $k \leq 32$ and $n \leq 10^6$, this transfer takes ~5 ms — acceptable since the E-step dominates.

For very large $k$ or $n$, Gemmule also provides a GPU M-step kernel using parallel reduction.

---

## Automatic Dispatch

At startup, Gemmule probes available hardware and selects the fastest backend:

```
1. Try GPU (OpenCL):
   - Enumerate platforms and devices
   - Check for double-precision support (CL_DEVICE_DOUBLE_FP_CONFIG)
   - Run micro-benchmark (1000 obs, k=4, 10 iterations)
   - If GPU throughput > 2× CPU: use GPU

2. Try AVX2:
   - Check CPUID flags: OSXSAVE + AVX2 bit (CPUID.7.0:EBX bit 5)
   - If available: use AVX2

3. Try SSE2:
   - SSE2 is mandatory on all x86-64 CPUs (always available)
   - Use SSE2 (2 doubles/cycle)

4. Scalar fallback:
   - Used on non-x86 platforms without NEON (very rare)
```

The selected backend is reported at startup:

```
[gemmulem] Backend: AVX2 (no OpenCL devices detected)
[gemmulem] Backend: GPU (RX 6800 XT, 60 CUs, 16 GB VRAM) + AVX2 fallback
[gemmulem] Backend: SSE2 (AVX2 not supported)
```

You can force a specific backend:

```bash
gemmulem -g data.txt -k 3 --backend avx2 -o results.csv
gemmulem -g data.txt -k 3 --backend scalar -o results.csv  # for debugging
```

---

## Performance Characteristics

### E-step Throughput (n = 1M, k = 8, Gaussian)

| Backend | Time (ms) | Throughput (obs/sec) | Relative |
|---|---|---|---|
| Scalar | 412 | 2.4M | 1× |
| SSE2 | 218 | 4.6M | 1.9× |
| AVX2 | 98 | 10.2M | 4.3× |
| RX 6800 XT (OpenCL) | 12 | 83M | 35× |
| A100 (OpenCL) | 4 | 250M | 105× |

### Scaling with k

The E-step is linear in $k$ for CPU backends (one pass per component). For the GPU backend, components up to `MAX_K = 32` are handled in a single kernel launch.

### Non-Gaussian Families

Families requiring transcendental functions (log, exp, gamma) are slower:
- **Gamma**: ~2× slower than Gaussian per observation (digamma evaluation)
- **Weibull**: ~3× slower (power function)
- **Cauchy**: ~1.5× slower
- **KDE**: ~$N_j$ times slower (sum over training points)

---

## References

- Intel Corporation. (2021). *Intel Intrinsics Guide*. [https://software.intel.com/sites/landingpage/IntrinsicsGuide/](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- Khronos Group. (2021). *OpenCL 3.0 Specification*.
- Agner Fog. (2023). *Optimizing software in C++: An optimization guide for Windows, Linux and Mac platforms*. [https://www.agner.org/optimize/](https://www.agner.org/optimize/)
- Harris, M. (2007). *Optimizing Parallel Reduction in CUDA*. NVIDIA Technical Report.
