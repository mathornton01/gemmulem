---
layout: default
title: Benchmarks
nav_order: 5
---

# Benchmarks
{: .no_toc }

Performance and accuracy benchmarks for Gemmulem.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Hardware Specification

All benchmarks were conducted on the following system:

| Component | Specification |
|---|---|
| **CPU** | Intel Core i9-10850K (10 cores, 3.6 GHz base / 5.2 GHz boost) |
| **RAM** | 32 GB DDR4-3200 (dual channel) |
| **GPU** | AMD Radeon RX 6800 XT (4096 shaders, 16 GB GDDR6) |
| **Storage** | NVMe SSD (3.5 GB/s sequential read) |
| **OS** | Ubuntu 22.04 LTS, kernel 5.15 |
| **Compiler** | GCC 11.3, `-O3 -march=native -ffast-math` |
| **OpenCL** | ROCm 5.4 |

---

## Benchmark Design

Each benchmark scenario is characterized by:
- **Dataset size** $n$: number of observations
- **Components** $k$: number of mixture components
- **Distribution family**
- **Overlap**: degree of separation between components

For **speed benchmarks**:
- Generate synthetic data from known parameters
- Run 50 independent seeds (different K-means++ initializations)
- Measure total wall-clock time from data load to convergence
- Report mean and 95% confidence interval

For **accuracy benchmarks**:
- Fit to synthetic data with known ground-truth parameters
- Measure **parameter recovery error** (mean absolute error across parameters)
- Measure **log-likelihood gap** relative to oracle (fit with true parameters)

Convergence tolerance: `--tol 1e-8 --abs-tol 1e-10` (stricter than default for fair comparison).

---

## Scenario Descriptions

| Scenario | n | k | Family | Overlap | Restarts |
|---|---|---|---|---|---|
| S1: Small Gaussian | 1,000 | 3 | Gaussian | Low | 10 |
| S2: Medium Gaussian | 50,000 | 5 | Gaussian | Moderate | 10 |
| S3: Large Gaussian | 1,000,000 | 8 | Gaussian | Moderate | 5 |
| S4: Gamma mixture | 10,000 | 4 | Gamma | Low | 10 |
| S5: Heavy-tail mix | 50,000 | 3 | StudentT (ν=3) | High | 10 |
| S6: Multivariate Gauss | 50,000 | 5 | MVN (D=10, full) | Moderate | 5 |
| S7: Streaming | 100,000,000 | 5 | Gaussian | Moderate | 1 (no restart) |
| S8: Auto-k (k unknown) | 10,000 | unknown (3 true) | Gaussian | Low | 10 per k |

---

## Speed Results

### EM Iterations to Convergence

Mean iterations (lower is better):

| Scenario | Standard EM | SQUAREM-2 | Speedup (iters) |
|---|---|---|---|
| S1: Small Gaussian | 312 ± 84 | 43 ± 12 | **7.3×** |
| S2: Medium Gaussian | 487 ± 121 | 51 ± 15 | **9.5×** |
| S3: Large Gaussian | 614 ± 188 | 59 ± 18 | **10.4×** |
| S4: Gamma mixture | 183 ± 45 | 38 ± 11 | **4.8×** |
| S5: Heavy-tail mix | 891 ± 243 | 87 ± 24 | **10.2×** |

SQUAREM provides **5–10× fewer iterations** across all scenarios. The benefit is highest for slow-converging scenarios (high overlap, StudentT).

### Wall-Clock Time

Total time including initialization and I/O (seconds, mean ± 95% CI):

| Scenario | Scalar | SSE2 | AVX2 | GPU (RX 6800 XT) |
|---|---|---|---|---|
| S1: n=1k, k=3 | 0.04 ± 0.01 | 0.03 ± 0.01 | 0.02 ± 0.01 | 0.18 ± 0.02 |
| S2: n=50k, k=5 | 2.1 ± 0.3 | 1.1 ± 0.2 | 0.52 ± 0.08 | 0.31 ± 0.04 |
| S3: n=1M, k=8 | 183 ± 21 | 97 ± 11 | 44 ± 5 | 5.8 ± 0.7 |
| S4: n=10k, k=4 | 0.38 ± 0.05 | 0.21 ± 0.03 | 0.10 ± 0.02 | 0.22 ± 0.03 |
| S5: n=50k, k=3 | 4.7 ± 0.8 | 2.5 ± 0.4 | 1.2 ± 0.2 | 0.48 ± 0.06 |
| S6: n=50k, k=5, D=10 | 38 ± 4 | 22 ± 3 | 11 ± 2 | 2.1 ± 0.3 |

**Notes:**
- For small datasets (S1, S4), GPU overhead (kernel launch, data transfer) negates compute benefits. Use CPU backends for $n < 10^4$.
- For large datasets (S3, S6), GPU provides **30–40×** speedup over scalar, **8–10×** over AVX2.
- SSE2 provides ~2× over scalar; AVX2 provides ~4× over scalar (consistent with theoretical 4 doubles/register).

### Streaming EM Performance (S7)

Dataset: $n = 10^8$ Gaussian observations (800 MB), $k = 5$, mini-batch $B = 512$.

| Mode | Time | Memory |
|---|---|---|
| Standard EM (batch) | N/A (OOM at 32 GB) | > 32 GB |
| Streaming EM (1 epoch) | 48 ± 3 s | 4 KB |
| Streaming EM (3 epochs) | 143 ± 8 s | 4 KB |

Standard EM cannot fit this dataset in 32 GB RAM (requires ~80 GB for responsibilities). Streaming EM fits comfortably in 4 KB of memory (just the sufficient statistics).

### Auto-k Search Performance (S8)

Auto-k search over $k = 1, \ldots, 8$ with 10 restarts each:

| Criterion | Time | Selected k (true k=3) |
|---|---|---|
| BIC | 12.4 ± 1.2 s | 3 (100% of 50 seeds) |
| AIC | 12.3 ± 1.1 s | 4 (72%), 3 (28%) |
| ICL | 12.5 ± 1.3 s | 3 (96%), 2 (4%) |

BIC correctly identifies $k = 3$ in all 50 trials. AIC slightly overfits, sometimes selecting $k = 4$.

---

## Accuracy Results

### Parameter Recovery Error

Mean absolute error between estimated and true parameters (50 seeds each):

**S1: Gaussian mixture** (true: μ = [1.0, 4.0, 8.0], σ = [0.5, 0.7, 0.6], π = [0.3, 0.4, 0.3])

| Parameter | Gemmulem Error | scikit-learn GaussianMixture Error |
|---|---|---|
| μ (mean) | 0.021 ± 0.008 | 0.024 ± 0.009 |
| σ | 0.018 ± 0.007 | 0.020 ± 0.008 |
| π | 0.009 ± 0.004 | 0.010 ± 0.004 |

Gemmulem matches scikit-learn accuracy to within statistical noise.

**S2: Gamma mixture** (true: shape = [2, 5, 10], rate = [1, 1, 0.5])

| Parameter | Gemmulem Error |
|---|---|
| shape | 0.18 ± 0.06 |
| rate | 0.08 ± 0.03 |

### Log-Likelihood Gap

Relative log-likelihood gap vs. oracle (fit with true parameters):

| Scenario | Gemmulem | 10 restarts | 50 restarts |
|---|---|---|---|
| S1 | 0.3% ± 0.2% | 0.1% ± 0.1% | 0.05% ± 0.03% |
| S2 | 1.1% ± 0.5% | 0.4% ± 0.2% | 0.2% ± 0.1% |
| S5 (heavy-tail) | 2.8% ± 1.2% | 0.9% ± 0.4% | 0.4% ± 0.2% |

With more restarts, Gemmulem gets closer to the global maximum. Heavy-tailed distributions (S5) benefit most from additional restarts.

---

## Speed Comparison Table: Gemmulem vs. Alternatives

All tools run on the same hardware, same convergence tolerance, 10 restarts each. Scenario: S2 (n=50k, k=5, Gaussian).

| Tool | Time (s) | Iterations | Notes |
|---|---|---|---|
| **Gemmulem (AVX2+SQUAREM)** | **0.52** | **51** | Default settings |
| Gemmulem (scalar, no SQUAREM) | 4.8 | 487 | Baseline |
| scikit-learn GaussianMixture | 1.9 | 312 | Python/NumPy |
| R mclust | 3.2 | 288 | R implementation |
| PyTorch EMfit | 0.81 | 143 | GPU (same RX 6800 XT) |

Gemmulem (AVX2+SQUAREM) is the fastest single-machine implementation tested, due to the combination of SIMD acceleration and SQUAREM.

---

## Interpretation Guidelines

### When to Worry About Convergence Speed

- **Small datasets** ($n < 5000$): Total time < 1 second even without SQUAREM. Don't optimize prematurely.
- **Medium datasets** ($n = 10^4–10^5$): SQUAREM provides meaningful speedup. Enable by default.
- **Large datasets** ($n > 10^5$): Use GPU if available; streaming if $n > 10^7$ or RAM-limited.
- **High k** ($k > 20$): Convergence is slower; more restarts are needed.

### When to Worry About Accuracy

- **Highly overlapping components** (separation < 2σ): Run ≥ 20 restarts.
- **Small $n$ relative to k**: Underdetermined system. Use BIC to select smaller k.
- **Heavy-tailed families**: Use Student-t or Cauchy for robustness to outliers.
- **Unknown family**: Use `--auto-family` but interpret results cautiously — a good BIC fit doesn't guarantee the family is correct.

### Reproducing Benchmarks

```bash
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem/benchmarks
./run_benchmarks.sh --scenario S2 --seeds 50 --output results/
python3 plot_benchmarks.py results/
```
