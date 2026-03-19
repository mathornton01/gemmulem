---
layout: default
title: Benchmarks
nav_order: 5
description: "Speed and accuracy benchmarks — Gemmule vs sklearn and others."
---

# Benchmarks
{: .fs-9 }

Rigorous head-to-head comparisons with statistical confidence intervals.
{: .fs-6 .fw-300 }

---

## Speed: Gemmule vs sklearn

All benchmarks: 50 seeds per scenario, 95% confidence intervals. Same convergence tolerance (1e-5). Single-threaded. CPU: Intel i9-10850K.

| Scenario | n | k | Gemmule | sklearn | Speedup |
|---|---|---|---|---|---|
| Well-separated 8σ | 10,000 | 3 | **14 ms** | 93 ms | **6.6×** |
| Overlapping 2σ | 10,000 | 3 | **29 ms** | 57 ms | **2.0×** |
| Unequal weights | 10,000 | 3 | **24 ms** | 68 ms | **2.8×** |
| Unequal variance | 10,000 | 3 | **31 ms** | 72 ms | **2.3×** |
| High-k | 20,000 | 8 | **61 ms** | 147 ms | **2.4×** |
| High-k tight | 10,000 | 8 | **48 ms** | 73 ms | **1.5×** |
| Large-n | 50,000 | 3 | **45 ms** | 159 ms | **3.5×** |
| Very large-n | 100,000 | 3 | **78 ms** | 312 ms | **4.0×** |

**Result: 8/8 wins. Accuracy tied on 7/8 (within 95% CI).**

### Why is Gemmule faster?

1. **AVX2 SIMD E-step**: processes 4 doubles per cycle with tiled memory access (TILE=128)
2. **O(n·k) k-means++**: maintains running min-distance instead of recomputing all centers
3. **Fused Lloyd iterations**: single-pass assign + accumulate (cache-friendly)
4. **SQUAREM acceleration**: quadratic convergence after iter 30
5. **No Python overhead**: pure C, no interpreter, no GIL

---

## Accuracy: Parameter Recovery

On well-separated Gaussian data (10σ), both Gemmule and sklearn achieve:

| Metric | Gemmule | sklearn |
|---|---|---|
| Mean recovery (MAE) | ≤ 0.02 | ≤ 0.02 |
| Variance recovery | ≤ 0.05 | ≤ 0.05 |
| Weight recovery | ≤ 0.01 | ≤ 0.01 |
| LL converges | ✅ | ✅ |

---

## Adaptive Distribution Finder

Gemmule's adaptive mode discovers both **k** (number of components) and the **distribution family** from data alone.

### Standard Benchmark (16 scenarios, 3 seeds each)

| Scenario | k Accuracy | Family Exact | Family Group |
|---|---|---|---|
| Gaussian k=2 | 100% | 100% | 100% |
| Gaussian k=3 | 100% | 100% | 100% |
| Gaussian k=5 | 100% | 33% | 33% |
| Exponential k=2 | 67% | 67% | 100% |
| Exponential k=3 | 100% | 0% | 100% |
| Gamma k=2 | 0% | 100% | 100% |
| Gamma k=3 | 67% | 100% | 100% |
| LogNormal k=2 | 100% | 100% | 100% |
| Weibull k=2 | 100% | 0% | 100% |
| Poisson k=2 | 100% | 100% | 100% |
| Poisson k=3 | 67% | 67% | 67% |
| Beta k=2 | 100% | 0% | 0% |
| Laplace k=2 | 100% | 100% | 100% |
| Logistic k=2 | 100% | 100% | 100% |
| Gumbel k=2 | 33% | 67% | 67% |
| **Average** | **77%** | **65%** | **79%** |

### Extended Benchmark (24 hard scenarios)

| Category | k Accuracy | Highlights |
|---|---|---|
| **Heavy overlap (2σ)** | 100% | Correctly finds 2 components |
| **99/1 imbalance** | 100% | Finds the 1% minority component |
| **Gaussian k=10** | 100% | All 10 components, correct family |
| **Large n=100K** | 100% | Perfect at scale |
| **3-family mix** | 100% | Correctly splits 3 different families |
| **Mixed scale (σ: 0.1 vs 10)** | 100% | Handles variance differences |

---

## Comparison with Other Libraries

| Feature | Gemmule | sklearn | R mixtools | R mclust | pomegranate |
|---|---|---|---|---|---|
| Language | C | Python | R | R | Python |
| Families | **35** | 1 | 3 | 1 | 5 |
| Auto-k | **5 methods** | ❌ | ❌ | BIC | ❌ |
| SIMD | **AVX2** | via NumPy | ❌ | ❌ | ❌ |
| GPU | **OpenCL** | ❌ | ❌ | ❌ | ❌ |
| Streaming | **✅** | ❌ | ❌ | ❌ | ❌ |
| Dependencies | **0** | NumPy, SciPy | Several | Several | Several |
| Multivariate | Full/Diag/Sph | Full/Diag/Sph | ❌ | **14 types** | Full |

---

## Test Coverage

| Suite | Tests | Assertions | Time |
|---|---|---|---|
| Unit tests (EM core) | 10 | ~45 | 0.02s |
| Distribution tests | 17 | ~88 | 0.25s |
| Pearson tests | 7 | ~36 | 4.1s |
| Adaptive tests | 6 | ~24 | 2.6s |
| Spectral/Online/MML | 8 | ~32 | 0.8s |
| Multivariate | 7 | ~28 | 0.2s |
| **Edge cases** | **50** | **344** | **4.7s** |
| **Total** | **105** | **~600** | **12.7s** |

Plus 26 Python correctness checks and 124 Python feature tests.

---

## Reproduce These Benchmarks

```bash
cd gemmulem

# Speed benchmark
python3 benchmark/final_fair.py

# Correctness validation
python3 benchmark/correctness_check.py

# Full feature test (124 tests)
python3 benchmark/full_feature_test.py

# Adaptive accuracy
python3 benchmark/adaptive_accuracy.py

# Extended hard scenarios
python3 benchmark/adaptive_extended.py
```
