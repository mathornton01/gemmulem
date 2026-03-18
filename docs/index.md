---
layout: default
title: Home
nav_order: 1
description: "Gemmule — General Mixed-Family Expectation Maximization"
permalink: /
---

# Gemmule
{: .fs-9 }

**General Mixed-Family Expectation Maximization**
{: .fs-6 .fw-300 }

Fit finite mixture models over 35 distribution families with a single command. SIMD-accelerated, SQUAREM-boosted, and streaming-ready.

[Get Started]({% link quickstart.md %}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/mathornton01/gemmulem){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What is Gemmule?

Gemmule is a high-performance C library and command-line tool for fitting **finite mixture models** using the **Expectation-Maximization (EM) algorithm**. It supports a broad catalog of 35 parametric distribution families — from the everyday Gaussian and Poisson to exotic families like the Burr, Nakagami, and Gompertz — plus kernel density estimation as a nonparametric fallback.

### Key Features

| Feature | Description |
|---|---|
| **35 distribution families** | Continuous, discrete, symmetric, skewed, heavy-tailed, bounded |
| **Automatic family selection** | BIC/AIC/ICL/VBEM/MML model selection criteria |
| **SQUAREM acceleration** | Superlinear convergence via squared iterative steps |
| **SIMD vectorization** | AVX2/SSE2 E-step for 4× throughput |
| **GPU support** | OpenCL kernel for massive parallel E-step |
| **Streaming EM** | Online updates for datasets that don't fit in RAM |
| **K-means++ initialization** | Reproducible, high-quality starting points |
| **Multivariate support** | Full, diagonal, and spherical covariance Gaussian mixtures |

---

## Quick Install

### Prerequisites

- CMake ≥ 3.15
- A C11-capable compiler (GCC ≥ 7, Clang ≥ 5, MSVC 2019+)
- (Optional) OpenCL SDK for GPU support

### Build from Source

```bash
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

For GPU support:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENCL=ON
```

For a static library only (no CLI):

```bash
cmake .. -DBUILD_CLI=OFF -DBUILD_SHARED_LIBS=OFF
```

Verify your installation:

```bash
gemmulem --version
# gemmulem 1.0.0 (AVX2+OpenCL)
```

---

## First Example

Suppose you have a file `data.txt` containing one observation per line:

```
2.31
5.87
1.94
6.12
2.05
5.44
...
```

Fit a 3-component Gaussian mixture model:

```bash
gemmulem -g data.txt -k 3 -o results.csv
```

Gemmule will:
1. Load your data
2. Initialize components with K-means++
3. Run EM until convergence (with SQUAREM acceleration)
4. Write per-component parameters and responsibilities to `results.csv`

### Sample Output

```
# Gemmule results — 3-component Gaussian mixture
# Log-likelihood: -1423.87  BIC: 2891.4  AIC: 2863.7
component,weight,mu,sigma
1,0.3412,2.15,0.47
2,0.3289,5.93,0.51
3,0.3299,9.78,0.44

# Per-observation responsibilities (first 5):
obs,gamma_1,gamma_2,gamma_3
1,0.9821,0.0174,0.0005
2,0.0031,0.9954,0.0015
...
```

### Automatic Family Selection

Not sure which distribution fits your data? Let Gemmule try all families and pick the best by BIC:

```bash
gemmulem -g data.txt -k 3 --auto-family --criterion bic -o results.csv
```

---

## Documentation Overview

| Section | What you'll find |
|---|---|
| [Quick Start]({% link quickstart.md %}) | Installation, first fit, reading output |
| [EM Algorithm]({% link theory/em-algorithm.md %}) | Full mathematical derivation of EM |
| [Initialization]({% link theory/initialization.md %}) | K-means++ and PRNG details |
| [SQUAREM]({% link theory/squarem.md %}) | Acceleration theory and implementation |
| [Model Selection]({% link theory/model-selection.md %}) | BIC, AIC, ICL, VBEM, MML |
| [Distributions]({% link theory/distributions.md %}) | All 35 families with formulas and M-step updates |
| [Multivariate]({% link theory/multivariate.md %}) | Multivariate Gaussian and Student-t mixtures |
| [SIMD & GPU]({% link theory/simd.md %}) | Vectorization and GPU acceleration |
| [Streaming EM]({% link theory/streaming.md %}) | Online EM for large datasets |
| [CLI Reference]({% link cli-reference.md %}) | Every flag, format, and exit code |
| [Benchmarks]({% link benchmarks.md %}) | Speed and accuracy results |
| [C API]({% link api-reference.md %}) | Public API for library users |
| [FAQ]({% link faq.md %}) | Common questions |

---

## Citation

If you use Gemmule in research, please cite:

```bibtex
@software{gemmulem,
  author  = {Thornton, Micah},
  title   = {Gemmule: General Mixed-Family Expectation Maximization},
  year    = {2024},
  url     = {https://github.com/mathornton01/gemmulem}
}
```

---

## License

Gemmule is released under the [MIT License](https://github.com/mathornton01/gemmulem/blob/main/LICENSE).
