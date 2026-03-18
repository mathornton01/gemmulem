# Gemmulem v2.0

**The most comprehensive univariate mixture model library.** 35 distribution families, 5 automatic model selection methods, spectral initialization, online EM — all in pure C with zero dependencies.

## Benchmark

| n | Gemmulem | sklearn GMM | Speedup |
|---:|---:|---:|---:|
| 1,000 | 14ms | 71ms | **5.1×** |
| 5,000 | 15ms | 45ms | **2.9×** |
| 10,000 | 20ms | 87ms | **4.3×** |
| 50,000 | 49ms | 124ms | **2.5×** |
| 100,000 | 90ms | 166ms | **1.8×** |

## Feature Comparison

| Feature | Gemmulem | sklearn | R mixtools | pomegranate |
|---|:---:|:---:|:---:|:---:|
| Distribution families | **35** | 1 | 3 | 5 |
| Auto k-selection methods | **5** | 1 | 0 | 0 |
| Spectral initialization | ✓ | ✗ | ✗ | ✗ |
| Online/stochastic EM | ✓ | ✗ | ✗ | ✗ |
| File-based streaming EM | ✓ | ✗ | ✗ | ✗ |
| SIMD-accelerated E-step | ✓ (AVX2/SSE2) | ✗ | ✗ | ✗ |
| GPU OpenCL E-step | ✓ | ✗ | ✗ | ✗ |
| Multivariate Gaussian mixture | ✓ | ✓ | ✗ | ✓ |
| Multivariate Student-t mixture | ✓ | ✗ | ✗ | ✗ |
| Multivariate auto-k | ✓ | ✗ | ✗ | ✗ |
| k-means++ init (10 restarts) | ✓ | ✓ | ✗ | ✗ |
| Nonparametric (KDE) | ✓ | ✗ | ✗ | ✗ |
| Cross-family adaptive EM | ✓ | ✗ | ✗ | ✗ |
| Pearson auto-type | ✓ | ✗ | ✗ | ✗ |
| SQUAREM acceleration | ✓ | ✗ | ✗ | ✗ |
| External dependencies | **none** | numpy | R core | PyTorch |

## Distribution Families (35)

**Real-valued (ℝ):** Gaussian, Student-t, Laplace, Cauchy, Logistic, Gumbel, Skew-Normal, Generalized Gaussian

**Positive (ℝ⁺):** Exponential, Gamma, Log-Normal, Weibull, Inverse Gaussian, Rayleigh, Pareto, Chi-squared, F, Log-Logistic, Nakagami, Lévy, Gompertz, Burr XII, Half-Normal, Maxwell-Boltzmann

**Unit interval [0,1]:** Beta, Kumaraswamy, Triangular

**Discrete:** Poisson, Binomial, Negative Binomial, Geometric, Zipf

**Meta/Special:** Pearson (auto-selects type from moments), Uniform, KDE (nonparametric)

## Model Selection Methods

| Method | Flag | Description |
|---|---|---|
| **BIC** | `--kmethod bic` | Bayesian Information Criterion (default) — conservative |
| **AIC** | `--kmethod aic` | Akaike — lighter penalty, finds more components |
| **ICL** | `--kmethod icl` | Integrated Classification Likelihood — penalizes overlap |
| **VBEM** | `--kmethod vbem` | Variational Bayes EM — genuine M-step over k |
| **MML** | `--kmethod mml` | Minimum Message Length (Wallace-Freeman) |

## Quick Start

### Install

```bash
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem && mkdir build && cd build
cmake .. && cmake --build .
```

Requires only a C/C++ compiler and CMake. No LAPACK, no BLAS, no external libraries.

### Basic Usage

```bash
# Fit 2-component Gaussian mixture
gemmulem -g data.txt -d Gaussian -k 2

# Fit Student-t mixture with 3 components
gemmulem -g data.txt -d StudentT -k 3

# Fit Gamma mixture with verbose output
gemmulem -g data.txt -d Gamma -k 3 -v

# Auto-detect number of components (BIC-driven)
gemmulem -g data.txt --adaptive --kmax 8

# Auto-detect with VBEM (Bayesian)
gemmulem -g data.txt --adaptive --kmethod vbem --kmax 10

# Online (mini-batch) EM for large in-memory datasets
gemmulem -g big_data.txt -d Gaussian -k 4 --online --batch-size 512

# Streaming EM for datasets too large to fit in memory
gemmulem -g huge_data.txt -d Gaussian -k 4 --stream --chunk-size 10000 --passes 10

# Multivariate Gaussian mixture (row-major CSV/space-separated input)
gemmulem -g mvdata.txt --mv -k 3 --cov full

# Multivariate Student-t with auto-k
gemmulem -g mvdata.txt --mvt --mv-autok --kmax 5

# Save output to file
gemmulem -g data.txt -d Gaussian -k 2 -o results.txt

# Verbose output
gemmulem -g data.txt -d Gaussian -k 2 -v
```

### Specify Distribution Family

Use `-d` / `--dist` with any family name:
```
Gaussian, StudentT, Laplace, Cauchy, Logistic, Gumbel, SkewNormal, GenGaussian,
Exponential, Gamma, LogNormal, Weibull, InvGaussian, Rayleigh, Pareto, ChiSquared,
F, LogLogistic, Nakagami, Levy, Gompertz, BurrXII, HalfNormal, MaxwellBoltzmann,
Beta, Kumaraswamy, Triangular, Poisson, Binomial, NegBinomial, Geometric, Zipf,
Pearson, Uniform, KDE
```

### Auto Mode

Exhaustive search over all compatible families × k values:
```bash
gemmulem -g data.txt --auto --kmax 5
```

### Adaptive Mode

Per-component family selection with automatic k:
```bash
gemmulem -g data.txt --adaptive --kmethod bic --kmax 8
```

Each component independently selects its best distribution family. Discovers both the number of components and the shape of each.

## EM Modes

### Standard EM
Full-data E-step and M-step each iteration. Deterministic, guaranteed monotone LL increase.

### Online/Stochastic EM
Mini-batch processing with Cappé & Moulines (2009) step-size schedule `η_t = (t+τ)^(-κ)`. Scales to millions of data points. Within 0.002% log-likelihood of standard EM on benchmark data.

```bash
gemmulem -g data.txt -d Gaussian -k 3 --online --batch-size 256
```

### Spectral Initialization
Gap-detection on sorted data provides provably good starting parameters for well-separated components. On synthetic N(−6,0.8) + N(6,0.8), recovers means of −6.00 and 6.02 *before EM even runs*.

## C/C++ API

```c
#include "distributions.h"

// Fixed family, fixed k
MixtureResult result;
UnmixGeneric(data, n, DIST_GAUSSIAN, k, 300, 1e-5, 0, &result);

// Adaptive: auto k + auto family
AdaptiveResult adaptive;
UnmixAdaptiveEx(data, n, k_max, 300, 1e-5, 0, KMETHOD_BIC, &adaptive);

// Online EM
MixtureResult online;
UnmixOnline(data, n, DIST_GAUSSIAN, k, 500, 1e-5, 256, 0, &online);

// Spectral initialization
double means[k], weights[k];
SpectralInit(data, n, k, means, weights);

// Cleanup
ReleaseMixtureResult(&result);
ReleaseAdaptiveResult(&adaptive);
```

## Legacy Modes

### Coarse Multinomial EM
The original Gemmulem mode for compatibility-pattern data:
```bash
gemmulem -i patterns.tsv -o output.txt
```

### R Extension
```R
library(rgemmulem)
rgemmulem_unmixgaussians(values, num_dist)
```

### Python Extension
```python
import pygemmulem
pygemmulem.unmixgaussians(values, num_distribution)
```

## Options Reference

### Input / Output

| Flag | Description | Default |
|---|---|---|
| `-g FILE` | Input data file (one value per line) | — |
| `-i FILE` | Compatibility matrix file (MN mode) | — |
| `-e FILE` | Exponential mixture values file | — |
| `-o FILE` | Output file | `<timestamp>.abdn.txt` |

### Distribution & Components

| Flag | Description | Default |
|---|---|---|
| `-d FAMILY` | Distribution family name | Gaussian |
| `-k N` | Number of mixture components | 3 |
| `--kmax N` | Max components for adaptive/auto/mv-autok | 8 |

### Convergence

| Flag | Description | Default |
|---|---|---|
| `-r TOL` | Relative convergence tolerance | 0.00001 |
| `-m N` | Maximum EM iterations | 1000 |
| `-c SEED` | Integer seed for RNG | — |

### Model Selection

| Flag | Description | Default |
|---|---|---|
| `--adaptive` | Adaptive per-component EM (auto-k + auto-family) | off |
| `--auto` | Exhaustive family × k search | off |
| `--kmethod M` | k-selection criterion: `bic` `aic` `icl` `vbem` `mml` | bic |

### Online / Streaming EM

| Flag | Description | Default |
|---|---|---|
| `--online` | Mini-batch (online) EM with step-size schedule | off |
| `--batch-size N` | Mini-batch size for `--online` | n/4 |
| `--stream` | File-based streaming EM (constant memory) | off |
| `--chunk-size N` | Rows per chunk for `--stream` | 10000 |
| `--passes N` | Number of full-data passes for `--stream` | 10 |

### Multivariate Modes

| Flag | Description | Default |
|---|---|---|
| `--mv` | Multivariate Gaussian mixture | off |
| `--mvt` | Multivariate Student-t mixture | off |
| `--mv-autok` | Auto-select k for multivariate mixture | off |
| `--dim D` | Dimensionality (auto-detected from data) | — |
| `--cov TYPE` | Covariance structure: `full` `diagonal` `spherical` | full |

### Output / Display

| Flag | Description | Default |
|---|---|---|
| `-v` | Verbose iteration-by-iteration output | off |
| `-t` | Print results to terminal (skip file write) | off |

## Building & Testing

```bash
mkdir build && cd build
cmake .. && cmake --build .
ctest --output-on-failure
```

6 test suites: unit_tests, distribution_tests, pearson_tests, adaptive_tests, spectral_online_mml_tests, multivariate_tests.

## License

GPL v3

## Contact

- Micah Thornton — Micah.Thornton@UTSouthwestern.edu
- Chanhee Park — Chanhee.Park@UTSouthwestern.edu

## Citation

If you use Gemmulem in your research, please cite:
```
Thornton, M. & Park, C. (2026). Gemmulem: General Mixed-Family Expectation
Maximization with Adaptive Model Selection. University of Texas Southwestern
Medical Center.
```
