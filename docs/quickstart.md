---
layout: default
title: Quick Start
nav_order: 2
---

# Quick Start
{: .no_toc }

Get up and running with Gemmule in under 10 minutes.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

### Linux

#### Ubuntu / Debian

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# (Optional) OpenCL for GPU support
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers

# Clone and build
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENCL=ON
make -j$(nproc)
sudo make install
```

#### Fedora / RHEL

```bash
sudo dnf install -y gcc cmake git ocl-icd-devel
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### macOS

```bash
# Install Homebrew if needed: https://brew.sh
brew install cmake git

git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
sudo make install
```

> **Note:** On Apple Silicon (M1/M2), Gemmule will use NEON SIMD instead of AVX2. Performance is similar.

### Windows

1. Install [Visual Studio 2022](https://visualstudio.microsoft.com/) with "Desktop development with C++"
2. Install [CMake](https://cmake.org/download/) and [Git](https://git-scm.com/)
3. Open **Developer Command Prompt for VS 2022** and run:

```cmd
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cmake --install . --config Release
```

Binaries will be installed to `C:\Program Files\gemmulem\bin\`. Add this to your PATH.

### Verify Installation

```bash
gemmulem --version
# gemmulem 1.0.0 (AVX2)    ← scalar fallback if no AVX2
# gemmulem 1.0.0 (AVX2+OpenCL)  ← with GPU support
```

---

## Your First Mixture Model Fit

### Step 1: Prepare Your Data

Gemmule expects a plain-text file with one numeric observation per line:

```
# data.txt — heights in cm (simulated bimodal)
158.2
162.7
170.1
172.4
159.8
171.0
180.3
...
```

Lines beginning with `#` are treated as comments and ignored. The file may contain integer or floating-point values.

For **multivariate data**, provide one vector per line, space- or comma-separated:

```
# multivariate.txt
1.2 3.4 5.6
2.1 3.9 5.0
...
```

### Step 2: Choose Number of Components

If you have domain knowledge, specify `k` directly. If not, use `--auto-k` to search over a range:

```bash
# Fixed k
gemmulem -g data.txt -k 2 -o results.csv

# Automatic k selection (tries k=1..8, picks best by BIC)
gemmulem -g data.txt --auto-k --k-max 8 -o results.csv
```

### Step 3: Run the Fit

```bash
gemmulem -g data.txt -k 3 -o results.csv
```

Gemmule will print progress to stderr:

```
[gemmulem] Loaded 1000 observations
[gemmulem] Initialized 3 components (K-means++)
[gemmulem] Iter    1: LL = -2341.87
[gemmulem] Iter    5: LL = -1893.41  (SQUAREM step)
[gemmulem] Iter   10: LL = -1731.22
[gemmulem] Iter   47: LL = -1423.87  (converged, |ΔLL| < 1e-6)
[gemmulem] BIC = 2891.4   AIC = 2863.7
```

### Step 4: Read the Output

The output CSV has two sections. First, the **component parameters**:

```csv
# Gemmule results
# family: gaussian   k: 3   n: 1000
# log_likelihood: -1423.87   bic: 2891.4   aic: 2863.7
component,weight,mu,sigma
1,0.3412,2.15,0.47
2,0.3289,5.93,0.51
3,0.3299,9.78,0.44
```

Then, the **responsibilities** (posterior probabilities, one row per observation):

```csv
obs,gamma_1,gamma_2,gamma_3,assignment
1,0.9821,0.0174,0.0005,1
2,0.0031,0.9954,0.0015,2
3,0.9773,0.0215,0.0012,1
...
```

The `assignment` column gives the hard-assignment (argmax of responsibilities).

#### Parameter naming by family

| Family | Parameters in output |
|---|---|
| Gaussian | `mu`, `sigma` |
| Exponential | `lambda` |
| Poisson | `lambda` |
| Gamma | `shape` (k), `rate` (β) |
| LogNormal | `mu`, `sigma` (log-scale) |
| Weibull | `shape` (k), `scale` (λ) |
| Beta | `alpha`, `beta` |
| Binomial | `n` (fixed), `p` |

See [Distributions]({% link theory/distributions.md %}) for the complete list.

---

## Choosing the Right Distribution Family

### Decision Guide

**Is your data continuous or discrete?**

- **Discrete** (counts): Poisson, Binomial, NegBinomial, Geometric, Zipf
- **Continuous**: proceed below

**Is your data bounded?**

- **[0, 1]**: Beta, Kumaraswamy
- **[0, ∞)**: Exponential, Gamma, LogNormal, Weibull, InvGaussian, Rayleigh, Pareto, HalfNormal, Maxwell, Levy, Gompertz
- **Unbounded (−∞, +∞)**: Gaussian, StudentT, Laplace, Cauchy, Logistic, Gumbel, SkewNormal, GenGaussian

**Shape characteristics:**

| If your data is... | Consider... |
|---|---|
| Bell-shaped, symmetric | Gaussian, StudentT, Laplace, Logistic |
| Right-skewed, unimodal | LogNormal, Gamma, Weibull, InvGaussian |
| Heavy-tailed | StudentT (small ν), Cauchy, Levy, Pareto |
| Asymmetric | SkewNormal, Gumbel, Gompertz |
| Multi-modal | Any family with k ≥ 2 components |
| Unknown / complex shape | KDE (nonparametric) |

**When in doubt, use `--auto-family`:**

```bash
gemmulem -g data.txt -k 3 --auto-family --criterion bic -o results.csv
```

This fits all applicable families and selects the one with the best BIC. Note: this is more expensive (~35× the work), but still fast on modern hardware.

### Automatic Pearson Family Classification

The Pearson system (Types I–VII) can automatically match your data's skewness and excess kurtosis:

```bash
gemmulem -g data.txt -k 3 -f pearson -o results.csv
```

Gemmule computes the **Pearson κ criterion**:

$$\kappa = \frac{\beta_1(\beta_2 + 3)^2}{4(4\beta_2 - 3\beta_1)(2\beta_2 - 3\beta_1 - 6)}$$

and automatically dispatches to the matching Pearson type. See [Distributions]({% link theory/distributions.md %}#pearson-system) for the full classification table.

---

## Common Workflows

### Comparing Multiple k Values

```bash
for k in 2 3 4 5; do
  gemmulem -g data.txt -k $k -o results_k${k}.csv --quiet
  echo "k=$k: $(grep '^# bic' results_k${k}.csv)"
done
```

### Using a Specific Random Seed

```bash
gemmulem -g data.txt -k 3 --seed 42 -o results.csv
```

This makes the K-means++ initialization fully reproducible.

### Running Multiple Restarts

```bash
# Try 20 random initializations, keep the best
gemmulem -g data.txt -k 3 --restarts 20 -o results.csv
```

### Streaming Mode for Large Files

```bash
# Process a 10GB file without loading it all into RAM
gemmulem -g huge_data.txt -k 5 --streaming -o results.csv
```

See [Streaming EM]({% link theory/streaming.md %}) for details.

### Multivariate Data

```bash
# Fit a 3-component multivariate Gaussian mixture (full covariance)
gemmulem -g multivariate.txt -k 3 --multivariate --cov-type full -o results.csv

# Diagonal covariance (faster, fewer parameters)
gemmulem -g multivariate.txt -k 3 --multivariate --cov-type diagonal -o results.csv
```

---

## Next Steps

- Understand the math: [EM Algorithm]({% link theory/em-algorithm.md %})
- See all CLI options: [CLI Reference]({% link cli-reference.md %})
- Explore all distributions: [Distributions]({% link theory/distributions.md %})
- Check performance numbers: [Benchmarks]({% link benchmarks.md %})
- Use Gemmule from C code: [C API Reference]({% link api-reference.md %})
