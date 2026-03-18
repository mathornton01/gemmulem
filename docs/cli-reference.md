---
layout: default
title: CLI Reference
nav_order: 4
---

# CLI Reference
{: .no_toc }

Complete reference for all `gemmulem` command-line options.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Synopsis

```
gemmulem [OPTIONS]
gemmulem -g <data-file> -k <components> -o <output-file>
gemmulem -g <data-file> --auto-k --k-max 10 -o <output-file>
```

---

## Core Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `-g, --input <file>` | path | (required) | Input data file |
| `-k, --components <n>` | int | — | Number of mixture components |
| `-o, --output <file>` | path | stdout | Output CSV file |
| `-f, --family <name>` | string | `gaussian` | Distribution family |
| `--auto-k` | flag | off | Automatically select k |
| `--k-min <n>` | int | 1 | Minimum k for auto-k search |
| `--k-max <n>` | int | 10 | Maximum k for auto-k search |
| `--auto-family` | flag | off | Try all families, pick best |
| `--criterion <name>` | string | `bic` | Model selection criterion |

### Examples

```bash
# 3-component Gaussian (default)
gemmulem -g data.txt -k 3 -o out.csv

# 5-component Weibull
gemmulem -g data.txt -k 5 -f weibull -o out.csv

# Auto-select k from 1–10 by BIC
gemmulem -g data.txt --auto-k --k-max 10 -o out.csv

# Try all families, pick best 3-component model by AIC
gemmulem -g data.txt -k 3 --auto-family --criterion aic -o out.csv
```

---

## Algorithm Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--max-iter <n>` | int | 1000 | Maximum EM iterations |
| `--tol <ε>` | float | 1e-6 | Relative convergence tolerance |
| `--abs-tol <ε>` | float | 1e-8 | Absolute LL convergence tolerance |
| `--no-squarem` | flag | off | Disable SQUAREM acceleration |
| `--squarem-warmup <n>` | int | 5 (Gaussian) / 0 (others) | Plain EM iterations before SQUAREM |
| `--restarts <n>` | int | auto (min(k², 10)) | K-means++ restart count |
| `--seed <n>` | int | random | PRNG seed for reproducibility |
| `--streaming` | flag | off | Enable streaming (online) EM |
| `--batch-size <n>` | int | 512 | Mini-batch size for streaming |
| `--step-size-exp <α>` | float | 0.6 | Step-size exponent ηₜ = (t+2)^(-α) |
| `--step-size <η>` | float | — | Fixed step-size (overrides exponent) |
| `--epochs <n>` | int | 1 | Passes over data (streaming) |
| `--shuffle` | flag | off | Shuffle data between epochs |
| `--vbem` | flag | off | Use variational Bayes EM |
| `--alpha0 <α>` | float | 1/k | Dirichlet concentration for VBEM |

### Convergence Behavior

EM stops when **either**:
1. `|ΔLL_relative| = |LL_new - LL_old| / (1 + |LL_old|) < --tol`
2. `|ΔLL_absolute| = |LL_new - LL_old| < --abs-tol`
3. `iteration > --max-iter`

If convergence is not achieved by `--max-iter`, a warning is printed to stderr and the best parameters found so far are reported.

---

## Distribution Family Options

| Flag | Description |
|---|---|
| `-f gaussian` | Normal (Gaussian) |
| `-f exponential` | Exponential |
| `-f poisson` | Poisson (discrete) |
| `-f gamma` | Gamma |
| `-f lognormal` | Log-normal |
| `-f weibull` | Weibull |
| `-f beta` | Beta (requires data in (0,1)) |
| `-f uniform` | Uniform |
| `-f pearson` | Auto-classified Pearson type |
| `-f studentt` | Student's t |
| `-f laplace` | Laplace (double exponential) |
| `-f cauchy` | Cauchy |
| `-f invgaussian` | Inverse Gaussian (Wald) |
| `-f rayleigh` | Rayleigh |
| `-f pareto` | Pareto |
| `-f logistic` | Logistic |
| `-f gumbel` | Gumbel (extreme value type I) |
| `-f skewnormal` | Skew-normal |
| `-f gengaussian` | Generalized Gaussian (exp. power) |
| `-f chisquared` | Chi-squared |
| `-f f` | F distribution |
| `-f loglogistic` | Log-logistic (Fisk) |
| `-f nakagami` | Nakagami-m |
| `-f levy` | Lévy |
| `-f gompertz` | Gompertz |
| `-f burr` | Burr Type XII |
| `-f halfnormal` | Half-normal |
| `-f maxwell` | Maxwell-Boltzmann |
| `-f kumaraswamy` | Kumaraswamy |
| `-f triangular` | Triangular |
| `-f binomial` | Binomial (requires `--n-trials`) |
| `-f negbinomial` | Negative binomial |
| `-f geometric` | Geometric |
| `-f zipf` | Zipf (zeta) |
| `-f kde` | Kernel density estimation |

### Family-Specific Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--n-trials <n>` | int | — | Number of trials (Binomial only) |
| `--kde-kernel <name>` | string | `epanechnikov` | KDE kernel (epanechnikov/gaussian/uniform) |
| `--kde-bandwidth <h>` | float | auto (Silverman) | Fixed bandwidth for KDE |
| `--pearson-criterion <κ>` | float | auto | Override Pearson κ for classification |

---

## Multivariate Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--multivariate` | flag | off | Enable multivariate mode |
| `--dimensions <D>` | int | auto | Number of dimensions (auto-detected) |
| `--cov-type <type>` | string | `full` | Covariance type: full/diagonal/spherical/tied |
| `--family <f>` | string | `gaussian` | `gaussian` or `studentt` |

---

## Input Format

### Univariate

One numeric value per line (whitespace and comments ignored):

```
# comment
1.23
4.56
-0.78
1e-3
```

Accepted numeric formats: integer, float, scientific notation, negative.

### Multivariate

One vector per line, space- or comma-separated:

```
1.2 3.4 5.6
2.1,3.9,5.0
-0.5 1.2 0.8
```

All lines must have the same number of values. Mixed delimiters are supported.

### Large Files

Files larger than available RAM should use `--streaming`. For NFS or slow storage, increase the read buffer:

```bash
gemmulem -g huge_data.txt --streaming --read-buffer 1048576 -k 5 -o results.csv
```

---

## Output Format

### CSV Output

The output file has two sections separated by a blank line.

**Section 1: Component parameters**

```csv
# Gemmulem v1.0.0 output
# family: gaussian   k: 3   n: 1000
# log_likelihood: -1423.87   bic: 2891.4   aic: 2863.7   icl: 2912.1
# converged: yes   iterations: 47
component,weight,mu,sigma
1,0.3412,2.15,0.47
2,0.3289,5.93,0.51
3,0.3299,9.78,0.44
```

**Section 2: Responsibilities**

```csv
obs,gamma_1,gamma_2,gamma_3,assignment
1,0.9821,0.0174,0.0005,1
2,0.0031,0.9954,0.0015,2
```

The `assignment` column is `argmax_j gamma_j` (1-indexed hard assignment).

### Suppressing Responsibilities

```bash
# Only output parameters (no per-observation responsibilities)
gemmulem -g data.txt -k 3 --no-responsibilities -o results.csv
```

### JSON Output

```bash
gemmulem -g data.txt -k 3 --format json -o results.json
```

### Quiet and Verbose Modes

```bash
gemmulem -g data.txt -k 3 --quiet -o results.csv   # suppress all stderr
gemmulem -g data.txt -k 3 --verbose -o results.csv  # detailed per-iteration output
```

---

## Model Selection Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--criterion <name>` | string | `bic` | Selection criterion: bic/aic/icl/vbem/mml |
| `--print-criteria` | flag | off | Print all criteria to stderr |
| `--k-penalty <λ>` | float | — | Custom penalty coefficient (overrides criterion) |

```bash
# Print all criteria for a 3-component fit
gemmulem -g data.txt -k 3 --print-criteria -o results.csv
# stderr: BIC=2891.4  AIC=2863.7  ICL=2912.1  MML=2874.2
```

---

## Performance Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backend <name>` | string | auto | Compute backend: auto/gpu/avx2/sse2/scalar |
| `--threads <n>` | int | auto | CPU threads for SIMD backend |
| `--tile-size <T>` | int | 128 | Cache tile size for SIMD |
| `--gpu-device <n>` | int | 0 | OpenCL device index |
| `--gpu-workgroup <n>` | int | 256 | OpenCL work-group size |

---

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Input file not found or not readable |
| 2 | Parse error in input file |
| 3 | Output file could not be created |
| 4 | Invalid argument (bad flag or value) |
| 5 | EM did not converge within max-iter |
| 6 | Numerical error (NaN/Inf in parameters) |
| 7 | Out of memory |
| 8 | OpenCL initialization failed |
| 9 | Family incompatible with data (e.g., negative values for Exponential) |

---

## Full Example: Auto-k with Multiple Families

```bash
gemmulem \
  -g data.txt \
  --auto-k --k-min 2 --k-max 8 \
  --auto-family \
  --criterion bic \
  --restarts 20 \
  --seed 42 \
  --print-criteria \
  --verbose \
  -o results.csv
```

This command:
1. Tries $k = 2, 3, \ldots, 8$ for all 35 families
2. Selects the combination with the best BIC
3. Uses 20 K-means++ restarts per (family, k) pair
4. Seeds the PRNG with 42 for reproducibility
5. Prints all criteria and detailed progress to stderr
