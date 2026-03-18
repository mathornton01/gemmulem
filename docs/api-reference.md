---
layout: default
title: C API Reference
nav_order: 6
---

# C API Reference
{: .no_toc }

Public API for embedding Gemmulem in C programs.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Header

```c
#include <gemmulem/gemmulem.h>
```

Link with `-lgemmulem` (dynamic) or `libgemmulem.a` (static).

---

## Data Structures

### `GemConfig`

Configuration for a mixture model fit. Must be initialized with `gem_config_default()` before setting individual fields.

```c
typedef struct {
    /* Model specification */
    int         k;              /* Number of components (or 0 for auto-k) */
    int         k_min;          /* Min k for auto-k search (default: 1) */
    int         k_max;          /* Max k for auto-k search (default: 10) */
    GemFamily   family;         /* Distribution family */
    GemCriterion criterion;     /* Model selection criterion */

    /* Algorithm control */
    int         max_iter;       /* Maximum EM iterations (default: 1000) */
    double      tol;            /* Relative convergence tolerance (1e-6) */
    double      abs_tol;        /* Absolute convergence tolerance (1e-8) */
    int         restarts;       /* K-means++ restart count (0 = auto) */
    uint64_t    seed;           /* PRNG seed (0 = random) */
    int         use_squarem;    /* Enable SQUAREM acceleration (default: 1) */
    int         squarem_warmup; /* Plain EM iterations before SQUAREM */

    /* Multivariate */
    int         multivariate;   /* Enable multivariate mode (default: 0) */
    int         dimensions;     /* Observation dimension (0 = auto-detect) */
    GemCovType  cov_type;       /* Covariance structure */

    /* Streaming */
    int         streaming;      /* Enable streaming EM (default: 0) */
    int         batch_size;     /* Mini-batch size (default: 512) */
    double      step_size_exp;  /* Step-size exponent (default: 0.6) */
    int         epochs;         /* Streaming epochs (default: 1) */

    /* Compute */
    GemBackend  backend;        /* Compute backend (GEM_BACKEND_AUTO) */
    int         threads;        /* CPU threads (0 = auto) */
    int         gpu_device;     /* OpenCL device index (default: 0) */

    /* Family-specific */
    int         n_trials;       /* Binomial: number of trials */
    GemKdeKernel kde_kernel;    /* KDE kernel type */
    double      kde_bandwidth;  /* KDE bandwidth (0 = Silverman) */
} GemConfig;
```

### `GemResult`

Result of a mixture model fit. Allocated by Gemmulem; must be freed with `gem_result_free()`.

```c
typedef struct {
    /* Fit quality */
    double  log_likelihood;     /* Final log-likelihood */
    double  bic;                /* Bayesian Information Criterion */
    double  aic;                /* Akaike Information Criterion */
    double  icl;                /* Integrated Complete-data Likelihood */
    double  mml;                /* Minimum Message Length */
    int     iterations;         /* EM iterations performed */
    int     converged;          /* 1 if converged, 0 if max_iter reached */
    int     k;                  /* Selected number of components */

    /* Component parameters (k arrays) */
    double *weights;            /* Mixing weights π_j [k] */
    GemDistParams *components;  /* Per-component parameters [k] */

    /* Responsibilities (n × k matrix, row-major) */
    double *responsibilities;   /* γ_ij [n * k], NULL if not computed */
    int    *assignments;        /* Hard assignments argmax_j γ_ij [n] */
    int     n;                  /* Number of observations */
} GemResult;
```

### `GemDistParams`

Parameters for a single mixture component. Field validity depends on the distribution family.

```c
typedef struct {
    GemFamily family;

    /* Location / scale (most families) */
    double mu;          /* Mean/location */
    double sigma;       /* Standard deviation/scale */
    double nu;          /* Degrees of freedom (StudentT) */

    /* Shape parameters */
    double shape;       /* Gamma shape / Weibull shape / Nakagami m */
    double rate;        /* Gamma rate / Exponential rate */
    double scale;       /* Weibull scale / Lévy c / Pareto x_m */

    /* Bounded distributions */
    double alpha;       /* Beta α / SkewNormal α */
    double beta_param;  /* Beta β (named to avoid conflict with beta()) */
    double lower;       /* Uniform/Triangular lower bound */
    double upper;       /* Uniform/Triangular upper bound */
    double mode;        /* Triangular mode */

    /* Discrete */
    double lambda;      /* Poisson/Exponential rate */
    double p;           /* Binomial/Geometric/NegBinomial success prob */
    int    n_trials;    /* Binomial number of trials */
    double r;           /* NegBinomial dispersion */
    double s;           /* Zipf exponent */

    /* Generalized Gaussian */
    double gen_alpha;   /* GenGaussian scale */
    double gen_beta;    /* GenGaussian shape */

    /* Multivariate */
    double *mean_vec;   /* Multivariate mean [D] */
    double *cov_matrix; /* Covariance matrix [D*D], row-major */
    int     dim;        /* Dimension D */

    /* KDE */
    double bandwidth;   /* KDE bandwidth */
    double *kde_data;   /* Pointer to component's training data [n_j] */
    int     kde_n;      /* Number of KDE training points */
} GemDistParams;
```

### Enumerations

```c
typedef enum {
    GEM_FAMILY_GAUSSIAN,
    GEM_FAMILY_EXPONENTIAL,
    GEM_FAMILY_POISSON,
    GEM_FAMILY_GAMMA,
    GEM_FAMILY_LOGNORMAL,
    GEM_FAMILY_WEIBULL,
    GEM_FAMILY_BETA,
    GEM_FAMILY_UNIFORM,
    GEM_FAMILY_PEARSON,
    GEM_FAMILY_STUDENTT,
    GEM_FAMILY_LAPLACE,
    GEM_FAMILY_CAUCHY,
    GEM_FAMILY_INVGAUSSIAN,
    GEM_FAMILY_RAYLEIGH,
    GEM_FAMILY_PARETO,
    GEM_FAMILY_LOGISTIC,
    GEM_FAMILY_GUMBEL,
    GEM_FAMILY_SKEWNORMAL,
    GEM_FAMILY_GENGAUSSIAN,
    GEM_FAMILY_CHISQUARED,
    GEM_FAMILY_F,
    GEM_FAMILY_LOGLOGISTIC,
    GEM_FAMILY_NAKAGAMI,
    GEM_FAMILY_LEVY,
    GEM_FAMILY_GOMPERTZ,
    GEM_FAMILY_BURR,
    GEM_FAMILY_HALFNORMAL,
    GEM_FAMILY_MAXWELL,
    GEM_FAMILY_KUMARASWAMY,
    GEM_FAMILY_TRIANGULAR,
    GEM_FAMILY_BINOMIAL,
    GEM_FAMILY_NEGBINOMIAL,
    GEM_FAMILY_GEOMETRIC,
    GEM_FAMILY_ZIPF,
    GEM_FAMILY_KDE
} GemFamily;

typedef enum {
    GEM_CRITERION_BIC,
    GEM_CRITERION_AIC,
    GEM_CRITERION_ICL,
    GEM_CRITERION_VBEM,
    GEM_CRITERION_MML
} GemCriterion;

typedef enum {
    GEM_COVTYPE_FULL,
    GEM_COVTYPE_DIAGONAL,
    GEM_COVTYPE_SPHERICAL,
    GEM_COVTYPE_TIED
} GemCovType;

typedef enum {
    GEM_BACKEND_AUTO,
    GEM_BACKEND_GPU,
    GEM_BACKEND_AVX2,
    GEM_BACKEND_SSE2,
    GEM_BACKEND_SCALAR
} GemBackend;

typedef enum {
    GEM_KDE_EPANECHNIKOV,
    GEM_KDE_GAUSSIAN,
    GEM_KDE_UNIFORM
} GemKdeKernel;
```

---

## Functions

### Configuration

#### `gem_config_default`

```c
GemConfig gem_config_default(void);
```

Returns a `GemConfig` with all fields set to their default values. **Always** call this before modifying individual fields.

```c
GemConfig cfg = gem_config_default();
cfg.k = 3;
cfg.family = GEM_FAMILY_GAUSSIAN;
```

---

### Core Fitting

#### `gem_fit`

```c
GemResult *gem_fit(
    const double *data,   /* Array of n observations */
    int           n,      /* Number of observations */
    const GemConfig *cfg  /* Configuration (NULL = use defaults) */
);
```

Fit a mixture model to scalar data. Returns a heap-allocated `GemResult*` that must be freed with `gem_result_free()`.

Returns `NULL` on fatal error; check `gem_errno()` for the error code.

```c
double data[] = {1.2, 3.4, 5.6, 2.1, 4.5, 6.7, 1.8, 3.2};
GemConfig cfg = gem_config_default();
cfg.k = 2;
cfg.family = GEM_FAMILY_GAUSSIAN;

GemResult *res = gem_fit(data, 8, &cfg);
if (!res) {
    fprintf(stderr, "gem_fit failed: %s\n", gem_strerror(gem_errno()));
    return 1;
}

printf("LL=%.2f  BIC=%.2f\n", res->log_likelihood, res->bic);
for (int j = 0; j < res->k; j++)
    printf("Component %d: weight=%.3f  mu=%.3f  sigma=%.3f\n",
           j+1, res->weights[j],
           res->components[j].mu, res->components[j].sigma);

gem_result_free(res);
```

---

#### `gem_fit_multivariate`

```c
GemResult *gem_fit_multivariate(
    const double *data,   /* Row-major matrix [n × D] */
    int           n,      /* Number of observations */
    int           D,      /* Dimension */
    const GemConfig *cfg  /* Configuration */
);
```

Fit a multivariate Gaussian or Student-t mixture.

```c
// 3D data, 100 observations
double data[100 * 3] = { /* ... */ };
GemConfig cfg = gem_config_default();
cfg.k = 2;
cfg.multivariate = 1;
cfg.dimensions = 3;
cfg.cov_type = GEM_COVTYPE_FULL;

GemResult *res = gem_fit_multivariate(data, 100, 3, &cfg);
```

---

#### `gem_fit_file`

```c
GemResult *gem_fit_file(
    const char   *path,   /* Path to data file */
    const GemConfig *cfg  /* Configuration */
);
```

Convenience wrapper: reads data from a text file and calls `gem_fit()`.

---

### Streaming API

#### `GemStream` (opaque type)

```c
typedef struct GemStream GemStream;
```

#### `gem_stream_create`

```c
GemStream *gem_stream_create(const GemConfig *cfg);
```

Create a streaming EM context. Does not start fitting; use `gem_stream_feed()` to push data.

#### `gem_stream_feed`

```c
int gem_stream_feed(GemStream *stream, const double *batch, int n_batch);
```

Feed a mini-batch of `n_batch` observations to the streaming EM. Returns 0 on success, non-zero on error.

#### `gem_stream_result`

```c
GemResult *gem_stream_result(GemStream *stream);
```

Finalize and retrieve the current fit. The stream remains valid; you can continue feeding data.

#### `gem_stream_free`

```c
void gem_stream_free(GemStream *stream);
```

Free a streaming context.

```c
GemConfig cfg = gem_config_default();
cfg.k = 3;
cfg.streaming = 1;

GemStream *stream = gem_stream_create(&cfg);

FILE *f = fopen("huge_data.txt", "r");
double batch[512];
int n;
while ((n = read_batch(f, batch, 512)) > 0)
    gem_stream_feed(stream, batch, n);

GemResult *res = gem_stream_result(stream);
printf("LL = %.2f\n", res->log_likelihood);

gem_result_free(res);
gem_stream_free(stream);
fclose(f);
```

---

### Density Evaluation

#### `gem_log_pdf`

```c
double gem_log_pdf(double x, const GemDistParams *params);
```

Evaluate $\log f(x \mid \theta)$ for a single observation.

#### `gem_log_pdf_batch`

```c
void gem_log_pdf_batch(
    const double      *x,       /* [n] observations */
    int                n,
    const GemDistParams *params,
    double            *log_p    /* [n] output */
);
```

Vectorized log-PDF evaluation using SIMD where available.

#### `gem_mixture_pdf`

```c
double gem_mixture_pdf(double x, const GemResult *result);
```

Evaluate the fitted mixture density at $x$.

---

### Memory Management

#### `gem_result_free`

```c
void gem_result_free(GemResult *result);
```

Free all memory associated with a `GemResult`. Safe to call on `NULL`.

---

### Error Handling

#### `gem_errno`

```c
int gem_errno(void);
```

Return the error code from the last failed Gemmulem call (thread-local).

#### `gem_strerror`

```c
const char *gem_strerror(int error_code);
```

Return a human-readable error message.

#### Error Codes

| Code | Constant | Description |
|---|---|---|
| 0 | `GEM_OK` | Success |
| 1 | `GEM_ERR_IO` | File I/O error |
| 2 | `GEM_ERR_PARSE` | Data parse error |
| 3 | `GEM_ERR_INVALID_ARG` | Invalid argument |
| 4 | `GEM_ERR_NO_CONVERGE` | Did not converge |
| 5 | `GEM_ERR_NUMERICAL` | NaN/Inf in parameters |
| 6 | `GEM_ERR_OOM` | Out of memory |
| 7 | `GEM_ERR_OPENCL` | OpenCL initialization failed |
| 8 | `GEM_ERR_FAMILY_COMPAT` | Family incompatible with data |

---

## Complete Usage Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <gemmulem/gemmulem.h>

int main(void) {
    /* Simulated bimodal data */
    double data[1000];
    /* ... fill data ... */

    /* Configure the fit */
    GemConfig cfg = gem_config_default();
    cfg.k = 2;
    cfg.family     = GEM_FAMILY_GAUSSIAN;
    cfg.restarts   = 10;
    cfg.seed       = 42;  /* reproducible */
    cfg.max_iter   = 500;
    cfg.tol        = 1e-7;

    /* Fit the model */
    GemResult *res = gem_fit(data, 1000, &cfg);
    if (!res) {
        fprintf(stderr, "Error: %s\n", gem_strerror(gem_errno()));
        return 1;
    }

    /* Print results */
    printf("Converged: %s in %d iterations\n",
           res->converged ? "yes" : "no (max_iter)", res->iterations);
    printf("Log-likelihood: %.4f\n", res->log_likelihood);
    printf("BIC: %.4f\n", res->bic);

    for (int j = 0; j < res->k; j++) {
        printf("Component %d: weight=%.4f  mu=%.4f  sigma=%.4f\n",
               j+1, res->weights[j],
               res->components[j].mu, res->components[j].sigma);
    }

    /* Classify a new observation */
    double new_obs = 2.5;
    printf("\nMixture density at x=%.1f: %.6f\n",
           new_obs, gem_mixture_pdf(new_obs, res));

    /* Evaluate component membership */
    for (int j = 0; j < res->k; j++) {
        double lp = gem_log_pdf(new_obs, &res->components[j]);
        printf("  log p(x | component %d) = %.4f\n", j+1, lp);
    }

    gem_result_free(res);
    return 0;
}
```

**Compile:**

```bash
gcc -O2 example.c -o example -lgemmulem -lm
```

---

## Thread Safety

| Function | Thread-safe? | Notes |
|---|---|---|
| `gem_fit()` | Yes | Each call is independent |
| `gem_stream_feed()` | No | Serialize access to a single stream |
| `gem_log_pdf()` | Yes | Pure function |
| `gem_log_pdf_batch()` | Yes | Independent per batch |
| `gem_config_default()` | Yes | No shared state |
| `gem_result_free()` | Yes | Frees its own allocation |
| `gem_errno()` | Yes | Thread-local storage |

You may safely call `gem_fit()` concurrently from multiple threads (e.g., for parallel auto-k search) as long as each call uses a separate `data` array and `GemConfig`.

The PRNG state is per-call (stored in `GemConfig` and copied at the start of `gem_fit()`), so concurrent calls with the same seed will produce the same results independently.

---

## Building with CMake (as a dependency)

```cmake
find_package(Gemmulem REQUIRED)
target_link_libraries(my_app PRIVATE Gemmulem::gemmulem)
```

Or via `pkg-config`:

```bash
gcc $(pkg-config --cflags gemmulem) my_app.c $(pkg-config --libs gemmulem) -o my_app
```
