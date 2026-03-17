/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 *
 * Generic distribution framework for GEMMULEM.
 * Each distribution family implements: pdf, weighted MLE, parameter count.
 * The generic EM engine uses these to unmix any mixture model.
 *
 * License: GPL v3
 */

#ifndef __DISTRIBUTIONS_H__
#define __DISTRIBUTIONS_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum parameters per component (e.g., mean + variance for Gaussian) */
#define DIST_MAX_PARAMS 4

/* Distribution family identifiers */
typedef enum {
    DIST_GAUSSIAN    = 0,
    DIST_EXPONENTIAL = 1,
    DIST_POISSON     = 2,
    DIST_GAMMA       = 3,
    DIST_LOGNORMAL   = 4,
    DIST_WEIBULL     = 5,
    DIST_BETA        = 6,
    DIST_UNIFORM     = 7,
    DIST_PEARSON     = 8,   /* Pearson system — auto-discovers distribution type */
    DIST_STUDENT_T   = 9,   /* Student-t: robust to outliers */
    DIST_LAPLACE     = 10,  /* Double-exponential: sharp-peaked, heavy-tailed */
    DIST_CAUCHY      = 11,  /* Cauchy: very heavy tails (undefined mean) */
    DIST_INVGAUSS    = 12,  /* Inverse-Gaussian: positive right-skewed */
    DIST_RAYLEIGH    = 13,  /* Rayleigh: positive, wind/signal amplitude */
    DIST_PARETO      = 14,  /* Pareto: power-law tails */
    DIST_LOGISTIC    = 15,  /* Logistic: similar to Gaussian, heavier tails */
    DIST_GUMBEL      = 16,  /* Gumbel (Extreme Value Type I): max/min modeling */
    DIST_SKEWNORMAL  = 17,  /* Skew-Normal: asymmetric generalization of Normal */
    DIST_GENGAUSS    = 18,  /* Generalized Gaussian: shape between Laplace and Gaussian */
    DIST_CHISQ       = 19,  /* Chi-squared: sum of squared normals */
    DIST_F           = 20,  /* F-distribution: ratio of chi-squareds */
    DIST_LOGLOGISTIC = 21,  /* Log-Logistic (Fisk): positive heavy-tailed */
    DIST_NAKAGAMI    = 22,  /* Nakagami: fading channel modeling */
    DIST_LEVY        = 23,  /* Lévy: stable, very heavy right tail */
    DIST_GOMPERTZ    = 24,  /* Gompertz: survival/actuarial modeling */
    DIST_BURR        = 25,  /* Burr Type XII: flexible positive distribution */
    DIST_HALFNORMAL  = 26,  /* Half-Normal: folded Gaussian, x>=0 */
    DIST_MAXWELL     = 27,  /* Maxwell-Boltzmann: particle speed distribution */
    DIST_KUMARASWAMY = 28,  /* Kumaraswamy: Beta-like on [0,1], closed-form CDF */
    DIST_TRIANGULAR  = 29,  /* Triangular: bounded [a,b] with mode c */
    DIST_BINOMIAL    = 30,  /* Binomial: discrete, n trials */
    DIST_NEGBINOM    = 31,  /* Negative Binomial: overdispersed counts */
    DIST_GEOMETRIC   = 32,  /* Geometric: trials until first success */
    DIST_ZIPF        = 33,  /* Zipf: power-law discrete (rank-frequency) */
    DIST_KDE         = 34,  /* Kernel Density Estimate: nonparametric, O(n²) */
    DIST_COUNT       = 35   /* sentinel: number of distributions */
} DistFamily;

/* Per-component parameters */
typedef struct {
    double p[DIST_MAX_PARAMS];
    int nparams;           /* how many of p[] are used */
} DistParams;

/* Distribution function table — one per family */
typedef struct {
    DistFamily family;
    const char* name;
    int num_params;       /* free parameters per component */

    /* Probability density function */
    double (*pdf)(double x, const DistParams* params);

    /* Log PDF (optional, for numerical stability; NULL = use log(pdf)) */
    double (*logpdf)(double x, const DistParams* params);

    /* Weighted MLE: given data + weights, estimate params for one component */
    void (*estimate)(const double* data, const double* weights, size_t n,
                     DistParams* out);

    /* Initialize parameters for k components from data */
    void (*init_params)(const double* data, size_t n, int k,
                        DistParams* out_params);

    /* Domain check: is this value valid for this distribution? */
    int (*valid)(double x);

} DistFunctions;

/* ====================================================================
 * Generic mixture EM result
 * ==================================================================== */
typedef struct {
    DistFamily family;
    int num_components;
    int iterations;
    double loglikelihood;
    double bic;
    double aic;
    double* mixing_weights;     /* [num_components] */
    DistParams* params;         /* [num_components] */
} MixtureResult;

/* ====================================================================
 * Model selection result
 * ==================================================================== */
typedef struct {
    DistFamily best_family;
    int best_k;
    double best_bic;
    int num_candidates;
    MixtureResult* candidates;  /* array of all tried models */
} ModelSelectResult;


/* ====================================================================
 * Public API
 * ==================================================================== */

/**
 * Get the function table for a distribution family.
 */
const DistFunctions* GetDistFunctions(DistFamily family);

/**
 * Get distribution name string.
 */
const char* GetDistName(DistFamily family);

/**
 * Generic mixture EM: unmix data into k components of the given family.
 *
 * @param data     Array of observed values
 * @param n        Number of observations
 * @param family   Distribution family
 * @param k        Number of mixture components
 * @param maxiter  Maximum EM iterations
 * @param rtole    Convergence tolerance (change in log-likelihood)
 * @param verbose  Print iteration info
 * @param result   Output (caller must call ReleaseMixtureResult)
 * @return 0 on success, <0 on error
 */
int UnmixGeneric(const double* data, size_t n, DistFamily family, int k,
                 int maxiter, double rtole, int verbose,
                 MixtureResult* result);

/**
 * Model selection: try all distribution families (or a subset) with
 * component counts from k_min to k_max, return the best by BIC.
 *
 * @param data       Array of observed values
 * @param n          Number of observations
 * @param families   Array of families to try (NULL = try all)
 * @param nfamilies  Length of families array (0 = try all)
 * @param k_min      Minimum components to try
 * @param k_max      Maximum components to try
 * @param maxiter    Max EM iterations per fit
 * @param rtole      Convergence tolerance
 * @param verbose    Print progress
 * @param result     Output (caller must call ReleaseModelSelectResult)
 * @return 0 on success
 */
int SelectBestMixture(const double* data, size_t n,
                      const DistFamily* families, int nfamilies,
                      int k_min, int k_max,
                      int maxiter, double rtole, int verbose,
                      ModelSelectResult* result);

/* ====================================================================
 * Component-count selection methods
 * ==================================================================== */
typedef enum {
    KMETHOD_BIC      = 0,  /* BIC-driven split-merge (default) */
    KMETHOD_AIC      = 1,  /* AIC-driven split-merge (less penalizing) */
    KMETHOD_ICL      = 2,  /* Integrated Complete-data Likelihood */
    KMETHOD_VBEM     = 3,  /* Variational Bayes EM — Dirichlet prior prunes k */
    KMETHOD_MML      = 4,  /* Minimum Message Length (Wallace-Freeman) */
    KMETHOD_COUNT    = 5
} KMethod;

/* ====================================================================
 * Adaptive Mixture Result (mixed-family, auto-k)
 * ==================================================================== */
typedef struct {
    int num_components;
    int iterations;
    double loglikelihood;
    double bic;
    double aic;
    double icl;                 /* ICL criterion (BIC + entropy) */
    KMethod kmethod;            /* which method was used */
    double* mixing_weights;     /* [num_components] */
    DistParams* params;         /* [num_components] */
    DistFamily* families;       /* [num_components] — family PER component */
} AdaptiveResult;

/**
 * Fully adaptive EM: discovers both the number of components (k)
 * AND the distribution family for each component independently.
 *
 * Algorithm:
 *   1. Start with k_init components (Gaussian)
 *   2. E-step: standard responsibilities using each component's current PDF
 *   3. M-step: for each component, try all valid families on its weighted
 *      data, pick the one with best weighted log-likelihood
 *   4. Split: if a component shows high kurtosis/bimodality, split it
 *   5. Merge: if two components are too similar, merge them
 *   6. Repeat until BIC stops improving
 *
 * @param data       Observed values
 * @param n          Number of observations
 * @param k_max      Maximum components to try (0 = auto, default 10)
 * @param maxiter    Max EM iterations per round
 * @param rtole      Convergence tolerance
 * @param verbose    Print progress
 * @param result     Output (caller must call ReleaseAdaptiveResult)
 * @return 0 on success
 */
int UnmixAdaptive(const double* data, size_t n,
                  int k_max, int maxiter, double rtole, int verbose,
                  AdaptiveResult* result);

/**
 * Adaptive EM with explicit k-selection method.
 * Same as UnmixAdaptive but allows choosing how k is determined.
 *
 * @param kmethod  KMETHOD_BIC (default split-merge), KMETHOD_AIC,
 *                 KMETHOD_ICL, or KMETHOD_VBEM (variational Bayes)
 */
int UnmixAdaptiveEx(const double* data, size_t n,
                    int k_max, int maxiter, double rtole, int verbose,
                    KMethod kmethod, AdaptiveResult* result);

/**
 * Get k-method name string.
 */
const char* GetKMethodName(KMethod m);

void ReleaseAdaptiveResult(AdaptiveResult* result);

/**
 * Spectral initialization: moment-based method for provably good
 * starting parameters. Uses Hankel matrix eigendecomposition.
 *
 * @param data       Observed values
 * @param n          Number of observations
 * @param k          Number of components to find
 * @param out_means  Output: array of k estimated means
 * @param out_weights Output: array of k estimated weights
 * @return 0 on success
 */
int SpectralInit(const double* data, size_t n, int k,
                 double* out_means, double* out_weights);

/**
 * Online/Stochastic EM: mini-batch processing for large datasets.
 * Uses Cappé & Moulines (2009) step-size schedule.
 */
int UnmixOnline(const double* data, size_t n, DistFamily family, int k,
                int maxiter, double rtole, int batch_size, int verbose,
                MixtureResult* result);

/**
 * Set global data pointer for KDE distribution (needed for PDF computation).
 * Must be called before using DIST_KDE in any EM function.
 */
void KDE_SetData(const double* data, size_t n);

/**
 * Release memory.
 */
void ReleaseMixtureResult(MixtureResult* result);
void ReleaseModelSelectResult(ModelSelectResult* result);

#ifdef __cplusplus
}
#endif

#endif /* __DISTRIBUTIONS_H__ */
