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
    DIST_COUNT       = 8   /* sentinel: number of distributions */
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

/**
 * Release memory.
 */
void ReleaseMixtureResult(MixtureResult* result);
void ReleaseModelSelectResult(ModelSelectResult* result);

#ifdef __cplusplus
}
#endif

#endif /* __DISTRIBUTIONS_H__ */
