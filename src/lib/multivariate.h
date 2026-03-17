/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * Multivariate Gaussian mixture model EM.
 * License: GPL v3
 */
#ifndef MULTIVARIATE_H
#define MULTIVARIATE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Covariance type */
typedef enum {
    COV_FULL     = 0,   /* Full covariance matrix (d² free params) */
    COV_DIAGONAL = 1,   /* Diagonal only (d free params) */
    COV_SPHERICAL = 2   /* σ²I (1 free param) */
} CovType;

/* Per-component multivariate Gaussian parameters */
typedef struct {
    int dim;              /* dimensionality */
    double* mean;         /* [dim] */
    double* cov;          /* [dim*dim] full covariance */
    double* cov_chol;     /* [dim*dim] lower Cholesky factor */
    double log_det;       /* log determinant of cov */
} MVGaussParams;

/* Result from multivariate EM */
typedef struct {
    int num_components;
    int dim;
    int iterations;
    double loglikelihood;
    double bic;
    double aic;
    CovType cov_type;
    double* mixing_weights;     /* [k] */
    MVGaussParams* components;  /* [k] */
} MVMixtureResult;

/**
 * Multivariate Gaussian mixture EM.
 *
 * @param data     Row-major n×d matrix of observations
 * @param n        Number of observations
 * @param d        Dimensionality
 * @param k        Number of components
 * @param cov_type Covariance structure
 * @param maxiter  Max EM iterations
 * @param rtole    Convergence tolerance
 * @param verbose  Print iteration info
 * @param result   Output
 * @return 0 on success
 */
int UnmixMVGaussian(const double* data, size_t n, int d, int k,
                    CovType cov_type, int maxiter, double rtole,
                    int verbose, MVMixtureResult* result);

/**
 * Release memory allocated by UnmixMVGaussian.
 */
void ReleaseMVMixtureResult(MVMixtureResult* result);

/**
 * Evaluate multivariate Gaussian PDF.
 */
double mvgauss_pdf(const double* x, const MVGaussParams* p);
double mvgauss_logpdf(const double* x, const MVGaussParams* p);

#ifdef __cplusplus
}
#endif

#endif /* MULTIVARIATE_H */
