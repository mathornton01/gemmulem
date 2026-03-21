/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * Complex-valued Gaussian mixture model EM.
 *
 * Supports:
 *   1. Circular symmetric (proper) complex Gaussian
 *      f(z) = (1/πσ²) exp(-|z-μ|²/σ²)
 *      Parameters: μ ∈ ℂ, σ² ∈ ℝ⁺
 *
 *   2. Non-circular (improper) complex Gaussian
 *      f(z) = (1/π√det(Γ)) exp(-½ z̃ᴴ Γ⁻¹ z̃)
 *      where z̃ = [z-μ; conj(z-μ)]ᵀ, Γ = [Σ C; C* Σ*]
 *      Parameters: μ ∈ ℂ, Σ ∈ ℂ (variance), C ∈ ℂ (pseudo-covariance)
 *
 *   3. Multivariate complex Gaussian (planned)
 *
 * Data format: interleaved real/imag doubles
 *   [re₀, im₀, re₁, im₁, ..., re_{n-1}, im_{n-1}]
 *
 * License: GPL v3
 */
#ifndef COMPLEX_EM_H
#define COMPLEX_EM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════════════════
 * Complex Gaussian model types
 * ════════════════════════════════════════════════════════════════════ */
typedef enum {
    CGAUSS_CIRCULAR    = 0,   /* Proper/circular: E[zz^T]=0 */
    CGAUSS_NONCIRCULAR = 1    /* Improper: E[zz^T]≠0 (has pseudo-covariance) */
} ComplexGaussType;

/* Per-component parameters for circular complex Gaussian */
typedef struct {
    double mu_re;         /* Real part of mean */
    double mu_im;         /* Imaginary part of mean */
    double var;           /* Variance σ² (real, positive) */
} CCircGaussParams;

/* Per-component parameters for non-circular complex Gaussian */
typedef struct {
    double mu_re;         /* Real part of mean */
    double mu_im;         /* Imaginary part of mean */
    double cov_re;        /* Re(Σ) — variance (real part) */
    double cov_im;        /* Im(Σ) — should be 0 for proper variance */
    double pcov_re;       /* Re(C) — pseudo-covariance real part */
    double pcov_im;       /* Im(C) — pseudo-covariance imag part */
} CNonCircGaussParams;

/* ════════════════════════════════════════════════════════════════════
 * Result structures
 * ════════════════════════════════════════════════════════════════════ */

/* Circular complex Gaussian mixture result */
typedef struct {
    int num_components;
    int iterations;
    double loglikelihood;
    double bic;
    double aic;
    double* mixing_weights;         /* [k] */
    CCircGaussParams* components;   /* [k] */
} CCircMixtureResult;

/* Non-circular complex Gaussian mixture result */
typedef struct {
    int num_components;
    int iterations;
    double loglikelihood;
    double bic;
    double aic;
    double* mixing_weights;             /* [k] */
    CNonCircGaussParams* components;    /* [k] */
} CNonCircMixtureResult;

/* ════════════════════════════════════════════════════════════════════
 * Public API — Circular symmetric complex Gaussian
 * ════════════════════════════════════════════════════════════════════ */

/**
 * Circular symmetric complex Gaussian mixture EM.
 *
 * Each observation z_n is a complex number stored as two doubles:
 *   data[2*n] = Re(z_n), data[2*n+1] = Im(z_n)
 *
 * Model: f(z|μ,σ²) = (1/πσ²) exp(-|z-μ|²/σ²)
 *
 * @param data     Interleaved real/imag array, length 2*n
 * @param n        Number of complex observations
 * @param k        Number of mixture components
 * @param maxiter  Maximum EM iterations
 * @param rtole    Convergence tolerance (relative change in LL)
 * @param verbose  Print iteration info
 * @param result   Output (caller must call ReleaseCCircResult)
 * @return 0 on success, <0 on error
 */
int UnmixComplexCircular(const double* data, size_t n, int k,
                         int maxiter, double rtole, int verbose,
                         CCircMixtureResult* result);

/**
 * Auto-k selection for circular complex Gaussian (BIC).
 */
int UnmixComplexCircularAutoK(const double* data, size_t n, int k_max,
                              int maxiter, double rtole, int verbose,
                              CCircMixtureResult* result);

void ReleaseCCircResult(CCircMixtureResult* result);

/* ════════════════════════════════════════════════════════════════════
 * Public API — Non-circular complex Gaussian
 * ════════════════════════════════════════════════════════════════════ */

/**
 * Non-circular (improper) complex Gaussian mixture EM.
 *
 * Uses the augmented covariance matrix Γ = [Σ C; C* Σ*].
 * Captures correlation between z and z* (e.g., BPSK, linearly modulated).
 *
 * @param data     Interleaved real/imag array, length 2*n
 * @param n        Number of complex observations
 * @param k        Number of components
 * @param maxiter  Max EM iterations
 * @param rtole    Convergence tolerance
 * @param verbose  Print iteration info
 * @param result   Output (caller must call ReleaseCNonCircResult)
 * @return 0 on success
 */
int UnmixComplexNonCircular(const double* data, size_t n, int k,
                            int maxiter, double rtole, int verbose,
                            CNonCircMixtureResult* result);

void ReleaseCNonCircResult(CNonCircMixtureResult* result);

/* ════════════════════════════════════════════════════════════════════
 * Multivariate complex Gaussian mixture (d-dimensional)
 * ════════════════════════════════════════════════════════════════════ */

/**
 * Per-component parameters for multivariate complex Gaussian.
 * The distribution is f(z|μ,Σ) = (1/(πᵈ det(Σ))) exp(-(z-μ)ᴴ Σ⁻¹ (z-μ))
 *
 * Covariance Σ is a d×d Hermitian positive-definite matrix stored as
 * 2*d*d doubles: cov[2*(i*d+j)] = Re(Σ[i,j]), cov[2*(i*d+j)+1] = Im(Σ[i,j])
 */
typedef struct {
    int dim;
    double* mean;       /* [2*dim]      interleaved re,im */
    double* cov;        /* [2*dim*dim]  Hermitian covariance, interleaved re,im */
    double* cov_chol;   /* [2*dim*dim]  lower Cholesky factor L of Σ = LLᴴ */
    double  log_det;    /* log det(Σ) = 2 Σᵢ log(L[i,i]) */
} MVComplexGaussParams;

typedef struct {
    int num_components;
    int dim;
    int iterations;
    double loglikelihood;
    double bic, aic;
    double* mixing_weights;          /* [k] */
    MVComplexGaussParams* components; /* [k] */
} MVComplexMixtureResult;

/**
 * Multivariate complex Gaussian mixture EM.
 *
 * Each observation z ∈ ℂᵈ is stored as 2d doubles:
 *   [re₁, im₁, re₂, im₂, ..., reₐ, imₐ]
 * Total data array: n × 2d doubles.
 *
 * @param data    Row-major n×2d array of interleaved re/im per dimension
 * @param n       Number of observations
 * @param d       Complex dimension
 * @param k       Number of mixture components
 * @param maxiter Max EM iterations
 * @param rtole   Relative convergence tolerance
 * @param verbose Print iteration info
 * @param result  Output (caller must call ReleaseMVComplexResult)
 * @return 0 on success, <0 on error
 */
int UnmixMVComplex(const double* data, size_t n, int d, int k,
                   int maxiter, double rtole, int verbose,
                   MVComplexMixtureResult* result);

void ReleaseMVComplexResult(MVComplexMixtureResult* result);

/* ════════════════════════════════════════════════════════════════════
 * Streaming Complex EM (Cappé & Moulines online EM)
 * ════════════════════════════════════════════════════════════════════ */

/**
 * Configuration for streaming complex EM.
 * Reads IQ data from a file in chunks — never loads more than chunk_size
 * complex observations into RAM at once.
 * Uses Cappé & Moulines (2009) step-size schedule: γₜ = (t + 2)^(-eta_decay)
 */
typedef struct {
    int num_components;
    int chunk_size;    /* Complex observations per chunk (default 10000) */
    int max_passes;    /* Passes over the file (default 10) */
    double rtole;      /* Convergence tolerance (default 1e-5) */
    int verbose;
    double eta_decay;  /* Step-size decay exponent ∈ (0.5, 1] (default 0.6) */
    ComplexGaussType type; /* Only CGAUSS_CIRCULAR supported */
} ComplexStreamConfig;

/**
 * Streaming EM for circular complex Gaussian mixture.
 * File format: one complex observation per line, "re im" or "re,im".
 *
 * @param filename  Path to IQ data file
 * @param config    Streaming configuration
 * @param result    Output (caller must call ReleaseCCircResult)
 * @return 0 on success, <0 on error
 */
int UnmixComplexStreaming(const char* filename, const ComplexStreamConfig* config,
                          CCircMixtureResult* result);

/* ════════════════════════════════════════════════════════════════════
 * Utility functions
 * ════════════════════════════════════════════════════════════════════ */

/**
 * Evaluate circular complex Gaussian PDF at point z = (re, im).
 */
double ccirc_gauss_pdf(double re, double im, const CCircGaussParams* p);
double ccirc_gauss_logpdf(double re, double im, const CCircGaussParams* p);

/**
 * Evaluate non-circular complex Gaussian PDF.
 */
double cnocirc_gauss_pdf(double re, double im, const CNonCircGaussParams* p);
double cnocirc_gauss_logpdf(double re, double im, const CNonCircGaussParams* p);

#ifdef __cplusplus
}
#endif

#endif /* COMPLEX_EM_H */
