/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 *
 * Pearson distribution system for GEMMULEM.
 * Unifies Normal, Beta, Gamma, Inv-Gamma, Beta-prime, Student-t,
 * and Type IV under a single parameterization via (mu, sigma, beta1, beta2).
 *
 * The distribution type emerges from the data during EM — no need
 * to pre-specify which distribution each component should be.
 *
 * License: GPL v3
 */

#ifndef __PEARSON_H__
#define __PEARSON_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pearson distribution types */
typedef enum {
    PEARSON_TYPE_0   = 0,   /* Normal */
    PEARSON_TYPE_I   = 1,   /* Beta (bounded) */
    PEARSON_TYPE_II  = 2,   /* Symmetric Beta */
    PEARSON_TYPE_III = 3,   /* Gamma */
    PEARSON_TYPE_IV  = 4,   /* Heavy-tailed (includes skewed t) */
    PEARSON_TYPE_V   = 5,   /* Inverse-Gamma */
    PEARSON_TYPE_VI  = 6,   /* Beta-prime (F-distribution) */
    PEARSON_TYPE_VII = 7,   /* Student-t (symmetric Type IV) */
    PEARSON_TYPE_UNDEFINED = -1
} PearsonType;

/* Full parameterization of a Pearson distribution */
typedef struct {
    /* Moment parameters (what EM estimates) */
    double mu;      /* mean */
    double sigma;   /* standard deviation */
    double beta1;   /* squared skewness = gamma1^2 */
    double beta2;   /* traditional kurtosis = gamma2 + 3 */

    /* Derived ODE coefficients: p'/p = -(a + x)/(b0 + b1*x + b2*x^2) */
    double a;       /* = a1 in the ODE (after centering on mu) */
    double b0, b1, b2_ode;  /* quadratic denominator */

    /* Type and support */
    PearsonType type;
    double support_lo;  /* lower bound of support (-INF if unbounded) */
    double support_hi;  /* upper bound of support (+INF if unbounded) */

    /* Type-specific derived parameters for efficient PDF evaluation */
    union {
        struct { double shape; double rate; } gamma;          /* Type III */
        struct { double alpha; double beta; } beta;           /* Type I */
        struct { double alpha; double m; double nu; double lambda; } type_iv;
        struct { double shape; double scale; } inv_gamma;     /* Type V */
        struct { double d1; double d2; } beta_prime;          /* Type VI */
        struct { double df; } student_t;                      /* Type VII */
    } p;

    /* Cached normalization constant (log scale) */
    double log_norm;
    int valid;
} PearsonParams;

/* ====================================================================
 * Public API
 * ==================================================================== */

/**
 * Classify a point (beta1, beta2) into a Pearson type.
 */
PearsonType pearson_classify(double beta1, double beta2);

/**
 * Get human-readable name for a Pearson type.
 */
const char* pearson_type_name(PearsonType type);

/**
 * Compute full Pearson parameterization from moments.
 * Returns 0 on success, <0 on error.
 */
int pearson_from_moments(double mu, double sigma, double beta1, double beta2,
                         PearsonParams* out);

/**
 * Evaluate the PDF at x.
 */
double pearson_pdf(double x, const PearsonParams* params);

/**
 * Evaluate the log-PDF at x.
 */
double pearson_logpdf(double x, const PearsonParams* params);

/**
 * Estimate Pearson parameters from weighted data (M-step).
 * Computes weighted mean, variance, skewness, kurtosis and calls
 * pearson_from_moments.
 */
int pearson_estimate(const double* data, const double* weights, size_t n,
                     PearsonParams* out);

#ifdef __cplusplus
}
#endif

#endif /* __PEARSON_H__ */
