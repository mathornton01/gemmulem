/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * SIMD-accelerated E-step for circular complex Gaussian mixture EM.
 * Uses AVX2 via simde headers — falls back to scalar when unavailable.
 * License: GPL v3
 */
#ifndef SIMD_COMPLEX_ESTEP_H
#define SIMD_COMPLEX_ESTEP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compute circular complex Gaussian E-step responsibilities.
 *
 * data[2*n]:  interleaved [re₀,im₀, re₁,im₁, ...]
 * log_w[k]:   log mixing weights
 * mu_re[k]:   component mean (real part)
 * mu_im[k]:   component mean (imaginary part)
 * var[k]:     component variance σ²
 * resp[k*n]:  output, column-major: resp[j*n+i] = P(comp j | z_i)
 *
 * Returns total log-likelihood.
 *
 * Complex circular Gaussian log-PDF:
 *   log p(z|μ,σ²) = -log(π) - log(σ²) - |z-μ|²/σ²
 */
double simd_complex_circular_estep(
    const double* __restrict__ data,
    size_t n,
    const double* __restrict__ log_w,
    const double* __restrict__ mu_re,
    const double* __restrict__ mu_im,
    const double* __restrict__ var,
    int k,
    double* __restrict__ resp
);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_COMPLEX_ESTEP_H */
