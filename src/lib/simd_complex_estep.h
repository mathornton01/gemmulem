/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * SIMD-accelerated E-step for circular complex Gaussian mixture EM.
 * Uses AVX2 (2 complex observations per __m256d register) via simde headers.
 * Falls back to scalar when AVX2 unavailable.
 * License: GPL v3
 */
#ifndef SIMD_COMPLEX_ESTEP_H
#define SIMD_COMPLEX_ESTEP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compute E-step responsibilities for n complex observations, k components.
 * Model: circular complex Gaussian CN(μⱼ, σⱼ²)
 *   log p(zᵢ|j) = log(wⱼ) - log(π) - log(σⱼ²) - |zᵢ-μⱼ|²/σⱼ²
 *
 * Data layout: interleaved re/im — data[2*i] = Re(zᵢ), data[2*i+1] = Im(zᵢ)
 * resp[j*n + i] = P(component j | zᵢ)  (normalised)
 * Returns total log-likelihood.
 *
 * With AVX2 the inner loop processes 2 complex observations per iteration
 * (4 doubles per __m256d register):
 *   v = [re₀, im₀, re₁, im₁]
 *   d = v - [μr, μi, μr, μi]
 *   sq = d*d  →  hadd → [|z₀-μ|², _, |z₁-μ|², _]
 */
double simd_complex_circular_estep(
    const double* data,    /* [2*n] interleaved re,im */
    size_t n,
    const double* log_w,   /* [k] log mixing weights */
    const double* mu_re,   /* [k] */
    const double* mu_im,   /* [k] */
    const double* var,     /* [k] variances σⱼ² */
    int k,
    double* resp           /* [k*n] output, resp[j*n+i] */
);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_COMPLEX_ESTEP_H */
