/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * SIMD-accelerated E-step for Gaussian mixture EM.
 * Uses AVX2 (8× float) via simde headers — no native intrinsics required.
 * Falls back to plain scalar when AVX2 unavailable.
 * License: GPL v3
 */
#ifndef SIMD_ESTEP_H
#define SIMD_ESTEP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compute Gaussian E-step responsibilities for n data points, k components.
 * Uses SIMD to process 8 data points per AVX2 iteration.
 *
 * resp[j*n + i] = P(component j | x_i)  normalized
 * Returns total log-likelihood.
 *
 * All arrays must be 32-byte aligned for best performance (will work unaligned too).
 */
double simd_gaussian_estep(
    const double* __restrict__ data,    /* [n] observations */
    size_t n,
    const double* __restrict__ log_w,   /* [k] log mixing weights */
    const double* __restrict__ mu,      /* [k] means */
    const double* __restrict__ var,     /* [k] variances */
    int k,
    double* __restrict__ resp           /* [k*n] output, resp[j*n+i] */
);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_ESTEP_H */
