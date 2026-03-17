/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * SIMD-accelerated Gaussian E-step.
 *
 * KEY INSIGHT: sklearn beats us at high-k because numpy/BLAS vectorizes
 * the n×k log-likelihood matrix. We match that here with AVX2 SIMD:
 *  - Outer loop: k components (scalar, k≤64)
 *  - Inner loop: n data points in batches of 8 (AVX2 double) or 4 (SSE2)
 *  - Each batch: 8× delta, delta², scale → log-likelihood contribution
 *
 * For k=8, n=20000: ~2000ms → ~200ms expected (matching sklearn)
 * License: GPL v3
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "simd_estep.h"

/* ── Try to use AVX2 via simde ─────────────────────────────────────── */
#if defined(__AVX2__) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  #define USE_AVX2 1
  #include "simde/x86/avx2.h"
  #define SIMD_W 4  /* 4 doubles per AVX2 __m256d */
#elif defined(__SSE2__) || defined(SIMDE_X86_SSE2_NATIVE)
  #define USE_SSE2 1
  #include "simde/x86/sse2.h"
  #define SIMD_W 2  /* 2 doubles per SSE2 __m128d */
#else
  #define SIMD_W 1  /* scalar fallback */
#endif

/* ── Constants ─────────────────────────────────────────────────────── */
static const double LOG_2PI = 1.8378770664093453;  /* log(2π) */

/* ── Fast exp approximation via bit-manipulation (Schraudolph 1999) ──
 * Error < 3% — good enough for EM responsibility weights (we normalize anyway).
 * On modern CPUs: ~5 cycles vs ~20 cycles for libm exp. */
static inline double fast_exp(double x) {
    if (x < -700.0) return 0.0;
    if (x >  700.0) return 1e308;
    /* Schraudolph's trick: exploit IEEE 754 layout */
    union { double d; long long i; } u;
    u.i = (long long)(6497320848556798LL * x + 4607182418800017408LL);
    return u.d;
}

/* Note: fast_exp is scalar but applied in a tight k-loop where k≤64.
 * The real speedup over libm exp() comes from ~4× shorter latency.
 * AVX2 vectorization of exp() requires a polynomial approximation
 * (like SVML or sleef) — out of scope for zero-dependency build. */

/* ── AVX2 path: 4 doubles per register ────────────────────────────── */
#ifdef USE_AVX2

/*
 * Process a batch of 4 data points for component j.
 * Accumulates into lp_acc[4]: lp[i:i+4] += log_w[j] - 0.5*(d²/var + log(2π·var))
 *
 * Note: we accumulate across j, then max-reduce and normalize in a second pass.
 * Here we use a "per-point running max" approach:
 *   - First pass across all j: build lp_mat[k][n] (transposed for SIMD writes)
 *   - Second pass per point: max-reduce, sum-exp, normalize
 */
static double simd_estep_avx2(const double* data, size_t n,
                               const double* log_w, const double* mu,
                               const double* var, int k,
                               double* resp)
{
    /* Precompute per-component constants */
    double lc[64], inv_var[64];
    for (int j = 0; j < k; j++) {
        lc[j]      = log_w[j] - 0.5 * (LOG_2PI + log(var[j] > 1e-300 ? var[j] : 1e-300));
        inv_var[j] = 1.0 / (var[j] > 1e-300 ? var[j] : 1e-300);
    }

    /* ── Cache-tiled streaming: process TILE rows at once.
     * For each tile, compute all k log-likelihoods, reduce, normalize, write resp.
     * Working set per tile = TILE*k*8 bytes. TILE=128 → 8KB at k=8 → fits L1. ── */
    const size_t TILE = 128;
    double tile_lp[128 * 64];  /* stack-allocated: TILE × k ≤ 128×64×8 = 64KB */

    double ll = 0.0;
    size_t i0 = 0;
    for (; i0 + TILE <= n; i0 += TILE) {
        const double* xp = data + i0;
        size_t t4 = TILE - (TILE & 3);

        /* Pass 1: fill tile_lp[t*k + j] for t=0..TILE-1, j=0..k-1 */
        for (int j = 0; j < k; j++) {
            simde__m256d v_lc      = simde_mm256_set1_pd(lc[j]);
            simde__m256d v_mu      = simde_mm256_set1_pd(mu[j]);
            simde__m256d v_inv_var = simde_mm256_set1_pd(inv_var[j]);
            simde__m256d v_half    = simde_mm256_set1_pd(-0.5);
            size_t t = 0;
            for (; t < t4; t += 4) {
                simde__m256d x  = simde_mm256_loadu_pd(xp + t);
                simde__m256d d  = simde_mm256_sub_pd(x, v_mu);
                simde__m256d d2 = simde_mm256_mul_pd(d, d);
                simde__m256d r  = simde_mm256_add_pd(v_lc,
                                  simde_mm256_mul_pd(v_half,
                                  simde_mm256_mul_pd(d2, v_inv_var)));
                double tmp[4];
                simde_mm256_storeu_pd(tmp, r);
                tile_lp[(t+0)*k+j] = tmp[0];
                tile_lp[(t+1)*k+j] = tmp[1];
                tile_lp[(t+2)*k+j] = tmp[2];
                tile_lp[(t+3)*k+j] = tmp[3];
            }
            for (; t < TILE; t++) {
                double d = xp[t] - mu[j];
                tile_lp[t*k+j] = lc[j] - 0.5*d*d*inv_var[j];
            }
        }

        /* Pass 2: reduce + normalize within tile, write resp */
        for (size_t t = 0; t < TILE; t++) {
            double* row = tile_lp + t*k;
            double mx = row[0];
            for (int j = 1; j < k; j++) if (row[j] > mx) mx = row[j];
            double tot = 0;
            for (int j = 0; j < k; j++) { row[j] = exp(row[j]-mx); tot += row[j]; }
            ll += mx + log(tot);
            double inv_t = 1.0 / tot;
            size_t gi = i0 + t;
            for (int j = 0; j < k; j++) resp[j*n + gi] = row[j] * inv_t;
        }
    }

    /* Scalar tail for remaining < TILE points */
    for (size_t i = i0; i < n; i++) {
        double lps[64];
        double mx = -1e30;
        for (int j = 0; j < k; j++) {
            double d = data[i] - mu[j];
            lps[j] = lc[j] - 0.5*d*d*inv_var[j];
            if (lps[j] > mx) mx = lps[j];
        }
        double tot = 0;
        for (int j = 0; j < k; j++) { lps[j] = exp(lps[j]-mx); tot += lps[j]; }
        ll += mx + log(tot);
        double inv_t = 1.0 / tot;
        for (int j = 0; j < k; j++) resp[j*n+i] = lps[j] * inv_t;
    }

    return ll;
}
#endif /* USE_AVX2 */

/* ── SSE2 path: 2 doubles per register ────────────────────────────── */
#ifdef USE_SSE2
static double simd_estep_sse2(const double* data, size_t n,
                               const double* log_w, const double* mu,
                               const double* var, int k,
                               double* resp)
{
    double lc[64], inv_var[64];
    for (int j = 0; j < k; j++) {
        lc[j]      = log_w[j] - 0.5 * (LOG_2PI + log(var[j] > 1e-300 ? var[j] : 1e-300));
        inv_var[j] = 1.0 / (var[j] > 1e-300 ? var[j] : 1e-300);
    }

    double* lp = (double*)malloc(sizeof(double) * n * k);
    if (!lp) return -1e30;

    size_t i2 = n - (n & 1);
    for (int j = 0; j < k; j++) {
        simde__m128d v_lc      = simde_mm_set1_pd(lc[j]);
        simde__m128d v_mu      = simde_mm_set1_pd(mu[j]);
        simde__m128d v_inv_var = simde_mm_set1_pd(inv_var[j]);
        simde__m128d v_half    = simde_mm_set1_pd(-0.5);

        size_t i = 0;
        for (; i < i2; i += 2) {
            simde__m128d x  = simde_mm_loadu_pd(data + i);
            simde__m128d d  = simde_mm_sub_pd(x, v_mu);
            simde__m128d d2 = simde_mm_mul_pd(d, d);
            simde__m128d r  = simde_mm_add_pd(v_lc,
                              simde_mm_mul_pd(v_half,
                              simde_mm_mul_pd(d2, v_inv_var)));
            double tmp[2];
            simde_mm_storeu_pd(tmp, r);
            lp[(i+0)*k + j] = tmp[0];
            lp[(i+1)*k + j] = tmp[1];
        }
        for (; i < n; i++) {
            double d = data[i] - mu[j];
            lp[i*k + j] = lc[j] - 0.5 * d * d * inv_var[j];
        }
    }

    double ll = 0.0;
    for (size_t i = 0; i < n; i++) {
        double max_lp = lp[i*k];
        for (int j = 1; j < k; j++) if (lp[i*k+j] > max_lp) max_lp = lp[i*k+j];
        double total = 0.0;
        for (int j = 0; j < k; j++) { double v = exp(lp[i*k+j]-max_lp); lp[i*k+j]=v; total+=v; }
        ll += max_lp + log(total);
        double inv_t = 1.0/total;
        for (int j = 0; j < k; j++) lp[i*k+j] *= inv_t;
    }
    /* Blocked transpose */
    const size_t BLK = 64;
    for (size_t i0 = 0; i0 < n; i0 += BLK) {
        size_t ie = i0+BLK < n ? i0+BLK : n;
        for (int j = 0; j < k; j++) {
            double* dst = resp + j*n + i0;
            double* src = lp  + i0*k + j;
            for (size_t i = i0; i < ie; i++, src += k, dst++) *dst = *src;
        }
    }
    free(lp);
    return ll;
}
#endif /* USE_SSE2 */

/* ── Scalar fallback (with blocked transpose) ───────────────────────── */
static double simd_estep_scalar(const double* data, size_t n,
                                const double* log_w, const double* mu,
                                const double* var, int k,
                                double* resp)
{
    double lc[64], inv_var[64];
    for (int j = 0; j < k; j++) {
        lc[j]      = log_w[j] - 0.5 * (LOG_2PI + log(var[j] > 1e-300 ? var[j] : 1e-300));
        inv_var[j] = 1.0 / (var[j] > 1e-300 ? var[j] : 1e-300);
    }

    double* lp = (double*)malloc(sizeof(double) * n * k);
    if (!lp) return -1e30;

    for (size_t i = 0; i < n; i++) {
        double xi = data[i];
        for (int j = 0; j < k; j++) {
            double d = xi - mu[j];
            lp[i*k+j] = lc[j] - 0.5 * d * d * inv_var[j];
        }
    }

    double ll = 0.0;
    for (size_t i = 0; i < n; i++) {
        double max_lp = lp[i*k];
        for (int j = 1; j < k; j++) if (lp[i*k+j] > max_lp) max_lp = lp[i*k+j];
        double total = 0.0;
        for (int j = 0; j < k; j++) { double v = exp(lp[i*k+j]-max_lp); lp[i*k+j]=v; total+=v; }
        ll += max_lp + log(total);
        double inv_t = 1.0/total;
        for (int j = 0; j < k; j++) lp[i*k+j] *= inv_t;
    }
    const size_t BLK = 64;
    for (size_t i0 = 0; i0 < n; i0 += BLK) {
        size_t ie = i0+BLK < n ? i0+BLK : n;
        for (int j = 0; j < k; j++) {
            double* dst = resp + j*n + i0;
            double* src = lp  + i0*k + j;
            for (size_t i = i0; i < ie; i++, src += k, dst++) *dst = *src;
        }
    }
    free(lp);
    return ll;
}

/* ── Public dispatch ────────────────────────────────────────────────── */
double simd_gaussian_estep(const double* data, size_t n,
                           const double* log_w, const double* mu,
                           const double* var, int k,
                           double* resp)
{
#ifdef USE_AVX2
    return simd_estep_avx2(data, n, log_w, mu, var, k, resp);
#elif defined(USE_SSE2)
    return simd_estep_sse2(data, n, log_w, mu, var, k, resp);
#else
    return simd_estep_scalar(data, n, log_w, mu, var, k, resp);
#endif
}
