/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * SIMD-accelerated E-step for circular complex Gaussian mixture EM.
 *
 * Each observation z is 2 doubles (re, im). Complex circular Gaussian:
 *   log p(z|μ,σ²) = -log(π) - log(σ²) - |z-μ|²/σ²
 *
 * AVX2 processes 2 complex observations per __m256d register:
 *   Load [re₀, im₀, re₁, im₁]
 *   Sub  [μr,  μi,  μr,  μi ]
 *   Square and hadd to get |dz|² per observation
 *
 * License: GPL v3
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "simd_complex_estep.h"

/* ── AVX2 detection via simde ──────────────────────────────────────── */
#if defined(__AVX2__) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  #define USE_AVX2_COMPLEX 1
  #include "simde/x86/avx2.h"
#endif

static const double LOG_PI = 1.1447298858494002;  /* log(π) */

/* ────────────────────────────────────────────────────────────────────
 * AVX2 path: process 2 complex observations per __m256d
 * ──────────────────────────────────────────────────────────────────── */
#ifdef USE_AVX2_COMPLEX

static double complex_estep_avx2(
    const double* data, size_t n,
    const double* log_w,
    const double* mu_re, const double* mu_im,
    const double* var, int k,
    double* resp)
{
    /* Precompute per-component constants:
     * lc[j] = log_w[j] - log(π) - log(σ²)
     * inv_var[j] = 1/σ²
     */
    double lc[64], iv[64];
    for (int j = 0; j < k; j++) {
        double v = var[j] > 1e-300 ? var[j] : 1e-300;
        lc[j] = log_w[j] - LOG_PI - log(v);
        iv[j] = 1.0 / v;
    }

    /* Cache-tiled: TILE complex observations at a time.
     * Working set per tile: TILE × k × 8 bytes.
     * TILE=128, k=8 → 8KB → fits L1 cache. */
    const size_t TILE = 128;
    double tile_lp[128 * 64];  /* TILE × k_max */

    double ll = 0.0;
    size_t i0 = 0;

    for (; i0 + TILE <= n; i0 += TILE) {
        const double* dp = data + 2 * i0;  /* pointer to start of tile in data */

        /* Pass 1: compute log p(z_i | comp j) for each (i, j) in tile */
        for (int j = 0; j < k; j++) {
            /* Broadcast component parameters:
             * v_mu = [μ_re, μ_im, μ_re, μ_im] */
            simde__m256d v_mu  = simde_mm256_set_pd(mu_im[j], mu_re[j],
                                                     mu_im[j], mu_re[j]);
            simde__m256d v_iv  = simde_mm256_set1_pd(iv[j]);
            simde__m256d v_lc  = simde_mm256_set1_pd(lc[j]);
            simde__m256d v_neg = simde_mm256_set1_pd(-1.0);

            /* Process 2 observations per iteration (4 doubles = 2 complex) */
            size_t t2 = TILE - (TILE & 1);  /* round down to even */
            size_t t = 0;
            for (; t < t2; t += 2) {
                /* Load [re_t, im_t, re_{t+1}, im_{t+1}] */
                simde__m256d z = simde_mm256_loadu_pd(dp + 2*t);

                /* d = z - mu */
                simde__m256d d = simde_mm256_sub_pd(z, v_mu);

                /* d² = d * d */
                simde__m256d d2 = simde_mm256_mul_pd(d, d);

                /* hadd pairs within 128-bit lanes:
                 * d2 = [dr₀², di₀², dr₁², di₁²]
                 * hadd(d2, d2) = [dr₀²+di₀², dr₀²+di₀², dr₁²+di₁², dr₁²+di₁²]
                 * We need elements [0] and [2] = |dz₀|² and |dz₁|² */
                simde__m256d mag2 = simde_mm256_hadd_pd(d2, d2);

                /* log p = lc - |dz|²/σ² = lc + (-1) * mag2 * iv */
                simde__m256d lp = simde_mm256_add_pd(v_lc,
                                  simde_mm256_mul_pd(v_neg,
                                  simde_mm256_mul_pd(mag2, v_iv)));

                /* Extract: hadd puts results at indices [0,1,2,3]
                 * where [0]=[1] = obs t, [2]=[3] = obs t+1 */
                double tmp[4];
                simde_mm256_storeu_pd(tmp, lp);
                tile_lp[t * k + j]       = tmp[0];  /* obs t */
                tile_lp[(t + 1) * k + j] = tmp[2];  /* obs t+1 */
            }

            /* Scalar tail within tile (at most 1 obs) */
            for (; t < TILE; t++) {
                double dr = dp[2*t]   - mu_re[j];
                double di = dp[2*t+1] - mu_im[j];
                tile_lp[t * k + j] = lc[j] - (dr*dr + di*di) * iv[j];
            }
        }

        /* Pass 2: log-sum-exp normalize, write resp (column-major) */
        for (size_t t = 0; t < TILE; t++) {
            double* row = tile_lp + t * k;
            double mx = row[0];
            for (int j = 1; j < k; j++) if (row[j] > mx) mx = row[j];
            double tot = 0.0;
            for (int j = 0; j < k; j++) { row[j] = exp(row[j] - mx); tot += row[j]; }
            ll += mx + log(tot);
            double inv_t = 1.0 / tot;
            size_t gi = i0 + t;
            for (int j = 0; j < k; j++) resp[j * n + gi] = row[j] * inv_t;
        }
    }

    /* Scalar tail for remaining < TILE observations */
    for (size_t i = i0; i < n; i++) {
        double re = data[2*i], im = data[2*i+1];
        double lps[64];
        double mx = -1e30;
        for (int j = 0; j < k; j++) {
            double dr = re - mu_re[j];
            double di = im - mu_im[j];
            lps[j] = lc[j] - (dr*dr + di*di) * iv[j];
            if (lps[j] > mx) mx = lps[j];
        }
        double tot = 0.0;
        for (int j = 0; j < k; j++) { lps[j] = exp(lps[j] - mx); tot += lps[j]; }
        ll += mx + log(tot);
        double inv_t = 1.0 / tot;
        for (int j = 0; j < k; j++) resp[j * n + i] = lps[j] * inv_t;
    }

    return ll;
}
#endif /* USE_AVX2_COMPLEX */

/* ────────────────────────────────────────────────────────────────────
 * Scalar fallback
 * ──────────────────────────────────────────────────────────────────── */
static double complex_estep_scalar(
    const double* data, size_t n,
    const double* log_w,
    const double* mu_re, const double* mu_im,
    const double* var, int k,
    double* resp)
{
    double lc[64], iv[64];
    for (int j = 0; j < k; j++) {
        double v = var[j] > 1e-300 ? var[j] : 1e-300;
        lc[j] = log_w[j] - LOG_PI - log(v);
        iv[j] = 1.0 / v;
    }

    /* Row-major lp buffer, then transpose to column-major resp */
    double* lp = (double*)malloc(sizeof(double) * n * k);
    if (!lp) return -1e30;

    for (size_t i = 0; i < n; i++) {
        double re = data[2*i], im = data[2*i+1];
        for (int j = 0; j < k; j++) {
            double dr = re - mu_re[j];
            double di = im - mu_im[j];
            lp[i*k + j] = lc[j] - (dr*dr + di*di) * iv[j];
        }
    }

    double ll = 0.0;
    for (size_t i = 0; i < n; i++) {
        double mx = lp[i*k];
        for (int j = 1; j < k; j++) if (lp[i*k+j] > mx) mx = lp[i*k+j];
        double tot = 0.0;
        for (int j = 0; j < k; j++) {
            double v = exp(lp[i*k+j] - mx);
            lp[i*k+j] = v;
            tot += v;
        }
        ll += mx + log(tot);
        double inv_t = 1.0 / tot;
        for (int j = 0; j < k; j++) lp[i*k+j] *= inv_t;
    }

    /* Blocked transpose: row-major lp[i*k+j] → column-major resp[j*n+i] */
    const size_t BLK = 64;
    for (size_t i0 = 0; i0 < n; i0 += BLK) {
        size_t ie = i0 + BLK < n ? i0 + BLK : n;
        for (int j = 0; j < k; j++) {
            for (size_t i = i0; i < ie; i++) {
                resp[j*n + i] = lp[i*k + j];
            }
        }
    }

    free(lp);
    return ll;
}

/* ────────────────────────────────────────────────────────────────────
 * Public dispatch
 * ──────────────────────────────────────────────────────────────────── */
double simd_complex_circular_estep(
    const double* data, size_t n,
    const double* log_w,
    const double* mu_re, const double* mu_im,
    const double* var, int k,
    double* resp)
{
#ifdef USE_AVX2_COMPLEX
    return complex_estep_avx2(data, n, log_w, mu_re, mu_im, var, k, resp);
#else
    return complex_estep_scalar(data, n, log_w, mu_re, mu_im, var, k, resp);
#endif
}
