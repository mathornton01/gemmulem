/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * SIMD-accelerated E-step for circular complex Gaussian mixture.
 *
 * Each observation zᵢ ∈ ℂ: data[2*i] = Re(zᵢ), data[2*i+1] = Im(zᵢ)
 * log p(zᵢ|j) = log(wⱼ) - log(π) - log(σⱼ²) - |zᵢ-μⱼ|²/σⱼ²
 *
 * AVX2 kernel: 2 complex observations per __m256d (4 doubles).
 *   v   = [re₀, im₀, re₁, im₁]
 *   mu  = [μr,  μi,  μr,  μi ]
 *   d   = v - mu  =  [dr₀, di₀, dr₁, di₁]
 *   sq  = d*d      =  [dr₀², di₀², dr₁², di₁²]
 *   hs  = hadd(sq,sq) = [dr₀²+di₀², dr₀²+di₀², dr₁²+di₁², dr₁²+di₁²]
 *   lp  = lc[j] + hs * (-inv_var)   (element 0 → obs 0, element 2 → obs 1)
 *
 * License: GPL v3
 */

#include "simd_complex_estep.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── AVX2 via simde ─────────────────────────────────────────────── */
#if defined(__AVX2__) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  #define USE_CX_AVX2 1
  #include "simde/x86/avx2.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Cache tile: TILE complex observations at a time.
 * Working set per tile = TILE * k * 8 bytes.  TILE=128, k=8 → 8KB (fits L1). */
#define CX_TILE 128

/* ════════════════════════════════════════════════════════════════════
 * AVX2 path
 * ════════════════════════════════════════════════════════════════════ */
#ifdef USE_CX_AVX2

static double cx_estep_avx2(const double* data, size_t n,
                             const double* log_w,
                             const double* mu_re,
                             const double* mu_im,
                             const double* var,
                             int k,
                             double* resp)
{
    /* Per-component constants:
     *   lc[j]      = log_w[j] - log(π) - log(σⱼ²)
     *   inv_var[j] = 1 / σⱼ²
     */
    double lc[64], inv_var[64];
    for (int j = 0; j < k; j++) {
        double v = var[j] > 1e-300 ? var[j] : 1e-300;
        lc[j]      = log_w[j] - log(M_PI) - log(v);
        inv_var[j] = 1.0 / v;
    }

    /* Stack-allocated tile: tile_lp[t*k + j] = log p(z_{i0+t} | j)   */
    double tile_lp[CX_TILE * 64];  /* 128 × 64 × 8 = 64 KB — fits stack */

    double ll   = 0.0;
    size_t i0   = 0;

    for (; i0 + (size_t)CX_TILE <= n; i0 += CX_TILE) {
        const double* xp = data + 2 * i0;   /* pointer to re of obs i0 */

        /* ── Pass 1: fill tile_lp[t*k+j] for all t, j ── */
        for (int j = 0; j < k; j++) {
            simde__m256d v_mu  = simde_mm256_set_pd(mu_im[j], mu_re[j],
                                                    mu_im[j], mu_re[j]);
            simde__m256d v_lc  = simde_mm256_set1_pd(lc[j]);
            simde__m256d v_niv = simde_mm256_set1_pd(-inv_var[j]);

            /* Process 2 complex observations (4 doubles) per AVX2 iter */
            size_t t = 0;
            size_t t4 = CX_TILE - (CX_TILE & 1);  /* even: always CX_TILE since 128 is even */
            for (; t < t4; t += 2) {
                /* xp + 2*t points to [re_t, im_t, re_{t+1}, im_{t+1}] — 4 doubles / 2 obs */
                simde__m256d v   = simde_mm256_loadu_pd(xp + 2*t);
                simde__m256d d   = simde_mm256_sub_pd(v, v_mu);
                simde__m256d sq  = simde_mm256_mul_pd(d, d);
                /* hadd: [sq[0]+sq[1], sq[0]+sq[1], sq[2]+sq[3], sq[2]+sq[3]]
                 *        = [|z_t-μ|², |z_t-μ|², |z_{t+1}-μ|², |z_{t+1}-μ|²]  */
                simde__m256d hs  = simde_mm256_hadd_pd(sq, sq);
                /* lp = lc[j] + (-inv_var) * dist²  */
                simde__m256d lp  = simde_mm256_add_pd(v_lc,
                                   simde_mm256_mul_pd(v_niv, hs));
                double tmp[4];
                simde_mm256_storeu_pd(tmp, lp);
                tile_lp[t       * k + j] = tmp[0];   /* obs t   */
                tile_lp[(t + 1) * k + j] = tmp[2];   /* obs t+1 */
            }
            /* Scalar tail (handles odd CX_TILE; currently CX_TILE=128 is even) */
            for (; t < (size_t)CX_TILE; t++) {
                double dr = xp[2*t]   - mu_re[j];
                double di = xp[2*t+1] - mu_im[j];
                tile_lp[t * k + j] = lc[j] - inv_var[j] * (dr*dr + di*di);
            }
        }

        /* ── Pass 2: log-sum-exp, normalise, write resp ── */
        for (size_t t = 0; t < (size_t)CX_TILE; t++) {
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

    /* ── Scalar tail for remaining < CX_TILE observations ── */
    for (size_t i = i0; i < n; i++) {
        double re_i = data[2*i], im_i = data[2*i+1];
        double lps[64];
        double mx = -1e30;
        for (int j = 0; j < k; j++) {
            double dr = re_i - mu_re[j];
            double di = im_i - mu_im[j];
            lps[j] = lc[j] - inv_var[j] * (dr*dr + di*di);
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
#endif /* USE_CX_AVX2 */


/* ════════════════════════════════════════════════════════════════════
 * Scalar fallback
 * ════════════════════════════════════════════════════════════════════ */
static double cx_estep_scalar(const double* data, size_t n,
                               const double* log_w,
                               const double* mu_re,
                               const double* mu_im,
                               const double* var,
                               int k,
                               double* resp)
{
    double lc[64], inv_var[64];
    for (int j = 0; j < k; j++) {
        double v = var[j] > 1e-300 ? var[j] : 1e-300;
        lc[j]      = log_w[j] - log(M_PI) - log(v);
        inv_var[j] = 1.0 / v;
    }

    /* Allocate temporary log-prob matrix [n × k] in row-major order */
    double* lp = (double*)malloc(n * (size_t)k * sizeof(double));
    if (!lp) return -1e30;

    for (size_t i = 0; i < n; i++) {
        double re_i = data[2*i], im_i = data[2*i+1];
        for (int j = 0; j < k; j++) {
            double dr = re_i - mu_re[j];
            double di = im_i - mu_im[j];
            lp[i*k + j] = lc[j] - inv_var[j] * (dr*dr + di*di);
        }
    }

    double ll = 0.0;
    for (size_t i = 0; i < n; i++) {
        double* row = lp + i * k;
        double mx = row[0];
        for (int j = 1; j < k; j++) if (row[j] > mx) mx = row[j];
        double tot = 0.0;
        for (int j = 0; j < k; j++) { row[j] = exp(row[j] - mx); tot += row[j]; }
        ll += mx + log(tot);
        double inv_t = 1.0 / tot;
        for (int j = 0; j < k; j++) row[j] *= inv_t;
    }

    /* Transpose from row-major lp[i*k+j] to column-major resp[j*n+i] */
    const size_t BLK = 64;
    for (size_t i0 = 0; i0 < n; i0 += BLK) {
        size_t ie = (i0 + BLK < n) ? i0 + BLK : n;
        for (int j = 0; j < k; j++) {
            double* dst = resp + j * n + i0;
            double* src = lp   + i0 * k + j;
            for (size_t i = i0; i < ie; i++, src += k, dst++) *dst = *src;
        }
    }

    free(lp);
    return ll;
}


/* ════════════════════════════════════════════════════════════════════
 * Public dispatch
 * ════════════════════════════════════════════════════════════════════ */
double simd_complex_circular_estep(const double* data, size_t n,
                                   const double* log_w,
                                   const double* mu_re,
                                   const double* mu_im,
                                   const double* var,
                                   int k,
                                   double* resp)
{
#ifdef USE_CX_AVX2
    return cx_estep_avx2(data, n, log_w, mu_re, mu_im, var, k, resp);
#else
    return cx_estep_scalar(data, n, log_w, mu_re, mu_im, var, k, resp);
#endif
}
