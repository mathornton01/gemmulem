/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * Complex-valued Gaussian mixture model EM.
 * License: GPL v3
 */

#include "complex_em.h"
#include "simd_complex_estep.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Threshold: use SIMD E-step when n >= this value */
#define SIMD_CIRC_THRESHOLD 1000

/* ════════════════════════════════════════════════════════════════════
 * Circular symmetric complex Gaussian PDF
 *
 * f(z|μ,σ²) = (1/πσ²) exp(-|z-μ|²/σ²)
 *
 * Log PDF = -log(π) - log(σ²) - |z-μ|²/σ²
 * ════════════════════════════════════════════════════════════════════ */

double ccirc_gauss_logpdf(double re, double im, const CCircGaussParams* p) {
    double dr = re - p->mu_re;
    double di = im - p->mu_im;
    double dist2 = dr*dr + di*di;
    return -log(M_PI) - log(p->var) - dist2 / p->var;
}

double ccirc_gauss_pdf(double re, double im, const CCircGaussParams* p) {
    return exp(ccirc_gauss_logpdf(re, im, p));
}

/* ════════════════════════════════════════════════════════════════════
 * Non-circular complex Gaussian PDF
 *
 * Augmented representation: z̃ = [z-μ; conj(z-μ)]
 * Augmented covariance: Γ = [Σ  C; C* Σ*]
 *   where Σ = E[(z-μ)(z-μ)*] (covariance)
 *         C = E[(z-μ)(z-μ)]  (pseudo-covariance)
 *
 * For scalar z: Σ ∈ ℝ⁺ (variance), C ∈ ℂ (pseudo-covariance)
 *
 * PDF: f(z) = (1 / π√det(Γ)) exp(-½ z̃ᴴ Γ⁻¹ z̃)
 *
 * det(Γ) = |Σ|² - |C|² = Σ² - |C|²  (since Σ is real for scalar)
 *
 * Γ⁻¹ = (1/det(Γ)) [Σ* -C; -C* Σ]
 *
 * z̃ᴴ Γ⁻¹ z̃ = (1/det(Γ)) [Σ|dz|² - Re(C·dz²) - Re(C*·dz*²) + ... ]
 *            = (1/det(Γ)) [Σ|dz|² - 2Re(C* · dz²)]
 *
 * where dz = z - μ
 * ════════════════════════════════════════════════════════════════════ */

double cnocirc_gauss_logpdf(double re, double im, const CNonCircGaussParams* p) {
    double dr = re - p->mu_re;
    double di = im - p->mu_im;

    /* Covariance Σ (real for scalar case) */
    double sigma = p->cov_re;

    /* Pseudo-covariance C = pcov_re + i*pcov_im */
    double c_re = p->pcov_re;
    double c_im = p->pcov_im;

    /* det(Γ) = Σ² - |C|² */
    double c_abs2 = c_re*c_re + c_im*c_im;
    double det = sigma*sigma - c_abs2;

    if (det <= 0.0) return -1e30;  /* degenerate */

    /* dz² = (dr + i·di)² = (dr²-di²) + 2i·dr·di */
    double dz2_re = dr*dr - di*di;
    double dz2_im = 2.0*dr*di;

    /* |dz|² = dr² + di² */
    double dz_abs2 = dr*dr + di*di;

    /* Re(C* · dz²) = c_re*dz2_re + c_im*dz2_im */
    double re_conj_c_dz2 = c_re*dz2_re + c_im*dz2_im;

    /* Exponent: -(1/det) [Σ·|dz|² - 2·Re(C*·dz²)]
     *
     * Derivation: z̃ᴴ Γ⁻¹ z̃ for scalar z with Γ = [Σ C; C* Σ]:
     *   Γ⁻¹ = (1/det) [Σ  -C; -C*  Σ]
     *   z̃ = [dz; dz*], z̃ᴴ = [dz*  dz]
     *   z̃ᴴ Γ⁻¹ z̃ = (1/det)[dz*·Σ·dz - dz*·C·dz* + dz·(-C*)·dz + dz·Σ·dz*]
     *             = (1/det)[2Σ|dz|² - 2Re(C·dz*²)]
     * Note: Re(C·dz*²) = Re(conj(C*·dz²)) = Re(C*·dz²) (since Re is invariant under conj)
     * So: z̃ᴴ Γ⁻¹ z̃ = (2/det)[Σ|dz|² - Re(C*·dz²)]
     * And the PDF exponent is -½ · z̃ᴴ Γ⁻¹ z̃ = -(1/det)[Σ|dz|² - Re(C*·dz²)]
     */
    double exponent = -1.0 / det * (sigma * dz_abs2 - re_conj_c_dz2);

    /* Normalization: -log(π) - 0.5*log(det(Γ)) */
    return -log(M_PI) - 0.5*log(det) + exponent;
}

double cnocirc_gauss_pdf(double re, double im, const CNonCircGaussParams* p) {
    return exp(cnocirc_gauss_logpdf(re, im, p));
}


/* ════════════════════════════════════════════════════════════════════
 * K-means++ initialization for complex data
 * ════════════════════════════════════════════════════════════════════ */
static void kmeans_init_complex(const double* data, size_t n, int k,
                                CCircGaussParams* params) {
    /* Pick first center uniformly at random */
    size_t c0 = (size_t)(((double)rand()/RAND_MAX) * n);
    if (c0 >= n) c0 = n-1;
    params[0].mu_re = data[2*c0];
    params[0].mu_im = data[2*c0+1];

    double* dist2 = (double*)malloc(n * sizeof(double));
    if (!dist2) return;

    for (int j = 1; j < k; j++) {
        /* Compute D²: min distance to existing centers */
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            double re_i = data[2*i], im_i = data[2*i+1];
            double mind2 = DBL_MAX;
            for (int c = 0; c < j; c++) {
                double dr = re_i - params[c].mu_re;
                double di = im_i - params[c].mu_im;
                double d2 = dr*dr + di*di;
                if (d2 < mind2) mind2 = d2;
            }
            dist2[i] = mind2;
            sum += mind2;
        }
        /* Roulette wheel selection */
        double target = ((double)rand()/RAND_MAX) * sum;
        double cumul = 0.0;
        size_t chosen = n-1;
        for (size_t i = 0; i < n; i++) {
            cumul += dist2[i];
            if (cumul >= target) { chosen = i; break; }
        }
        params[j].mu_re = data[2*chosen];
        params[j].mu_im = data[2*chosen+1];
    }

    /* Initialize variances from data spread around each center */
    for (int j = 0; j < k; j++) {
        double total_d2 = 0.0;
        int count = 0;
        for (size_t i = 0; i < n; i++) {
            /* Assign to nearest center */
            double re_i = data[2*i], im_i = data[2*i+1];
            double dr = re_i - params[j].mu_re;
            double di = im_i - params[j].mu_im;
            double d2 = dr*dr + di*di;
            /* Check if this is the nearest center */
            int nearest = 1;
            for (int c = 0; c < k; c++) {
                if (c == j) continue;
                double dr2 = re_i - params[c].mu_re;
                double di2 = im_i - params[c].mu_im;
                if (dr2*dr2+di2*di2 < d2) { nearest = 0; break; }
            }
            if (nearest) { total_d2 += d2; count++; }
        }
        params[j].var = (count > 1) ? total_d2 / count : 1.0;
        if (params[j].var < 1e-10) params[j].var = 1e-10;
    }

    free(dist2);
}


/* ════════════════════════════════════════════════════════════════════
 * Circular symmetric complex Gaussian mixture EM
 * ════════════════════════════════════════════════════════════════════ */

int UnmixComplexCircular(const double* data, size_t n, int k,
                         int maxiter, double rtole, int verbose,
                         CCircMixtureResult* result) {
    if (!data || n < 1 || k < 1 || !result) return -1;

    memset(result, 0, sizeof(*result));

    /* resp is stored column-major: resp[j*n + i] = r_{ij}
     * This layout is required by simd_complex_circular_estep and is
     * also cache-friendly for the M-step inner loops. */
    double* weights = (double*)calloc(k, sizeof(double));
    CCircGaussParams* comps = (CCircGaussParams*)calloc(k, sizeof(CCircGaussParams));
    double* resp = (double*)calloc((size_t)n * k, sizeof(double));

    if (!weights || !comps || !resp) {
        free(weights); free(comps); free(resp);
        return -2;
    }

    /* Initialize: k-means++ centers + equal weights */
    kmeans_init_complex(data, n, k, comps);
    for (int j = 0; j < k; j++) weights[j] = 1.0 / k;

    /* Temporary arrays for SIMD path */
    double* log_w_tmp  = (double*)malloc(k * sizeof(double));
    double* mu_re_tmp  = (double*)malloc(k * sizeof(double));
    double* mu_im_tmp  = (double*)malloc(k * sizeof(double));
    double* var_tmp    = (double*)malloc(k * sizeof(double));
    if (!log_w_tmp || !mu_re_tmp || !mu_im_tmp || !var_tmp) {
        free(log_w_tmp); free(mu_re_tmp); free(mu_im_tmp); free(var_tmp);
        free(weights); free(comps); free(resp);
        return -2;
    }

    double prev_ll = -1e30;
    int iter;
    int use_simd = (n >= SIMD_CIRC_THRESHOLD);

    for (iter = 0; iter < maxiter; iter++) {

        double ll;

        if (use_simd) {
            /* ── SIMD E-step ── */
            for (int j = 0; j < k; j++) {
                log_w_tmp[j] = log(weights[j]);
                mu_re_tmp[j] = comps[j].mu_re;
                mu_im_tmp[j] = comps[j].mu_im;
                var_tmp[j]   = comps[j].var;
            }
            ll = simd_complex_circular_estep(data, n,
                                             log_w_tmp, mu_re_tmp, mu_im_tmp,
                                             var_tmp, k, resp);
        } else {
            /* ── Scalar E-step: write column-major resp[j*n + i] ── */
            ll = 0.0;
            for (size_t i = 0; i < n; i++) {
                double re = data[2*i], im = data[2*i+1];
                double max_logp = -1e30;
                double lps[64];
                for (int j = 0; j < k; j++) {
                    lps[j] = log(weights[j]) + ccirc_gauss_logpdf(re, im, &comps[j]);
                    if (lps[j] > max_logp) max_logp = lps[j];
                }
                double sum_exp = 0.0;
                for (int j = 0; j < k; j++) {
                    lps[j] = exp(lps[j] - max_logp);
                    sum_exp += lps[j];
                }
                double inv_s = 1.0 / sum_exp;
                for (int j = 0; j < k; j++) resp[j*n + i] = lps[j] * inv_s;
                ll += max_logp + log(sum_exp);
            }
        }

        if (verbose && (iter < 5 || iter % 10 == 0 || iter == maxiter-1)) {
            printf("  iter %3d: LL = %.6f\n", iter, ll);
        }

        /* Check convergence */
        if (iter > 0 && fabs(ll - prev_ll) < rtole * fabs(prev_ll)) {
            if (verbose) printf("  Converged at iter %d (LL=%.6f)\n", iter, ll);
            break;
        }
        prev_ll = ll;

        /* ── M-step: column-major resp[j*n + i] ── */
        for (int j = 0; j < k; j++) {
            double nj = 0.0;
            double sum_re = 0.0, sum_im = 0.0;
            const double* rj = resp + j * n;

            for (size_t i = 0; i < n; i++) {
                double r = rj[i];
                nj     += r;
                sum_re += r * data[2*i];
                sum_im += r * data[2*i+1];
            }

            if (nj < 1e-10) {
                /* Dead component — reinitialize */
                size_t ri = (size_t)(((double)rand()/RAND_MAX) * n);
                if (ri >= n) ri = n-1;
                comps[j].mu_re = data[2*ri];
                comps[j].mu_im = data[2*ri+1];
                comps[j].var = 1.0;
                weights[j] = 1e-3;
                continue;
            }

            comps[j].mu_re = sum_re / nj;
            comps[j].mu_im = sum_im / nj;

            /* Variance: σ² = (1/N_j) Σ rᵢⱼ |zᵢ - μⱼ|² */
            double sum_d2 = 0.0;
            for (size_t i = 0; i < n; i++) {
                double dr = data[2*i]   - comps[j].mu_re;
                double di = data[2*i+1] - comps[j].mu_im;
                sum_d2 += rj[i] * (dr*dr + di*di);
            }
            comps[j].var = sum_d2 / nj;
            if (comps[j].var < 1e-10) comps[j].var = 1e-10;

            weights[j] = nj / n;
        }
    }

    free(log_w_tmp); free(mu_re_tmp); free(mu_im_tmp); free(var_tmp);

    /* Compute final log-likelihood */
    double final_ll = 0.0;
    for (size_t i = 0; i < n; i++) {
        double re = data[2*i], im = data[2*i+1];
        double max_logp = -1e30;
        double lps[64]; /* max 64 components */
        for (int j = 0; j < k && j < 64; j++) {
            lps[j] = log(weights[j]) + ccirc_gauss_logpdf(re, im, &comps[j]);
            if (lps[j] > max_logp) max_logp = lps[j];
        }
        double sum = 0.0;
        for (int j = 0; j < k && j < 64; j++) sum += exp(lps[j] - max_logp);
        final_ll += max_logp + log(sum);
    }

    /* BIC/AIC: k components × 3 params each + (k-1) weights */
    int nparams = k * 3 + (k - 1);
    result->num_components = k;
    result->iterations = iter;
    result->loglikelihood = final_ll;
    result->bic = -2.0*final_ll + nparams * log((double)n);
    result->aic = -2.0*final_ll + 2.0*nparams;
    result->mixing_weights = weights;
    result->components = comps;

    free(resp);
    return 0;
}


/* ════════════════════════════════════════════════════════════════════
 * Auto-k for circular complex Gaussian (BIC)
 * ════════════════════════════════════════════════════════════════════ */

int UnmixComplexCircularAutoK(const double* data, size_t n, int k_max,
                              int maxiter, double rtole, int verbose,
                              CCircMixtureResult* result) {
    if (!data || n < 1 || !result) return -1;
    if (k_max <= 0) k_max = 10;

    double best_bic = 1e30;
    CCircMixtureResult best = {0};
    int best_found = 0;

    for (int k = 1; k <= k_max; k++) {
        CCircMixtureResult trial = {0};
        int ret = UnmixComplexCircular(data, n, k, maxiter, rtole, 0, &trial);
        if (ret != 0) continue;

        if (verbose) {
            printf("  k=%d: BIC=%.2f, LL=%.2f\n", k, trial.bic, trial.loglikelihood);
        }

        if (trial.bic < best_bic) {
            if (best_found) ReleaseCCircResult(&best);
            best = trial;
            best_bic = trial.bic;
            best_found = 1;
        } else {
            ReleaseCCircResult(&trial);
        }
    }

    if (!best_found) return -3;
    *result = best;
    return 0;
}


void ReleaseCCircResult(CCircMixtureResult* result) {
    if (result) {
        free(result->mixing_weights);
        free(result->components);
        memset(result, 0, sizeof(*result));
    }
}


/* ════════════════════════════════════════════════════════════════════
 * Non-circular complex Gaussian mixture EM
 * ════════════════════════════════════════════════════════════════════ */

int UnmixComplexNonCircular(const double* data, size_t n, int k,
                            int maxiter, double rtole, int verbose,
                            CNonCircMixtureResult* result) {
    if (!data || n < 1 || k < 1 || !result) return -1;

    memset(result, 0, sizeof(*result));

    double* weights = (double*)calloc(k, sizeof(double));
    CNonCircGaussParams* comps = (CNonCircGaussParams*)calloc(k, sizeof(CNonCircGaussParams));
    double* resp = (double*)calloc(n * k, sizeof(double));

    if (!weights || !comps || !resp) {
        free(weights); free(comps); free(resp);
        return -2;
    }

    /* Initialize with circular params, zero pseudo-covariance */
    {
        CCircGaussParams* circ = (CCircGaussParams*)calloc(k, sizeof(CCircGaussParams));
        if (circ) {
            kmeans_init_complex(data, n, k, circ);
            for (int j = 0; j < k; j++) {
                comps[j].mu_re = circ[j].mu_re;
                comps[j].mu_im = circ[j].mu_im;
                comps[j].cov_re = circ[j].var;
                comps[j].cov_im = 0.0;
                comps[j].pcov_re = 0.0;
                comps[j].pcov_im = 0.0;
            }
            free(circ);
        }
    }
    for (int j = 0; j < k; j++) weights[j] = 1.0 / k;

    double prev_ll = -1e30;
    int iter;

    for (iter = 0; iter < maxiter; iter++) {

        /* ── E-step ── */
        double ll = 0.0;
        for (size_t i = 0; i < n; i++) {
            double re = data[2*i], im = data[2*i+1];
            double max_logp = -1e30;
            for (int j = 0; j < k; j++) {
                double logp = log(weights[j]) + cnocirc_gauss_logpdf(re, im, &comps[j]);
                resp[i*k + j] = logp;
                if (logp > max_logp) max_logp = logp;
            }
            double sum_exp = 0.0;
            for (int j = 0; j < k; j++) {
                resp[i*k + j] = exp(resp[i*k + j] - max_logp);
                sum_exp += resp[i*k + j];
            }
            for (int j = 0; j < k; j++) resp[i*k + j] /= sum_exp;
            ll += max_logp + log(sum_exp);
        }

        if (verbose && (iter < 5 || iter % 10 == 0))
            printf("  iter %3d: LL = %.6f\n", iter, ll);

        if (iter > 0 && fabs(ll - prev_ll) < rtole * fabs(prev_ll)) {
            if (verbose) printf("  Converged at iter %d\n", iter);
            break;
        }
        prev_ll = ll;

        /* ── M-step ── */
        for (int j = 0; j < k; j++) {
            double nj = 0.0;
            double sum_re = 0.0, sum_im = 0.0;

            for (size_t i = 0; i < n; i++) {
                double r = resp[i*k + j];
                nj += r;
                sum_re += r * data[2*i];
                sum_im += r * data[2*i+1];
            }

            if (nj < 1e-10) {
                size_t ri = (size_t)(((double)rand()/RAND_MAX) * n);
                if (ri >= n) ri = n-1;
                comps[j].mu_re = data[2*ri];
                comps[j].mu_im = data[2*ri+1];
                comps[j].cov_re = 1.0;
                comps[j].pcov_re = 0.0;
                comps[j].pcov_im = 0.0;
                weights[j] = 1e-3;
                continue;
            }

            comps[j].mu_re = sum_re / nj;
            comps[j].mu_im = sum_im / nj;

            /* Covariance Σ = (1/N_j) Σ r_{ij} |dz_i|² */
            double sum_var = 0.0;
            /* Pseudo-covariance C = (1/N_j) Σ r_{ij} dz_i² */
            double sum_pc_re = 0.0, sum_pc_im = 0.0;

            for (size_t i = 0; i < n; i++) {
                double r = resp[i*k + j];
                double dr = data[2*i] - comps[j].mu_re;
                double di = data[2*i+1] - comps[j].mu_im;

                sum_var += r * (dr*dr + di*di);
                /* dz² = (dr+i·di)² = (dr²-di²) + i·2·dr·di */
                sum_pc_re += r * (dr*dr - di*di);
                sum_pc_im += r * 2.0*dr*di;
            }

            comps[j].cov_re = sum_var / nj;
            comps[j].cov_im = 0.0; /* Always real for scalar variance */
            comps[j].pcov_re = sum_pc_re / nj;
            comps[j].pcov_im = sum_pc_im / nj;

            /* Ensure det(Γ) > 0: Σ² > |C|² */
            double pc_abs2 = comps[j].pcov_re*comps[j].pcov_re +
                             comps[j].pcov_im*comps[j].pcov_im;
            if (pc_abs2 >= comps[j].cov_re * comps[j].cov_re * 0.99) {
                /* Shrink pseudo-covariance to maintain positive definiteness */
                double scale = 0.95 * comps[j].cov_re / sqrt(pc_abs2 + 1e-30);
                comps[j].pcov_re *= scale;
                comps[j].pcov_im *= scale;
            }

            if (comps[j].cov_re < 1e-10) comps[j].cov_re = 1e-10;
            weights[j] = nj / n;
        }
    }

    /* Final log-likelihood */
    double final_ll = 0.0;
    for (size_t i = 0; i < n; i++) {
        double re = data[2*i], im = data[2*i+1];
        double max_logp = -1e30;
        double lps[64];
        for (int j = 0; j < k && j < 64; j++) {
            lps[j] = log(weights[j]) + cnocirc_gauss_logpdf(re, im, &comps[j]);
            if (lps[j] > max_logp) max_logp = lps[j];
        }
        double sum = 0.0;
        for (int j = 0; j < k && j < 64; j++) sum += exp(lps[j] - max_logp);
        final_ll += max_logp + log(sum);
    }

    /* Non-circular has 6 params per component: mu_re, mu_im, Σ, C_re, C_im + weight */
    int nparams = k * 5 + (k - 1);
    result->num_components = k;
    result->iterations = iter;
    result->loglikelihood = final_ll;
    result->bic = -2.0*final_ll + nparams * log((double)n);
    result->aic = -2.0*final_ll + 2.0*nparams;
    result->mixing_weights = weights;
    result->components = comps;

    free(resp);
    return 0;
}


void ReleaseCNonCircResult(CNonCircMixtureResult* result) {
    if (result) {
        free(result->mixing_weights);
        free(result->components);
        memset(result, 0, sizeof(*result));
    }
}


/* ════════════════════════════════════════════════════════════════════
 * Feature 1: Multivariate Complex Gaussian Mixture EM
 *
 * f(z|μ,Σ) = (1/(πᵈ det(Σ))) exp(-(z-μ)ᴴ Σ⁻¹ (z-μ))
 *
 * Covariance Σ is Hermitian positive-definite.
 * Stored flat: cov[2*(i*d+j)] = Re(Σ[i,j]), cov[2*(i*d+j)+1] = Im(Σ[i,j])
 * ════════════════════════════════════════════════════════════════════ */

#define MVC_COV_REG  1e-6   /* Diagonal regularisation when Cholesky fails */

/* ── Complex Cholesky: Σ = L Lᴴ, L lower triangular ──────────────── */
/* L[j,j] real & positive; L[i,j] complex for i>j; upper is zero.     */
static int complex_cholesky(const double* A, int d, double* L) {
    memset(L, 0, sizeof(double) * 2 * d * d);
    for (int j = 0; j < d; j++) {
        /* Diagonal element – must be real & positive for Hermitian A  */
        double sum = 0.0;
        for (int kk = 0; kk < j; kk++) {
            double lr = L[2*(j*d+kk)];
            double li = L[2*(j*d+kk)+1];
            sum += lr*lr + li*li;
        }
        double val = A[2*(j*d+j)] - sum;   /* A[j,j] is real (Hermitian diagonal) */
        if (val <= 0.0) return -1;
        L[2*(j*d+j)]   = sqrt(val);
        L[2*(j*d+j)+1] = 0.0;

        /* Sub-diagonal column elements */
        double ljj = L[2*(j*d+j)];
        for (int i = j+1; i < d; i++) {
            /* L[i,j] = (1/L[j,j]) * (A[i,j] - Σₖ<ⱼ L[i,k]·conj(L[j,k])) */
            double s_re = A[2*(i*d+j)];
            double s_im = A[2*(i*d+j)+1];
            for (int kk = 0; kk < j; kk++) {
                double lir = L[2*(i*d+kk)];
                double lii = L[2*(i*d+kk)+1];
                double ljr = L[2*(j*d+kk)];
                double lji = L[2*(j*d+kk)+1];
                /* L[i,k] * conj(L[j,k]) */
                s_re -= lir*ljr + lii*lji;
                s_im -= lii*ljr - lir*lji;
            }
            L[2*(i*d+j)]   = s_re / ljj;
            L[2*(i*d+j)+1] = s_im / ljj;
        }
    }
    return 0;
}

/* ── Forward substitution: Ly = b (complex L lower triangular) ────── */
static void complex_solve_lower(const double* L, int d,
                                const double* b, double* y) {
    for (int i = 0; i < d; i++) {
        double sr = b[2*i];
        double si = b[2*i+1];
        for (int j = 0; j < i; j++) {
            double lr = L[2*(i*d+j)];
            double li = L[2*(i*d+j)+1];
            double yr = y[2*j];
            double yi = y[2*j+1];
            sr -= lr*yr - li*yi;
            si -= lr*yi + li*yr;
        }
        double diag = L[2*(i*d+i)];  /* real, positive */
        y[2*i]   = sr / diag;
        y[2*i+1] = si / diag;
    }
}

/* ── Log-determinant from Cholesky: 2 Σᵢ log(L[i,i]) ──────────────── */
static double complex_chol_logdet(const double* L, int d) {
    double ld = 0.0;
    for (int i = 0; i < d; i++) ld += log(L[2*(i*d+i)]);
    return 2.0 * ld;
}

/* ── (z-μ)ᴴ Σ⁻¹ (z-μ) = ||y||² where Ly = (z-μ) ──────────────────── */
static double complex_quad_form(const double* L, int d, const double* diff) {
    double* y = (double*)malloc(2 * d * sizeof(double));
    if (!y) return 1e30;
    complex_solve_lower(L, d, diff, y);
    double q = 0.0;
    for (int i = 0; i < d; i++) q += y[2*i]*y[2*i] + y[2*i+1]*y[2*i+1];
    free(y);
    return q;
}

/* ── MV complex Gaussian log-PDF ──────────────────────────────────── */
static double mv_complex_logpdf(const double* z, const MVComplexGaussParams* p) {
    int d = p->dim;
    double* diff = (double*)malloc(2 * d * sizeof(double));
    if (!diff) return -1e30;
    for (int i = 0; i < d; i++) {
        diff[2*i]   = z[2*i]   - p->mean[2*i];
        diff[2*i+1] = z[2*i+1] - p->mean[2*i+1];
    }
    double q = complex_quad_form(p->cov_chol, d, diff);
    free(diff);
    return -(double)d * log(M_PI) - p->log_det - q;
}

/* ── Allocate / free MVComplexGaussParams ─────────────────────────── */
static int alloc_mv_complex_params(MVComplexGaussParams* p, int d) {
    p->dim      = d;
    p->mean     = (double*)calloc(2 * d,     sizeof(double));
    p->cov      = (double*)calloc(2 * d * d, sizeof(double));
    p->cov_chol = (double*)calloc(2 * d * d, sizeof(double));
    p->log_det  = 0.0;
    if (!p->mean || !p->cov || !p->cov_chol) {
        free(p->mean); free(p->cov); free(p->cov_chol);
        p->mean = p->cov = p->cov_chol = NULL;
        return -1;
    }
    return 0;
}

static void free_mv_complex_params(MVComplexGaussParams* p) {
    free(p->mean); free(p->cov); free(p->cov_chol);
    p->mean = p->cov = p->cov_chol = NULL;
}

/* Regularise diagonal and recompute Cholesky; also stores log_det.     */
static int update_mv_complex_cholesky(MVComplexGaussParams* p) {
    int d = p->dim;
    /* Add small regularisation to Re(diagonal) */
    for (int i = 0; i < d; i++) p->cov[2*(i*d+i)] += MVC_COV_REG;
    int rc = complex_cholesky(p->cov, d, p->cov_chol);
    if (rc != 0) {
        /* Heavier regularisation on failure */
        for (int i = 0; i < d; i++) p->cov[2*(i*d+i)] += 1e-3;
        rc = complex_cholesky(p->cov, d, p->cov_chol);
    }
    for (int i = 0; i < d; i++) p->cov[2*(i*d+i)] -= MVC_COV_REG;
    if (rc != 0) return -1;
    p->log_det = complex_chol_logdet(p->cov_chol, d);
    return 0;
}

/* ── K-means++ init for multivariate complex data ─────────────────── */
static void kmeans_init_mv_complex(const double* data, size_t n, int d, int k,
                                   MVComplexGaussParams* params,
                                   double* mixing_weights) {
    int d2 = 2 * d;
    /* Choose first center uniformly */
    size_t c0 = (size_t)(((double)rand() / RAND_MAX) * n);
    if (c0 >= n) c0 = n - 1;
    memcpy(params[0].mean, data + c0 * d2, d2 * sizeof(double));

    double* dist2 = (double*)malloc(n * sizeof(double));
    if (!dist2) return;

    for (int jj = 1; jj < k; jj++) {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double* zi = data + i * d2;
            double mind2 = DBL_MAX;
            for (int c = 0; c < jj; c++) {
                double d2c = 0.0;
                for (int dd = 0; dd < d; dd++) {
                    double dr = zi[2*dd]   - params[c].mean[2*dd];
                    double di = zi[2*dd+1] - params[c].mean[2*dd+1];
                    d2c += dr*dr + di*di;
                }
                if (d2c < mind2) mind2 = d2c;
            }
            dist2[i] = mind2;
            sum += mind2;
        }
        double target = ((double)rand() / RAND_MAX) * sum;
        double cumul = 0.0;
        size_t chosen = n - 1;
        for (size_t i = 0; i < n; i++) {
            cumul += dist2[i];
            if (cumul >= target) { chosen = i; break; }
        }
        memcpy(params[jj].mean, data + chosen * d2, d2 * sizeof(double));
    }
    free(dist2);

    /* Initialise covariance as scaled identity, equal weights */
    /* Estimate per-component variance from nearest-center distances */
    double* total_d2 = (double*)calloc(k, sizeof(double));
    int* counts = (int*)calloc(k, sizeof(int));
    if (total_d2 && counts) {
        for (size_t i = 0; i < n; i++) {
            const double* zi = data + i * d2;
            int nearest = 0;
            double mind2 = DBL_MAX;
            for (int c = 0; c < k; c++) {
                double d2c = 0.0;
                for (int dd = 0; dd < d; dd++) {
                    double dr = zi[2*dd]   - params[c].mean[2*dd];
                    double di = zi[2*dd+1] - params[c].mean[2*dd+1];
                    d2c += dr*dr + di*di;
                }
                if (d2c < mind2) { mind2 = d2c; nearest = c; }
            }
            total_d2[nearest] += mind2;
            counts[nearest]++;
        }
        for (int jj = 0; jj < k; jj++) {
            double var = (counts[jj] > 0) ? total_d2[jj] / (counts[jj] * d) : 1.0;
            if (var < 1e-10) var = 1.0;
            /* Covariance = var * I  (only real diagonal) */
            memset(params[jj].cov, 0, 2 * d * d * sizeof(double));
            for (int dd = 0; dd < d; dd++)
                params[jj].cov[2*(dd*d+dd)] = var;
            mixing_weights[jj] = 1.0 / k;
            update_mv_complex_cholesky(&params[jj]);
        }
    }
    free(total_d2);
    free(counts);
}

/* ── Public API ───────────────────────────────────────────────────── */

int UnmixMVComplex(const double* data, size_t n, int d, int k,
                   int maxiter, double rtole, int verbose,
                   MVComplexMixtureResult* result) {
    if (!data || n < 1 || d < 1 || k < 1 || !result) return -1;

    memset(result, 0, sizeof(*result));
    result->num_components = k;
    result->dim = d;

    double* weights = (double*)malloc(k * sizeof(double));
    MVComplexGaussParams* comps =
        (MVComplexGaussParams*)calloc(k, sizeof(MVComplexGaussParams));
    double* resp = (double*)malloc((size_t)n * k * sizeof(double));

    if (!weights || !comps || !resp) goto oom;

    for (int jj = 0; jj < k; jj++) {
        if (alloc_mv_complex_params(&comps[jj], d) != 0) goto oom;
    }

    /* Initialise */
    kmeans_init_mv_complex(data, n, d, k, comps, weights);

    double prev_ll = -1e30;
    int iter;
    int d2 = 2 * d;

    for (iter = 0; iter < maxiter; iter++) {

        /* ── E-step ── */
        double ll = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double* zi = data + i * d2;
            double max_logp = -1e30;
            for (int jj = 0; jj < k; jj++) {
                double logp = log(weights[jj]) + mv_complex_logpdf(zi, &comps[jj]);
                resp[i*k + jj] = logp;
                if (logp > max_logp) max_logp = logp;
            }
            double sum_exp = 0.0;
            for (int jj = 0; jj < k; jj++) {
                resp[i*k + jj] = exp(resp[i*k + jj] - max_logp);
                sum_exp += resp[i*k + jj];
            }
            for (int jj = 0; jj < k; jj++) resp[i*k + jj] /= sum_exp;
            ll += max_logp + log(sum_exp);
        }

        if (verbose && (iter < 5 || iter % 10 == 0 || iter == maxiter-1))
            printf("  iter %3d: LL = %.6f\n", iter, ll);

        if (iter > 0 && fabs(ll - prev_ll) < rtole * fabs(prev_ll)) {
            if (verbose) printf("  Converged at iter %d (LL=%.6f)\n", iter, ll);
            break;
        }
        prev_ll = ll;

        /* ── M-step ── */
        for (int jj = 0; jj < k; jj++) {
            double nj = 0.0;

            /* Weighted mean */
            double* new_mean = comps[jj].mean;
            memset(new_mean, 0, d2 * sizeof(double));
            for (size_t i = 0; i < n; i++) {
                double r = resp[i*k + jj];
                nj += r;
                const double* zi = data + i * d2;
                for (int dd = 0; dd < d2; dd++) new_mean[dd] += r * zi[dd];
            }

            if (nj < 1e-10) {
                /* Dead component — reinitialise */
                size_t ri = (size_t)(((double)rand() / RAND_MAX) * n);
                if (ri >= n) ri = n - 1;
                memcpy(comps[jj].mean, data + ri * d2, d2 * sizeof(double));
                memset(comps[jj].cov, 0, 2*d*d*sizeof(double));
                for (int dd = 0; dd < d; dd++) comps[jj].cov[2*(dd*d+dd)] = 1.0;
                weights[jj] = 1e-3;
                update_mv_complex_cholesky(&comps[jj]);
                continue;
            }

            for (int dd = 0; dd < d2; dd++) new_mean[dd] /= nj;

            /* Weighted Hermitian covariance:
             * Σ[p,q] = (1/Nⱼ) Σᵢ rᵢⱼ (zᵢ[p]-μ[p]) conj(zᵢ[q]-μ[q])
             * Re(Σ[p,q]) = (1/Nⱼ) Σᵢ rᵢⱼ (drᵢₚ·drᵢᵧ + diᵢₚ·diᵢᵧ)
             * Im(Σ[p,q]) = (1/Nⱼ) Σᵢ rᵢⱼ (diᵢₚ·drᵢᵧ - drᵢₚ·diᵢᵧ)
             */
            double* cov = comps[jj].cov;
            memset(cov, 0, 2*d*d*sizeof(double));
            for (size_t i = 0; i < n; i++) {
                double r = resp[i*k + jj];
                if (r < 1e-15) continue;
                const double* zi = data + i * d2;
                for (int p = 0; p < d; p++) {
                    double drp = zi[2*p]   - new_mean[2*p];
                    double dip = zi[2*p+1] - new_mean[2*p+1];
                    for (int q = 0; q < d; q++) {
                        double drq = zi[2*q]   - new_mean[2*q];
                        double diq = zi[2*q+1] - new_mean[2*q+1];
                        /* (z[p]-μ[p]) conj(z[q]-μ[q]) */
                        cov[2*(p*d+q)]   += r * (drp*drq + dip*diq);
                        cov[2*(p*d+q)+1] += r * (dip*drq - drp*diq);
                    }
                }
            }
            for (int e = 0; e < 2*d*d; e++) cov[e] /= nj;

            weights[jj] = nj / n;
            if (update_mv_complex_cholesky(&comps[jj]) != 0) {
                /* Fallback: reset to identity */
                memset(cov, 0, 2*d*d*sizeof(double));
                for (int dd = 0; dd < d; dd++) cov[2*(dd*d+dd)] = 1.0;
                update_mv_complex_cholesky(&comps[jj]);
            }
        }
        /* Renormalise weights */
        double wsum = 0.0;
        for (int jj = 0; jj < k; jj++) wsum += weights[jj];
        if (wsum > 0) for (int jj = 0; jj < k; jj++) weights[jj] /= wsum;
    }

    /* Final log-likelihood */
    double final_ll = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double* zi = data + i * d2;
        double max_logp = -1e30;
        double lps[64];
        for (int jj = 0; jj < k && jj < 64; jj++) {
            lps[jj] = log(weights[jj]) + mv_complex_logpdf(zi, &comps[jj]);
            if (lps[jj] > max_logp) max_logp = lps[jj];
        }
        double sum = 0.0;
        for (int jj = 0; jj < k && jj < 64; jj++) sum += exp(lps[jj] - max_logp);
        final_ll += max_logp + log(sum);
    }

    /* Parameters per component: 2d (mean) + d² (complex cov, d² complex entries)
     * Hermitian constraint halves independent entries; plus (k-1) weights. */
    int nparams = k * (2*d + d*d) + (k - 1);
    result->loglikelihood = final_ll;
    result->bic = -2.0*final_ll + nparams * log((double)n);
    result->aic = -2.0*final_ll + 2.0*nparams;
    result->iterations = iter;
    result->mixing_weights = weights;
    result->components = comps;

    free(resp);
    return 0;

oom:
    free(resp);
    free(weights);
    if (comps) {
        for (int jj = 0; jj < k; jj++) free_mv_complex_params(&comps[jj]);
        free(comps);
    }
    return -2;
}

void ReleaseMVComplexResult(MVComplexMixtureResult* result) {
    if (!result) return;
    if (result->components) {
        for (int jj = 0; jj < result->num_components; jj++)
            free_mv_complex_params(&result->components[jj]);
        free(result->components);
    }
    free(result->mixing_weights);
    memset(result, 0, sizeof(*result));
}


/* ════════════════════════════════════════════════════════════════════
 * Feature 3: Streaming Complex EM (Cappé & Moulines 2009)
 *
 * Processes IQ data from a file in chunks.  Never loads more than
 * chunk_size complex observations into RAM at once.
 *
 * Step-size schedule: γₜ = (t + 2)^(-eta_decay)
 * ════════════════════════════════════════════════════════════════════ */

int UnmixComplexStreaming(const char* filename,
                          const ComplexStreamConfig* config,
                          CCircMixtureResult* result) {
    if (!filename || !config || !result) return -1;

    int k          = config->num_components;
    int chunk_size = config->chunk_size  > 0 ? config->chunk_size  : 10000;
    int max_passes = config->max_passes  > 0 ? config->max_passes  : 10;
    double rtole   = config->rtole       > 0 ? config->rtole       : 1e-5;
    double decay   = config->eta_decay   > 0 ? config->eta_decay   : 0.6;

    if (k < 1) return -1;

    memset(result, 0, sizeof(*result));

    /* ── Pass 0: count lines and collect global stats for init ────── */
    FILE* fp = fopen(filename, "r");
    if (!fp) return -3;

    size_t total_n = 0;
    double g_sum_re = 0, g_sum_im = 0;
    double g_sum2 = 0;
    double g_min_re = 1e30, g_max_re = -1e30;
    double g_min_im = 1e30, g_max_im = -1e30;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        double re = 0, im = 0;
        /* Accept "re im" or "re,im" */
        char* comma = strchr(line, ',');
        if (comma) { *comma = ' '; }
        if (sscanf(line, "%lf %lf", &re, &im) != 2) continue;
        g_sum_re += re; g_sum_im += im;
        g_sum2 += re*re + im*im;
        if (re < g_min_re) g_min_re = re;
        if (re > g_max_re) g_max_re = re;
        if (im < g_min_im) g_min_im = im;
        if (im > g_max_im) g_max_im = im;
        total_n++;
    }
    fclose(fp);

    if (total_n == 0) return -4;

    double g_mean_re = g_sum_re / total_n;
    double g_mean_im = g_sum_im / total_n;
    double g_var = g_sum2 / total_n - (g_mean_re*g_mean_re + g_mean_im*g_mean_im);
    if (g_var < 1e-10) g_var = 1.0;

    if (config->verbose)
        printf("  [stream] n=%zu  mean=(%.4f,%.4f)  var=%.4f\n",
               total_n, g_mean_re, g_mean_im, g_var);

    /* ── Allocate result ─────────────────────────────────────────── */
    result->num_components = k;
    result->mixing_weights = (double*)malloc(k * sizeof(double));
    result->components     = (CCircGaussParams*)malloc(k * sizeof(CCircGaussParams));
    if (!result->mixing_weights || !result->components) goto stream_oom;

    /* ── Initialise: spread centers over IQ bounding box ─────────── */
    for (int jj = 0; jj < k; jj++) {
        double frac = (k > 1) ? (double)jj / (k - 1) : 0.5;
        result->components[jj].mu_re = g_min_re + frac * (g_max_re - g_min_re);
        result->components[jj].mu_im = g_min_im + frac * (g_max_im - g_min_im);
        result->components[jj].var   = g_var / k;
        if (result->components[jj].var < 1e-10)
            result->components[jj].var = 1e-10;
        result->mixing_weights[jj] = 1.0 / k;
    }

    /* ── Sufficient statistics (Cappé & Moulines) ───────────────── */
    /* s0[j]: weighted count, s1_re[j]/s1_im[j]: weighted mean sums,
     * s2[j]: weighted |z|² sum (for variance) */
    double* s0     = (double*)calloc(k, sizeof(double));
    double* s1_re  = (double*)calloc(k, sizeof(double));
    double* s1_im  = (double*)calloc(k, sizeof(double));
    double* s2     = (double*)calloc(k, sizeof(double));
    double* chunk  = (double*)malloc(2 * chunk_size * sizeof(double));
    double* c_resp = (double*)malloc((size_t)k * chunk_size * sizeof(double));
    if (!s0 || !s1_re || !s1_im || !s2 || !chunk || !c_resp) {
        free(s0); free(s1_re); free(s1_im); free(s2);
        free(chunk); free(c_resp);
        goto stream_oom;
    }

    /* Seed sufficient stats from initial params */
    for (int jj = 0; jj < k; jj++) {
        s0[jj]    = result->mixing_weights[jj];
        s1_re[jj] = result->mixing_weights[jj] * result->components[jj].mu_re;
        s1_im[jj] = result->mixing_weights[jj] * result->components[jj].mu_im;
        double var = result->components[jj].var;
        double mu2 = result->components[jj].mu_re * result->components[jj].mu_re
                   + result->components[jj].mu_im * result->components[jj].mu_im;
        s2[jj]    = result->mixing_weights[jj] * (var + mu2);
    }

    double prev_ll  = -1e30;
    int global_step = 0;

    for (int pass = 0; pass < max_passes; pass++) {
        fp = fopen(filename, "r");
        if (!fp) break;

        double pass_ll = 0.0;
        size_t pass_n  = 0;

        while (1) {
            /* Read a chunk */
            int n_read = 0;
            while (n_read < chunk_size && fgets(line, sizeof(line), fp)) {
                if (line[0] == '#' || line[0] == '\n') continue;
                double re = 0, im = 0;
                char* comma = strchr(line, ',');
                if (comma) { *comma = ' '; }
                if (sscanf(line, "%lf %lf", &re, &im) != 2) continue;
                chunk[2*n_read]   = re;
                chunk[2*n_read+1] = im;
                n_read++;
            }
            if (n_read == 0) break;

            double eta = pow((double)(global_step + 2), -decay);
            global_step++;

            /* ── E-step on chunk ── */
            double chunk_ll = 0.0;
            for (int i = 0; i < n_read; i++) {
                double re_i = chunk[2*i], im_i = chunk[2*i+1];
                double max_logp = -1e30;
                for (int jj = 0; jj < k; jj++) {
                    double lp = log(result->mixing_weights[jj])
                              + ccirc_gauss_logpdf(re_i, im_i, &result->components[jj]);
                    c_resp[jj * n_read + i] = lp;
                    if (lp > max_logp) max_logp = lp;
                }
                double sum_e = 0.0;
                for (int jj = 0; jj < k; jj++) {
                    c_resp[jj * n_read + i] = exp(c_resp[jj * n_read + i] - max_logp);
                    sum_e += c_resp[jj * n_read + i];
                }
                for (int jj = 0; jj < k; jj++) c_resp[jj * n_read + i] /= sum_e;
                chunk_ll += max_logp + log(sum_e);
            }
            pass_ll += chunk_ll;
            pass_n  += (size_t)n_read;

            /* ── Online M-step: update sufficient statistics ── */
            for (int jj = 0; jj < k; jj++) {
                double nw = 0, nwx_re = 0, nwx_im = 0, nwxx = 0;
                for (int i = 0; i < n_read; i++) {
                    double r  = c_resp[jj * n_read + i];
                    double re_i = chunk[2*i], im_i = chunk[2*i+1];
                    nw     += r;
                    nwx_re += r * re_i;
                    nwx_im += r * im_i;
                    nwxx   += r * (re_i*re_i + im_i*im_i);
                }
                /* Normalise by chunk size to get per-obs averages */
                nw     /= n_read;
                nwx_re /= n_read;
                nwx_im /= n_read;
                nwxx   /= n_read;

                /* Exponential moving average */
                s0[jj]    = (1 - eta) * s0[jj]    + eta * nw;
                s1_re[jj] = (1 - eta) * s1_re[jj] + eta * nwx_re;
                s1_im[jj] = (1 - eta) * s1_im[jj] + eta * nwx_im;
                s2[jj]    = (1 - eta) * s2[jj]    + eta * nwxx;
            }

            /* Reconstruct parameters */
            double wsum = 0.0;
            for (int jj = 0; jj < k; jj++) wsum += s0[jj];
            for (int jj = 0; jj < k; jj++) {
                result->mixing_weights[jj] = s0[jj] / (wsum > 1e-15 ? wsum : 1e-15);
                double s0j = s0[jj] > 1e-15 ? s0[jj] : 1e-15;
                double mu_re = s1_re[jj] / s0j;
                double mu_im = s1_im[jj] / s0j;
                double var   = s2[jj] / s0j - (mu_re*mu_re + mu_im*mu_im);
                if (var < 1e-10) var = 1e-10;
                result->components[jj].mu_re = mu_re;
                result->components[jj].mu_im = mu_im;
                result->components[jj].var   = var;
            }
        }
        fclose(fp);

        double avg_ll = pass_ll / (pass_n > 0 ? (double)pass_n : 1.0);
        if (config->verbose)
            printf("  [stream] pass %d/%d  avg_LL=%.6f  eta=%.4f\n",
                   pass + 1, max_passes, avg_ll, pow((double)(global_step + 1), -decay));

        if (pass > 0 && fabs(avg_ll - prev_ll) < rtole) {
            if (config->verbose) printf("  [stream] converged at pass %d\n", pass + 1);
            break;
        }
        prev_ll = avg_ll;
    }

    /* ── Final full-pass LL ───────────────────────────────────────── */
    fp = fopen(filename, "r");
    double final_ll = 0.0;
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            double re = 0, im = 0;
            char* comma = strchr(line, ',');
            if (comma) { *comma = ' '; }
            if (sscanf(line, "%lf %lf", &re, &im) != 2) continue;
            double max_logp = -1e30;
            double lps[64];
            for (int jj = 0; jj < k && jj < 64; jj++) {
                lps[jj] = log(result->mixing_weights[jj])
                        + ccirc_gauss_logpdf(re, im, &result->components[jj]);
                if (lps[jj] > max_logp) max_logp = lps[jj];
            }
            double s = 0.0;
            for (int jj = 0; jj < k && jj < 64; jj++) s += exp(lps[jj] - max_logp);
            final_ll += max_logp + log(s);
        }
        fclose(fp);
    }

    result->loglikelihood = final_ll;
    result->iterations    = global_step;
    int nfree = k * 3 + (k - 1);
    result->bic = -2.0 * final_ll + nfree * log((double)total_n);
    result->aic = -2.0 * final_ll + 2.0 * nfree;

    free(s0); free(s1_re); free(s1_im); free(s2);
    free(chunk); free(c_resp);
    return 0;

stream_oom:
    free(result->mixing_weights); result->mixing_weights = NULL;
    free(result->components);     result->components     = NULL;
    return -2;
}
