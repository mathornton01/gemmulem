/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * Complex-valued Gaussian mixture model EM.
 * License: GPL v3
 */

#include "complex_em.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    /* Allocate */
    double* weights = (double*)calloc(k, sizeof(double));
    CCircGaussParams* comps = (CCircGaussParams*)calloc(k, sizeof(CCircGaussParams));
    double* resp = (double*)calloc(n * k, sizeof(double));

    if (!weights || !comps || !resp) {
        free(weights); free(comps); free(resp);
        return -2;
    }

    /* Initialize: k-means++ centers + equal weights */
    kmeans_init_complex(data, n, k, comps);
    for (int j = 0; j < k; j++) weights[j] = 1.0 / k;

    double prev_ll = -1e30;
    int iter;

    for (iter = 0; iter < maxiter; iter++) {

        /* ── E-step: compute responsibilities ── */
        double ll = 0.0;
        for (size_t i = 0; i < n; i++) {
            double re = data[2*i], im = data[2*i+1];
            double max_logp = -1e30;

            /* Compute log(w_j * pdf_j) for each component */
            for (int j = 0; j < k; j++) {
                double logp = log(weights[j]) + ccirc_gauss_logpdf(re, im, &comps[j]);
                resp[i*k + j] = logp;
                if (logp > max_logp) max_logp = logp;
            }

            /* Log-sum-exp for numerical stability */
            double sum_exp = 0.0;
            for (int j = 0; j < k; j++) {
                resp[i*k + j] = exp(resp[i*k + j] - max_logp);
                sum_exp += resp[i*k + j];
            }

            /* Normalize */
            for (int j = 0; j < k; j++) {
                resp[i*k + j] /= sum_exp;
            }

            ll += max_logp + log(sum_exp);
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

        /* ── M-step: update parameters ── */
        for (int j = 0; j < k; j++) {
            double nj = 0.0;
            double sum_re = 0.0, sum_im = 0.0;

            /* Weighted sums for mean */
            for (size_t i = 0; i < n; i++) {
                double r = resp[i*k + j];
                nj += r;
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

            /* Update mean */
            comps[j].mu_re = sum_re / nj;
            comps[j].mu_im = sum_im / nj;

            /* Update variance: σ² = (1/N_j) Σ r_{ij} |z_i - μ_j|² */
            double sum_d2 = 0.0;
            for (size_t i = 0; i < n; i++) {
                double dr = data[2*i] - comps[j].mu_re;
                double di = data[2*i+1] - comps[j].mu_im;
                sum_d2 += resp[i*k + j] * (dr*dr + di*di);
            }
            comps[j].var = sum_d2 / nj;
            if (comps[j].var < 1e-10) comps[j].var = 1e-10;

            /* Update weight */
            weights[j] = nj / n;
        }
    }

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
