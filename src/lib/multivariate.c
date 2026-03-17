/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * Multivariate Gaussian mixture EM — pure C, no LAPACK.
 * License: GPL v3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "multivariate.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MV_PDF_FLOOR 1e-300
#define MV_COV_REG   1e-6    /* Regularization added to diagonal */

/* ════════════════════════════════════════════════════════════════════
 * Linear algebra helpers (no LAPACK dependency)
 * ════════════════════════════════════════════════════════════════════ */

/* Cholesky decomposition: A = L L^T (lower triangular L)
 * Returns 0 on success, -1 if not positive definite */
static int cholesky(const double* A, int d, double* L) {
    memset(L, 0, sizeof(double) * d * d);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L[i*d+k] * L[j*d+k];

            if (i == j) {
                double val = A[i*d+i] - sum;
                if (val <= 0) return -1;  /* not positive definite */
                L[i*d+j] = sqrt(val);
            } else {
                L[i*d+j] = (A[i*d+j] - sum) / L[j*d+j];
            }
        }
    }
    return 0;
}

/* Log-determinant from Cholesky factor: log|A| = 2 * Σ log(L_ii) */
static double log_det_cholesky(const double* L, int d) {
    double ld = 0;
    for (int i = 0; i < d; i++) ld += log(L[i*d+i]);
    return 2.0 * ld;
}

/* Solve L*y = b (forward substitution, L lower triangular) */
static void solve_lower(const double* L, int d, const double* b, double* y) {
    for (int i = 0; i < d; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++) s -= L[i*d+j] * y[j];
        y[i] = s / L[i*d+i];
    }
}

/* Compute (x-μ)^T Σ^{-1} (x-μ) using Cholesky factor */
static double mahalanobis_sq(const double* x, const double* mean,
                              const double* L_chol, int d) {
    double* diff = (double*)malloc(sizeof(double) * d);
    double* y = (double*)malloc(sizeof(double) * d);

    for (int i = 0; i < d; i++) diff[i] = x[i] - mean[i];
    solve_lower(L_chol, d, diff, y);

    double mah = 0;
    for (int i = 0; i < d; i++) mah += y[i] * y[i];

    free(diff);
    free(y);
    return mah;
}


/* ════════════════════════════════════════════════════════════════════
 * Multivariate Gaussian PDF
 * ════════════════════════════════════════════════════════════════════ */

double mvgauss_logpdf(const double* x, const MVGaussParams* p) {
    double mah = mahalanobis_sq(x, p->mean, p->cov_chol, p->dim);
    return -0.5 * (p->dim * log(2 * M_PI) + p->log_det + mah);
}

double mvgauss_pdf(const double* x, const MVGaussParams* p) {
    double lp = mvgauss_logpdf(x, p);
    return exp(lp);
}


/* ════════════════════════════════════════════════════════════════════
 * Allocation helpers
 * ════════════════════════════════════════════════════════════════════ */

static void alloc_mvparams(MVGaussParams* p, int d) {
    p->dim = d;
    p->mean = (double*)calloc(d, sizeof(double));
    p->cov = (double*)calloc(d * d, sizeof(double));
    p->cov_chol = (double*)calloc(d * d, sizeof(double));
    p->log_det = 0;
}

static void free_mvparams(MVGaussParams* p) {
    free(p->mean); free(p->cov); free(p->cov_chol);
    p->mean = p->cov = p->cov_chol = NULL;
}

static int update_cholesky(MVGaussParams* p) {
    int d = p->dim;
    /* Add regularization to diagonal */
    for (int i = 0; i < d; i++)
        p->cov[i*d+i] += MV_COV_REG;

    int rc = cholesky(p->cov, d, p->cov_chol);
    if (rc != 0) return -1;
    p->log_det = log_det_cholesky(p->cov_chol, d);

    /* Remove regularization from stored cov */
    for (int i = 0; i < d; i++)
        p->cov[i*d+i] -= MV_COV_REG;
    return 0;
}


/* ════════════════════════════════════════════════════════════════════
 * Multivariate Gaussian Mixture EM
 * ════════════════════════════════════════════════════════════════════ */

int UnmixMVGaussian(const double* data, size_t n, int d, int k,
                    CovType cov_type, int maxiter, double rtole,
                    int verbose, MVMixtureResult* result)
{
    if (!data || n == 0 || d <= 0 || k <= 0 || !result) return -1;

    /* Allocate result */
    result->num_components = k;
    result->dim = d;
    result->cov_type = cov_type;
    result->mixing_weights = (double*)malloc(sizeof(double) * k);
    result->components = (MVGaussParams*)malloc(sizeof(MVGaussParams) * k);

    for (int j = 0; j < k; j++) {
        alloc_mvparams(&result->components[j], d);
        result->mixing_weights[j] = 1.0 / k;
    }

    /* ─── Initialization: K-means++ style ─── */
    {
        /* Pick first center randomly (use data point n/2) */
        int first = (int)(n / 2);
        memcpy(result->components[0].mean, &data[first * d], sizeof(double) * d);

        /* Pick remaining centers: furthest-point heuristic */
        double* min_dist = (double*)malloc(sizeof(double) * n);
        for (size_t i = 0; i < n; i++) min_dist[i] = 1e30;

        for (int j = 1; j < k; j++) {
            /* Update min distances to nearest existing center */
            for (size_t i = 0; i < n; i++) {
                double dist = 0;
                for (int dd = 0; dd < d; dd++) {
                    double diff = data[i*d+dd] - result->components[j-1].mean[dd];
                    dist += diff * diff;
                }
                if (dist < min_dist[i]) min_dist[i] = dist;
            }
            /* Pick point with max min_dist */
            size_t best = 0;
            for (size_t i = 1; i < n; i++)
                if (min_dist[i] > min_dist[best]) best = i;
            memcpy(result->components[j].mean, &data[best * d], sizeof(double) * d);
        }
        free(min_dist);

        /* Initialize covariances from global variance */
        double* global_var = (double*)calloc(d, sizeof(double));
        double* global_mean = (double*)calloc(d, sizeof(double));
        for (size_t i = 0; i < n; i++)
            for (int dd = 0; dd < d; dd++)
                global_mean[dd] += data[i*d+dd];
        for (int dd = 0; dd < d; dd++) global_mean[dd] /= n;
        for (size_t i = 0; i < n; i++)
            for (int dd = 0; dd < d; dd++) {
                double diff = data[i*d+dd] - global_mean[dd];
                global_var[dd] += diff * diff;
            }
        for (int dd = 0; dd < d; dd++) global_var[dd] /= n;

        for (int j = 0; j < k; j++) {
            memset(result->components[j].cov, 0, sizeof(double) * d * d);
            for (int dd = 0; dd < d; dd++)
                result->components[j].cov[dd*d+dd] = global_var[dd] / k;
            update_cholesky(&result->components[j]);
        }
        free(global_var);
        free(global_mean);
    }

    /* Responsibilities: n × k */
    double* resp = (double*)malloc(sizeof(double) * n * k);
    double prev_ll = -1e30;

    /* ─── EM loop ─── */
    int iter;
    for (iter = 0; iter < maxiter; iter++) {

        /* ─── E-step ─── */
        double ll = 0;
        for (size_t i = 0; i < n; i++) {
            const double* xi = &data[i * d];
            double total = 0;
            double max_lp = -1e30;

            /* Log-sum-exp trick for numerical stability */
            double lps[64];  /* k <= 64 */
            int kk = k < 64 ? k : 64;
            for (int j = 0; j < kk; j++) {
                lps[j] = log(result->mixing_weights[j]) +
                         mvgauss_logpdf(xi, &result->components[j]);
                if (lps[j] > max_lp) max_lp = lps[j];
            }
            for (int j = 0; j < kk; j++) {
                double v = exp(lps[j] - max_lp);
                resp[i*k+j] = v;
                total += v;
            }
            for (int j = 0; j < kk; j++)
                resp[i*k+j] /= total;

            ll += max_lp + log(total);
        }

        double delta = fabs(ll - prev_ll);
        if (verbose) {
            printf("  [MV-Gauss k=%d d=%d] iter %d  LL=%.4f  delta=%.2e\n",
                   k, d, iter, ll, delta);
        }

        if (iter > 0 && delta < rtole) {
            iter++;
            break;
        }
        prev_ll = ll;

        /* ─── M-step ─── */
        for (int j = 0; j < k; j++) {
            double nj = 0;
            for (size_t i = 0; i < n; i++) nj += resp[i*k+j];

            if (nj < 1e-10) {
                result->mixing_weights[j] = 1e-10;
                continue;
            }

            /* Update weight */
            result->mixing_weights[j] = nj / n;

            /* Update mean */
            memset(result->components[j].mean, 0, sizeof(double) * d);
            for (size_t i = 0; i < n; i++) {
                double r = resp[i*k+j];
                for (int dd = 0; dd < d; dd++)
                    result->components[j].mean[dd] += r * data[i*d+dd];
            }
            for (int dd = 0; dd < d; dd++)
                result->components[j].mean[dd] /= nj;

            /* Update covariance */
            memset(result->components[j].cov, 0, sizeof(double) * d * d);

            if (cov_type == COV_FULL) {
                for (size_t i = 0; i < n; i++) {
                    double r = resp[i*k+j];
                    for (int a = 0; a < d; a++) {
                        double da = data[i*d+a] - result->components[j].mean[a];
                        for (int b = a; b < d; b++) {
                            double db = data[i*d+b] - result->components[j].mean[b];
                            result->components[j].cov[a*d+b] += r * da * db;
                        }
                    }
                }
                /* Symmetrize and normalize */
                for (int a = 0; a < d; a++) {
                    for (int b = a; b < d; b++) {
                        result->components[j].cov[a*d+b] /= nj;
                        result->components[j].cov[b*d+a] = result->components[j].cov[a*d+b];
                    }
                }
            } else if (cov_type == COV_DIAGONAL) {
                for (size_t i = 0; i < n; i++) {
                    double r = resp[i*k+j];
                    for (int dd = 0; dd < d; dd++) {
                        double diff = data[i*d+dd] - result->components[j].mean[dd];
                        result->components[j].cov[dd*d+dd] += r * diff * diff;
                    }
                }
                for (int dd = 0; dd < d; dd++)
                    result->components[j].cov[dd*d+dd] /= nj;
            } else { /* COV_SPHERICAL */
                double total_var = 0;
                for (size_t i = 0; i < n; i++) {
                    double r = resp[i*k+j];
                    for (int dd = 0; dd < d; dd++) {
                        double diff = data[i*d+dd] - result->components[j].mean[dd];
                        total_var += r * diff * diff;
                    }
                }
                total_var /= (nj * d);
                for (int dd = 0; dd < d; dd++)
                    result->components[j].cov[dd*d+dd] = total_var;
            }

            /* Update Cholesky */
            if (update_cholesky(&result->components[j]) != 0) {
                /* Reset to identity if Cholesky fails */
                memset(result->components[j].cov, 0, sizeof(double) * d * d);
                for (int dd = 0; dd < d; dd++)
                    result->components[j].cov[dd*d+dd] = 1.0;
                update_cholesky(&result->components[j]);
            }
        }

        /* Normalize weights */
        double wsum = 0;
        for (int j = 0; j < k; j++) wsum += result->mixing_weights[j];
        for (int j = 0; j < k; j++) result->mixing_weights[j] /= wsum;
    }

    result->iterations = iter;
    result->loglikelihood = prev_ll;

    /* Compute BIC/AIC */
    int nfree;
    switch (cov_type) {
        case COV_FULL:      nfree = k * (d + d*(d+1)/2) + k - 1; break;
        case COV_DIAGONAL:  nfree = k * (d + d) + k - 1; break;
        case COV_SPHERICAL: nfree = k * (d + 1) + k - 1; break;
        default:            nfree = k * (d + d*(d+1)/2) + k - 1;
    }
    result->bic = -2 * result->loglikelihood + nfree * log((double)n);
    result->aic = -2 * result->loglikelihood + 2 * nfree;

    free(resp);
    return 0;
}


void ReleaseMVMixtureResult(MVMixtureResult* r) {
    if (!r) return;
    if (r->components) {
        for (int j = 0; j < r->num_components; j++)
            free_mvparams(&r->components[j]);
        free(r->components);
        r->components = NULL;
    }
    free(r->mixing_weights);
    r->mixing_weights = NULL;
}


/* ════════════════════════════════════════════════════════════════════
 * Multivariate Student-t Mixture EM
 *
 * Each component has mean μ, scale matrix Σ, and degrees of freedom ν.
 * The E-step computes both responsibilities and per-point weights u_i,j
 * that down-weight outliers (the "robustness" mechanism).
 *
 * u_i,j = (ν_j + d) / (ν_j + δ²_i,j)
 * where δ²_i,j = Mahalanobis distance of x_i from component j.
 * ════════════════════════════════════════════════════════════════════ */

/* Multivariate Student-t log-PDF */
static double mvt_logpdf(const double* x, int d, const double* mean,
                         const double* L_chol, double log_det, double nu)
{
    /* Compute Mahalanobis distance */
    double* diff = (double*)malloc(sizeof(double) * d);
    double* y = (double*)malloc(sizeof(double) * d);
    for (int i = 0; i < d; i++) diff[i] = x[i] - mean[i];
    solve_lower(L_chol, d, diff, y);
    double mah = 0;
    for (int i = 0; i < d; i++) mah += y[i] * y[i];
    free(diff); free(y);

    /* log Γ((ν+d)/2) - log Γ(ν/2) - (d/2) log(νπ) - 0.5 log|Σ|
     * - ((ν+d)/2) log(1 + δ²/ν) */
    double lp = lgamma((nu + d) / 2.0) - lgamma(nu / 2.0)
              - (d / 2.0) * log(nu * M_PI)
              - 0.5 * log_det
              - ((nu + d) / 2.0) * log(1.0 + mah / nu);
    return lp;
}

static void alloc_mvt_params(MVStudentTParams* p, int d) {
    p->dim = d;
    p->mean = (double*)calloc(d, sizeof(double));
    p->cov = (double*)calloc(d * d, sizeof(double));
    p->cov_chol = (double*)calloc(d * d, sizeof(double));
    p->log_det = 0;
    p->nu = 5.0;  /* default */
}

static void free_mvt_params(MVStudentTParams* p) {
    free(p->mean); free(p->cov); free(p->cov_chol);
    p->mean = p->cov = p->cov_chol = NULL;
}

static int update_cholesky_t(MVStudentTParams* p) {
    int d = p->dim;
    for (int i = 0; i < d; i++)
        p->cov[i*d+i] += MV_COV_REG;
    int rc = cholesky(p->cov, d, p->cov_chol);
    if (rc != 0) return -1;
    p->log_det = log_det_cholesky(p->cov_chol, d);
    for (int i = 0; i < d; i++)
        p->cov[i*d+i] -= MV_COV_REG;
    return 0;
}

/* Digamma for ν optimization */
static double digamma_mv(double x) {
    if (x < 6) return digamma_mv(x + 1) - 1.0 / x;
    return log(x) - 1.0/(2*x) - 1.0/(12*x*x) + 1.0/(120*x*x*x*x);
}

int UnmixMVStudentT(const double* data, size_t n, int d, int k,
                    CovType cov_type, int maxiter, double rtole,
                    int verbose, MVStudentTResult* result)
{
    if (!data || n == 0 || d <= 0 || k <= 0 || !result) return -1;

    result->num_components = k;
    result->dim = d;
    result->cov_type = cov_type;
    result->mixing_weights = (double*)malloc(sizeof(double) * k);
    result->components = (MVStudentTParams*)malloc(sizeof(MVStudentTParams) * k);

    for (int j = 0; j < k; j++) {
        alloc_mvt_params(&result->components[j], d);
        result->mixing_weights[j] = 1.0 / k;
        result->components[j].nu = 5.0;
    }

    /* Initialize: same as Gaussian (k-means++ style) */
    {
        int first = (int)(n / 2);
        memcpy(result->components[0].mean, &data[first * d], sizeof(double) * d);

        double* min_dist = (double*)malloc(sizeof(double) * n);
        for (size_t i = 0; i < n; i++) min_dist[i] = 1e30;

        for (int j = 1; j < k; j++) {
            for (size_t i = 0; i < n; i++) {
                double dist = 0;
                for (int dd = 0; dd < d; dd++) {
                    double diff = data[i*d+dd] - result->components[j-1].mean[dd];
                    dist += diff * diff;
                }
                if (dist < min_dist[i]) min_dist[i] = dist;
            }
            size_t best = 0;
            for (size_t i = 1; i < n; i++)
                if (min_dist[i] > min_dist[best]) best = i;
            memcpy(result->components[j].mean, &data[best * d], sizeof(double) * d);
        }
        free(min_dist);

        double* global_var = (double*)calloc(d, sizeof(double));
        double* global_mean = (double*)calloc(d, sizeof(double));
        for (size_t i = 0; i < n; i++)
            for (int dd = 0; dd < d; dd++)
                global_mean[dd] += data[i*d+dd];
        for (int dd = 0; dd < d; dd++) global_mean[dd] /= n;
        for (size_t i = 0; i < n; i++)
            for (int dd = 0; dd < d; dd++) {
                double diff = data[i*d+dd] - global_mean[dd];
                global_var[dd] += diff * diff;
            }
        for (int dd = 0; dd < d; dd++) global_var[dd] /= n;

        for (int j = 0; j < k; j++) {
            memset(result->components[j].cov, 0, sizeof(double) * d * d);
            for (int dd = 0; dd < d; dd++)
                result->components[j].cov[dd*d+dd] = global_var[dd] / k;
            update_cholesky_t(&result->components[j]);
        }
        free(global_var); free(global_mean);
    }

    double* resp = (double*)malloc(sizeof(double) * n * k);
    double* u_weights = (double*)malloc(sizeof(double) * n * k);  /* per-point weights */
    double prev_ll = -1e30;

    int iter;
    for (iter = 0; iter < maxiter; iter++) {

        /* E-step: compute responsibilities AND u-weights */
        double ll = 0;
        for (size_t i = 0; i < n; i++) {
            const double* xi = &data[i * d];
            double max_lp = -1e30;
            double lps[64];
            int kk = k < 64 ? k : 64;

            for (int j = 0; j < kk; j++) {
                lps[j] = log(result->mixing_weights[j]) +
                         mvt_logpdf(xi, d, result->components[j].mean,
                                    result->components[j].cov_chol,
                                    result->components[j].log_det,
                                    result->components[j].nu);
                if (lps[j] > max_lp) max_lp = lps[j];
            }

            double total = 0;
            for (int j = 0; j < kk; j++) {
                double v = exp(lps[j] - max_lp);
                resp[i*k+j] = v;
                total += v;
            }
            for (int j = 0; j < kk; j++) resp[i*k+j] /= total;
            ll += max_lp + log(total);

            /* Compute u-weights: u_ij = (ν_j + d) / (ν_j + δ²_ij) */
            for (int j = 0; j < kk; j++) {
                double mah = mahalanobis_sq(xi, result->components[j].mean,
                                            result->components[j].cov_chol, d);
                double nu_j = result->components[j].nu;
                u_weights[i*k+j] = (nu_j + d) / (nu_j + mah);
            }
        }

        if (verbose)
            printf("  [MV-T k=%d d=%d] iter %d  LL=%.4f  delta=%.2e  nu=[",
                   k, d, iter, ll, ll - prev_ll);
        if (verbose) {
            for (int j = 0; j < k; j++) printf("%.1f%s", result->components[j].nu, j<k-1?",":"");
            printf("]\n");
        }

        if (iter > 0 && fabs(ll - prev_ll) < rtole) { iter++; break; }
        prev_ll = ll;

        /* M-step */
        for (int j = 0; j < k; j++) {
            double nj = 0, nj_u = 0;
            for (size_t i = 0; i < n; i++) {
                nj += resp[i*k+j];
                nj_u += resp[i*k+j] * u_weights[i*k+j];
            }
            if (nj < 1e-10) { result->mixing_weights[j] = 1e-10; continue; }

            result->mixing_weights[j] = nj / n;

            /* Update mean: weighted by r_ij * u_ij */
            memset(result->components[j].mean, 0, sizeof(double) * d);
            for (size_t i = 0; i < n; i++) {
                double ru = resp[i*k+j] * u_weights[i*k+j];
                for (int dd = 0; dd < d; dd++)
                    result->components[j].mean[dd] += ru * data[i*d+dd];
            }
            for (int dd = 0; dd < d; dd++)
                result->components[j].mean[dd] /= nj_u;

            /* Update scale matrix: weighted by r_ij * u_ij */
            memset(result->components[j].cov, 0, sizeof(double) * d * d);
            if (cov_type == COV_FULL) {
                for (size_t i = 0; i < n; i++) {
                    double ru = resp[i*k+j] * u_weights[i*k+j];
                    for (int a = 0; a < d; a++) {
                        double da = data[i*d+a] - result->components[j].mean[a];
                        for (int b = a; b < d; b++) {
                            double db = data[i*d+b] - result->components[j].mean[b];
                            result->components[j].cov[a*d+b] += ru * da * db;
                        }
                    }
                }
                for (int a = 0; a < d; a++)
                    for (int b = a; b < d; b++) {
                        result->components[j].cov[a*d+b] /= nj;
                        result->components[j].cov[b*d+a] = result->components[j].cov[a*d+b];
                    }
            } else if (cov_type == COV_DIAGONAL) {
                for (size_t i = 0; i < n; i++) {
                    double ru = resp[i*k+j] * u_weights[i*k+j];
                    for (int dd = 0; dd < d; dd++) {
                        double diff = data[i*d+dd] - result->components[j].mean[dd];
                        result->components[j].cov[dd*d+dd] += ru * diff * diff;
                    }
                }
                for (int dd = 0; dd < d; dd++)
                    result->components[j].cov[dd*d+dd] /= nj;
            } else { /* spherical */
                double tv = 0;
                for (size_t i = 0; i < n; i++) {
                    double ru = resp[i*k+j] * u_weights[i*k+j];
                    for (int dd = 0; dd < d; dd++) {
                        double diff = data[i*d+dd] - result->components[j].mean[dd];
                        tv += ru * diff * diff;
                    }
                }
                tv /= (nj * d);
                for (int dd = 0; dd < d; dd++)
                    result->components[j].cov[dd*d+dd] = tv;
            }

            if (update_cholesky_t(&result->components[j]) != 0) {
                memset(result->components[j].cov, 0, sizeof(double) * d * d);
                for (int dd = 0; dd < d; dd++)
                    result->components[j].cov[dd*d+dd] = 1.0;
                update_cholesky_t(&result->components[j]);
            }

            /* Update ν via fixed-point iteration (Peel & McLachlan 2000) */
            double nu_old = result->components[j].nu;
            for (int nu_iter = 0; nu_iter < 20; nu_iter++) {
                double nu = result->components[j].nu;
                /* Fixed-point: ψ(ν/2+d/2) - log(ν/2+d/2) + 1 + E[log u - u] + ψ(ν/2) - log(ν/2) = 0
                 * Simplified update: solve for ν using bisection */
                double sum_log_u = 0, sum_u = 0;
                for (size_t i = 0; i < n; i++) {
                    double w = resp[i*k+j] / nj;
                    sum_log_u += w * log(u_weights[i*k+j]);
                    sum_u += w * u_weights[i*k+j];
                }
                /* Newton step for ν */
                double target = 1.0 + sum_log_u - sum_u
                              + digamma_mv((nu + d) / 2.0) - log((nu + d) / 2.0)
                              - digamma_mv(nu / 2.0) + log(nu / 2.0);
                /* Approximate gradient */
                double eps = 0.01;
                double target2 = 1.0 + sum_log_u - sum_u
                               + digamma_mv((nu+eps+d)/2.0) - log((nu+eps+d)/2.0)
                               - digamma_mv((nu+eps)/2.0) + log((nu+eps)/2.0);
                double grad = (target2 - target) / eps;
                if (fabs(grad) > 1e-10) {
                    double nu_new = nu - target / grad;
                    if (nu_new < 1.0) nu_new = 1.0;
                    if (nu_new > 200.0) nu_new = 200.0;
                    result->components[j].nu = nu_new;
                    if (fabs(nu_new - nu) < 0.01) break;
                } else break;
            }
        }

        double wsum = 0;
        for (int j = 0; j < k; j++) wsum += result->mixing_weights[j];
        for (int j = 0; j < k; j++) result->mixing_weights[j] /= wsum;
    }

    result->iterations = iter;
    result->loglikelihood = prev_ll;

    int nfree;
    switch (cov_type) {
        case COV_FULL:      nfree = k * (d + d*(d+1)/2 + 1) + k - 1; break; /* +1 for ν */
        case COV_DIAGONAL:  nfree = k * (d + d + 1) + k - 1; break;
        case COV_SPHERICAL: nfree = k * (d + 1 + 1) + k - 1; break;
        default:            nfree = k * (d + d*(d+1)/2 + 1) + k - 1;
    }
    result->bic = -2 * prev_ll + nfree * log((double)n);
    result->aic = -2 * prev_ll + 2 * nfree;

    free(resp); free(u_weights);
    return 0;
}

void ReleaseMVStudentTResult(MVStudentTResult* r) {
    if (!r) return;
    if (r->components) {
        for (int j = 0; j < r->num_components; j++)
            free_mvt_params(&r->components[j]);
        free(r->components);
        r->components = NULL;
    }
    free(r->mixing_weights);
    r->mixing_weights = NULL;
}


/* ════════════════════════════════════════════════════════════════════
 * Multivariate Auto-k (BIC-driven)
 * ════════════════════════════════════════════════════════════════════ */

int UnmixMVAutoK(const double* data, size_t n, int d, int k_max,
                 CovType cov_type, int maxiter, double rtole,
                 int verbose, MVAutoKResult* result)
{
    if (!data || n == 0 || d <= 0 || k_max <= 0 || !result) return -1;

    result->dim = d;
    result->cov_type = cov_type;
    result->best_bic = 1e30;
    result->best_k = 1;

    MVMixtureResult best;
    memset(&best, 0, sizeof(best));
    int best_initialized = 0;

    for (int k = 1; k <= k_max; k++) {
        MVMixtureResult r;
        int rc = UnmixMVGaussian(data, n, d, k, cov_type, maxiter, rtole, 0, &r);
        if (rc != 0) continue;

        if (verbose)
            printf("  [MV-AutoK] k=%d  LL=%.2f  BIC=%.2f  AIC=%.2f%s\n",
                   k, r.loglikelihood, r.bic, r.aic,
                   r.bic < result->best_bic ? "  <-- new best" : "");

        if (r.bic < result->best_bic) {
            if (best_initialized) ReleaseMVMixtureResult(&best);
            result->best_bic = r.bic;
            result->best_ll = r.loglikelihood;
            result->best_k = k;
            best = r;
            best_initialized = 1;
        } else {
            ReleaseMVMixtureResult(&r);
            /* If BIC increased for 2 consecutive k values, stop early */
            if (k > result->best_k + 1) break;
        }
    }

    if (best_initialized) {
        result->best_model = best;
    }

    if (verbose)
        printf("  [MV-AutoK] Selected k=%d  BIC=%.2f\n", result->best_k, result->best_bic);

    return 0;
}

void ReleaseMVAutoKResult(MVAutoKResult* r) {
    if (!r) return;
    ReleaseMVMixtureResult(&r->best_model);
}
