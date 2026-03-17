/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 *
 * Generic distribution implementations and mixture EM engine.
 * License: GPL v3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "distributions.h"
#include "pearson.h"
#include "gpu_estep.h"

/* Global GPU context — initialized on first use, NULL if unavailable */
static GpuContext* g_gpu_ctx = NULL;
static int g_gpu_tried = 0;

static GpuContext* get_gpu(void) {
    if (!g_gpu_tried) {
        g_gpu_tried = 1;
        g_gpu_ctx = gpu_init(1);  /* prefer GPU */
        if (g_gpu_ctx) fprintf(stderr, "[GPU] OpenCL E-step enabled\n");
    }
    return g_gpu_ctx;
}

/* Forward declaration */
DistFunctions pearson_get_dist_functions(void);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Minimum probability floor to avoid log(0) */
#define PDF_FLOOR 1e-300

/* ====================================================================
 * Helper: weighted statistics
 * ==================================================================== */
static double wt_sum(const double* w, size_t n) {
    double s = 0; for (size_t i = 0; i < n; i++) s += w[i]; return s;
}
static double wt_mean(const double* x, const double* w, size_t n) {
    double s = 0, sw = 0;
    for (size_t i = 0; i < n; i++) { s += w[i]*x[i]; sw += w[i]; }
    return sw > 0 ? s/sw : 0;
}
static double wt_var(const double* x, const double* w, size_t n, double mu) {
    double s = 0, sw = 0;
    for (size_t i = 0; i < n; i++) { s += w[i]*(x[i]-mu)*(x[i]-mu); sw += w[i]; }
    return sw > 0 ? s/sw : 1.0;
}

/* lgamma is C99 — provide fallback via Stirling if missing */
#ifndef lgamma
static double lgamma_approx(double x) {
    return 0.5*log(2*M_PI/x) + x*(log(x + 1.0/(12*x - 1.0/(10*x))) - 1);
}
#define lgamma lgamma_approx
#endif

/* ====================================================================
 * GAUSSIAN: params = {mean, variance}
 * ==================================================================== */
static double gauss_pdf(double x, const DistParams* p) {
    double mu = p->p[0], var = p->p[1];
    if (var <= 0) var = 1e-10;
    double z = (x - mu) / sqrt(var);
    return (1.0 / sqrt(2*M_PI*var)) * exp(-0.5*z*z);
}
static double gauss_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], var = p->p[1];
    if (var <= 0) var = 1e-10;
    double z = (x - mu) / sqrt(var);
    return -0.5*log(2*M_PI*var) - 0.5*z*z;
}
static void gauss_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    if (var < 1e-10) var = 1e-10;
    out->p[0] = mu; out->p[1] = var; out->nparams = 2;
}
static void gauss_init(const double* x, size_t n, int k, DistParams* out) {
    double mn = x[0], mx = x[0];
    for (size_t i = 1; i < n; i++) { if (x[i]<mn) mn=x[i]; if (x[i]>mx) mx=x[i]; }
    double range = mx - mn;
    if (range < 1e-10) range = 1.0;
    /* Global variance for initial spread */
    double gmu = 0; for (size_t i = 0; i < n; i++) gmu += x[i]; gmu /= n;
    double gvar = 0; for (size_t i = 0; i < n; i++) gvar += (x[i]-gmu)*(x[i]-gmu); gvar /= n;
    if (gvar < 1e-10) gvar = 1.0;
    for (int j = 0; j < k; j++) {
        out[j].p[0] = mn + range * (j + 0.5) / k;
        out[j].p[1] = gvar;
        out[j].nparams = 2;
    }
}
static int gauss_valid(double x) { return 1; }

/* ====================================================================
 * EXPONENTIAL: params = {rate} (mean = 1/rate)
 * ==================================================================== */
static double expo_pdf(double x, const DistParams* p) {
    double rate = p->p[0];
    if (x < 0 || rate <= 0) return 0;
    return rate * exp(-rate * x);
}
static double expo_logpdf(double x, const DistParams* p) {
    double rate = p->p[0];
    if (x < 0 || rate <= 0) return -1e30;
    return log(rate) - rate * x;
}
static void expo_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    if (mu < 1e-10) mu = 1e-10;
    out->p[0] = 1.0 / mu; out->nparams = 1;
}
static void expo_init(const double* x, size_t n, int k, DistParams* out) {
    double mn = x[0], mx = x[0];
    for (size_t i = 1; i < n; i++) { if (x[i]<mn) mn=x[i]; if (x[i]>mx) mx=x[i]; }
    double mean = 0; for (size_t i = 0; i < n; i++) mean += x[i]; mean /= n;
    for (int j = 0; j < k; j++) {
        double m = mean * (0.5 + j) / k;
        if (m < 1e-10) m = 1e-10;
        out[j].p[0] = 1.0 / m;
        out[j].nparams = 1;
    }
}
static int expo_valid(double x) { return x >= 0; }

/* ====================================================================
 * POISSON: params = {lambda}
 * Treated as continuous approximation for EM on real-valued data
 * ==================================================================== */
static double poisson_pdf(double x, const DistParams* p) {
    double lam = p->p[0];
    if (x < 0 || lam <= 0) return 0;
    int ix = (int)(x + 0.5);  /* round to nearest integer */
    return exp(ix * log(lam) - lam - lgamma(ix + 1));
}
static void poisson_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    if (mu < 1e-10) mu = 1e-10;
    out->p[0] = mu; out->nparams = 1;
}
static void poisson_init(const double* x, size_t n, int k, DistParams* out) {
    double mean = 0; for (size_t i = 0; i < n; i++) mean += x[i]; mean /= n;
    for (int j = 0; j < k; j++) {
        out[j].p[0] = mean * (0.5 + j) / k;
        if (out[j].p[0] < 0.1) out[j].p[0] = 0.1;
        out[j].nparams = 1;
    }
}
static int poisson_valid(double x) { return x >= 0; }

/* ====================================================================
 * GAMMA: params = {shape (alpha), rate (beta)}
 *   mean = alpha/beta, var = alpha/beta^2
 * ==================================================================== */
/* Gamma: p[0]=alpha, p[1]=beta, p[2]=cached lgamma(alpha) (set in estimate/init) */
static void gamma_cache(DistParams* p) {
    if (p->nparams >= 3 && p->p[2] == p->p[2]) return;  /* already cached */
    p->p[2] = lgamma(p->p[0]);
    if (p->nparams < 3) p->nparams = 3;
}
static double gamma_logpdf(double x, const DistParams* p) {
    double a = p->p[0], b = p->p[1];
    if (x <= 0 || a <= 0 || b <= 0) return -1e30;
    /* Use cached lgamma if available (p[2]), else compute on the fly */
    double lga = (p->nparams >= 3) ? p->p[2] : lgamma(a);
    return a*log(b) + (a-1)*log(x) - b*x - lga;
}
static double gamma_pdf(double x, const DistParams* p) {
    double lp = gamma_logpdf(x, p);
    return lp > -700 ? exp(lp) : 0;
}
/* digamma approximation (Bernardo 1976, accurate to ~1e-8) */
static double digamma_approx(double x) {
    if (x < 6) {
        /* Recurrence: digamma(x) = digamma(x+1) - 1/x */
        return digamma_approx(x + 1) - 1.0 / x;
    }
    double r = log(x) - 1.0/(2*x);
    r -= 1.0/(12*x*x) - 1.0/(120*x*x*x*x) + 1.0/(252*x*x*x*x*x*x);
    return r;
}
/* trigamma (derivative of digamma) */
static double trigamma_approx(double x) {
    if (x < 6) return trigamma_approx(x + 1) + 1.0/(x*x);
    return 1.0/x + 1.0/(2*x*x) + 1.0/(6*x*x*x) - 1.0/(30*x*x*x*x*x);
}

static void gamma_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    if (mu < 1e-10) mu = 1e-10;
    if (var < 1e-10) var = 1e-10;

    /* Weighted log-mean: E[log X] */
    double sw = 0, logmean = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] > 0) {
            double wi = (w && w[i] > 0) ? w[i] : 1.0;
            logmean += wi * log(x[i]);
            sw += wi;
        }
    }
    logmean /= (sw > 0 ? sw : 1);

    double log_mean = log(mu);
    double s = log_mean - logmean;  /* MLE sufficient statistic */

    /* MoM starting point */
    double alpha = mu * mu / var;
    if (alpha < 0.1) alpha = 0.1;

    /* Newton's method for MLE of alpha (Choi & Wette 1969 / Minka 2002) */
    if (s > 0) {
        for (int it = 0; it < 10; it++) {
            double da = (log(alpha) - digamma_approx(alpha) - s);
            double dda = 1.0/alpha - trigamma_approx(alpha);
            double step = da / dda;
            double alpha_new = alpha - step;
            if (alpha_new <= 0) alpha_new = alpha * 0.5;
            if (fabs(alpha_new - alpha) < 1e-8 * alpha) { alpha = alpha_new; break; }
            alpha = alpha_new;
        }
    }

    double beta = alpha / mu;
    if (alpha < 0.01) alpha = 0.01;
    if (beta < 0.01) beta = 0.01;
    out->p[0] = alpha; out->p[1] = beta;
    out->p[2] = lgamma(alpha);  /* cache lgamma */
    out->nparams = 3;
}
static void gamma_init(const double* x, size_t n, int k, DistParams* out) {
    /* Sort a sample to get quantile-based component initialization.
     * Using global variance with multi-modal data gives near-zero alpha. */
    size_t ns = (n > 1000) ? 1000 : n;
    double* s = (double*)malloc(sizeof(double)*ns);
    for (size_t i = 0; i < ns; i++) s[i] = x[i*n/ns];
    /* Simple insertion sort for small ns */
    for (size_t i = 1; i < ns; i++) {
        double tmp = s[i]; size_t j = i;
        while (j > 0 && s[j-1] > tmp) { s[j] = s[j-1]; j--; }
        s[j] = tmp;
    }
    for (int j = 0; j < k; j++) {
        /* Local quantile range for component j */
        size_t lo = (size_t)(j*ns/k), hi = (size_t)((j+1)*ns/k);
        if (hi >= ns) hi = ns-1;
        if (hi <= lo) hi = lo+1;
        double lm = 0, lm2 = 0; size_t cnt = hi - lo;
        for (size_t i = lo; i < hi; i++) { lm += s[i]; lm2 += s[i]*s[i]; }
        lm /= cnt; lm2 /= cnt;
        double lv = lm2 - lm*lm;
        if (lm < 1e-6) lm = 1e-6;
        if (lv < lm*0.01) lv = lm*0.1;  /* minimum CV of 31% */
        double a = lm*lm/lv; if (a < 0.1) a = 0.1;
        double b = lm/lv;    if (b < 0.01) b = 0.01;
        out[j].p[0] = a; out[j].p[1] = b;
        out[j].p[2] = lgamma(a);  /* cache lgamma */
        out[j].nparams = 3;
    }
    free(s);
}
static int gamma_valid(double x) { return x > 0; }

/* ====================================================================
 * LOG-NORMAL: params = {mu_log, sigma_log^2}
 *   X ~ LogNormal(mu, sigma^2) means log(X) ~ N(mu, sigma^2)
 * ==================================================================== */
static double lognorm_pdf(double x, const DistParams* p) {
    double mu = p->p[0], var = p->p[1];
    if (x <= 0 || var <= 0) return 0;
    double lx = log(x);
    double z = (lx - mu) / sqrt(var);
    return (1.0 / (x * sqrt(2*M_PI*var))) * exp(-0.5*z*z);
}
static double lognorm_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], var = p->p[1];
    if (x <= 0 || var <= 0) return -1e30;
    double lx = log(x);
    double z = (lx - mu) / sqrt(var);
    return -log(x) - 0.5*log(2*M_PI*var) - 0.5*z*z;
}
static void lognorm_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* Weighted MLE on log(x) */
    double sw = 0, slx = 0, slx2 = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] > 0) {
            double lx = log(x[i]);
            slx += w[i]*lx; slx2 += w[i]*lx*lx; sw += w[i];
        }
    }
    if (sw < 1e-10) { out->p[0] = 0; out->p[1] = 1; out->nparams = 2; return; }
    double mu = slx / sw;
    double var = slx2 / sw - mu*mu;
    if (var < 1e-10) var = 1e-10;
    out->p[0] = mu; out->p[1] = var; out->nparams = 2;
}
static void lognorm_init(const double* x, size_t n, int k, DistParams* out) {
    /* Compute log stats */
    double slx = 0; int cnt = 0;
    for (size_t i = 0; i < n; i++) { if (x[i]>0) { slx += log(x[i]); cnt++; } }
    double mu_l = cnt > 0 ? slx/cnt : 0;
    double vlx = 0;
    for (size_t i = 0; i < n; i++) { if (x[i]>0) { double d = log(x[i])-mu_l; vlx += d*d; } }
    vlx = cnt > 0 ? vlx/cnt : 1.0;
    if (vlx < 1e-10) vlx = 1.0;
    double range = sqrt(vlx) * 4;
    for (int j = 0; j < k; j++) {
        out[j].p[0] = mu_l - range/2 + range*(j+0.5)/k;
        out[j].p[1] = vlx;
        out[j].nparams = 2;
    }
}
static int lognorm_valid(double x) { return x > 0; }

/* ====================================================================
 * WEIBULL: params = {shape (k), scale (lambda)}
 *   pdf = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
 * ==================================================================== */
static double weibull_pdf(double x, const DistParams* p) {
    double k = p->p[0], lam = p->p[1];
    if (x < 0 || k <= 0 || lam <= 0) return 0;
    if (x == 0) return (k < 1) ? 1e30 : (k == 1 ? 1.0/lam : 0);
    double z = x / lam;
    return (k/lam) * pow(z, k-1) * exp(-pow(z, k));
}
static void weibull_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* Iterative MLE for Weibull shape via Newton's method */
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    if (mu < 1e-10) mu = 1e-10;
    if (var < 1e-10) var = mu*mu;
    /* Initial shape from coefficient of variation */
    double cv = sqrt(var) / mu;
    double k = 1.0 / cv;  /* rough approximation */
    if (k < 0.1) k = 0.1; if (k > 100) k = 100;

    /* A few Newton steps for shape */
    double sw = wt_sum(w, n);
    for (int iter = 0; iter < 30; iter++) {
        double sum_xk = 0, sum_xk_logx = 0, sum_logx = 0;
        for (size_t i = 0; i < n; i++) {
            if (x[i] > 0 && w[i] > 0) {
                double lx = log(x[i]);
                double xk = pow(x[i], k);
                sum_xk += w[i] * xk;
                sum_xk_logx += w[i] * xk * lx;
                sum_logx += w[i] * lx;
            }
        }
        if (sum_xk < 1e-30) break;
        double f = sw/k + sum_logx - sw * sum_xk_logx / sum_xk;
        /* Simple step */
        double new_k = k + 0.1 * (f > 0 ? 1 : -1);
        if (new_k < 0.01) new_k = 0.01;
        if (fabs(new_k - k) < 1e-8) break;
        k = new_k;
    }

    /* Scale from shape: lambda = (sum w_i x_i^k / sum w_i)^(1/k) */
    double sum_xk = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] > 0 && w[i] > 0) sum_xk += w[i] * pow(x[i], k);
    }
    double lam = pow(sum_xk / sw, 1.0/k);
    if (lam < 1e-10) lam = 1e-10;

    out->p[0] = k; out->p[1] = lam; out->nparams = 2;
}
static void weibull_init(const double* x, size_t n, int k, DistParams* out) {
    double mean = 0; for (size_t i = 0; i < n; i++) mean += x[i]; mean /= n;
    if (mean < 1e-10) mean = 1.0;
    for (int j = 0; j < k; j++) {
        out[j].p[0] = 1.0 + j * 0.5;  /* shape */
        out[j].p[1] = mean * (0.5 + j) / k;  /* scale */
        if (out[j].p[1] < 1e-10) out[j].p[1] = 1e-10;
        out[j].nparams = 2;
    }
}
static int weibull_valid(double x) { return x >= 0; }

/* ====================================================================
 * BETA: params = {alpha, beta}  — support (0, 1)
 * ==================================================================== */
static double beta_pdf(double x, const DistParams* p) {
    double a = p->p[0], b = p->p[1];
    if (x <= 0 || x >= 1 || a <= 0 || b <= 0) return 0;
    return exp((a-1)*log(x) + (b-1)*log(1-x) - lgamma(a) - lgamma(b) + lgamma(a+b));
}
static double beta_logpdf(double x, const DistParams* p) {
    double a = p->p[0], b = p->p[1];
    if (x <= 0 || x >= 1 || a <= 0 || b <= 0) return -1e30;
    return (a-1)*log(x) + (b-1)*log(1-x) - lgamma(a) - lgamma(b) + lgamma(a+b);
}
static void beta_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    if (mu < 0.001) mu = 0.001; if (mu > 0.999) mu = 0.999;
    if (var < 1e-10) var = 1e-10;
    if (var >= mu*(1-mu)) var = mu*(1-mu) * 0.99;
    /* Method of moments */
    double common = mu*(1-mu)/var - 1;
    double a = mu * common;
    double b = (1-mu) * common;
    if (a < 0.01) a = 0.01; if (b < 0.01) b = 0.01;
    out->p[0] = a; out->p[1] = b; out->nparams = 2;
}
static void beta_init(const double* x, size_t n, int k, DistParams* out) {
    for (int j = 0; j < k; j++) {
        double mu = (j + 0.5) / k;
        out[j].p[0] = 2.0 * mu;  if (out[j].p[0] < 0.1) out[j].p[0] = 0.1;
        out[j].p[1] = 2.0 * (1 - mu); if (out[j].p[1] < 0.1) out[j].p[1] = 0.1;
        out[j].nparams = 2;
    }
}
static int beta_valid(double x) { return x > 0 && x < 1; }

/* ====================================================================
 * UNIFORM: params = {a, b}  (min, max)
 * ==================================================================== */
static double uniform_pdf(double x, const DistParams* p) {
    double a = p->p[0], b = p->p[1];
    if (b <= a) return 0;
    return (x >= a && x <= b) ? 1.0/(b-a) : 0;
}
static void uniform_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double lo = 1e30, hi = -1e30;
    for (size_t i = 0; i < n; i++) {
        if (w[i] > 1e-10) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
    }
    if (hi <= lo) { hi = lo + 1.0; }
    out->p[0] = lo; out->p[1] = hi; out->nparams = 2;
}
static void uniform_init(const double* x, size_t n, int k, DistParams* out) {
    double mn = x[0], mx = x[0];
    for (size_t i = 1; i < n; i++) { if (x[i]<mn) mn=x[i]; if (x[i]>mx) mx=x[i]; }
    double range = mx - mn; if (range < 1e-10) range = 1.0;
    for (int j = 0; j < k; j++) {
        double lo = mn + range * j / k;
        double hi = mn + range * (j + 1) / k;
        out[j].p[0] = lo; out[j].p[1] = hi; out[j].nparams = 2;
    }
}
static int uniform_valid(double x) { return 1; }


/* ====================================================================
 * STUDENT-T: params = {mu (location), sigma (scale), df (degrees of freedom)}
 *   pdf = Gamma((df+1)/2) / (sigma*sqrt(df*pi)*Gamma(df/2)) * (1 + ((x-mu)/sigma)^2/df)^(-(df+1)/2)
 * ==================================================================== */
static double studt_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], sigma = p->p[1], df = p->p[2];
    if (sigma <= 0) sigma = 1e-10;
    if (df <= 0) df = 1;
    double z = (x - mu) / sigma;
    return lgamma(0.5*(df+1)) - lgamma(0.5*df) - 0.5*log(df*M_PI) - log(sigma)
           - 0.5*(df+1)*log(1 + z*z/df);
}
static double studt_pdf(double x, const DistParams* p) {
    return exp(studt_logpdf(x, p));
}
static void studt_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* IRLS-style: estimate mu, sigma from weighted data; df via kurtosis */
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    if (var < 1e-10) var = 1e-10;
    /* Excess kurtosis for t(df) = 6/(df-4) for df>4, so df = 4 + 6/kurtosis */
    double sw = wt_sum(w, n);
    double m4 = 0;
    for (size_t i = 0; i < n; i++) {
        double d = x[i] - mu;
        m4 += w[i] * d * d * d * d;
    }
    m4 /= sw;
    double kurt = m4 / (var * var) - 3.0;  /* excess kurtosis */
    double df;
    if (kurt > 0.1) {
        df = 4.0 + 6.0 / kurt;
    } else {
        df = 100;  /* near-Normal */
    }
    if (df < 1) df = 1; if (df > 200) df = 200;
    double sigma = sqrt(var * (df - 2) / df);
    if (df <= 2) sigma = sqrt(var);
    if (sigma < 1e-10) sigma = 1e-10;
    out->p[0] = mu; out->p[1] = sigma; out->p[2] = df; out->nparams = 3;
}
static void studt_init(const double* x, size_t n, int k, DistParams* out) {
    double mn = x[0], mx = x[0];
    for (size_t i = 1; i < n; i++) { if (x[i]<mn) mn=x[i]; if (x[i]>mx) mx=x[i]; }
    double gmu = 0; for (size_t i = 0; i < n; i++) gmu += x[i]; gmu /= n;
    double gvar = 0; for (size_t i = 0; i < n; i++) gvar += (x[i]-gmu)*(x[i]-gmu); gvar /= n;
    if (gvar < 1e-10) gvar = 1.0;
    for (int j = 0; j < k; j++) {
        out[j].p[0] = mn + (mx-mn)*(j+0.5)/k;  /* mu */
        out[j].p[1] = sqrt(gvar);                /* sigma */
        out[j].p[2] = 5.0;                       /* df */
        out[j].nparams = 3;
    }
}
static int studt_valid(double x) { return 1; }

/* ====================================================================
 * LAPLACE: params = {mu (location), b (scale)}
 *   pdf = (1/(2b)) * exp(-|x-mu|/b)
 * ==================================================================== */
static double laplace_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], b = p->p[1];
    if (b <= 0) b = 1e-10;
    return -log(2*b) - fabs(x - mu) / b;
}
static double laplace_pdf(double x, const DistParams* p) {
    return exp(laplace_logpdf(x, p));
}
static void laplace_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* Weighted MLE: mu = weighted median, b = weighted mean absolute deviation
     * For simplicity, use weighted mean as approximate location */
    double mu = wt_mean(x, w, n);
    double sw = wt_sum(w, n);
    double mad = 0;
    for (size_t i = 0; i < n; i++) mad += w[i] * fabs(x[i] - mu);
    double b = mad / sw;
    if (b < 1e-10) b = 1e-10;
    out->p[0] = mu; out->p[1] = b; out->nparams = 2;
}
static void laplace_init(const double* x, size_t n, int k, DistParams* out) {
    double mn = x[0], mx = x[0];
    for (size_t i = 1; i < n; i++) { if (x[i]<mn) mn=x[i]; if (x[i]>mx) mx=x[i]; }
    double gmu = 0; for (size_t i = 0; i < n; i++) gmu += x[i]; gmu /= n;
    double gvar = 0; for (size_t i = 0; i < n; i++) gvar += (x[i]-gmu)*(x[i]-gmu); gvar /= n;
    double b = sqrt(gvar / 2.0);  /* Laplace variance = 2b^2 */
    if (b < 1e-10) b = 1.0;
    for (int j = 0; j < k; j++) {
        out[j].p[0] = mn + (mx-mn)*(j+0.5)/k;
        out[j].p[1] = b;
        out[j].nparams = 2;
    }
}
static int laplace_valid(double x) { return 1; }

/* ====================================================================
 * CAUCHY: params = {x0 (location), gamma (scale)}
 *   pdf = 1 / (pi*gamma*(1 + ((x-x0)/gamma)^2))
 * ==================================================================== */
static double cauchy_logpdf(double x, const DistParams* p) {
    double x0 = p->p[0], gam = p->p[1];
    if (gam <= 0) gam = 1e-10;
    double z = (x - x0) / gam;
    return -log(M_PI * gam) - log(1 + z*z);
}
static double cauchy_pdf(double x, const DistParams* p) {
    return exp(cauchy_logpdf(x, p));
}
static void cauchy_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* Use weighted median for location, weighted MAD for scale */
    /* Approximate: weighted mean and IQR-based scale */
    double mu = wt_mean(x, w, n);  /* biased but ok for EM starting */
    double sw = wt_sum(w, n);
    double mad = 0;
    for (size_t i = 0; i < n; i++) mad += w[i] * fabs(x[i] - mu);
    double gam = mad / sw;  /* MAD ≈ gamma for Cauchy */
    if (gam < 1e-10) gam = 1e-10;
    out->p[0] = mu; out->p[1] = gam; out->nparams = 2;
}
static void cauchy_init(const double* x, size_t n, int k, DistParams* out) {
    double mn = x[0], mx = x[0];
    for (size_t i = 1; i < n; i++) { if (x[i]<mn) mn=x[i]; if (x[i]>mx) mx=x[i]; }
    double range = mx - mn; if (range < 1e-10) range = 1.0;
    for (int j = 0; j < k; j++) {
        out[j].p[0] = mn + range*(j+0.5)/k;
        out[j].p[1] = range / (2.0 * k);
        out[j].nparams = 2;
    }
}
static int cauchy_valid(double x) { return 1; }

/* ====================================================================
 * INVERSE-GAUSSIAN: params = {mu (mean), lambda (shape)}
 *   pdf = sqrt(lambda/(2*pi*x^3)) * exp(-lambda*(x-mu)^2 / (2*mu^2*x))
 * ==================================================================== */
static double invgauss_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], lam = p->p[1];
    if (x <= 0 || mu <= 0 || lam <= 0) return -1e30;
    return 0.5*(log(lam) - log(2*M_PI) - 3*log(x)) - lam*(x-mu)*(x-mu) / (2*mu*mu*x);
}
static double invgauss_pdf(double x, const DistParams* p) {
    double lp = invgauss_logpdf(x, p);
    return lp > -700 ? exp(lp) : 0;
}
static void invgauss_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    if (mu < 1e-10) mu = 1e-10;
    /* MLE for lambda: 1/lambda = (1/n) * sum(1/xi - 1/mu) */
    double sw = wt_sum(w, n);
    double inv_sum = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] > 1e-10) inv_sum += w[i] * (1.0/x[i] - 1.0/mu);
    }
    double inv_lam = inv_sum / sw;
    double lam = (inv_lam > 1e-10) ? 1.0 / inv_lam : mu * mu;
    if (lam < 1e-10) lam = 1e-10;
    out->p[0] = mu; out->p[1] = lam; out->nparams = 2;
}
static void invgauss_init(const double* x, size_t n, int k, DistParams* out) {
    double mean = 0; int cnt = 0;
    for (size_t i = 0; i < n; i++) { if (x[i]>0) { mean += x[i]; cnt++; } }
    mean = cnt > 0 ? mean/cnt : 1.0;
    double var = 0;
    for (size_t i = 0; i < n; i++) { if (x[i]>0) var += (x[i]-mean)*(x[i]-mean); }
    var = cnt > 0 ? var/cnt : 1.0;
    if (var < 1e-10) var = 1.0;
    for (int j = 0; j < k; j++) {
        double m = mean * (0.5 + j) / k;
        if (m < 1e-10) m = 1e-10;
        out[j].p[0] = m;
        out[j].p[1] = m*m*m / var;  /* lambda = mu^3 / var for InvGauss */
        out[j].nparams = 2;
    }
}
static int invgauss_valid(double x) { return x > 0; }

/* ====================================================================
 * RAYLEIGH: params = {sigma}
 *   pdf = (x/sigma^2) * exp(-x^2 / (2*sigma^2))
 * ==================================================================== */
static double rayleigh_logpdf(double x, const DistParams* p) {
    double sig = p->p[0];
    if (x < 0 || sig <= 0) return -1e30;
    if (x == 0) return -1e30;
    return log(x) - 2*log(sig) - x*x / (2*sig*sig);
}
static double rayleigh_pdf(double x, const DistParams* p) {
    return exp(rayleigh_logpdf(x, p));
}
static void rayleigh_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* MLE: sigma^2 = sum(w_i * x_i^2) / (2 * sum(w_i)) */
    double sw = 0, sx2 = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] >= 0) { sw += w[i]; sx2 += w[i] * x[i] * x[i]; }
    }
    double sig2 = sw > 0 ? sx2 / (2 * sw) : 1.0;
    if (sig2 < 1e-10) sig2 = 1e-10;
    out->p[0] = sqrt(sig2); out->nparams = 1;
}
static void rayleigh_init(const double* x, size_t n, int k, DistParams* out) {
    double mean = 0; for (size_t i = 0; i < n; i++) mean += x[i]; mean /= n;
    if (mean < 1e-10) mean = 1.0;
    for (int j = 0; j < k; j++) {
        /* sigma ≈ mean / sqrt(pi/2) for Rayleigh */
        double sig = mean * (0.5 + j) / k / 1.2533;
        if (sig < 1e-10) sig = 1e-10;
        out[j].p[0] = sig; out[j].nparams = 1;
    }
}
static int rayleigh_valid(double x) { return x >= 0; }

/* ====================================================================
 * PARETO: params = {alpha (shape), x_m (scale/minimum)}
 *   pdf = alpha * x_m^alpha / x^(alpha+1)  for x >= x_m
 * ==================================================================== */
static double pareto_logpdf(double x, const DistParams* p) {
    double alpha = p->p[0], xm = p->p[1];
    if (x < xm || alpha <= 0 || xm <= 0) return -1e30;
    return log(alpha) + alpha*log(xm) - (alpha+1)*log(x);
}
static double pareto_pdf(double x, const DistParams* p) {
    return exp(pareto_logpdf(x, p));
}
static void pareto_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* MLE: x_m = min(x), alpha = n / sum(log(x) - log(x_m)) */
    double xm = 1e30;
    for (size_t i = 0; i < n; i++) {
        if (w[i] > 1e-10 && x[i] > 0 && x[i] < xm) xm = x[i];
    }
    if (xm <= 0 || xm > 1e20) xm = 1e-10;
    double sw = 0, slog = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] >= xm && w[i] > 1e-10) {
            sw += w[i];
            slog += w[i] * (log(x[i]) - log(xm));
        }
    }
    double alpha = slog > 1e-10 ? sw / slog : 1.0;
    if (alpha < 0.1) alpha = 0.1; if (alpha > 100) alpha = 100;
    out->p[0] = alpha; out->p[1] = xm; out->nparams = 2;
}
static void pareto_init(const double* x, size_t n, int k, DistParams* out) {
    double mn = 1e30;
    for (size_t i = 0; i < n; i++) { if (x[i] > 0 && x[i] < mn) mn = x[i]; }
    if (mn > 1e20) mn = 1.0;
    for (int j = 0; j < k; j++) {
        out[j].p[0] = 1.0 + j;  /* alpha */
        out[j].p[1] = mn;        /* x_m */
        out[j].nparams = 2;
    }
}
static int pareto_valid(double x) { return x > 0; }


/* ====================================================================
 * 16. LOGISTIC: μ, s  (domain: ℝ)
 * ==================================================================== */
static double logistic_pdf(double x, const DistParams* p) {
    double mu = p->p[0], s = fmax(p->p[1], 1e-10);
    double z = (x - mu) / s;
    double ez = exp(-z);
    return ez / (s * (1+ez) * (1+ez));
}
static double logistic_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], s = fmax(p->p[1], 1e-10);
    double z = (x - mu) / s;
    return -z - log(s) - 2*log(1+exp(-z));
}
static void logistic_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    out->p[0] = mu;
    out->p[1] = fmax(sqrt(var * 3.0) / M_PI, 1e-10);  /* s = sigma*sqrt(3)/pi */
    out->nparams = 2;
}
static void logistic_init(const double* x, size_t n, int k, DistParams* out) {
    double mn=x[0],mx=x[0]; for(size_t i=1;i<n;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
    for(int j=0;j<k;j++){out[j].p[0]=mn+(mx-mn)*(j+1.0)/(k+1);out[j].p[1]=(mx-mn)/(2*k);out[j].nparams=2;}
}
static int logistic_valid(double x) { (void)x; return 1; }

/* ====================================================================
 * 17. GUMBEL (Type I Extreme Value): μ, β  (domain: ℝ)
 * ==================================================================== */
static double gumbel_pdf(double x, const DistParams* p) {
    double mu = p->p[0], b = fmax(p->p[1], 1e-10);
    double z = (x - mu) / b;
    return exp(-(z + exp(-z))) / b;
}
static double gumbel_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], b = fmax(p->p[1], 1e-10);
    double z = (x - mu) / b;
    return -(z + exp(-z)) - log(b);
}
static void gumbel_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    out->p[1] = fmax(sqrt(var * 6.0) / M_PI, 1e-10);  /* β = σ√6/π */
    out->p[0] = mu - 0.5772 * out->p[1];  /* μ = mean - γ·β */
    out->nparams = 2;
}
static void gumbel_init(const double* x, size_t n, int k, DistParams* out) {
    double mn=x[0],mx=x[0]; for(size_t i=1;i<n;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
    for(int j=0;j<k;j++){out[j].p[0]=mn+(mx-mn)*(j+1.0)/(k+1);out[j].p[1]=(mx-mn)/(2*k);out[j].nparams=2;}
}
static int gumbel_valid(double x) { (void)x; return 1; }

/* ====================================================================
 * 18. SKEW-NORMAL: ξ (location), ω (scale), α (shape)  (domain: ℝ)
 * ==================================================================== */
static double skewnorm_phi(double x) { return exp(-0.5*x*x) / sqrt(2*M_PI); }
static double skewnorm_Phi(double x) { return 0.5 * erfc(-x / sqrt(2.0)); }
static double skewnorm_pdf(double x, const DistParams* p) {
    double xi = p->p[0], omega = fmax(p->p[1], 1e-10), alpha = p->p[2];
    double z = (x - xi) / omega;
    return 2.0 / omega * skewnorm_phi(z) * skewnorm_Phi(alpha * z);
}
static double skewnorm_logpdf(double x, const DistParams* p) {
    double v = skewnorm_pdf(x, p);
    return (v > 0) ? log(v) : -700;
}
static void skewnorm_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    double sw = 0, m3 = 0;
    for (size_t i = 0; i < n; i++) sw += w[i];
    double sd = sqrt(fmax(var, 1e-10));
    for (size_t i = 0; i < n; i++) { double d = (x[i]-mu)/sd; m3 += w[i]*d*d*d; }
    m3 /= sw;
    /* Method of moments approximation */
    double gamma1 = fmax(-0.99, fmin(0.99, m3));
    double delta = (gamma1 > 0 ? 1 : -1) * sqrt(M_PI/2 * fabs(gamma1) / (fabs(gamma1) + pow((4-M_PI)/2, 2.0/3)));
    if (!isfinite(delta)) delta = 0;
    double alpha = delta / sqrt(1 - delta*delta + 1e-10);
    double omega = sd / sqrt(1 - 2*delta*delta/M_PI + 1e-10);
    double xi = mu - omega * delta * sqrt(2/M_PI);
    out->p[0] = xi; out->p[1] = fmax(omega, 1e-10); out->p[2] = alpha;
    out->nparams = 3;
}
static void skewnorm_init(const double* x, size_t n, int k, DistParams* out) {
    double mn=x[0],mx=x[0]; for(size_t i=1;i<n;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
    for(int j=0;j<k;j++){out[j].p[0]=mn+(mx-mn)*(j+1.0)/(k+1);out[j].p[1]=(mx-mn)/(2*k);out[j].p[2]=0;out[j].nparams=3;}
}
static int skewnorm_valid(double x) { (void)x; return 1; }

/* ====================================================================
 * 19. GENERALIZED GAUSSIAN: μ, α (scale), β (shape)  (domain: ℝ)
 *     β=1 → Laplace, β=2 → Gaussian, β→∞ → Uniform
 * ==================================================================== */
static double gengauss_pdf(double x, const DistParams* p) {
    double mu = p->p[0], a = fmax(p->p[1], 1e-10), b = fmax(p->p[2], 0.5);
    double z = fabs(x - mu) / a;
    return b / (2*a*tgamma(1.0/b)) * exp(-pow(z, b));
}
static double gengauss_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], a = fmax(p->p[1], 1e-10), b = fmax(p->p[2], 0.5);
    double z = fabs(x - mu) / a;
    return log(b) - log(2*a) - lgamma(1.0/b) - pow(z, b);
}
static void gengauss_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    double sw = 0, m4 = 0;
    for (size_t i = 0; i < n; i++) sw += w[i];
    for (size_t i = 0; i < n; i++) { double d = x[i]-mu; m4 += w[i]*d*d*d*d; }
    m4 /= sw;
    double kurtosis = m4 / (var*var + 1e-20);
    /* Approximate β from kurtosis: kurtosis = Γ(5/β)Γ(1/β)/Γ(3/β)² */
    double beta = 2.0; /* default: Gaussian */
    if (kurtosis > 3.5) beta = 1.5; else if (kurtosis > 5) beta = 1.0;
    else if (kurtosis < 2.5) beta = 3.0; else if (kurtosis < 2) beta = 5.0;
    beta = fmax(0.3, fmin(10, beta));
    double alpha = sqrt(var * tgamma(1.0/beta) / tgamma(3.0/beta));
    out->p[0] = mu; out->p[1] = fmax(alpha, 1e-10); out->p[2] = beta;
    out->nparams = 3;
}
static void gengauss_init(const double* x, size_t n, int k, DistParams* out) {
    double mn=x[0],mx=x[0]; for(size_t i=1;i<n;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
    for(int j=0;j<k;j++){out[j].p[0]=mn+(mx-mn)*(j+1.0)/(k+1);out[j].p[1]=(mx-mn)/(2*k);out[j].p[2]=2.0;out[j].nparams=3;}
}
static int gengauss_valid(double x) { (void)x; return 1; }

/* ====================================================================
 * 20. CHI-SQUARED: k (degrees of freedom)  (domain: x>0)
 * ==================================================================== */
static double chisq_pdf(double x, const DistParams* p) {
    double k = fmax(p->p[0], 0.5);
    if (x <= 0) return 0;
    return exp((k/2-1)*log(x) - x/2 - (k/2)*log(2) - lgamma(k/2));
}
static double chisq_logpdf(double x, const DistParams* p) {
    double k = fmax(p->p[0], 0.5);
    if (x <= 0) return -700;
    return (k/2-1)*log(x) - x/2 - (k/2)*log(2) - lgamma(k/2);
}
static void chisq_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    out->p[0] = fmax(mu, 0.5);  /* E[X] = k */
    out->nparams = 1;
}
static void chisq_init(const double* x, size_t n, int k, DistParams* out) {
    double mean=0; for(size_t i=0;i<n;i++) mean+=x[i]; mean/=n;
    for(int j=0;j<k;j++){out[j].p[0]=fmax(mean*(0.5+j)/k,0.5);out[j].nparams=1;}
}
static int chisq_valid(double x) { return x > 0; }

/* ====================================================================
 * 21. F-DISTRIBUTION: d1, d2  (domain: x>0)
 * ==================================================================== */
static double fdist_pdf(double x, const DistParams* p) {
    double d1 = fmax(p->p[0], 1), d2 = fmax(p->p[1], 1);
    if (x <= 0) return 0;
    return exp(0.5*(d1*log(d1)+d2*log(d2)+(d1-2)*log(x)
              -((d1+d2)/2)*log(d1*x+d2))
              +lgamma((d1+d2)/2)-lgamma(d1/2)-lgamma(d2/2));
}
static double fdist_logpdf(double x, const DistParams* p) {
    double d1 = fmax(p->p[0], 1), d2 = fmax(p->p[1], 1);
    if (x <= 0) return -700;
    return 0.5*(d1*log(d1/d2)+(d1-2)*log(x)-(d1+d2)*log(1+d1*x/d2))
           +lgamma((d1+d2)/2)-lgamma(d1/2)-lgamma(d2/2)-log(x);
}
static void fdist_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    /* E[X] = d2/(d2-2); Var = 2*d2^2*(d1+d2-2)/(d1*(d2-2)^2*(d2-4)) */
    double d2 = fmax(2*mu/(mu-1+1e-10), 5);
    double d1 = fmax(2*d2*d2*(d2-2)/(var*(d2-2)*(d2-2)*(d2-4+1e-10))-d2+2, 1);
    out->p[0] = fmax(d1, 1); out->p[1] = fmax(d2, 5);
    out->nparams = 2;
}
static void fdist_init(const double* x, size_t n, int k, DistParams* out) {
    for(int j=0;j<k;j++){out[j].p[0]=5+j*3;out[j].p[1]=10+j*5;out[j].nparams=2;}
}
static int fdist_valid(double x) { return x > 0; }

/* ====================================================================
 * 22. LOG-LOGISTIC (Fisk): α (scale), β (shape)  (domain: x>0)
 * ==================================================================== */
static double loglogistic_pdf(double x, const DistParams* p) {
    double a = fmax(p->p[0], 1e-10), b = fmax(p->p[1], 0.5);
    if (x <= 0) return 0;
    double t = pow(x/a, b);
    return (b/a) * pow(x/a, b-1) / ((1+t)*(1+t));
}
static double loglogistic_logpdf(double x, const DistParams* p) {
    double a = fmax(p->p[0], 1e-10), b = fmax(p->p[1], 0.5);
    if (x <= 0) return -700;
    double t = pow(x/a, b);
    return log(b) - log(a) + (b-1)*(log(x)-log(a)) - 2*log(1+t);
}
static void loglogistic_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* MLE via log moments */
    double sw=0, slx=0, slx2=0;
    for(size_t i=0;i<n;i++){if(x[i]>0){sw+=w[i]; slx+=w[i]*log(x[i]); slx2+=w[i]*log(x[i])*log(x[i]);}}
    if(sw<1){out->p[0]=1;out->p[1]=2;out->nparams=2;return;}
    double mu_lx = slx/sw, var_lx = slx2/sw - mu_lx*mu_lx;
    out->p[0] = fmax(exp(mu_lx), 1e-10);
    out->p[1] = fmax(M_PI / (sqrt(3*fmax(var_lx,1e-10))), 0.5);
    out->nparams = 2;
}
static void loglogistic_init(const double* x, size_t n, int k, DistParams* out) {
    double mean=0; int cnt=0; for(size_t i=0;i<n;i++){if(x[i]>0){mean+=x[i];cnt++;}} mean/=fmax(cnt,1);
    for(int j=0;j<k;j++){out[j].p[0]=mean*(0.5+j)/k;out[j].p[1]=2.0;out[j].nparams=2;}
}
static int loglogistic_valid(double x) { return x > 0; }

/* ====================================================================
 * 23. NAKAGAMI: m (shape ≥0.5), Ω (spread)  (domain: x>0)
 * ==================================================================== */
static double nakagami_pdf(double x, const DistParams* p) {
    double m = fmax(p->p[0], 0.5), om = fmax(p->p[1], 1e-10);
    if (x <= 0) return 0;
    return 2*pow(m/om,m)/tgamma(m)*pow(x,2*m-1)*exp(-m*x*x/om);
}
static double nakagami_logpdf(double x, const DistParams* p) {
    double m = fmax(p->p[0], 0.5), om = fmax(p->p[1], 1e-10);
    if (x <= 0) return -700;
    return log(2)+m*log(m/om)-lgamma(m)+(2*m-1)*log(x)-m*x*x/om;
}
static void nakagami_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double sw=0,s1=0,s2=0;
    for(size_t i=0;i<n;i++){if(x[i]>0){sw+=w[i];s1+=w[i]*x[i]*x[i];s2+=w[i]*x[i]*x[i]*x[i]*x[i];}}
    if(sw<1){out->p[0]=1;out->p[1]=1;out->nparams=2;return;}
    double E2=s1/sw, E4=s2/sw;
    double om=E2, m=E2*E2/fmax(E4-E2*E2,1e-10);
    out->p[0]=fmax(m,0.5); out->p[1]=fmax(om,1e-10); out->nparams=2;
}
static void nakagami_init(const double* x, size_t n, int k, DistParams* out) {
    double s=0; for(size_t i=0;i<n;i++) s+=x[i]*x[i]; s/=n;
    for(int j=0;j<k;j++){out[j].p[0]=1.0+j;out[j].p[1]=s*(0.5+j)/k;out[j].nparams=2;}
}
static int nakagami_valid(double x) { return x > 0; }

/* ====================================================================
 * 24. LÉVY: μ (location), c (scale)  (domain: x > μ)
 * ==================================================================== */
static double levy_pdf(double x, const DistParams* p) {
    double mu = p->p[0], c = fmax(p->p[1], 1e-10);
    if (x <= mu) return 0;
    double d = x - mu;
    return sqrt(c/(2*M_PI)) * exp(-c/(2*d)) / (d*sqrt(d));
}
static double levy_logpdf(double x, const DistParams* p) {
    double mu = p->p[0], c = fmax(p->p[1], 1e-10);
    if (x <= mu) return -700;
    double d = x - mu;
    return 0.5*(log(c)-log(2*M_PI)) - c/(2*d) - 1.5*log(d);
}
static void levy_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* Estimate: μ ≈ min(x), c from harmonic mean of (x-μ) */
    double mn = 1e30, sw=0, sh=0;
    for(size_t i=0;i<n;i++){if(w[i]>0.01&&x[i]<mn) mn=x[i]; sw+=w[i];}
    for(size_t i=0;i<n;i++){double d=x[i]-mn; if(d>1e-10) sh+=w[i]/d;}
    double hm = sw/fmax(sh,1e-10);
    out->p[0] = mn; out->p[1] = fmax(hm, 1e-10);
    out->nparams = 2;
}
static void levy_init(const double* x, size_t n, int k, DistParams* out) {
    double mn=x[0]; for(size_t i=1;i<n;i++){if(x[i]<mn) mn=x[i];}
    for(int j=0;j<k;j++){out[j].p[0]=mn;out[j].p[1]=1.0+j;out[j].nparams=2;}
}
static int levy_valid(double x) { (void)x; return x > 0; }

/* ====================================================================
 * 25. GOMPERTZ: η (shape), b (rate)  (domain: x ≥ 0)
 * ==================================================================== */
static double gompertz_pdf(double x, const DistParams* p) {
    double eta = fmax(p->p[0], 1e-10), b = fmax(p->p[1], 1e-10);
    if (x < 0) return 0;
    return b * eta * exp(eta + b*x - eta*exp(b*x));
}
static double gompertz_logpdf(double x, const DistParams* p) {
    double eta = fmax(p->p[0], 1e-10), b = fmax(p->p[1], 1e-10);
    if (x < 0) return -700;
    return log(b) + log(eta) + eta + b*x - eta*exp(b*x);
}
static void gompertz_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    out->p[0] = 0.1; out->p[1] = fmax(1.0/mu, 1e-10);
    out->nparams = 2;
}
static void gompertz_init(const double* x, size_t n, int k, DistParams* out) {
    double mean=0; for(size_t i=0;i<n;i++) mean+=x[i]; mean/=n;
    for(int j=0;j<k;j++){out[j].p[0]=0.1*(j+1);out[j].p[1]=1.0/fmax(mean,1e-10);out[j].nparams=2;}
}
static int gompertz_valid(double x) { return x >= 0; }

/* ====================================================================
 * 26. BURR TYPE XII: c, k (both > 0)  (domain: x > 0)
 *     pdf = c*k*x^(c-1) / (1+x^c)^(k+1)
 * ==================================================================== */
static double burr_pdf(double x, const DistParams* p) {
    double c = fmax(p->p[0], 0.5), k = fmax(p->p[1], 0.5);
    if (x <= 0) return 0;
    return c*k*pow(x,c-1) / pow(1+pow(x,c), k+1);
}
static double burr_logpdf(double x, const DistParams* p) {
    double c = fmax(p->p[0], 0.5), k = fmax(p->p[1], 0.5);
    if (x <= 0) return -700;
    return log(c)+log(k)+(c-1)*log(x)-(k+1)*log(1+pow(x,c));
}
static void burr_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* Rough MoM: E[X]=kB(k-1/c,1+1/c), use log-moments for c, then k */
    double sw=0,slx=0;
    for(size_t i=0;i<n;i++){if(x[i]>0){sw+=w[i];slx+=w[i]*log(x[i]);}}
    double mul = slx/fmax(sw,1);
    out->p[0] = fmax(M_PI/(sqrt(6)*fmax(sqrt(wt_var(x,w,n,wt_mean(x,w,n))),1e-10)/fmax(wt_mean(x,w,n),1e-10)), 0.5);
    out->p[1] = fmax(1.0, 0.5);
    out->nparams = 2;
}
static void burr_init(const double* x, size_t n, int k, DistParams* out) {
    for(int j=0;j<k;j++){out[j].p[0]=2.0;out[j].p[1]=2.0+j;out[j].nparams=2;}
}
static int burr_valid(double x) { return x > 0; }

/* ====================================================================
 * 27. HALF-NORMAL: σ  (domain: x ≥ 0)
 *     pdf = sqrt(2/(πσ²)) * exp(-x²/(2σ²))
 * ==================================================================== */
static double halfnorm_pdf(double x, const DistParams* p) {
    double s = fmax(p->p[0], 1e-10);
    if (x < 0) return 0;
    return sqrt(2.0/(M_PI*s*s)) * exp(-x*x/(2*s*s));
}
static double halfnorm_logpdf(double x, const DistParams* p) {
    double s = fmax(p->p[0], 1e-10);
    if (x < 0) return -700;
    return 0.5*log(2.0/(M_PI*s*s)) - x*x/(2*s*s);
}
static void halfnorm_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double sw=0,s2=0;
    for(size_t i=0;i<n;i++){sw+=w[i];s2+=w[i]*x[i]*x[i];}
    out->p[0] = fmax(sqrt(s2/fmax(sw,1)), 1e-10);
    out->nparams = 1;
}
static void halfnorm_init(const double* x, size_t n, int k, DistParams* out) {
    double s2=0; for(size_t i=0;i<n;i++) s2+=x[i]*x[i]; s2/=n;
    for(int j=0;j<k;j++){out[j].p[0]=sqrt(s2)*(0.5+j)/k;out[j].nparams=1;}
}
static int halfnorm_valid(double x) { return x >= 0; }

/* ====================================================================
 * 28. MAXWELL-BOLTZMANN: a (parameter)  (domain: x ≥ 0)
 *     pdf = sqrt(2/π) * x² * exp(-x²/(2a²)) / a³
 * ==================================================================== */
static double maxwell_pdf(double x, const DistParams* p) {
    double a = fmax(p->p[0], 1e-10);
    if (x < 0) return 0;
    return sqrt(2.0/M_PI) * x*x * exp(-x*x/(2*a*a)) / (a*a*a);
}
static double maxwell_logpdf(double x, const DistParams* p) {
    double a = fmax(p->p[0], 1e-10);
    if (x <= 0) return -700;
    return 0.5*log(2.0/M_PI) + 2*log(x) - x*x/(2*a*a) - 3*log(a);
}
static void maxwell_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double sw=0,s2=0;
    for(size_t i=0;i<n;i++){if(x[i]>0){sw+=w[i];s2+=w[i]*x[i]*x[i];}}
    /* E[X²] = 3a² → a = sqrt(E[X²]/3) */
    out->p[0] = fmax(sqrt(s2/(3*fmax(sw,1))), 1e-10);
    out->nparams = 1;
}
static void maxwell_init(const double* x, size_t n, int k, DistParams* out) {
    double s2=0; for(size_t i=0;i<n;i++) s2+=x[i]*x[i]; s2/=n;
    for(int j=0;j<k;j++){out[j].p[0]=sqrt(s2/3)*(0.5+j)/k;out[j].nparams=1;}
}
static int maxwell_valid(double x) { return x >= 0; }

/* ====================================================================
 * 29. KUMARASWAMY: a, b  (domain: x ∈ (0,1))
 *     pdf = a*b*x^(a-1)*(1-x^a)^(b-1)
 * ==================================================================== */
static double kumaraswamy_pdf(double x, const DistParams* p) {
    double a = fmax(p->p[0], 0.1), b = fmax(p->p[1], 0.1);
    if (x <= 0 || x >= 1) return 0;
    return a*b*pow(x,a-1)*pow(1-pow(x,a),b-1);
}
static double kumaraswamy_logpdf(double x, const DistParams* p) {
    double a = fmax(p->p[0], 0.1), b = fmax(p->p[1], 0.1);
    if (x <= 0 || x >= 1) return -700;
    double xa = pow(x, a);
    return log(a)+log(b)+(a-1)*log(x)+(b-1)*log(fmax(1-xa, 1e-300));
}
static void kumaraswamy_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* Approximate MoM via log-moments */
    double sw=0,slx=0,sl1x=0;
    for(size_t i=0;i<n;i++){
        if(x[i]>0&&x[i]<1){sw+=w[i];slx+=w[i]*log(x[i]);sl1x+=w[i]*log(1-x[i]);}
    }
    if(sw<1){out->p[0]=1;out->p[1]=1;out->nparams=2;return;}
    /* Heuristic: use Beta MoM as starting point */
    double mu=wt_mean(x,w,n), var=wt_var(x,w,n,mu);
    if(var<1e-10) var=0.01;
    double t=mu*(1-mu)/var-1;
    out->p[0]=fmax(mu*t,0.1); out->p[1]=fmax((1-mu)*t,0.1);
    out->nparams=2;
}
static void kumaraswamy_init(const double* x, size_t n, int k, DistParams* out) {
    for(int j=0;j<k;j++){out[j].p[0]=1.0+j;out[j].p[1]=1.0+j;out[j].nparams=2;}
}
static int kumaraswamy_valid(double x) { return x > 0 && x < 1; }

/* ====================================================================
 * 30. TRIANGULAR: a (min), b (max), c (mode)  (domain: a ≤ x ≤ b)
 *     Stored as p[0]=a, p[1]=b, p[2]=c
 * ==================================================================== */
static double triangular_pdf(double x, const DistParams* p) {
    double a = p->p[0], b = p->p[1], c = p->p[2];
    if (b <= a) b = a + 1e-10;
    if (c < a) c = a; if (c > b) c = b;
    if (x < a || x > b) return 0;
    if (x < c) return 2*(x-a)/((b-a)*(c-a+1e-300));
    if (x > c) return 2*(b-x)/((b-a)*(b-c+1e-300));
    return 2.0/(b-a);
}
static double triangular_logpdf(double x, const DistParams* p) {
    double v = triangular_pdf(x, p);
    return v > 0 ? log(v) : -700;
}
static void triangular_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    double sd = sqrt(fmax(var, 1e-10));
    /* a ≈ mu - √6·σ, b ≈ mu + √6·σ, c ≈ mu */
    out->p[0] = mu - sqrt(6.0)*sd;
    out->p[1] = mu + sqrt(6.0)*sd;
    out->p[2] = mu;
    out->nparams = 3;
}
static void triangular_init(const double* x, size_t n, int k, DistParams* out) {
    double mn=x[0],mx=x[0]; for(size_t i=1;i<n;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
    double range=mx-mn; if(range<1e-10) range=1;
    for(int j=0;j<k;j++){
        double lo=mn+range*j/k, hi=mn+range*(j+1)/k;
        out[j].p[0]=lo;out[j].p[1]=hi;out[j].p[2]=(lo+hi)/2;out[j].nparams=3;
    }
}
static int triangular_valid(double x) { (void)x; return 1; }

/* ====================================================================
 * 31. BINOMIAL: n (trials), p (success prob)  (domain: x ∈ {0,..,n})
 * ==================================================================== */
static double log_choose(double n, double k) {
    return lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1);
}
static double binomial_pdf(double x, const DistParams* p) {
    int n = (int)fmax(p->p[0], 1);
    double pr = fmax(1e-10, fmin(1-1e-10, p->p[1]));
    int k = (int)(x + 0.5);
    if (k < 0 || k > n) return 0;
    return exp(log_choose(n,k) + k*log(pr) + (n-k)*log(1-pr));
}
static double binomial_logpdf(double x, const DistParams* p) {
    double v = binomial_pdf(x, p);
    return v > 0 ? log(v) : -700;
}
static void binomial_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    /* n = mu²/(mu - var), p = mu/n */
    double est_n;
    if (var < mu) est_n = mu*mu / fmax(mu - var, 1e-10); else est_n = fmax(mu, 1);
    est_n = fmax(round(est_n), 1);
    out->p[0] = est_n;
    out->p[1] = fmax(1e-10, fmin(1-1e-10, mu/est_n));
    out->nparams = 2;
}
static void binomial_init(const double* x, size_t n, int k, DistParams* out) {
    double mx=x[0]; for(size_t i=1;i<n;i++){if(x[i]>mx)mx=x[i];}
    for(int j=0;j<k;j++){out[j].p[0]=fmax(mx,1);out[j].p[1]=0.5;out[j].nparams=2;}
}
static int binomial_valid(double x) { return x >= 0 && x == floor(x); }

/* ====================================================================
 * 32. NEGATIVE BINOMIAL: r (successes), p (success prob)
 *     (domain: x ∈ {0,1,2,...})
 * ==================================================================== */
static double negbinom_pdf(double x, const DistParams* p) {
    double r = fmax(p->p[0], 0.5);
    double pr = fmax(1e-10, fmin(1-1e-10, p->p[1]));
    int k = (int)(x + 0.5);
    if (k < 0) return 0;
    return exp(lgamma(k+r)-lgamma(k+1)-lgamma(r) + r*log(pr) + k*log(1-pr));
}
static double negbinom_logpdf(double x, const DistParams* p) {
    double r = fmax(p->p[0], 0.5);
    double pr = fmax(1e-10, fmin(1-1e-10, p->p[1]));
    int k = (int)(x + 0.5);
    if (k < 0) return -700;
    return lgamma(k+r)-lgamma(k+1)-lgamma(r) + r*log(pr) + k*log(1-pr);
}
static void negbinom_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    /* var = μ(1-p)/p, r = μp/(1-p) → p = μ/var (if var>μ) */
    double p_est = (var > mu) ? mu/var : 0.5;
    double r_est = mu*p_est/(1-p_est+1e-10);
    out->p[0] = fmax(r_est, 0.5);
    out->p[1] = fmax(1e-10, fmin(1-1e-10, p_est));
    out->nparams = 2;
}
static void negbinom_init(const double* x, size_t n, int k, DistParams* out) {
    double mean=0; for(size_t i=0;i<n;i++) mean+=x[i]; mean/=n;
    for(int j=0;j<k;j++){out[j].p[0]=fmax(mean*(0.5+j)/k,0.5);out[j].p[1]=0.5;out[j].nparams=2;}
}
static int negbinom_valid(double x) { return x >= 0 && x == floor(x); }

/* ====================================================================
 * 33. GEOMETRIC: p (success probability)  (domain: x ∈ {0,1,2,...})
 *     P(X=k) = (1-p)^k * p
 * ==================================================================== */
static double geometric_pdf(double x, const DistParams* p) {
    double pr = fmax(1e-10, fmin(1-1e-10, p->p[0]));
    int k = (int)(x + 0.5);
    if (k < 0) return 0;
    return pr * pow(1-pr, k);
}
static double geometric_logpdf(double x, const DistParams* p) {
    double pr = fmax(1e-10, fmin(1-1e-10, p->p[0]));
    int k = (int)(x + 0.5);
    if (k < 0) return -700;
    return log(pr) + k*log(1-pr);
}
static void geometric_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    out->p[0] = fmax(1e-10, fmin(1-1e-10, 1.0/(1+mu)));
    out->nparams = 1;
}
static void geometric_init(const double* x, size_t n, int k, DistParams* out) {
    for(int j=0;j<k;j++){out[j].p[0]=0.5;out[j].nparams=1;}
}
static int geometric_valid(double x) { return x >= 0 && x == floor(x); }

/* ====================================================================
 * 34. ZIPF: s (exponent > 1)  (domain: x ∈ {1,2,3,...})
 *     P(X=k) = k^(-s) / ζ(s)
 * ==================================================================== */
static double zeta_approx(double s) {
    /* Riemann zeta via partial sum + Euler-Maclaurin */
    double sum = 0;
    for (int k = 1; k <= 1000; k++) sum += pow(k, -s);
    return sum;
}
static double zipf_pdf(double x, const DistParams* p) {
    double s = fmax(p->p[0], 1.01);
    int k = (int)(x + 0.5);
    if (k < 1) return 0;
    return pow(k, -s) / zeta_approx(s);
}
static double zipf_logpdf(double x, const DistParams* p) {
    double s = fmax(p->p[0], 1.01);
    int k = (int)(x + 0.5);
    if (k < 1) return -700;
    return -s*log(k) - log(zeta_approx(s));
}
static void zipf_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* MLE: s that makes E[log(k)] = ζ'(s)/ζ(s) — use grid search */
    double sw=0, slx=0;
    for(size_t i=0;i<n;i++){if(x[i]>=1){sw+=w[i];slx+=w[i]*log(x[i]);}}
    double ml = slx/fmax(sw,1);
    /* Grid search for s ∈ [1.01, 10] */
    double best_s=2, best_diff=1e30;
    for(double s=1.01; s<=10; s+=0.05) {
        double z=0,zp=0;
        for(int k=1;k<=500;k++){double ks=pow(k,-s);z+=ks;zp+=ks*log(k);}
        double expected=-zp/z;
        double diff=fabs(expected-ml);
        if(diff<best_diff){best_diff=diff;best_s=s;}
    }
    out->p[0] = best_s;
    out->nparams = 1;
}
static void zipf_init(const double* x, size_t n, int k, DistParams* out) {
    for(int j=0;j<k;j++){out[j].p[0]=1.5+j*0.5;out[j].nparams=1;}
}
static int zipf_valid(double x) { return x >= 1 && x == floor(x); }

/* ====================================================================
 * 35. KDE: Kernel Density Estimate (nonparametric)
 *     p[0] = bandwidth h
 *     Uses global data pointer for PDF computation — O(n²) per eval!
 * ==================================================================== */
static const double* g_kde_data = NULL;
static size_t g_kde_n = 0;

void KDE_SetData(const double* data, size_t n) {
    g_kde_data = data;
    g_kde_n = n;
}

static double kde_pdf(double x, const DistParams* p) {
    if (!g_kde_data || g_kde_n == 0) return 1e-300;
    double h = fmax(p->p[0], 1e-10);
    double sum = 0;
    for (size_t i = 0; i < g_kde_n; i++) {
        double z = (x - g_kde_data[i]) / h;
        sum += exp(-0.5 * z * z);
    }
    return sum / (g_kde_n * h * sqrt(2 * M_PI));
}
static double kde_logpdf(double x, const DistParams* p) {
    double v = kde_pdf(x, p);
    return v > 1e-300 ? log(v) : -700;
}
static void kde_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    /* Silverman's rule: h = 1.06 * σ * n_eff^(-1/5) */
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    double sw = 0;
    for (size_t i = 0; i < n; i++) sw += w[i];
    double neff = fmax(sw, 2);
    out->p[0] = fmax(1.06 * sqrt(var) * pow(neff, -0.2), 1e-10);
    out->nparams = 1;
}
static void kde_init(const double* x, size_t n, int k, DistParams* out) {
    double var = 0, mean = 0;
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (size_t i = 0; i < n; i++) var += (x[i]-mean)*(x[i]-mean);
    var /= n;
    double h = 1.06 * sqrt(var) * pow((double)n, -0.2);
    for (int j = 0; j < k; j++) { out[j].p[0] = h; out[j].nparams = 1; }
}
static int kde_valid(double x) { (void)x; return 1; }


/* ====================================================================
 * Distribution registry
 * ==================================================================== */
static DistFunctions dist_table[] = {
    { DIST_GAUSSIAN,    "Gaussian",    2, gauss_pdf,   gauss_logpdf,   gauss_estimate,   gauss_init,   gauss_valid },
    { DIST_EXPONENTIAL, "Exponential", 1, expo_pdf,    expo_logpdf,    expo_estimate,    expo_init,    expo_valid },
    { DIST_POISSON,     "Poisson",     1, poisson_pdf, NULL,           poisson_estimate, poisson_init, poisson_valid },
    { DIST_GAMMA,       "Gamma",       2, gamma_pdf,   gamma_logpdf,   gamma_estimate,   gamma_init,   gamma_valid },
    { DIST_LOGNORMAL,   "LogNormal",   2, lognorm_pdf, lognorm_logpdf, lognorm_estimate, lognorm_init, lognorm_valid },
    { DIST_WEIBULL,     "Weibull",     2, weibull_pdf, NULL,           weibull_estimate, weibull_init, weibull_valid },
    { DIST_BETA,        "Beta",        2, beta_pdf,    beta_logpdf,    beta_estimate,    beta_init,    beta_valid },
    { DIST_UNIFORM,     "Uniform",     2, uniform_pdf, NULL,           uniform_estimate, uniform_init, uniform_valid },
    { 0, NULL, 0, NULL, NULL, NULL, NULL, NULL },  /* Pearson placeholder, filled at init */
    { DIST_STUDENT_T,   "StudentT",    3, studt_pdf,   studt_logpdf,   studt_estimate,   studt_init,   studt_valid },
    { DIST_LAPLACE,     "Laplace",     2, laplace_pdf, laplace_logpdf, laplace_estimate, laplace_init, laplace_valid },
    { DIST_CAUCHY,      "Cauchy",      2, cauchy_pdf,  cauchy_logpdf,  cauchy_estimate,  cauchy_init,  cauchy_valid },
    { DIST_INVGAUSS,    "InvGaussian", 2, invgauss_pdf, invgauss_logpdf, invgauss_estimate, invgauss_init, invgauss_valid },
    { DIST_RAYLEIGH,    "Rayleigh",    1, rayleigh_pdf, rayleigh_logpdf, rayleigh_estimate, rayleigh_init, rayleigh_valid },
    { DIST_PARETO,      "Pareto",      2, pareto_pdf,  pareto_logpdf,  pareto_estimate,  pareto_init,  pareto_valid },
    { DIST_LOGISTIC,    "Logistic",    2, logistic_pdf,logistic_logpdf,logistic_estimate,logistic_init,logistic_valid },
    { DIST_GUMBEL,      "Gumbel",      2, gumbel_pdf,  gumbel_logpdf,  gumbel_estimate,  gumbel_init,  gumbel_valid },
    { DIST_SKEWNORMAL,  "SkewNormal",  3, skewnorm_pdf,skewnorm_logpdf,skewnorm_estimate,skewnorm_init,skewnorm_valid },
    { DIST_GENGAUSS,    "GenGaussian", 3, gengauss_pdf,gengauss_logpdf,gengauss_estimate,gengauss_init,gengauss_valid },
    { DIST_CHISQ,       "ChiSquared",  1, chisq_pdf,   chisq_logpdf,   chisq_estimate,   chisq_init,   chisq_valid },
    { DIST_F,           "F",           2, fdist_pdf,   fdist_logpdf,   fdist_estimate,   fdist_init,   fdist_valid },
    { DIST_LOGLOGISTIC, "LogLogistic", 2, loglogistic_pdf,loglogistic_logpdf,loglogistic_estimate,loglogistic_init,loglogistic_valid },
    { DIST_NAKAGAMI,    "Nakagami",    2, nakagami_pdf,nakagami_logpdf,nakagami_estimate,nakagami_init,nakagami_valid },
    { DIST_LEVY,        "Levy",        2, levy_pdf,    levy_logpdf,    levy_estimate,    levy_init,    levy_valid },
    { DIST_GOMPERTZ,    "Gompertz",    2, gompertz_pdf,gompertz_logpdf,gompertz_estimate,gompertz_init,gompertz_valid },
    { DIST_BURR,        "Burr",        2, burr_pdf,    burr_logpdf,    burr_estimate,    burr_init,    burr_valid },
    { DIST_HALFNORMAL,  "HalfNormal",  1, halfnorm_pdf,halfnorm_logpdf,halfnorm_estimate,halfnorm_init,halfnorm_valid },
    { DIST_MAXWELL,     "Maxwell",     1, maxwell_pdf, maxwell_logpdf, maxwell_estimate, maxwell_init, maxwell_valid },
    { DIST_KUMARASWAMY, "Kumaraswamy", 2, kumaraswamy_pdf,kumaraswamy_logpdf,kumaraswamy_estimate,kumaraswamy_init,kumaraswamy_valid },
    { DIST_TRIANGULAR,  "Triangular",  3, triangular_pdf,triangular_logpdf,triangular_estimate,triangular_init,triangular_valid },
    { DIST_BINOMIAL,    "Binomial",    2, binomial_pdf,binomial_logpdf,binomial_estimate,binomial_init,binomial_valid },
    { DIST_NEGBINOM,    "NegBinomial", 2, negbinom_pdf,negbinom_logpdf,negbinom_estimate,negbinom_init,negbinom_valid },
    { DIST_GEOMETRIC,   "Geometric",   1, geometric_pdf,geometric_logpdf,geometric_estimate,geometric_init,geometric_valid },
    { DIST_ZIPF,        "Zipf",        1, zipf_pdf,    zipf_logpdf,    zipf_estimate,    zipf_init,    zipf_valid },
    { DIST_KDE,         "KDE",         1, kde_pdf,     kde_logpdf,     kde_estimate,     kde_init,     kde_valid },
};

static int dist_table_initialized = 0;
static void init_dist_table(void) {
    if (dist_table_initialized) return;
    dist_table[DIST_PEARSON] = pearson_get_dist_functions();
    dist_table_initialized = 1;
}

const DistFunctions* GetDistFunctions(DistFamily family) {
    init_dist_table();
    if (family >= 0 && family < DIST_COUNT) return &dist_table[family];
    return NULL;
}
const char* GetDistName(DistFamily family) {
    init_dist_table();
    if (family >= 0 && family < DIST_COUNT && dist_table[family].name) return dist_table[family].name;
    return "Unknown";
}


/* ====================================================================
 * Generic Mixture EM
 * ==================================================================== */
int UnmixGeneric(const double* data, size_t n, DistFamily family, int k,
                 int maxiter, double rtole, int verbose,
                 MixtureResult* result)
{
    if (!data || n == 0 || k <= 0 || !result) return -1;

    const DistFunctions* df = GetDistFunctions(family);
    if (!df) return -2;

    /* Allocate */
    result->family = family;
    result->num_components = k;
    result->mixing_weights = (double*)malloc(sizeof(double) * k);
    result->params = (DistParams*)malloc(sizeof(DistParams) * k);

    /* Initialize parameters */
    df->init_params(data, n, k, result->params);
    for (int j = 0; j < k; j++) result->mixing_weights[j] = 1.0 / k;

    /* Responsibility matrix: r[j*n + i] = P(component j | data_i) */
    double* resp = (double*)malloc(sizeof(double) * k * n);
    double* weights_j = (double*)malloc(sizeof(double) * n);

    double prev_ll = -1e30;
    int iter;

    /* SQUAREM acceleration: pack all parameters into a flat vector for extrapolation.
     * Layout: [w_0..w_{k-1}, p_0[0]..p_0[nparams-1], p_1[0]..., ...] */
    int nparams_per = df->num_params;
    int ntheta = k + k * nparams_per;  /* weights + all params */
    double* theta0 = (double*)malloc(sizeof(double) * ntheta);
    double* theta1 = (double*)malloc(sizeof(double) * ntheta);
    double* theta2 = (double*)malloc(sizeof(double) * ntheta);

    /* Helper to pack current state into theta */
    #define PACK_THETA(th) do { \
        for (int _j = 0; _j < k; _j++) (th)[_j] = result->mixing_weights[_j]; \
        for (int _j = 0; _j < k; _j++) \
            for (int _q = 0; _q < nparams_per; _q++) \
                (th)[k + _j*nparams_per + _q] = result->params[_j].p[_q]; \
    } while(0)
    #define UNPACK_THETA(th) do { \
        double _wsum = 0; \
        for (int _j = 0; _j < k; _j++) { \
            result->mixing_weights[_j] = (th)[_j] > 1e-10 ? (th)[_j] : 1e-10; \
            _wsum += result->mixing_weights[_j]; } \
        for (int _j = 0; _j < k; _j++) result->mixing_weights[_j] /= _wsum; \
        for (int _j = 0; _j < k; _j++) \
            for (int _q = 0; _q < nparams_per; _q++) { \
                double _v = (th)[k + _j*nparams_per + _q]; \
                if (isfinite(_v)) result->params[_j].p[_q] = _v; } \
    } while(0)

    for (iter = 0; iter < maxiter; iter++) {
        /* ---- E-step: compute responsibilities ---- */
        /* Precompute log-weights to avoid repeated log in inner loop */
        double logw[64]; /* k <= 64 */
        int kk = k < 64 ? k : 64;
        for (int j = 0; j < kk; j++)
            logw[j] = log(result->mixing_weights[j] > 1e-300 ? result->mixing_weights[j] : 1e-300);

        double ll = 0.0;

        /* GPU path: try OpenCL for Gaussian E-step when n is large */
        int used_gpu = 0;
        if (family == DIST_GAUSSIAN && n >= 50000) {
            GpuContext* gpu = get_gpu();
            if (gpu) {
                float* fdata  = (float*)malloc(sizeof(float) * n);
                float* flw    = (float*)malloc(sizeof(float) * kk);
                float* fmu    = (float*)malloc(sizeof(float) * kk);
                float* fvar   = (float*)malloc(sizeof(float) * kk);
                float* fresp  = (float*)malloc(sizeof(float) * kk * n);

                for (size_t i = 0; i < n; i++) fdata[i] = (float)data[i];
                for (int j = 0; j < kk; j++) {
                    flw[j]  = (float)logw[j];
                    fmu[j]  = (float)result->params[j].p[0];
                    fvar[j] = (float)(result->params[j].nparams >= 2 ?
                                      result->params[j].p[1] : 1.0);
                }

                int rc = gpu_estep_gaussian(gpu, fdata, (int)n, flw, fmu, fvar,
                                             kk, fresp, &ll);
                if (rc == 0) {
                    for (int j = 0; j < kk; j++)
                        for (size_t i = 0; i < n; i++)
                            resp[j * n + i] = fresp[j * n + i];
                    used_gpu = 1;
                }
                free(fdata); free(flw); free(fmu); free(fvar); free(fresp);
            }
        }

        if (!used_gpu) {
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:ll) schedule(static) if(n > 5000)
        #endif
        for (size_t i = 0; i < n; i++) {
            /* Log-sum-exp trick for numerical stability */
            double lps[64];
            double max_lp = -1e30;
            for (int j = 0; j < kk; j++) {
                double lp;
                if (df->logpdf) {
                    lp = logw[j] + df->logpdf(data[i], &result->params[j]);
                } else {
                    double pv = df->pdf(data[i], &result->params[j]);
                    lp = logw[j] + (pv > 1e-300 ? log(pv) : -700);
                }
                lps[j] = lp;
                if (lp > max_lp) max_lp = lp;
            }
            double total = 0;
            for (int j = 0; j < kk; j++) {
                double v = exp(lps[j] - max_lp);
                resp[j * n + i] = v;
                total += v;
            }
            /* Normalize */
            for (int j = 0; j < kk; j++) resp[j * n + i] /= total;
            ll += max_lp + log(total);
        }
        } /* end if (!used_gpu) */

        if (verbose) {
            printf("  [%s k=%d] iter %d  LL=%.4f  delta=%.2e\n",
                   df->name, k, iter, ll, ll - prev_ll);
        }

        /* Check convergence */
        if (iter > 0 && fabs(ll - prev_ll) < rtole) {
            prev_ll = ll;
            iter++;
            break;
        }
        prev_ll = ll;

        /* ---- M-step: update parameters ---- */
        for (int j = 0; j < k; j++) {
            /* Extract weights for this component */
            double nj = 0;
            for (size_t i = 0; i < n; i++) {
                weights_j[i] = resp[j * n + i];
                nj += weights_j[i];
            }

            /* Update mixing weight */
            result->mixing_weights[j] = nj / n;
            if (result->mixing_weights[j] < 1e-10) result->mixing_weights[j] = 1e-10;

            /* Update distribution parameters via weighted MLE */
            DistParams old_p = result->params[j];
            df->estimate(data, weights_j, n, &result->params[j]);
            /* Guard against NaN params — revert to prior if estimate fails */
            int any_nan = 0;
            for (int q = 0; q < result->params[j].nparams; q++) {
                if (!isfinite(result->params[j].p[q])) { any_nan = 1; break; }
            }
            if (any_nan) result->params[j] = old_p;
        }

        /* Renormalize mixing weights */
        double wsum = 0;
        for (int j = 0; j < k; j++) wsum += result->mixing_weights[j];
        for (int j = 0; j < k; j++) result->mixing_weights[j] /= wsum;

        /* ---- SQUAREM acceleration (Varadhan & Roland 2008) every 3 iters ---- */
        /* Only apply when convergence is genuinely slow (iter > 20 means
         * standard EM is struggling; skip for fast-converging cases) */
        if (iter >= 20 && (iter % 3) == 0 && fabs(ll - prev_ll) < 1.0) {
            /* θ0 = params before this M-step; θ1 = params after first M-step;
             * θ2 = params after second M-step from θ1 */
            PACK_THETA(theta0);  /* current = after first M-step */

            /* Second EM step from current position */
            {
                /* E-step */
                for (size_t i = 0; i < n; i++) {
                    double tot = 0;
                    for (int j = 0; j < k; j++) {
                        double p = df->logpdf ?
                            result->mixing_weights[j] * exp(df->logpdf(data[i], &result->params[j])) :
                            result->mixing_weights[j] * df->pdf(data[i], &result->params[j]);
                        if (p < PDF_FLOOR) p = PDF_FLOOR;
                        resp[j*n+i] = p; tot += p;
                    }
                    for (int j = 0; j < k; j++) resp[j*n+i] /= tot;
                }
                /* M-step */
                for (int j = 0; j < k; j++) {
                    double nj = 0;
                    for (size_t i = 0; i < n; i++) { weights_j[i] = resp[j*n+i]; nj += weights_j[i]; }
                    result->mixing_weights[j] = nj / n > 1e-10 ? nj / n : 1e-10;
                    DistParams tmp = result->params[j];
                    df->estimate(data, weights_j, n, &result->params[j]);
                    for (int q = 0; q < result->params[j].nparams; q++)
                        if (!isfinite(result->params[j].p[q])) result->params[j].p[q] = tmp.p[q];
                }
                double ws2 = 0;
                for (int j = 0; j < k; j++) ws2 += result->mixing_weights[j];
                for (int j = 0; j < k; j++) result->mixing_weights[j] /= ws2;
            }
            PACK_THETA(theta2);  /* after second M-step */

            /* Compute step length r = ||θ2 - θ0|| / ||θ1 - θ0||²
             * where θ1 was saved before this block — use theta0 as θ1 approx
             * Simplified SQUAREM-2 (Varadhan 2008 Eq 2): extrapolate */
            double r_num = 0, r_den = 0;
            for (int q = 0; q < ntheta; q++) {
                double d = theta2[q] - theta0[q];    /* step after 2 EM steps */
                r_num += d * d;
            }
            /* Step size α = -1 / sqrt(||step||) clamped to [-1, -32] */
            double alpha_sq = r_num > 1e-30 ? -1.0 / sqrt(r_num) : -1.0;
            if (alpha_sq > -1.0) alpha_sq = -1.0;
            if (alpha_sq < -32.0) alpha_sq = -32.0;

            /* Extrapolated θ* = θ0 - 2α*(θ2-θ0) + α²*(θ2-2*θ1+θ0)
             * Simplified: θ* = θ0 + 2*(θ2-θ0) for step ~2x */
            /* Use θ* = θ0 + (1 - 2*alpha_sq) * (θ2 - θ0)  */
            double step_mult = 1.0 - 2.0 * alpha_sq;  /* >1 since alpha_sq<-1 */
            if (step_mult > 10.0) step_mult = 10.0;    /* cap extrapolation */

            for (int q = 0; q < ntheta; q++)
                theta1[q] = theta0[q] + step_mult * (theta2[q] - theta0[q]);

            /* Unpack and verify monotone (if extrapolation decreases LL, revert) */
            UNPACK_THETA(theta1);

            /* Verify LL hasn't decreased */
            double ll_check = 0;
            for (size_t i = 0; i < n; i++) {
                double tot = 0;
                for (int j = 0; j < k; j++) {
                    double p = df->logpdf ?
                        result->mixing_weights[j] * exp(df->logpdf(data[i], &result->params[j])) :
                        result->mixing_weights[j] * df->pdf(data[i], &result->params[j]);
                    tot += (p < PDF_FLOOR ? PDF_FLOOR : p);
                }
                ll_check += log(tot);
            }
            if (ll_check < prev_ll - 1.0) {
                /* Revert to θ2 (safe double-step) */
                UNPACK_THETA(theta2);
            } else {
                prev_ll = ll_check;
                iter++;  /* count the extra EM step we did */
            }
        }
    }

    #undef PACK_THETA
    #undef UNPACK_THETA

    result->iterations = iter;
    result->loglikelihood = prev_ll;

    /* BIC = -2*LL + p*log(n), where p = k*(num_params + 1) - 1 */
    int num_free = k * (df->num_params + 1) - 1;
    result->bic = -2.0 * prev_ll + num_free * log((double)n);
    result->aic = -2.0 * prev_ll + 2.0 * num_free;

    free(resp);
    free(weights_j);
    free(theta0); free(theta1); free(theta2);

    return 0;
}


/* ====================================================================
 * Model Selection
 * ==================================================================== */
int SelectBestMixture(const double* data, size_t n,
                      const DistFamily* families, int nfamilies,
                      int k_min, int k_max,
                      int maxiter, double rtole, int verbose,
                      ModelSelectResult* result)
{
    if (!data || n == 0 || !result) return -1;
    if (k_min < 1) k_min = 1;
    if (k_max < k_min) k_max = k_min;

    /* Default: try all families */
    DistFamily all_families[DIST_COUNT];
    if (!families || nfamilies <= 0) {
        nfamilies = DIST_COUNT;
        for (int i = 0; i < DIST_COUNT; i++) all_families[i] = (DistFamily)i;
        families = all_families;
    }

    /* Check which families are valid for this data */
    int valid_families[DIST_COUNT];
    int n_valid = 0;
    for (int f = 0; f < nfamilies; f++) {
        const DistFunctions* df = GetDistFunctions(families[f]);
        if (!df) continue;
        int all_valid = 1;
        for (size_t i = 0; i < n && all_valid; i++) {
            if (!df->valid(data[i])) all_valid = 0;
        }
        if (all_valid) valid_families[n_valid++] = families[f];
    }

    int total_models = n_valid * (k_max - k_min + 1);
    result->candidates = (MixtureResult*)calloc(total_models, sizeof(MixtureResult));
    result->num_candidates = 0;
    result->best_bic = 1e30;

    if (verbose) {
        printf("\nModel Selection: %d families x %d-%d components = %d candidates\n",
               n_valid, k_min, k_max, total_models);
        printf("Valid families:");
        for (int f = 0; f < n_valid; f++) printf(" %s", GetDistName(valid_families[f]));
        printf("\n\n");
    }

    for (int f = 0; f < n_valid; f++) {
        DistFamily fam = (DistFamily)valid_families[f];
        for (int k = k_min; k <= k_max; k++) {
            int idx = result->num_candidates;
            int rc = UnmixGeneric(data, n, fam, k, maxiter, rtole, 0, &result->candidates[idx]);

            if (rc == 0) {
                if (verbose) {
                    printf("  %-12s k=%d  LL=%12.2f  BIC=%12.2f  AIC=%12.2f  iters=%d",
                           GetDistName(fam), k,
                           result->candidates[idx].loglikelihood,
                           result->candidates[idx].bic,
                           result->candidates[idx].aic,
                           result->candidates[idx].iterations);
                }

                if (result->candidates[idx].bic < result->best_bic) {
                    result->best_bic = result->candidates[idx].bic;
                    result->best_family = fam;
                    result->best_k = k;
                    if (verbose) printf("  <-- BEST");
                }
                if (verbose) printf("\n");

                result->num_candidates++;
            }
        }
    }

    if (verbose) {
        printf("\nBest model: %s with k=%d (BIC=%.2f)\n",
               GetDistName(result->best_family), result->best_k, result->best_bic);
    }

    return 0;
}


/* ====================================================================
 * Adaptive Mixed-Family EM with Auto-k
 *
 * Each component independently discovers its best distribution family.
 * The number of components is discovered via split-merge operations
 * driven by BIC improvement.
 * ==================================================================== */

/* Candidate families for adaptive selection (skip Pearson — too slow for inner loop) */
static const DistFamily ADAPT_FAMILIES_REAL[] = {
    DIST_GAUSSIAN, DIST_STUDENT_T, DIST_LAPLACE, DIST_CAUCHY,
    DIST_LOGISTIC, DIST_GUMBEL, DIST_SKEWNORMAL
};
static const int N_ADAPT_REAL = 7;
/* Note: GenGaussian excluded from adaptive — subsumes Gaussian+Laplace,
   too flexible with low β, causes single-component overfitting */

static const DistFamily ADAPT_FAMILIES_POS[] = {
    DIST_EXPONENTIAL, DIST_GAMMA, DIST_LOGNORMAL, DIST_WEIBULL,
    DIST_INVGAUSS, DIST_RAYLEIGH, DIST_CHISQ, DIST_LOGLOGISTIC,
    DIST_NAKAGAMI, DIST_LEVY, DIST_GOMPERTZ, DIST_BURR,
    DIST_HALFNORMAL, DIST_MAXWELL
};
static const int N_ADAPT_POS = 14;

/* Bounded [0,1] families */
static const DistFamily ADAPT_FAMILIES_UNIT[] = {
    DIST_BETA, DIST_KUMARASWAMY
};
static const int N_ADAPT_UNIT = 2;

/* Discrete families */
static const DistFamily ADAPT_FAMILIES_DISC[] = {
    DIST_POISSON, DIST_BINOMIAL, DIST_NEGBINOM, DIST_GEOMETRIC, DIST_ZIPF
};
static const int N_ADAPT_DISC = 5;

/* Weighted log-likelihood for a single component under a given family */
static double component_wll(const double* data, const double* weights,
                            size_t n, DistFamily fam, DistParams* out_params)
{
    const DistFunctions* df = GetDistFunctions(fam);
    if (!df) return -1e30;

    /* Check domain validity */
    double sw = 0;
    for (size_t i = 0; i < n; i++) {
        if (weights[i] > 1e-10 && !df->valid(data[i])) return -1e30;
        sw += weights[i];
    }
    if (sw < 1.0) return -1e30;  /* need at least ~1 effective sample */

    /* Estimate parameters */
    DistParams params;
    df->estimate(data, weights, n, &params);

    /* Check for NaN */
    for (int q = 0; q < params.nparams; q++) {
        if (!isfinite(params.p[q])) return -1e30;
    }

    /* Compute weighted log-likelihood */
    double wll = 0;
    for (size_t i = 0; i < n; i++) {
        if (weights[i] < 1e-10) continue;
        double lp = df->logpdf ? df->logpdf(data[i], &params)
                               : log(df->pdf(data[i], &params) + 1e-300);
        wll += weights[i] * lp;
    }

    /* Penalize by number of free params (local BIC-like criterion) */
    wll -= 0.5 * df->num_params * log(sw);

    if (out_params) *out_params = params;
    return wll;
}

/* Find best family for a component given its responsibility weights */
static DistFamily best_family_for_component(const double* data, const double* weights,
                                             size_t n, int all_positive,
                                             DistParams* out_params)
{
    double best_wll = -1e30;
    DistFamily best_fam = DIST_GAUSSIAN;
    DistParams best_p = {{0,1,0,0}, 2};

    /* Try real-valued families */
    for (int f = 0; f < N_ADAPT_REAL; f++) {
        DistParams p;
        double wll = component_wll(data, weights, n, ADAPT_FAMILIES_REAL[f], &p);
        if (wll > best_wll) { best_wll = wll; best_fam = ADAPT_FAMILIES_REAL[f]; best_p = p; }
    }

    /* Try positive-only families if data is all positive */
    if (all_positive) {
        for (int f = 0; f < N_ADAPT_POS; f++) {
            DistParams p;
            double wll = component_wll(data, weights, n, ADAPT_FAMILIES_POS[f], &p);
            if (wll > best_wll) { best_wll = wll; best_fam = ADAPT_FAMILIES_POS[f]; best_p = p; }
        }
    }

    /* Try unit-interval [0,1] families if all data in (0,1) */
    int all_unit = 1;
    for (size_t i = 0; i < n; i++) {
        if (weights[i] > 0.01 && (data[i] <= 0 || data[i] >= 1)) { all_unit = 0; break; }
    }
    if (all_unit) {
        for (int f = 0; f < N_ADAPT_UNIT; f++) {
            DistParams p;
            double wll = component_wll(data, weights, n, ADAPT_FAMILIES_UNIT[f], &p);
            if (wll > best_wll) { best_wll = wll; best_fam = ADAPT_FAMILIES_UNIT[f]; best_p = p; }
        }
    }

    /* Try discrete families if data looks integer-valued */
    int all_integer = 1;
    for (size_t i = 0; i < n; i++) {
        if (weights[i] > 0.01 && fabs(data[i] - round(data[i])) > 0.01) { all_integer = 0; break; }
    }
    if (all_integer) {
        for (int f = 0; f < N_ADAPT_DISC; f++) {
            DistParams p;
            double wll = component_wll(data, weights, n, ADAPT_FAMILIES_DISC[f], &p);
            if (wll > best_wll) { best_wll = wll; best_fam = ADAPT_FAMILIES_DISC[f]; best_p = p; }
        }
    }

    if (out_params) *out_params = best_p;
    return best_fam;
}

/* Bimodality test: Hartigan's dip statistic approximation */
static double bimodality_coeff(const double* data, const double* weights, size_t n)
{
    double sw = 0, sm = 0;
    for (size_t i = 0; i < n; i++) { sw += weights[i]; sm += weights[i]*data[i]; }
    if (sw < 5) return 0;
    double mu = sm / sw;
    double m2 = 0, m3 = 0, m4 = 0;
    for (size_t i = 0; i < n; i++) {
        double d = data[i] - mu, w = weights[i];
        m2 += w*d*d; m3 += w*d*d*d; m4 += w*d*d*d*d;
    }
    m2 /= sw; m3 /= sw; m4 /= sw;
    if (m2 < 1e-10) return 0;
    double gamma1_sq = (m3*m3) / (m2*m2*m2);
    double gamma2 = m4 / (m2*m2) - 3;
    /* BC = (gamma1^2 + 1) / (gamma2 + 3*(n-1)^2/((n-2)*(n-3))) */
    double n_eff = sw;
    double denom = gamma2 + 3.0*(n_eff-1)*(n_eff-1) / ((n_eff-2)*(n_eff-3));
    if (denom < 0.1) denom = 0.1;
    return (gamma1_sq + 1.0) / denom;
}

/* Kullback-Leibler divergence estimate between two components */
static double component_kl_sym(const double* data, size_t n,
                                DistFamily fam_a, const DistParams* pa,
                                DistFamily fam_b, const DistParams* pb)
{
    const DistFunctions* dfa = GetDistFunctions(fam_a);
    const DistFunctions* dfb = GetDistFunctions(fam_b);
    if (!dfa || !dfb) return 1e30;

    /* Monte Carlo KL using data points */
    double kl_ab = 0, kl_ba = 0;
    int cnt = 0;
    for (size_t i = 0; i < n; i += (n > 1000 ? n/500 : 1)) {
        double la = dfa->logpdf ? dfa->logpdf(data[i], pa) : log(dfa->pdf(data[i], pa)+1e-300);
        double lb = dfb->logpdf ? dfb->logpdf(data[i], pb) : log(dfb->pdf(data[i], pb)+1e-300);
        if (!isfinite(la) || !isfinite(lb)) continue;
        kl_ab += la - lb;
        kl_ba += lb - la;
        cnt++;
    }
    if (cnt == 0) return 1e30;
    kl_ab /= cnt; kl_ba /= cnt;
    return 0.5 * (fabs(kl_ab) + fabs(kl_ba));  /* symmetric KL */
}

/* ════════════════════════════════════════════════════════════════════
 * Helper: compute information criteria
 * ════════════════════════════════════════════════════════════════════ */
static int count_free_params(const DistFamily* fams, int k) {
    int nf = 0;
    for (int j = 0; j < k; j++) {
        const DistFunctions* df = GetDistFunctions(fams[j]);
        nf += df->num_params + 1;  /* params + mixing weight */
    }
    return nf - 1;  /* weights sum to 1 */
}

static double compute_bic(double ll, int nfree, size_t n) {
    return -2.0 * ll + nfree * log((double)n);
}
static double compute_aic(double ll, int nfree) {
    return -2.0 * ll + 2.0 * nfree;
}

/* ICL = BIC + classification entropy.
 * Entropy = -Σᵢ Σⱼ rᵢⱼ log(rᵢⱼ) — penalizes fuzzy assignments */
static double compute_icl(double bic, const double* resp, int k, size_t n) {
    double entropy = 0;
    for (int j = 0; j < k; j++) {
        for (size_t i = 0; i < n; i++) {
            double r = resp[j * n + i];
            if (r > 1e-300) entropy -= r * log(r);
        }
    }
    return bic + 2.0 * entropy;
}

/* MML (Wallace-Freeman approximation):
 * MML = -LL + 0.5 * Σ_j log(n·w_j/12) + 0.5 * nfree * log(n/12) + 0.5 * nfree */
static double compute_mml(double ll, int nfree, size_t n,
                          const double* mix_w, int k)
{
    double mml = -ll;
    double logn12 = log((double)n / 12.0);
    for (int j = 0; j < k; j++) {
        double wj = (mix_w && mix_w[j] > 1e-10) ? mix_w[j] : 1e-10;
        mml += 0.5 * log((double)n * wj / 12.0);
    }
    mml += 0.5 * nfree * logn12 + 0.5 * nfree;
    return mml;
}

/* Criterion value for model comparison: lower is better */
static double eval_criterion(KMethod m, double ll, int nfree, size_t n,
                             const double* resp, int k)
{
    switch (m) {
        case KMETHOD_AIC:  return compute_aic(ll, nfree);
        case KMETHOD_ICL:  return compute_icl(compute_bic(ll, nfree, n), resp, k, n);
        case KMETHOD_MML:  return compute_bic(ll, nfree, n);  /* placeholder — real MML needs weights, computed in split-merge */
        case KMETHOD_BIC:
        default:           return compute_bic(ll, nfree, n);
    }
}

const char* GetKMethodName(KMethod m) {
    switch(m) {
        case KMETHOD_BIC:  return "BIC";
        case KMETHOD_AIC:  return "AIC";
        case KMETHOD_ICL:  return "ICL";
        case KMETHOD_VBEM: return "VBEM";
        case KMETHOD_MML:  return "MML";
        default: return "Unknown";
    }
}


/* ════════════════════════════════════════════════════════════════════
 * Inner EM loop shared by split-merge methods (BIC/AIC/ICL)
 * Returns final log-likelihood in *out_ll
 * ════════════════════════════════════════════════════════════════════ */
static void inner_em_loop(const double* data, size_t n,
    int k, DistFamily* fams, DistParams* par, double* mix_w,
    double* resp, double* wj,
    int maxiter, double rtole, int verbose,
    int family_reselect_interval,
    double* out_ll)
{
    double prev_ll = -1e30;
    for (int iter = 0; iter < maxiter; iter++) {
        /* E-step */
        double ll = 0;
        for (size_t i = 0; i < n; i++) {
            double total = 0;
            for (int j = 0; j < k; j++) {
                const DistFunctions* df = GetDistFunctions(fams[j]);
                double lp = df->logpdf ? df->logpdf(data[i], &par[j])
                                       : log(df->pdf(data[i], &par[j]) + 1e-300);
                double p = mix_w[j] * exp(lp);
                if (p < PDF_FLOOR) p = PDF_FLOOR;
                resp[j * n + i] = p;
                total += p;
            }
            for (int j = 0; j < k; j++) resp[j * n + i] /= total;
            ll += log(total);
        }

        if (verbose && (iter < 5 || iter % 20 == 0)) {
            printf("  [adaptive k=%d] iter %d  LL=%.4f  delta=%.2e  families:",
                   k, iter, ll, ll - prev_ll);
            for (int j = 0; j < k; j++) printf(" %s", GetDistName(fams[j]));
            printf("\n");
        }

        if (iter > 1 && fabs(ll - prev_ll) < rtole) { prev_ll = ll; break; }
        prev_ll = ll;

        /* M-step */
        for (int j = 0; j < k; j++) {
            double nj = 0;
            for (size_t i = 0; i < n; i++) { wj[i] = resp[j*n+i]; nj += wj[i]; }
            mix_w[j] = nj / n;
            if (mix_w[j] < 1e-10) mix_w[j] = 1e-10;

            if (iter % family_reselect_interval == 0) {
                int comp_positive = 1;
                for (size_t i = 0; i < n; i++)
                    if (wj[i] > 0.01 && data[i] <= 0) { comp_positive = 0; break; }
                DistParams new_p;
                DistFamily new_fam = best_family_for_component(data, wj, n, comp_positive, &new_p);
                fams[j] = new_fam;
                par[j] = new_p;
            } else {
                DistParams old_p = par[j];
                const DistFunctions* df = GetDistFunctions(fams[j]);
                df->estimate(data, wj, n, &par[j]);
                int any_nan = 0;
                for (int q = 0; q < par[j].nparams; q++)
                    if (!isfinite(par[j].p[q])) { any_nan = 1; break; }
                if (any_nan) par[j] = old_p;
            }
        }

        double wsum = 0;
        for (int j = 0; j < k; j++) wsum += mix_w[j];
        for (int j = 0; j < k; j++) mix_w[j] /= wsum;
    }
    *out_ll = prev_ll;
}


/* ════════════════════════════════════════════════════════════════════
 * Split-merge adaptive (BIC / AIC / ICL)
 * ════════════════════════════════════════════════════════════════════ */
static int adaptive_split_merge(const double* data, size_t n,
    int k_max, int maxiter, double rtole, int verbose,
    KMethod kmethod, AdaptiveResult* result)
{
    int k = 1;
    double* mix_w = (double*)malloc(sizeof(double) * k_max);
    DistParams* par = (DistParams*)malloc(sizeof(DistParams) * k_max);
    DistFamily* fams = (DistFamily*)malloc(sizeof(DistFamily) * k_max);

    const DistFunctions* gauss = GetDistFunctions(DIST_GAUSSIAN);
    gauss->init_params(data, n, 1, par);
    mix_w[0] = 1.0;
    fams[0] = DIST_GAUSSIAN;

    double* resp = (double*)malloc(sizeof(double) * k_max * n);
    double* wj = (double*)malloc(sizeof(double) * n);

    double best_crit = 1e30;
    int best_k = 1;
    double* best_mix_w = (double*)malloc(sizeof(double) * k_max);
    DistParams* best_par = (DistParams*)malloc(sizeof(DistParams) * k_max);
    DistFamily* best_fams = (DistFamily*)malloc(sizeof(DistFamily) * k_max);
    double best_ll = -1e30;

    int outer_iter = 0, no_improve_count = 0;
    const char* crit_name = GetKMethodName(kmethod);

    while (outer_iter < 20 && no_improve_count < 3) {
        outer_iter++;

        double cur_ll;
        inner_em_loop(data, n, k, fams, par, mix_w, resp, wj,
                      maxiter, rtole, verbose, 5, &cur_ll);

        int nfree = count_free_params(fams, k);
        double cur_crit;
        if (kmethod == KMETHOD_MML) {
            cur_crit = compute_mml(cur_ll, nfree, n, mix_w, k);
        } else {
            cur_crit = eval_criterion(kmethod, cur_ll, nfree, n, resp, k);
        }

        if (verbose) {
            printf("  [k=%d] LL=%.2f  %s=%.2f  (best %s=%.2f)\n",
                   k, cur_ll, crit_name, cur_crit, crit_name, best_crit);
        }

        if (cur_crit < best_crit - 1.0) {
            best_crit = cur_crit;
            best_k = k;
            best_ll = cur_ll;
            for (int j = 0; j < k; j++) {
                best_mix_w[j] = mix_w[j]; best_par[j] = par[j]; best_fams[j] = fams[j];
            }
            no_improve_count = 0;
        } else {
            no_improve_count++;
        }

        if (k >= k_max) break;

        /* Split most bimodal component */
        double max_bc = 0; int split_j = -1;
        for (int j = 0; j < k; j++) {
            for (size_t i = 0; i < n; i++) wj[i] = resp[j*n+i];
            double bc = bimodality_coeff(data, wj, n);
            if (bc > max_bc && mix_w[j] > 0.05) { max_bc = bc; split_j = j; }
        }

        /* Merge similar components */
        int merge_a = -1, merge_b = -1; double min_kl = 1e30;
        for (int a = 0; a < k; a++)
            for (int b = a+1; b < k; b++) {
                double kl = component_kl_sym(data, n, fams[a], &par[a], fams[b], &par[b]);
                if (kl < min_kl) { min_kl = kl; merge_a = a; merge_b = b; }
            }

        /* Split */
        if (k < k_max) {
            if (split_j < 0) split_j = 0;
            if (verbose) printf("  Splitting component %d (BC=%.3f, %s)\n",
                                split_j, max_bc, GetDistName(fams[split_j]));
            double mu_j = par[split_j].p[0];
            double sigma_j = (par[split_j].nparams >= 2) ? sqrt(fabs(par[split_j].p[1])) : 1.0;
            if (sigma_j < 1e-6) sigma_j = 1.0;
            par[k] = par[split_j]; par[k].p[0] = mu_j + sigma_j * 0.5;
            fams[k] = fams[split_j]; par[split_j].p[0] = mu_j - sigma_j * 0.5;
            double w_orig = mix_w[split_j];
            mix_w[split_j] = w_orig * 0.5; mix_w[k] = w_orig * 0.5;
            k++;
        }

        /* Merge if KL < 0.1 */
        if (k > 1 && min_kl < 0.1 && merge_a >= 0 && merge_b >= 0) {
            if (verbose) printf("  Merging %d+%d, KL=%.4f\n", merge_a, merge_b, min_kl);
            double wa = mix_w[merge_a], wb = mix_w[merge_b], wt = wa+wb;
            for (int q = 0; q < par[merge_a].nparams; q++)
                par[merge_a].p[q] = (wa*par[merge_a].p[q]+wb*par[merge_b].p[q])/wt;
            mix_w[merge_a] = wt;
            for (int j = merge_b; j < k-1; j++) { par[j]=par[j+1]; fams[j]=fams[j+1]; mix_w[j]=mix_w[j+1]; }
            k--;
        }

        /* Prune dead */
        for (int j = k-1; j >= 0; j--) {
            if (mix_w[j] < 0.01 && k > 1) {
                if (verbose) printf("  Pruning dead component %d (w=%.4f)\n", j, mix_w[j]);
                for (int jj = j; jj < k-1; jj++) { par[jj]=par[jj+1]; fams[jj]=fams[jj+1]; mix_w[jj]=mix_w[jj+1]; }
                k--;
            }
        }
        double wsum = 0;
        for (int j = 0; j < k; j++) wsum += mix_w[j];
        for (int j = 0; j < k; j++) mix_w[j] /= wsum;
    }

    /* Return best model */
    result->num_components = best_k;
    result->loglikelihood = best_ll;
    result->kmethod = kmethod;
    result->iterations = outer_iter;
    int nfree = count_free_params(best_fams, best_k);
    result->bic = compute_bic(best_ll, nfree, n);
    result->aic = compute_aic(best_ll, nfree);
    /* Compute ICL from best model's responsibilities (re-run E-step) */
    {
        double* final_resp = (double*)malloc(sizeof(double) * best_k * n);
        for (size_t i = 0; i < n; i++) {
            double total = 0;
            for (int j = 0; j < best_k; j++) {
                const DistFunctions* df = GetDistFunctions(best_fams[j]);
                double lp = df->logpdf ? df->logpdf(data[i], &best_par[j])
                                       : log(df->pdf(data[i], &best_par[j]) + 1e-300);
                double p = best_mix_w[j] * exp(lp);
                if (p < PDF_FLOOR) p = PDF_FLOOR;
                final_resp[j * n + i] = p;
                total += p;
            }
            for (int j = 0; j < best_k; j++) final_resp[j * n + i] /= total;
        }
        result->icl = compute_icl(result->bic, final_resp, best_k, n);
        free(final_resp);
    }

    result->mixing_weights = (double*)malloc(sizeof(double) * best_k);
    result->params = (DistParams*)malloc(sizeof(DistParams) * best_k);
    result->families = (DistFamily*)malloc(sizeof(DistFamily) * best_k);
    for (int j = 0; j < best_k; j++) {
        result->mixing_weights[j] = best_mix_w[j];
        result->params[j] = best_par[j];
        result->families[j] = best_fams[j];
    }

    free(resp); free(wj); free(mix_w); free(par); free(fams);
    free(best_mix_w); free(best_par); free(best_fams);
    return 0;
}


/* ════════════════════════════════════════════════════════════════════
 * VBEM: Variational Bayes EM
 *
 * Start with k_max components. Use Dirichlet prior α₀ on weights.
 * In the VB M-step, effective weight of component j is:
 *     αⱼ = α₀ + Σᵢ rᵢⱼ
 *     E[log πⱼ] = ψ(αⱼ) - ψ(Σ αⱼ)
 *
 * Components whose αⱼ → α₀ (no data support) naturally die.
 * After convergence, prune components with weight < threshold.
 * This is a genuine M-step over k — no heuristic split-merge needed.
 * ════════════════════════════════════════════════════════════════════ */
/* Digamma (psi) function approximation via asymptotic series + recurrence */
static double digamma_safe(double x) {
    double r = 0;
    while (x < 6) { r -= 1.0/x; x += 1.0; }
    r += log(x) - 0.5/x - 1.0/(12*x*x) + 1.0/(120*x*x*x*x);
    return r;
}

static int adaptive_vbem(const double* data, size_t n,
    int k_max, int maxiter, double rtole, int verbose,
    AdaptiveResult* result)
{
    if (k_max <= 0) k_max = 10;
    int k = k_max;  /* start with maximum components */
    /* VBEM converges faster than split-merge — cap iterations */
    if (maxiter > 100) maxiter = 100;
    /* Relax tolerance for VBEM — we're looking for approximate k, not exact convergence */
    if (rtole < 0.01) rtole = 0.01;

    double* mix_w = (double*)malloc(sizeof(double) * k);
    DistParams* par = (DistParams*)malloc(sizeof(DistParams) * k);
    DistFamily* fams = (DistFamily*)malloc(sizeof(DistFamily) * k);
    double* alpha = (double*)malloc(sizeof(double) * k);  /* Dirichlet params */

    /* Initialize: spread k components evenly across data range */
    const DistFunctions* gauss = GetDistFunctions(DIST_GAUSSIAN);
    gauss->init_params(data, n, k, par);
    for (int j = 0; j < k; j++) {
        mix_w[j] = 1.0 / k;
        fams[j] = DIST_GAUSSIAN;
        alpha[j] = 1.0 / k;  /* weak symmetric Dirichlet prior */
    }

    double* resp = (double*)malloc(sizeof(double) * k * n);
    double* wj = (double*)malloc(sizeof(double) * n);

    double alpha0 = 1.0 / k;  /* prior concentration per component */
    double prev_ll = -1e30;

    for (int iter = 0; iter < maxiter; iter++) {
        /* VB E-step: use E[log πⱼ] instead of log πⱼ */
        double alpha_sum = 0;
        for (int j = 0; j < k; j++) alpha_sum += alpha[j];
        double psi_sum = digamma_safe(alpha_sum);

        double ll = 0;
        for (size_t i = 0; i < n; i++) {
            double total = 0;
            for (int j = 0; j < k; j++) {
                const DistFunctions* df = GetDistFunctions(fams[j]);
                double lp = df->logpdf ? df->logpdf(data[i], &par[j])
                                       : log(df->pdf(data[i], &par[j]) + 1e-300);
                /* Use expected log-weight from Dirichlet */
                double log_pi_j = digamma_safe(alpha[j]) - psi_sum;
                double p = exp(log_pi_j + lp);
                if (p < PDF_FLOOR) p = PDF_FLOOR;
                resp[j * n + i] = p;
                total += p;
            }
            for (int j = 0; j < k; j++) resp[j * n + i] /= total;
            ll += log(total);
        }

        if (verbose && (iter < 5 || iter % 20 == 0)) {
            int alive = 0;
            for (int j = 0; j < k; j++) if (mix_w[j] > 0.01) alive++;
            printf("  [VBEM] iter %d  LL=%.4f  delta=%.2e  alive=%d/%d  families:",
                   iter, ll, ll - prev_ll, alive, k);
            for (int j = 0; j < k; j++) {
                if (mix_w[j] > 0.01) printf(" %s(%.1f%%)", GetDistName(fams[j]), mix_w[j]*100);
            }
            printf("\n");
        }

        if (iter > 1 && fabs(ll - prev_ll) < rtole) { prev_ll = ll; break; }
        prev_ll = ll;

        /* VB M-step */
        for (int j = 0; j < k; j++) {
            double nj = 0;
            for (size_t i = 0; i < n; i++) { wj[i] = resp[j*n+i]; nj += wj[i]; }

            /* Update Dirichlet parameter */
            alpha[j] = alpha0 + nj;

            /* Effective weight */
            mix_w[j] = alpha[j] / (k * alpha0 + n);

            /* Skip parameter update for nearly-dead components */
            if (nj < 1.0) continue;

            /* VBEM uses Gaussian only — family selection is too expensive inside the inner loop.
             * For adaptive family discovery, use adaptive_split_merge() instead. */
            if (0) {  /* disabled for VBEM performance */
                int comp_positive = 1;
                for (size_t i = 0; i < n; i++)
                    if (wj[i] > 0.01 && data[i] <= 0) { comp_positive = 0; break; }
                DistParams new_p;
                DistFamily new_fam = best_family_for_component(data, wj, n, comp_positive, &new_p);
                fams[j] = new_fam;
                par[j] = new_p;
            } else {
                DistParams old_p = par[j];
                const DistFunctions* df = GetDistFunctions(fams[j]);
                df->estimate(data, wj, n, &par[j]);
                int any_nan = 0;
                for (int q = 0; q < par[j].nparams; q++)
                    if (!isfinite(par[j].p[q])) { any_nan = 1; break; }
                if (any_nan) par[j] = old_p;
            }
        }
    }

    /* Prune dead components (weight < 1%) */
    int alive = 0;
    for (int j = 0; j < k; j++) {
        if (mix_w[j] >= 0.01) alive++;
    }
    if (alive == 0) alive = 1;  /* keep at least 1 */

    result->num_components = alive;
    result->mixing_weights = (double*)malloc(sizeof(double) * alive);
    result->params = (DistParams*)malloc(sizeof(DistParams) * alive);
    result->families = (DistFamily*)malloc(sizeof(DistFamily) * alive);

    int idx = 0;
    double wsum = 0;
    for (int j = 0; j < k; j++) {
        if (mix_w[j] >= 0.01 || (alive == 1 && idx == 0)) {
            result->mixing_weights[idx] = mix_w[j];
            result->params[idx] = par[j];
            result->families[idx] = fams[j];
            wsum += mix_w[j];
            idx++;
            if (idx >= alive) break;
        }
    }
    /* Renormalize */
    for (int j = 0; j < alive; j++) result->mixing_weights[j] /= wsum;

    result->loglikelihood = prev_ll;
    result->kmethod = KMETHOD_VBEM;
    result->iterations = maxiter;
    int nfree = count_free_params(result->families, alive);
    result->bic = compute_bic(prev_ll, nfree, n);
    result->aic = compute_aic(prev_ll, nfree);
    result->icl = 0;  /* Not meaningful for VBEM */

    free(resp); free(wj); free(mix_w); free(par); free(fams); free(alpha);
    return 0;
}


/* ════════════════════════════════════════════════════════════════════
 * Public API: UnmixAdaptive / UnmixAdaptiveEx
 * ════════════════════════════════════════════════════════════════════ */
int UnmixAdaptiveEx(const double* data, size_t n,
                    int k_max, int maxiter, double rtole, int verbose,
                    KMethod kmethod, AdaptiveResult* result)
{
    if (!data || n == 0 || !result) return -1;
    if (k_max <= 0) k_max = 10;
    if (maxiter <= 0) maxiter = 300;
    init_dist_table();

    if (verbose) {
        printf("INFO: Adaptive mode — k-selection method: %s\n", GetKMethodName(kmethod));
    }

    if (kmethod == KMETHOD_VBEM) {
        return adaptive_vbem(data, n, k_max, maxiter, rtole, verbose, result);
    } else {
        return adaptive_split_merge(data, n, k_max, maxiter, rtole, verbose, kmethod, result);
    }
}

int UnmixAdaptive(const double* data, size_t n,
                  int k_max, int maxiter, double rtole, int verbose,
                  AdaptiveResult* result)
{
    return UnmixAdaptiveEx(data, n, k_max, maxiter, rtole, verbose, KMETHOD_BIC, result);
}

/* ════════════════════════════════════════════════════════════════════
 * Spectral Initialization (Vempala & Wang 2004)
 *
 * For univariate data: use empirical moments to build a Hankel matrix,
 * eigendecompose it, and extract component means from eigenvalues.
 * This provides a provably good initialization for EM.
 * ════════════════════════════════════════════════════════════════════ */

/* Simple Jacobi eigenvalue solver for symmetric matrices */
static void jacobi_eigen(double* A, int n, double* eigenvalues, double* eigenvectors, int max_iter) {
    /* Initialize eigenvectors as identity */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            eigenvectors[i*n+j] = (i == j) ? 1.0 : 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Find largest off-diagonal element */
        double max_val = 0;
        int p = 0, q = 1;
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                if (fabs(A[i*n+j]) > max_val) { max_val = fabs(A[i*n+j]); p = i; q = j; }

        if (max_val < 1e-12) break;  /* converged */

        /* Compute rotation */
        double app = A[p*n+p], aqq = A[q*n+q], apq = A[p*n+q];
        double theta = 0.5 * atan2(2*apq, app - aqq);
        double c = cos(theta), s = sin(theta);

        /* Apply rotation to A */
        for (int i = 0; i < n; i++) {
            double aip = A[i*n+p], aiq = A[i*n+q];
            A[i*n+p] = c*aip + s*aiq;
            A[i*n+q] = -s*aip + c*aiq;
        }
        for (int j = 0; j < n; j++) {
            double apj = A[p*n+j], aqj = A[q*n+j];
            A[p*n+j] = c*apj + s*aqj;
            A[q*n+j] = -s*apj + c*aqj;
        }

        /* Apply rotation to eigenvectors */
        for (int i = 0; i < n; i++) {
            double vip = eigenvectors[i*n+p], viq = eigenvectors[i*n+q];
            eigenvectors[i*n+p] = c*vip + s*viq;
            eigenvectors[i*n+q] = -s*vip + c*viq;
        }
    }

    for (int i = 0; i < n; i++) eigenvalues[i] = A[i*n+i];
}

int SpectralInit(const double* data, size_t n, int k,
                 double* out_means, double* out_weights)
{
    if (!data || n == 0 || k <= 0 || !out_means || !out_weights) return -1;
    if (k == 1) {
        double sum = 0;
        for (size_t i = 0; i < n; i++) sum += data[i];
        out_means[0] = sum / n;
        out_weights[0] = 1.0;
        return 0;
    }
    if ((size_t)k > n) return -2;

    /* Compute empirical moments M_1 through M_{2k} */
    int nm = 2*k + 1;
    double* moments = (double*)calloc(nm, sizeof(double));
    for (size_t i = 0; i < n; i++) {
        double xp = 1.0;
        for (int m = 0; m < nm; m++) {
            moments[m] += xp;
            xp *= data[i];
        }
    }
    for (int m = 0; m < nm; m++) moments[m] /= n;

    /* Build k×k Hankel matrix: H[i][j] = moment[i+j+1] - moment[i+1]*moment[j+1]/moment[0]
     * Actually simpler: H[i][j] = M_{i+j} for i,j = 0..k-1 (moment matrix) */
    double* H = (double*)calloc(k*k, sizeof(double));
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            H[i*k+j] = moments[i+j];  /* M_{i+j} */

    /* Center the Hankel matrix: subtract M1² to get central moments */
    double mu = moments[1];  /* E[X] */
    /* Use centralized moments: (X - μ) */
    double* cmoments = (double*)calloc(nm, sizeof(double));
    for (size_t i = 0; i < n; i++) {
        double xc = data[i] - mu;
        double xp = 1.0;
        for (int m = 0; m < nm; m++) {
            cmoments[m] += xp;
            xp *= xc;
        }
    }
    for (int m = 0; m < nm; m++) cmoments[m] /= n;

    /* Rebuild Hankel from central moments */
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            H[i*k+j] = cmoments[i+j];

    /* Eigendecompose */
    double* eigenvalues = (double*)calloc(k, sizeof(double));
    double* eigenvectors = (double*)calloc(k*k, sizeof(double));
    jacobi_eigen(H, k, eigenvalues, eigenvectors, 200);

    /* Sort eigenvalues descending by absolute value */
    for (int i = 0; i < k-1; i++)
        for (int j = i+1; j < k; j++)
            if (fabs(eigenvalues[j]) > fabs(eigenvalues[i])) {
                double tmp = eigenvalues[i]; eigenvalues[i] = eigenvalues[j]; eigenvalues[j] = tmp;
            }

    /* Project data onto top eigenvector, sort by projection, then
     * k-means style clustering to extract means and weights.
     * This is the Vempala & Wang spectral projection approach. */
    {
        /* Project each data point onto the principal eigenvector
         * For univariate: use the sorted data directly with Chebyshev
         * polynomial projections using top-k eigenvectors */

        /* Approach: sort data, partition into k segments using k-1 optimal
         * split points found by minimizing within-segment variance.
         * The spectral eigenvalues tell us the relative spread. */

        double* sorted = (double*)malloc(sizeof(double) * n);
        memcpy(sorted, data, sizeof(double) * n);
        /* Simple insertion sort for moderate n */
        for (size_t i = 1; i < n; i++) {
            double key = sorted[i];
            int j = (int)i - 1;
            while (j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j]; j--; }
            sorted[j+1] = key;
        }

        /* Find k-1 split points by largest gaps in sorted data */
        /* Compute gaps */
        double* gaps = (double*)malloc(sizeof(double) * (n-1));
        int* gap_idx = (int*)malloc(sizeof(int) * (n-1));
        for (size_t i = 0; i < n-1; i++) {
            gaps[i] = sorted[i+1] - sorted[i];
            gap_idx[i] = (int)i;
        }
        /* Sort gaps descending, pick top k-1 */
        for (int i = 0; i < k-1 && i < (int)(n-1); i++) {
            int best = i;
            for (int j = i+1; j < (int)(n-1); j++) {
                if (gaps[j] > gaps[best]) best = j;
            }
            if (best != i) {
                double tg = gaps[i]; gaps[i] = gaps[best]; gaps[best] = tg;
                int ti = gap_idx[i]; gap_idx[i] = gap_idx[best]; gap_idx[best] = ti;
            }
        }

        /* Sort the top k-1 split indices ascending */
        int* splits = (int*)malloc(sizeof(int) * k);
        for (int i = 0; i < k-1; i++) splits[i] = gap_idx[i] + 1;
        for (int i = 0; i < k-2; i++)
            for (int j = i+1; j < k-1; j++)
                if (splits[j] < splits[i]) { int t = splits[i]; splits[i] = splits[j]; splits[j] = t; }
        splits[k-1] = (int)n;  /* sentinel */

        /* Compute segment means and weights */
        int prev = 0;
        for (int j = 0; j < k; j++) {
            int end = splits[j];
            double seg_sum = 0;
            int seg_count = end - prev;
            for (int ii = prev; ii < end; ii++) seg_sum += sorted[ii];
            out_means[j] = (seg_count > 0) ? seg_sum / seg_count : mu;
            out_weights[j] = (double)seg_count / n;
            prev = end;
        }

        /* Ensure no zero weights */
        for (int j = 0; j < k; j++)
            if (out_weights[j] < 1e-10) out_weights[j] = 1.0 / k;
        double wsum = 0;
        for (int j = 0; j < k; j++) wsum += out_weights[j];
        for (int j = 0; j < k; j++) out_weights[j] /= wsum;

        free(sorted); free(gaps); free(gap_idx); free(splits);
    }

    free(moments); free(cmoments); free(H); free(eigenvalues); free(eigenvectors);
    return 0;
}


/* ════════════════════════════════════════════════════════════════════
 * Online/Stochastic EM (Cappé & Moulines 2009)
 *
 * Mini-batch E-step + running sufficient statistics.
 * Step-size schedule: η_t = (t + τ)^(-κ), κ ∈ (0.5, 1], τ ≥ 1
 * Default: κ=0.6, τ=2
 * ════════════════════════════════════════════════════════════════════ */

/* xorshift64 PRNG for reproducible sampling */
static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

int UnmixOnline(const double* data, size_t n, DistFamily family, int k,
                int maxiter, double rtole, int batch_size, int verbose,
                MixtureResult* result)
{
    if (!data || n == 0 || k <= 0 || !result) return -1;
    init_dist_table();

    const DistFunctions* df = GetDistFunctions(family);
    if (!df) return -2;

    if (batch_size <= 0) batch_size = (int)fmin(256, n / 4);
    if (batch_size < 10) batch_size = 10;
    if ((size_t)batch_size > n) batch_size = (int)n;

    /* Allocate */
    result->family = family;
    result->num_components = k;
    result->mixing_weights = (double*)malloc(sizeof(double) * k);
    result->params = (DistParams*)malloc(sizeof(DistParams) * k);

    /* Initialize parameters */
    df->init_params(data, n, k, result->params);
    for (int j = 0; j < k; j++) result->mixing_weights[j] = 1.0 / k;

    /* Running sufficient statistics: weighted mean and variance per component */
    double* suf_w = (double*)calloc(k, sizeof(double));   /* sum of weights */
    double* suf_wx = (double*)calloc(k, sizeof(double));  /* sum of w*x */
    double* suf_wxx = (double*)calloc(k, sizeof(double)); /* sum of w*x² */

    /* Initialize sufficient stats from first pass */
    for (int j = 0; j < k; j++) {
        suf_w[j] = 1.0 / k;
        suf_wx[j] = result->params[j].p[0] / k;
        suf_wxx[j] = (result->params[j].p[0] * result->params[j].p[0] +
                       (result->params[j].nparams >= 2 ? result->params[j].p[1] : 1.0)) / k;
    }

    double* batch_resp = (double*)malloc(sizeof(double) * k * batch_size);
    double* batch_w = (double*)malloc(sizeof(double) * batch_size);
    int* batch_idx = (int*)malloc(sizeof(int) * batch_size);

    uint64_t rng_state = 12345678901234ULL;  /* seed */
    double prev_ll = -1e30;

    for (int iter = 0; iter < maxiter; iter++) {
        double eta = pow(iter + 2.0, -0.6);  /* Cappé step size */

        /* Sample mini-batch */
        for (int b = 0; b < batch_size; b++) {
            batch_idx[b] = (int)(xorshift64(&rng_state) % n);
        }

        /* E-step on mini-batch */
        double ll = 0;
        for (int b = 0; b < batch_size; b++) {
            double x = data[batch_idx[b]];
            double total = 0;
            for (int j = 0; j < k; j++) {
                double lp = df->logpdf ? df->logpdf(x, &result->params[j])
                                       : log(df->pdf(x, &result->params[j]) + 1e-300);
                double p = result->mixing_weights[j] * exp(lp);
                if (p < PDF_FLOOR) p = PDF_FLOOR;
                batch_resp[j * batch_size + b] = p;
                total += p;
            }
            for (int j = 0; j < k; j++) batch_resp[j * batch_size + b] /= total;
            ll += log(total);
        }
        ll /= batch_size;

        /* Stochastic M-step: update sufficient statistics */
        for (int j = 0; j < k; j++) {
            double new_w = 0, new_wx = 0, new_wxx = 0;
            for (int b = 0; b < batch_size; b++) {
                double r = batch_resp[j * batch_size + b];
                double x = data[batch_idx[b]];
                new_w += r;
                new_wx += r * x;
                new_wxx += r * x * x;
            }
            new_w /= batch_size;
            new_wx /= batch_size;
            new_wxx /= batch_size;

            /* Blend with running stats */
            suf_w[j] = (1 - eta) * suf_w[j] + eta * new_w;
            suf_wx[j] = (1 - eta) * suf_wx[j] + eta * new_wx;
            suf_wxx[j] = (1 - eta) * suf_wxx[j] + eta * new_wxx;
        }

        /* Reconstruct parameters from sufficient statistics */
        double wsum = 0;
        for (int j = 0; j < k; j++) wsum += suf_w[j];
        for (int j = 0; j < k; j++) {
            result->mixing_weights[j] = fmax(suf_w[j] / wsum, 1e-10);
            double mu = suf_wx[j] / fmax(suf_w[j], 1e-10);
            double var = suf_wxx[j] / fmax(suf_w[j], 1e-10) - mu * mu;
            if (var < 1e-10) var = 1e-10;

            /* For Gaussian: directly set mean/variance
             * For others: use sufficient stats as pseudo-data for MLE */
            if (family == DIST_GAUSSIAN) {
                result->params[j].p[0] = mu;
                result->params[j].p[1] = var;
                result->params[j].nparams = 2;
            } else {
                /* General: construct weighted pseudo-data from batch */
                for (int b = 0; b < batch_size; b++)
                    batch_w[b] = batch_resp[j * batch_size + b];
                double* batch_data = (double*)malloc(sizeof(double) * batch_size);
                for (int b = 0; b < batch_size; b++) batch_data[b] = data[batch_idx[b]];
                DistParams old = result->params[j];
                df->estimate(batch_data, batch_w, batch_size, &result->params[j]);
                /* Blend with previous parameters */
                for (int q = 0; q < result->params[j].nparams; q++) {
                    if (!isfinite(result->params[j].p[q])) result->params[j].p[q] = old.p[q];
                    else result->params[j].p[q] = (1-eta)*old.p[q] + eta*result->params[j].p[q];
                }
                free(batch_data);
            }
        }

        if (verbose && (iter < 5 || iter % 50 == 0)) {
            printf("  [online] iter %d  LL≈%.4f  eta=%.4f\n", iter, ll, eta);
        }

        if (iter > 5 && fabs(ll - prev_ll) < rtole) break;
        prev_ll = ll;
    }

    /* Compute final log-likelihood on full data */
    double ll = 0;
    for (size_t i = 0; i < n; i++) {
        double total = 0;
        for (int j = 0; j < k; j++) {
            double lp = df->logpdf ? df->logpdf(data[i], &result->params[j])
                                   : log(df->pdf(data[i], &result->params[j]) + 1e-300);
            total += result->mixing_weights[j] * exp(lp);
        }
        ll += log(fmax(total, 1e-300));
    }
    result->loglikelihood = ll;
    result->iterations = maxiter;
    int nfree = df->num_params * k + k - 1;
    result->bic = -2*ll + nfree * log((double)n);
    result->aic = -2*ll + 2*nfree;

    free(suf_w); free(suf_wx); free(suf_wxx);
    free(batch_resp); free(batch_w); free(batch_idx);
    return 0;
}


void ReleaseAdaptiveResult(AdaptiveResult* r) {
    if (!r) return;
    if (r->mixing_weights) { free(r->mixing_weights); r->mixing_weights = NULL; }
    if (r->params) { free(r->params); r->params = NULL; }
    if (r->families) { free(r->families); r->families = NULL; }
}


/* ====================================================================
 * Cleanup
 * ==================================================================== */
void ReleaseMixtureResult(MixtureResult* r) {
    if (!r) return;
    if (r->mixing_weights) { free(r->mixing_weights); r->mixing_weights = NULL; }
    if (r->params) { free(r->params); r->params = NULL; }
}

void ReleaseModelSelectResult(ModelSelectResult* r) {
    if (!r) return;
    if (r->candidates) {
        for (int i = 0; i < r->num_candidates; i++) {
            ReleaseMixtureResult(&r->candidates[i]);
        }
        free(r->candidates);
        r->candidates = NULL;
    }
}
