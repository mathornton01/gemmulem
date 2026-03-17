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

#include "distributions.h"
#include "pearson.h"

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
static double gamma_pdf(double x, const DistParams* p) {
    double a = p->p[0], b = p->p[1];
    if (x <= 0 || a <= 0 || b <= 0) return 0;
    return exp(a*log(b) + (a-1)*log(x) - b*x - lgamma(a));
}
static double gamma_logpdf(double x, const DistParams* p) {
    double a = p->p[0], b = p->p[1];
    if (x <= 0 || a <= 0 || b <= 0) return -1e30;
    return a*log(b) + (a-1)*log(x) - b*x - lgamma(a);
}
static void gamma_estimate(const double* x, const double* w, size_t n, DistParams* out) {
    double mu = wt_mean(x, w, n);
    double var = wt_var(x, w, n, mu);
    if (mu < 1e-10) mu = 1e-10;
    if (var < 1e-10) var = 1e-10;
    /* Method of moments: alpha = mu^2/var, beta = mu/var */
    double alpha = mu*mu / var;
    double beta = mu / var;
    if (alpha < 0.01) alpha = 0.01;
    if (beta < 0.01) beta = 0.01;
    out->p[0] = alpha; out->p[1] = beta; out->nparams = 2;
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
        out[j].p[0] = a; out[j].p[1] = b; out[j].nparams = 2;
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

    for (iter = 0; iter < maxiter; iter++) {
        /* ---- E-step: compute responsibilities ---- */
        double ll = 0.0;
        for (size_t i = 0; i < n; i++) {
            double total = 0;
            for (int j = 0; j < k; j++) {
                double p;
                if (df->logpdf) {
                    p = result->mixing_weights[j] * exp(df->logpdf(data[i], &result->params[j]));
                } else {
                    p = result->mixing_weights[j] * df->pdf(data[i], &result->params[j]);
                }
                if (p < PDF_FLOOR) p = PDF_FLOOR;
                resp[j * n + i] = p;
                total += p;
            }
            /* Normalize */
            for (int j = 0; j < k; j++) resp[j * n + i] /= total;
            ll += log(total);
        }

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
    }

    result->iterations = iter;
    result->loglikelihood = prev_ll;

    /* BIC = -2*LL + p*log(n), where p = k*(num_params + 1) - 1 */
    int num_free = k * (df->num_params + 1) - 1;
    result->bic = -2.0 * prev_ll + num_free * log((double)n);
    result->aic = -2.0 * prev_ll + 2.0 * num_free;

    free(resp);
    free(weights_j);

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
