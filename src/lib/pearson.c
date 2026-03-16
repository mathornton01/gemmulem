/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 *
 * Pearson distribution system implementation.
 * License: GPL v3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "pearson.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

/* ====================================================================
 * Type classification
 *
 * Uses the kappa criterion from Pearson (1916):
 *   kappa = beta1*(beta2+3)^2 / (4*(4*beta2-3*beta1)*(2*beta2-3*beta1-6))
 *
 * Boundary lines in (beta1, beta2) space:
 *   Type III line: beta2 = 3 + 1.5*beta1  (Gamma/chi-sq)
 *   Impossible region: beta2 < 1 + beta1
 * ==================================================================== */

PearsonType pearson_classify(double beta1, double beta2)
{
    /* Validity check: beta2 >= 1 + beta1 always */
    if (beta2 < 1.0 + beta1 - 0.05) return PEARSON_TYPE_UNDEFINED;
    if (beta1 < 0) beta1 = 0;

    /* Wide Normal zone: catches nearly-normal distributions from finite samples.
     * Sampling noise on 10K points gives beta1 ≈ 0.0003, beta2 ≈ 2.97–3.03.
     * Require meaningful skewness AND kurtosis deviation before leaving Normal. */
    if (beta1 < 0.05 && fabs(beta2 - 3.0) < 0.5) return PEARSON_TYPE_0;

    /* Symmetric (beta1 very small, kurtosis differs from Normal) */
    if (beta1 < 0.05) {
        return (beta2 < 3.0) ? PEARSON_TYPE_II : PEARSON_TYPE_VII;
    }

    /* Compute discriminant quantities */
    double denom1 = 4.0 * beta2 - 3.0 * beta1;
    double denom2 = 2.0 * beta2 - 3.0 * beta1 - 6.0;

    /* Type III line: beta2 ≈ 3 + 1.5*beta1 (gamma boundary) */
    if (fabs(denom2) < 0.05 * (1.0 + beta1)) return PEARSON_TYPE_III;

    /* Avoid division by zero */
    if (fabs(denom1) < 1e-10 || fabs(denom2) < 1e-10) return PEARSON_TYPE_IV;

    double kappa = beta1 * (beta2 + 3.0) * (beta2 + 3.0) /
                   (4.0 * denom1 * denom2);

    /* Classification by kappa */
    if (kappa < -0.05)  return PEARSON_TYPE_I;      /* Beta (bounded) */
    if (kappa < 0.05)   return PEARSON_TYPE_III;     /* Near Gamma boundary */
    if (kappa < 0.95)   return PEARSON_TYPE_IV;      /* Heavy-tailed skewed */
    if (kappa < 1.05)   return PEARSON_TYPE_V;       /* Inverse-Gamma */
    return PEARSON_TYPE_VI;                           /* Beta-prime */
}

const char* pearson_type_name(PearsonType type)
{
    switch (type) {
        case PEARSON_TYPE_0:   return "Normal (Type 0)";
        case PEARSON_TYPE_I:   return "Beta (Type I)";
        case PEARSON_TYPE_II:  return "Symmetric Beta (Type II)";
        case PEARSON_TYPE_III: return "Gamma (Type III)";
        case PEARSON_TYPE_IV:  return "Type IV";
        case PEARSON_TYPE_V:   return "Inv-Gamma (Type V)";
        case PEARSON_TYPE_VI:  return "Beta-prime (Type VI)";
        case PEARSON_TYPE_VII: return "Student-t (Type VII)";
        default: return "Undefined";
    }
}

/* ====================================================================
 * ODE coefficient computation from moments
 *
 * From Pearson (1916, p. 437):
 *   r = 6*(beta2 - beta1 - 1) / (2*beta2 - 3*beta1 - 6)
 *   a = sqrt(beta1) * sigma * (beta2 + 3) / (10*beta2 - 12*beta1 - 18) * sign
 *
 * (We work in centered coordinates x' = x - mu)
 * ==================================================================== */

/* Log of gamma function */
static double my_lgamma(double x) { return lgamma(x); }

/* Log of beta function B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b) */
static double my_lbeta(double a, double b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

int pearson_from_moments(double mu, double sigma, double beta1, double beta2,
                         PearsonParams* out)
{
    if (!out || sigma <= 0) return -1;

    memset(out, 0, sizeof(PearsonParams));
    out->mu = mu;
    out->sigma = sigma;
    out->beta1 = beta1;
    out->beta2 = beta2;
    out->type = pearson_classify(beta1, beta2);
    out->support_lo = -INFINITY;
    out->support_hi = INFINITY;
    out->valid = 0;

    double var = sigma * sigma;
    double gamma1 = (beta1 > 0) ? sqrt(beta1) : 0;

    /* Compute ODE coefficients (centered on mu) */
    double D = 10.0 * beta2 - 12.0 * beta1 - 18.0;
    if (fabs(D) < 1e-10) D = 1e-10;

    out->b2_ode = -(2.0 * beta2 - 3.0 * beta1 - 6.0) / D;
    out->b1 = -gamma1 * sigma * (beta2 + 3.0) / D;
    out->b0 = -var * (4.0 * beta2 - 3.0 * beta1) / D;
    out->a = out->b1;  /* a1 = b1 for centered form */

    /* Type-specific parameterization for closed-form PDF */
    switch (out->type) {
    case PEARSON_TYPE_0: {
        /* Normal(mu, sigma^2) */
        out->log_norm = -0.5 * log(2.0 * M_PI * var);
        out->valid = 1;
        break;
    }

    case PEARSON_TYPE_I:
    case PEARSON_TYPE_II: {
        /* Beta distribution on (a1, a2) where a1, a2 are roots */
        /* From discriminant: roots of b0 + b1*x + b2_ode*x^2 = 0 */
        double disc = out->b1 * out->b1 - 4.0 * out->b0 * out->b2_ode;
        if (disc < 0) { out->valid = 0; break; }
        double sq = sqrt(disc);
        double r1 = (-out->b1 - sq) / (2.0 * out->b2_ode);
        double r2 = (-out->b1 + sq) / (2.0 * out->b2_ode);
        if (r1 > r2) { double t = r1; r1 = r2; r2 = t; }

        out->support_lo = mu + r1;
        out->support_hi = mu + r2;

        /* m1, m2 shape parameters */
        double range = r2 - r1;
        if (range < 1e-10) { out->valid = 0; break; }
        double m1 = -(1.0 + out->a / (out->b2_ode * r1));
        double m2 = 1.0 + out->a / (out->b2_ode * r2);

        if (m1 <= -1.0 || m2 <= -1.0) { out->valid = 0; break; }

        /* Store as beta params: alpha = m1+1, beta = m2+1 */
        out->p.beta.alpha = m1 + 1.0;
        out->p.beta.beta = m2 + 1.0;
        if (out->p.beta.alpha <= 0 || out->p.beta.beta <= 0) { out->valid = 0; break; }

        out->log_norm = -(my_lbeta(out->p.beta.alpha, out->p.beta.beta) +
                         (out->p.beta.alpha + out->p.beta.beta - 1.0) * log(range));
        out->valid = 1;
        break;
    }

    case PEARSON_TYPE_III: {
        /* Gamma: shape = 4/beta1, rate = 2/(sigma*sqrt(beta1))
         * Require beta1 > 0.05 to avoid degenerate large-shape Gamma.
         * Small beta1 collapses to Normal. */
        if (beta1 < 0.05) {
            out->type = PEARSON_TYPE_0;
            out->log_norm = -0.5 * log(2.0 * M_PI * var);
            out->valid = 1;
            break;
        }
        double shape = 4.0 / beta1;   /* > 80 when beta1 < 0.05 -- blocked above */
        double scale_param = 0.5 * sigma * sqrt(beta1);
        if (scale_param < 1e-10) scale_param = 1e-10;
        double loc = mu - shape * scale_param;

        out->p.gamma.shape = shape;
        out->p.gamma.rate = 1.0 / scale_param;

        /* Determine skew direction */
        if (gamma1 >= 0) {
            out->support_lo = loc;
            out->support_hi = INFINITY;
        } else {
            /* Reflected gamma (negated) */
            out->support_lo = -INFINITY;
            out->support_hi = loc + 2.0 * (mu - loc);
        }

        /* Numerical stability check for lgamma */
        if (shape > 1e6) { out->valid = 0; break; }
        out->log_norm = shape * log(out->p.gamma.rate) - my_lgamma(shape);
        out->valid = 1;
        break;
    }

    case PEARSON_TYPE_IV: {
        /* Type IV: p(x) ∝ (1 + ((x-λ)/α)^2)^(-m) * exp(-ν*arctan((x-λ)/α))
         * Parameters from moments:
         *   r = 6*(beta2 - beta1 - 1) / (2*beta2 - 3*beta1 - 6)
         *   m = (r + 2) / 2
         *   nu involves gamma1
         */
        double denom2 = 2.0 * beta2 - 3.0 * beta1 - 6.0;
        if (fabs(denom2) < 1e-10) { out->valid = 0; break; }
        double r = 6.0 * (beta2 - beta1 - 1.0) / denom2;
        double m = (r + 2.0) / 2.0;
        if (m <= 0.5) { out->valid = 0; break; }

        /* nu and alpha */
        double nu = -r * (r - 2.0) * gamma1 / sqrt(16.0 * (r - 1.0) - beta1 * (r - 2.0) * (r - 2.0));
        double alpha2 = var * (16.0 * (r - 1.0) - beta1 * (r - 2.0) * (r - 2.0)) / 16.0;
        if (alpha2 <= 0) { out->valid = 0; break; }
        double alpha = sqrt(alpha2);

        /* location */
        double lambda = mu - sigma * nu / (2.0 * sqrt(r - 1.0 - beta1 * (r - 2.0) * (r - 2.0) / 16.0));

        out->p.type_iv.m = m;
        out->p.type_iv.nu = nu;
        out->p.type_iv.alpha = alpha;
        out->p.type_iv.lambda = lambda;

        /* Normalization: involves complex gamma function
         * log(C) = log(|Gamma(m + i*nu/2)|) - m*log(alpha) - log(B(m-1/2, 1/2)) */
        /* Use approximation: |Gamma(m + iy)|^2 = pi*y / (sinh(pi*y)) * prod */
        /* For now, use numerical integration to get normalization */
        /* Compute by quadrature over [-50*sigma, 50*sigma] */
        double sum = 0;
        double step = sigma * 0.01;
        for (double x = lambda - 50.0*sigma; x <= lambda + 50.0*sigma; x += step) {
            double z = (x - lambda) / alpha;
            double lp = -m * log(1.0 + z*z) - nu * atan(z);
            sum += exp(lp) * step;
        }
        out->log_norm = (sum > 0) ? -log(sum) : 0;
        out->valid = (sum > 0);
        break;
    }

    case PEARSON_TYPE_V: {
        /* Inverse-Gamma: shape and scale from moments */
        /* For InvGamma(alpha, beta): mean = beta/(alpha-1), var = beta^2/((alpha-1)^2*(alpha-2)) */
        /* Requires alpha > 2 for finite variance */
        if (beta2 <= 3.0) { out->valid = 0; break; }

        /* From moments of inverse-gamma: beta2 = 3 + 6/(alpha-3) for alpha > 4 */
        double alpha = 3.0 + 6.0 / (beta2 - 3.0);
        if (alpha <= 2.0) { out->valid = 0; break; }
        double beta_param = mu * (alpha - 1.0);

        out->p.inv_gamma.shape = alpha;
        out->p.inv_gamma.scale = beta_param;
        out->support_lo = 0;
        out->log_norm = alpha * log(beta_param) - my_lgamma(alpha);
        out->valid = (beta_param > 0);
        break;
    }

    case PEARSON_TYPE_VI: {
        /* Beta-prime: ratio of two chi-squared variables
         * F(d1, d2) = (d1/d2) * BetaPrime(d1/2, d2/2)
         * For now, use moment-based parameterization */
        /* mean = d2/(d2-2), var = 2*d2^2*(d1+d2-2)/(d1*(d2-2)^2*(d2-4)) */
        /* This is complex — use numerical Type IV machinery as fallback */
        /* Treat as Type IV with appropriate parameters */
        double denom2 = 2.0 * beta2 - 3.0 * beta1 - 6.0;
        if (fabs(denom2) < 1e-10) { out->valid = 0; break; }

        /* Fall through to Type IV-style computation with numerical normalization */
        double r = 6.0 * (beta2 - beta1 - 1.0) / denom2;
        double disc = out->b1 * out->b1 - 4.0 * out->b0 * out->b2_ode;
        if (disc < 0) {
            /* Should have positive discriminant for Type VI */
            out->valid = 0;
            break;
        }
        double sq = sqrt(disc);
        double r1 = (-out->b1 - sq) / (2.0 * out->b2_ode);
        double r2 = (-out->b1 + sq) / (2.0 * out->b2_ode);
        /* Both roots same sign for Type VI */
        if (r1 > r2) { double t = r1; r1 = r2; r2 = t; }

        /* Power-form PDF: p(x) ∝ (x - a1)^m1 * (a2 - x)^m2 */
        double m1 = -(1.0 + out->a / (out->b2_ode * r1));
        double m2 = 1.0 + out->a / (out->b2_ode * r2);

        out->p.beta_prime.d1 = 2.0 * (m1 + 1.0);
        out->p.beta_prime.d2 = 2.0 * (m2 + 1.0);
        out->support_lo = mu + r1;
        out->support_hi = INFINITY;

        /* Numerical normalization */
        double sum = 0;
        double step = sigma * 0.01;
        for (double x = out->support_lo + step*0.5; x < mu + 50*sigma; x += step) {
            double y1 = x - out->support_lo;
            double y2 = (mu + r2) - x;
            if (y1 > 0 && y2 > 0) {
                double lp = m1 * log(y1) + m2 * log(y2);
                sum += exp(lp) * step;
            }
        }
        out->log_norm = (sum > 0) ? -log(sum) : 0;
        out->valid = (sum > 0);
        break;
    }

    case PEARSON_TYPE_VII: {
        /* Student-t: beta1=0, beta2 > 3 */
        /* df = 6 / (beta2 - 3) + 4  for beta2 in (3, 9) giving df > 4 */
        if (beta2 <= 3.0) { out->valid = 0; break; }
        double df = 2.0 * (5.0 * beta2 - 9.0) / (beta2 - 3.0);
        if (df <= 0) df = 1.0;  /* Cauchy limit */

        out->p.student_t.df = df;
        /* Non-standardized t: scaled by sigma */
        /* scale = sigma * sqrt((df-2)/df) for df > 2 */
        double scale = (df > 2) ? sigma * sqrt((df - 2.0) / df) : sigma;
        out->log_norm = my_lgamma(0.5*(df+1.0)) - my_lgamma(0.5*df)
                       - 0.5*log(df*M_PI) - log(scale);
        out->valid = 1;
        break;
    }

    default:
        out->valid = 0;
        break;
    }

    return out->valid ? 0 : -1;
}


/* ====================================================================
 * PDF evaluation
 * ==================================================================== */

double pearson_logpdf(double x, const PearsonParams* params)
{
    if (!params || !params->valid) return -1e30;

    switch (params->type) {
    case PEARSON_TYPE_0: {
        double z = (x - params->mu) / params->sigma;
        return params->log_norm - 0.5 * z * z;
    }

    case PEARSON_TYPE_I:
    case PEARSON_TYPE_II: {
        if (x <= params->support_lo || x >= params->support_hi) return -1e30;
        double range = params->support_hi - params->support_lo;
        double y = (x - params->support_lo) / range;
        double a = params->p.beta.alpha;
        double b = params->p.beta.beta;
        return params->log_norm + (a - 1.0)*log(y) + (b - 1.0)*log(1.0 - y);
    }

    case PEARSON_TYPE_III: {
        double shape = params->p.gamma.shape;
        double rate = params->p.gamma.rate;
        double loc = params->mu - shape / rate;
        double y = x - loc;
        if (y <= 0) return -1e30;
        return params->log_norm + (shape - 1.0)*log(y) - rate*y;
    }

    case PEARSON_TYPE_IV: {
        double m = params->p.type_iv.m;
        double nu = params->p.type_iv.nu;
        double alpha = params->p.type_iv.alpha;
        double lambda = params->p.type_iv.lambda;
        double z = (x - lambda) / alpha;
        return params->log_norm - m * log(1.0 + z*z) - nu * atan(z);
    }

    case PEARSON_TYPE_V: {
        if (x <= 0) return -1e30;
        double a = params->p.inv_gamma.shape;
        double b = params->p.inv_gamma.scale;
        return params->log_norm - (a + 1.0)*log(x) - b/x;
    }

    case PEARSON_TYPE_VI: {
        if (x <= params->support_lo) return -1e30;
        /* Use stored numerical normalization with ODE-form */
        double y = x - params->mu;  /* centered */
        double qval = params->b0 + params->b1*y + params->b2_ode*y*y;
        if (qval <= 0) return -1e30;
        /* Integrate ODE: log p = integral of -(a + y)/(b0 + b1*y + b2*y^2) dy */
        /* For numerical stability, use the power form */
        double disc = params->b1*params->b1 - 4.0*params->b0*params->b2_ode;
        if (disc < 0) return -1e30;
        double sq = sqrt(disc);
        double r1 = (-params->b1 - sq) / (2.0*params->b2_ode);
        double r2 = (-params->b1 + sq) / (2.0*params->b2_ode);
        if (r1 > r2) { double t = r1; r1 = r2; r2 = t; }
        double m1 = -(1.0 + params->a / (params->b2_ode * r1));
        double m2 = 1.0 + params->a / (params->b2_ode * r2);
        double y1 = (x - params->mu) - r1;
        double y2 = r2 - (x - params->mu);
        if (y1 <= 0 || y2 <= 0) return -1e30;
        return params->log_norm + m1*log(y1) + m2*log(y2);
    }

    case PEARSON_TYPE_VII: {
        double df = params->p.student_t.df;
        double scale = (df > 2) ? params->sigma * sqrt((df - 2.0)/df) : params->sigma;
        double z = (x - params->mu) / scale;
        return params->log_norm - 0.5*(df+1.0)*log(1.0 + z*z/df);
    }

    default:
        return -1e30;
    }
}

double pearson_pdf(double x, const PearsonParams* params)
{
    double lp = pearson_logpdf(x, params);
    return (lp > -700) ? exp(lp) : 0.0;
}


/* ====================================================================
 * Moment estimation from weighted data (M-step)
 * ==================================================================== */

int pearson_estimate(const double* data, const double* weights, size_t n,
                     PearsonParams* out)
{
    if (!data || !weights || n == 0 || !out) return -1;

    /* Compute weighted moments using a two-pass approach:
     * Pass 1: compute mean and variance (robust).
     * Pass 2: compute higher moments with Winsorization — cap data
     *   influence at 4 sigma from the mean to prevent distant
     *   low-responsibility points from inflating beta1/beta2. */
    double sw = 0;
    for (size_t i = 0; i < n; i++) sw += weights[i];
    if (sw <= 0) return -1;

    /* Robust moment estimation for EM context.
     *
     * Problem: In EM, each component has responsibility weights on ALL data
     * points. Low-responsibility distant points (e.g., 3% weight on a point
     * 10σ away) massively inflate higher moments. Standard moment-based
     * Pearson estimation breaks.
     *
     * Solution: Compute moments ONLY from "core" points — those with
     * responsibility above a threshold. This mimics what the component
     * would look like if it only owned its own cluster. */

    /* Mean (from all weights — this is well-behaved) */
    double mu = 0;
    for (size_t i = 0; i < n; i++) mu += weights[i] * data[i];
    mu /= sw;

    /* Identify core points: responsibility > mean responsibility.
     * Mean responsibility = sw / n.  Core = above 0.1 * mean. */
    double mean_resp = sw / n;
    double threshold = mean_resp * 0.1;
    if (threshold < 1e-10) threshold = 1e-10;

    /* Pass 1: Robust variance from core points (resp > threshold) */
    double m2_core = 0, sw_core = 0;
    for (size_t i = 0; i < n; i++) {
        if (weights[i] < threshold) continue;
        double d = data[i] - mu;
        m2_core += weights[i] * d * d;
        sw_core += weights[i];
    }
    if (sw_core < sw * 0.3) {
        m2_core = 0; sw_core = 0;
        for (size_t i = 0; i < n; i++) {
            m2_core += weights[i] * (data[i]-mu)*(data[i]-mu);
            sw_core += weights[i];
        }
    }
    double sigma_est = sqrt(m2_core / sw_core);
    if (sigma_est < 1e-10) sigma_est = 1e-10;

    /* Pass 2: ALL moments with Winsorization at ±3σ from mean.
     * This ensures higher moments reflect the component's local shape,
     * not contamination from distant points owned by other components. */
    double clip = 3.0 * sigma_est;
    double m2 = 0, m3 = 0, m4 = 0;
    double sw_clip = 0;
    for (size_t i = 0; i < n; i++) {
        double d = data[i] - mu;
        double w = weights[i];
        /* Hard-clip: points beyond ±3σ contribute at the boundary */
        if (d > clip) d = clip;
        if (d < -clip) d = -clip;
        double d2 = d * d;
        m2 += w * d2;
        m3 += w * d2 * d;
        m4 += w * d2 * d2;
        sw_clip += w;
    }
    m2 /= sw;
    m3 /= sw;
    m4 /= sw;

    double sigma = sqrt(m2);
    if (sigma < 1e-10) sigma = 1e-10;

    /* beta1 = gamma1^2 = m3^2 / m2^3 */
    double beta1 = (m2 > 1e-20) ? (m3 * m3) / (m2 * m2 * m2) : 0.0;

    /* beta2 = m4 / m2^2  (traditional kurtosis) */
    double beta2 = (m2 > 1e-20) ? m4 / (m2 * m2) : 3.0;

    /* Clamp to valid region: beta2 >= 1 + beta1 */
    if (beta2 < 1.0 + beta1 + 0.05) beta2 = 1.0 + beta1 + 0.05;

    /* Hard cap: avoid numerically extreme distributions.
     * Beta1 > 10 → extremely skewed; cap to 10.
     * Beta2 > 20 → very heavy-tailed; cap to 20. */
    if (beta1 > 10.0) beta1 = 10.0;
    if (beta2 > 20.0) beta2 = 20.0;

    /* Regularize toward Normal based on effective sample count.
     * The less data assigned to a component, the more we trust
     * the prior (Normal) over noisy higher moment estimates. */
    {
        double reg_strength;
        if (sw < 5.0)        reg_strength = 0.95;
        else if (sw < 20.0)  reg_strength = 0.7;
        else if (sw < 100.0) reg_strength = 0.4;
        else if (sw < 500.0) reg_strength = 0.1;
        else                 reg_strength = 0.0;

        beta1 = beta1 * (1.0 - reg_strength);
        beta2 = beta2 * (1.0 - reg_strength) + 3.0 * reg_strength;
    }

    return pearson_from_moments(mu, sigma, beta1, beta2, out);
}


/* ====================================================================
 * Register Pearson as a DistFamily in the distributions framework
 * ==================================================================== */
#include "distributions.h"

/* Adapter functions matching the DistFunctions interface */

static double pearson_dist_pdf(double x, const DistParams* p) {
    PearsonParams pp;
    if (pearson_from_moments(p->p[0], p->p[1], p->p[2], p->p[3], &pp) < 0) return 0;
    return pearson_pdf(x, &pp);
}

static double pearson_dist_logpdf(double x, const DistParams* p) {
    PearsonParams pp;
    if (pearson_from_moments(p->p[0], p->p[1], p->p[2], p->p[3], &pp) < 0) {
        /* Fallback to Normal */
        double mu = p->p[0], sigma = p->p[1];
        if (sigma <= 0) sigma = 1.0;
        double z = (x - mu) / sigma;
        return -0.5*log(2*3.14159265358979*sigma*sigma) - 0.5*z*z;
    }
    return pearson_logpdf(x, &pp);
}

static void pearson_dist_estimate(const double* data, const double* weights,
                                   size_t n, DistParams* out) {
    PearsonParams pp;
    if (pearson_estimate(data, weights, n, &pp) == 0) {
        out->p[0] = pp.mu;
        out->p[1] = pp.sigma;
        out->p[2] = pp.beta1;
        out->p[3] = pp.beta2;
    } else {
        /* Fallback to Normal */
        double sw = 0, sm = 0;
        for (size_t i = 0; i < n; i++) { sw += weights[i]; sm += weights[i]*data[i]; }
        double mu2 = sw > 0 ? sm/sw : 0;
        double sv = 0;
        for (size_t i = 0; i < n; i++) sv += weights[i]*(data[i]-mu2)*(data[i]-mu2);
        double sig = sw > 0 ? sqrt(sv/sw) : 1;
        out->p[0] = mu2; out->p[1] = sig; out->p[2] = 0; out->p[3] = 3.0;
    }
    out->nparams = 4;
}

static void pearson_dist_init(const double* data, size_t n, int k,
                               DistParams* out) {
    /* Initialization using evenly-spaced quantiles of the data.
     * Sample up to 2000 points to get approximate quantiles efficiently. */
    size_t ns = (n > 2000) ? 2000 : n;
    double* sorted = (double*)malloc(sizeof(double) * ns);
    /* Evenly subsample */
    for (size_t i = 0; i < ns; i++) sorted[i] = data[i * n / ns];
    /* Sort the sample */
    for (size_t i = 0; i < ns-1; i++) {
        for (size_t j = i+1; j < ns; j++) {
            if (sorted[j] < sorted[i]) {
                double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
            }
        }
    }

    double mu = 0; for (size_t i = 0; i < n; i++) mu += data[i]; mu /= n;
    double var = 0; for (size_t i = 0; i < n; i++) var += (data[i]-mu)*(data[i]-mu); var /= n;
    double sigma = sqrt(var); if (sigma < 1e-10) sigma = 1.0;

    /* Component sigma = global sigma / sqrt(k) for non-overlapping init */
    double comp_sigma = sigma / sqrt((double)k);
    if (comp_sigma < sigma * 0.2) comp_sigma = sigma * 0.2;

    /* Place means at evenly-spaced quantiles of the sorted sample */
    for (int j = 0; j < k; j++) {
        size_t qi = (size_t)((j + 0.5) * ns / k);
        if (qi >= ns) qi = ns - 1;
        out[j].p[0] = sorted[qi];      /* mu at quantile */
        out[j].p[1] = comp_sigma;      /* sigma */
        out[j].p[2] = 0.0;             /* beta1 — start Normal */
        out[j].p[3] = 3.0;             /* beta2 — start Normal */
        out[j].nparams = 4;
    }

    free(sorted);
}

static int pearson_dist_valid(double x) { return 1; }

/* This function is called from distributions.c to register Pearson */
DistFunctions pearson_get_dist_functions(void) {
    DistFunctions df;
    df.family = DIST_PEARSON;
    df.name = "Pearson";
    df.num_params = 4;  /* mu, sigma, beta1, beta2 */
    df.pdf = pearson_dist_pdf;
    df.logpdf = pearson_dist_logpdf;
    df.estimate = pearson_dist_estimate;
    df.init_params = pearson_dist_init;
    df.valid = pearson_dist_valid;
    return df;
}
