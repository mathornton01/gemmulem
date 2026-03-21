/*
 * Tests for complex-valued Gaussian mixture EM.
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * License: GPL v3
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../src/lib/complex_em.h"
#include "../src/lib/simd_complex_estep.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    if (fabs((a)-(b)) > (tol)) { \
        printf("  FAIL: %s: %.6f != %.6f (tol=%.6f, line %d)\n", msg, (double)(a), (double)(b), (double)(tol), __LINE__); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* ── Box-Muller for generating complex Gaussian samples ── */
static void box_muller(double* out_re, double* out_im) {
    double u1 = ((double)rand()+1.0) / (RAND_MAX+1.0);
    double u2 = ((double)rand()+1.0) / (RAND_MAX+1.0);
    double r = sqrt(-2.0 * log(u1));
    *out_re = r * cos(2.0*M_PI*u2);
    *out_im = r * sin(2.0*M_PI*u2);
}

/* Generate n samples from circular complex Gaussian CN(mu, sigma²) */
static void gen_circular(double* data, size_t n,
                         double mu_re, double mu_im, double sigma2) {
    double s = sqrt(sigma2 / 2.0); /* Each component gets half the variance */
    for (size_t i = 0; i < n; i++) {
        double re, im;
        box_muller(&re, &im);
        data[2*i]   = mu_re + s * re;
        data[2*i+1] = mu_im + s * im;
    }
}

/* Generate n samples from non-circular complex Gaussian with pseudo-covariance */
static void gen_noncircular(double* data, size_t n,
                            double mu_re, double mu_im,
                            double sigma2, double pcov_re, double pcov_im) {
    /* Generate via augmented real representation:
     * [Re(z); Im(z)] ~ N([mu_re; mu_im], Sigma_real)
     * where Sigma_real = 0.5 * [[sigma2+pcov_re, pcov_im],
     *                            [pcov_im, sigma2-pcov_re]]
     */
    double a11 = 0.5 * (sigma2 + pcov_re);
    double a12 = 0.5 * pcov_im;
    double a22 = 0.5 * (sigma2 - pcov_re);

    /* Cholesky: L such that L*L^T = Sigma_real */
    double l11 = sqrt(a11);
    double l21 = a12 / l11;
    double l22 = sqrt(a22 - l21*l21);

    for (size_t i = 0; i < n; i++) {
        double z1, z2;
        box_muller(&z1, &z2);
        data[2*i]   = mu_re + l11*z1;
        data[2*i+1] = mu_im + l21*z1 + l22*z2;
    }
}


/* ════════════════════════════════════════════════════════════════════
 * Test: Circular PDF normalization
 * ════════════════════════════════════════════════════════════════════ */
static void test_circular_pdf(void) {
    printf("test_circular_pdf...\n");
    CCircGaussParams p = {0.0, 0.0, 1.0};

    /* PDF at origin should be 1/π ≈ 0.31831 */
    double pdf0 = ccirc_gauss_pdf(0.0, 0.0, &p);
    ASSERT_NEAR(pdf0, 1.0/M_PI, 1e-6, "PDF at origin for CN(0,1)");

    /* PDF should decrease with distance */
    double pdf1 = ccirc_gauss_pdf(1.0, 0.0, &p);
    ASSERT(pdf1 < pdf0, "PDF decreases with distance");
    ASSERT_NEAR(pdf1, exp(-1.0)/M_PI, 1e-6, "PDF at (1,0) for CN(0,1)");

    /* Log PDF consistency */
    double logp = ccirc_gauss_logpdf(0.5, 0.3, &p);
    double p_direct = ccirc_gauss_pdf(0.5, 0.3, &p);
    ASSERT_NEAR(logp, log(p_direct), 1e-10, "logpdf == log(pdf)");
}


/* ════════════════════════════════════════════════════════════════════
 * Test: Single circular Gaussian recovery
 * ════════════════════════════════════════════════════════════════════ */
static void test_single_circular(void) {
    printf("test_single_circular...\n");
    srand(42);

    double mu_re = 3.0, mu_im = -2.0, var = 4.0;
    size_t n = 5000;
    double* data = (double*)malloc(2*n*sizeof(double));
    gen_circular(data, n, mu_re, mu_im, var);

    CCircMixtureResult result = {0};
    int ret = UnmixComplexCircular(data, n, 1, 200, 1e-8, 0, &result);

    ASSERT(ret == 0, "returns success");
    ASSERT(result.num_components == 1, "k=1");
    ASSERT_NEAR(result.components[0].mu_re, mu_re, 0.2, "mean_re recovery");
    ASSERT_NEAR(result.components[0].mu_im, mu_im, 0.2, "mean_im recovery");
    ASSERT_NEAR(result.components[0].var, var, 0.5, "variance recovery");

    ReleaseCCircResult(&result);
    free(data);
}


/* ════════════════════════════════════════════════════════════════════
 * Test: Two well-separated circular Gaussians
 * ════════════════════════════════════════════════════════════════════ */
static void test_two_circular(void) {
    printf("test_two_circular...\n");
    srand(123);

    size_t n = 8000;
    double* data = (double*)malloc(2*n*sizeof(double));

    /* Component 1: CN(5+3i, 1.0), 60% */
    gen_circular(data, (size_t)(n*0.6), 5.0, 3.0, 1.0);
    /* Component 2: CN(-3-2i, 2.0), 40% */
    gen_circular(data + 2*(size_t)(n*0.6), (size_t)(n*0.4), -3.0, -2.0, 2.0);

    CCircMixtureResult result = {0};
    int ret = UnmixComplexCircular(data, n, 2, 300, 1e-8, 0, &result);

    ASSERT(ret == 0, "returns success");
    ASSERT(result.num_components == 2, "k=2");

    /* Sort by mu_re to identify components */
    int c1 = (result.components[0].mu_re < result.components[1].mu_re) ? 0 : 1;
    int c2 = 1 - c1;

    ASSERT_NEAR(result.components[c1].mu_re, -3.0, 0.3, "comp1 mean_re");
    ASSERT_NEAR(result.components[c1].mu_im, -2.0, 0.3, "comp1 mean_im");
    ASSERT_NEAR(result.components[c1].var, 2.0, 0.5, "comp1 variance");

    ASSERT_NEAR(result.components[c2].mu_re, 5.0, 0.3, "comp2 mean_re");
    ASSERT_NEAR(result.components[c2].mu_im, 3.0, 0.3, "comp2 mean_im");
    ASSERT_NEAR(result.components[c2].var, 1.0, 0.3, "comp2 variance");

    ASSERT_NEAR(result.mixing_weights[c1], 0.4, 0.05, "weight1");
    ASSERT_NEAR(result.mixing_weights[c2], 0.6, 0.05, "weight2");

    ReleaseCCircResult(&result);
    free(data);
}


/* ════════════════════════════════════════════════════════════════════
 * Test: Auto-k selection (circular)
 * ════════════════════════════════════════════════════════════════════ */
static void test_autok_circular(void) {
    printf("test_autok_circular...\n");
    srand(77);

    size_t n = 6000;
    double* data = (double*)malloc(2*n*sizeof(double));

    /* 3 well-separated components */
    gen_circular(data,                            2000, 10.0, 0.0, 1.0);
    gen_circular(data + 2*2000,                   2000, -10.0, 0.0, 1.0);
    gen_circular(data + 2*4000,                   2000, 0.0, 15.0, 1.0);

    CCircMixtureResult result = {0};
    int ret = UnmixComplexCircularAutoK(data, n, 6, 200, 1e-8, 0, &result);

    ASSERT(ret == 0, "returns success");
    ASSERT(result.num_components == 3, "auto-k finds k=3");

    ReleaseCCircResult(&result);
    free(data);
}


/* ════════════════════════════════════════════════════════════════════
 * Test: Non-circular PDF
 * ════════════════════════════════════════════════════════════════════ */
static void test_noncircular_pdf(void) {
    printf("test_noncircular_pdf...\n");

    /* When pseudo-covariance is 0, should match circular */
    CNonCircGaussParams nc = {0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    CCircGaussParams c = {0.0, 0.0, 1.0};

    double nc_pdf = cnocirc_gauss_pdf(0.5, 0.3, &nc);
    double c_pdf = ccirc_gauss_pdf(0.5, 0.3, &c);
    ASSERT_NEAR(nc_pdf, c_pdf, 1e-10, "Non-circular with C=0 matches circular");

    /* Non-zero pseudo-covariance should give different results */
    CNonCircGaussParams nc2 = {0.0, 0.0, 2.0, 0.0, 1.0, 0.0};
    double nc2_pdf_a = cnocirc_gauss_pdf(1.0, 0.0, &nc2);
    double nc2_pdf_b = cnocirc_gauss_pdf(0.0, 1.0, &nc2);
    /* With C_re > 0, real axis should be stretched → PDF at (1,0) ≠ PDF at (0,1) */
    ASSERT(fabs(nc2_pdf_a - nc2_pdf_b) > 0.01, "Non-circular breaks rotational symmetry");
}


/* ════════════════════════════════════════════════════════════════════
 * Test: Non-circular parameter recovery
 * ════════════════════════════════════════════════════════════════════ */
static void test_noncircular_recovery(void) {
    printf("test_noncircular_recovery...\n");
    srand(999);

    double mu_re = 1.0, mu_im = 2.0;
    double sigma2 = 3.0;
    double pcov_re = 1.5, pcov_im = 0.5;

    size_t n = 10000;
    double* data = (double*)malloc(2*n*sizeof(double));
    gen_noncircular(data, n, mu_re, mu_im, sigma2, pcov_re, pcov_im);

    CNonCircMixtureResult result = {0};
    int ret = UnmixComplexNonCircular(data, n, 1, 300, 1e-8, 0, &result);

    ASSERT(ret == 0, "returns success");
    ASSERT_NEAR(result.components[0].mu_re, mu_re, 0.2, "mean_re");
    ASSERT_NEAR(result.components[0].mu_im, mu_im, 0.2, "mean_im");
    ASSERT_NEAR(result.components[0].cov_re, sigma2, 0.5, "variance");
    ASSERT_NEAR(result.components[0].pcov_re, pcov_re, 0.5, "pseudo-cov real");
    ASSERT_NEAR(result.components[0].pcov_im, pcov_im, 0.5, "pseudo-cov imag");

    ReleaseCNonCircResult(&result);
    free(data);
}


/* ════════════════════════════════════════════════════════════════════
 * Test: Edge cases
 * ════════════════════════════════════════════════════════════════════ */
static void test_edge_cases(void) {
    printf("test_edge_cases...\n");

    /* Single observation */
    double data1[2] = {1.0, 2.0};
    CCircMixtureResult r1 = {0};
    int ret = UnmixComplexCircular(data1, 1, 1, 100, 1e-6, 0, &r1);
    ASSERT(ret == 0, "single observation succeeds");
    ASSERT_NEAR(r1.components[0].mu_re, 1.0, 1e-6, "single obs mu_re");
    ASSERT_NEAR(r1.components[0].mu_im, 2.0, 1e-6, "single obs mu_im");
    ReleaseCCircResult(&r1);

    /* Null/invalid inputs */
    CCircMixtureResult r2 = {0};
    ASSERT(UnmixComplexCircular(NULL, 100, 1, 100, 1e-6, 0, &r2) < 0, "null data returns error");
    ASSERT(UnmixComplexCircular(data1, 0, 1, 100, 1e-6, 0, &r2) < 0, "n=0 returns error");
    ASSERT(UnmixComplexCircular(data1, 1, 0, 100, 1e-6, 0, &r2) < 0, "k=0 returns error");

    /* Identical observations */
    double data_ident[10] = {5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0};
    CCircMixtureResult r3 = {0};
    ret = UnmixComplexCircular(data_ident, 5, 1, 100, 1e-6, 0, &r3);
    ASSERT(ret == 0, "identical observations succeed");
    ASSERT_NEAR(r3.components[0].mu_re, 5.0, 1e-4, "identical mu_re");
    ASSERT_NEAR(r3.components[0].mu_im, 3.0, 1e-4, "identical mu_im");
    ReleaseCCircResult(&r3);
}


/* ════════════════════════════════════════════════════════════════════
 * Test: IQ signal scenario (QPSK-like)
 * ════════════════════════════════════════════════════════════════════ */
static void test_qpsk_scenario(void) {
    printf("test_qpsk_scenario...\n");
    srand(2026);

    /* 4 constellation points: (1,1), (1,-1), (-1,1), (-1,-1) with noise */
    size_t n = 8000;
    double* data = (double*)malloc(2*n*sizeof(double));
    double noise_var = 0.1;

    gen_circular(data,              2000,  1.0,  1.0, noise_var);
    gen_circular(data + 2*2000,     2000,  1.0, -1.0, noise_var);
    gen_circular(data + 2*4000,     2000, -1.0,  1.0, noise_var);
    gen_circular(data + 2*6000,     2000, -1.0, -1.0, noise_var);

    CCircMixtureResult result = {0};
    int ret = UnmixComplexCircular(data, n, 4, 300, 1e-8, 0, &result);

    ASSERT(ret == 0, "QPSK succeeds");
    ASSERT(result.num_components == 4, "k=4");

    /* Verify all 4 constellation points recovered */
    int found[4] = {0,0,0,0};
    double targets[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    for (int j = 0; j < 4; j++) {
        for (int t = 0; t < 4; t++) {
            double dr = result.components[j].mu_re - targets[t][0];
            double di = result.components[j].mu_im - targets[t][1];
            if (dr*dr+di*di < 0.25) found[t] = 1;
        }
    }
    ASSERT(found[0] && found[1] && found[2] && found[3], "all 4 QPSK points recovered");

    /* Weights should be ~0.25 each */
    for (int j = 0; j < 4; j++) {
        ASSERT_NEAR(result.mixing_weights[j], 0.25, 0.05, "QPSK weight ~0.25");
    }

    ReleaseCCircResult(&result);
    free(data);
}


/* ════════════════════════════════════════════════════════════════════
 * Feature 1 tests: Multivariate Complex Gaussian
 * ════════════════════════════════════════════════════════════════════ */

/* Generate d-dimensional circular complex Gaussian samples.
 * mean: [2*d], cov_chol: lower Cholesky [2*d*d] (complex, stored interleaved).
 * Output: n*2*d doubles (row-major, interleaved re/im per dim). */
static void gen_mv_complex(double* data, size_t n, int d,
                            const double* mean, const double* cov_chol) {
    for (size_t i = 0; i < n; i++) {
        /* Generate d independent CN(0,1) and transform */
        double* z = data + i * 2 * d;
        double white[64*2]; /* temp: d CN(0,1) samples */
        for (int dd = 0; dd < d; dd++) {
            double re, im;
            box_muller(&re, &im);
            /* box_muller gives N(0,1); scale to N(0,0.5) for CN(0,1) */
            white[2*dd]   = re * 0.70710678118654752;  /* 1/sqrt(2) */
            white[2*dd+1] = im * 0.70710678118654752;
        }
        /* Apply lower Cholesky: z = mean + L * white */
        for (int p = 0; p < d; p++) {
            double re = mean[2*p], im = mean[2*p+1];
            for (int q = 0; q <= p; q++) {
                double lr = cov_chol[2*(p*d+q)];
                double li = cov_chol[2*(p*d+q)+1];
                double wr = white[2*q];
                double wi = white[2*q+1];
                re += lr*wr - li*wi;
                im += lr*wi + li*wr;
            }
            z[2*p]   = re;
            z[2*p+1] = im;
        }
    }
}

static void test_mv_complex_cholesky(void) {
    printf("test_mv_complex_cholesky...\n");

    /* Known 2×2 Hermitian positive-definite matrix:
     * A = [ 4    2+i ]
     *     [ 2-i  5   ]
     * det(A) = 4*5 - (4+1) = 15, positive definite
     * Expected L:
     *   L[0,0] = sqrt(4) = 2 (real)
     *   L[1,0] = (2-i) / 2 = 1 - 0.5i
     *   L[1,1] = sqrt(5 - |1-0.5i|^2) = sqrt(5 - 1.25) = sqrt(3.75)
     */
    /* We test via UnmixMVComplex on trivial data with known covariance */
    /* Build Cholesky by hand and verify via log_det */
    double L00 = sqrt(4.0);          /* 2.0 */
    double L10r = 2.0 / L00;         /* 1.0 */
    double L10i = -1.0 / L00;        /* -0.5 */
    double L11 = sqrt(5.0 - (L10r*L10r + L10i*L10i)); /* sqrt(5 - 1.25) */
    double expected_logdet = 2.0 * (log(L00) + log(L11));


    /* Generate data from this distribution and run EM (k=1) */
    srand(2026);
    int d = 2;
    size_t n = 2000;
    double mean_true[4] = {1.0, 0.5, -0.5, 1.0};
    double chol_true[8] = {
        L00, 0.0,   0.0, 0.0,     /* row 0: [L00, 0] */
        L10r, L10i,  L11, 0.0     /* row 1: [L10, L11] */
    };

    double* obs = (double*)malloc(n * 2 * d * sizeof(double));
    gen_mv_complex(obs, n, d, mean_true, chol_true);

    MVComplexMixtureResult result = {0};
    int ret = UnmixMVComplex(obs, n, d, 1, 200, 1e-8, 0, &result);

    ASSERT(ret == 0, "MV complex Cholesky: EM returns success");
    ASSERT(result.num_components == 1, "MV complex Cholesky: k=1");

    /* Check that log_det is approximately correct */
    double got_logdet = result.components[0].log_det;
    ASSERT_NEAR(got_logdet, expected_logdet, 0.5,
                "MV complex Cholesky: log_det recovered");

    /* Check mean recovery */
    ASSERT_NEAR(result.components[0].mean[0], mean_true[0], 0.15, "MV Chol mean[0]");
    ASSERT_NEAR(result.components[0].mean[1], mean_true[1], 0.15, "MV Chol mean[1]");
    ASSERT_NEAR(result.components[0].mean[2], mean_true[2], 0.15, "MV Chol mean[2]");
    ASSERT_NEAR(result.components[0].mean[3], mean_true[3], 0.15, "MV Chol mean[3]");

    ReleaseMVComplexResult(&result);
    free(obs);
}

static void test_mv_complex_single(void) {
    printf("test_mv_complex_single...\n");
    srand(42);

    int d = 2;
    size_t n = 3000;
    double mean_true[4] = {2.0, -1.0, 0.5, 3.0};

    /* Σ = 2*I (diagonal, real) — Cholesky L = sqrt(2)*I */
    double chol_true[8] = {
        1.41421356, 0.0,  0.0, 0.0,
        0.0, 0.0,  1.41421356, 0.0
    };

    double* obs = (double*)malloc(n * 2 * d * sizeof(double));
    gen_mv_complex(obs, n, d, mean_true, chol_true);

    MVComplexMixtureResult result = {0};
    int ret = UnmixMVComplex(obs, n, d, 1, 200, 1e-8, 0, &result);

    ASSERT(ret == 0, "MV single: EM success");
    ASSERT_NEAR(result.components[0].mean[0], mean_true[0], 0.15, "MV single mean[0]");
    ASSERT_NEAR(result.components[0].mean[1], mean_true[1], 0.15, "MV single mean[1]");
    ASSERT_NEAR(result.components[0].mean[2], mean_true[2], 0.15, "MV single mean[2]");
    ASSERT_NEAR(result.components[0].mean[3], mean_true[3], 0.15, "MV single mean[3]");

    /* Covariance diagonal should be ~2 */
    ASSERT_NEAR(result.components[0].cov[0], 2.0, 0.4, "MV single cov[0,0].re");
    ASSERT_NEAR(result.components[0].cov[6], 2.0, 0.4, "MV single cov[1,1].re");

    ReleaseMVComplexResult(&result);
    free(obs);
}

static void test_mv_complex_two(void) {
    printf("test_mv_complex_two...\n");
    srand(123);

    int d = 2;
    size_t n1 = 2000, n2 = 2000;
    size_t n = n1 + n2;

    /* Component 1: mean=(5+2i, 1-3i), Σ = I */
    double mean1[4] = {5.0, 2.0, 1.0, -3.0};
    double chol1[8] = { 1,0, 0,0,  0,0, 1,0 };

    /* Component 2: mean=(-5-2i, -1+3i), Σ = I */
    double mean2[4] = {-5.0, -2.0, -1.0, 3.0};
    double chol2[8] = { 1,0, 0,0,  0,0, 1,0 };

    double* obs = (double*)malloc(n * 2 * d * sizeof(double));
    gen_mv_complex(obs,             n1, d, mean1, chol1);
    gen_mv_complex(obs + n1*2*d,    n2, d, mean2, chol2);

    MVComplexMixtureResult result = {0};
    int ret = UnmixMVComplex(obs, n, d, 2, 200, 1e-8, 0, &result);

    ASSERT(ret == 0, "MV two: EM success");
    ASSERT(result.num_components == 2, "MV two: k=2");

    /* Identify components by first mean's real part */
    int c1 = (result.components[0].mean[0] > 0) ? 0 : 1;
    int c2 = 1 - c1;

    ASSERT_NEAR(result.components[c1].mean[0],  5.0, 0.3, "MV two c1 mu[0]");
    ASSERT_NEAR(result.components[c1].mean[1],  2.0, 0.3, "MV two c1 mu[1]");
    ASSERT_NEAR(result.components[c2].mean[0], -5.0, 0.3, "MV two c2 mu[0]");
    ASSERT_NEAR(result.components[c2].mean[1], -2.0, 0.3, "MV two c2 mu[1]");

    ASSERT_NEAR(result.mixing_weights[c1], 0.5, 0.05, "MV two weight c1");
    ASSERT_NEAR(result.mixing_weights[c2], 0.5, 0.05, "MV two weight c2");

    ReleaseMVComplexResult(&result);
    free(obs);
}


/* ════════════════════════════════════════════════════════════════════
 * Feature 2 test: SIMD E-step vs scalar
 * ════════════════════════════════════════════════════════════════════ */

static void test_simd_matches_scalar(void) {
    printf("test_simd_matches_scalar...\n");
    srand(77);

    size_t n = 2048;
    int k = 4;

    double* data   = (double*)malloc(2 * n * sizeof(double));
    double* resp_s = (double*)malloc((size_t)k * n * sizeof(double));
    double* resp_v = (double*)malloc((size_t)k * n * sizeof(double));
    double log_w[4], mu_re[4], mu_im[4], var[4];

    /* Generate random parameters and data */
    for (int j = 0; j < k; j++) {
        log_w[j] = log(0.25);
        mu_re[j] = (j % 2 == 0) ? 3.0 : -3.0;
        mu_im[j] = (j < 2)      ? 3.0 : -3.0;
        var[j]   = 1.0 + j * 0.5;
    }
    for (size_t i = 0; i < n; i++) {
        double re, im;
        box_muller(&re, &im);
        data[2*i]   = re * 2.0 + 1.0;
        data[2*i+1] = im * 2.0 - 1.0;
    }

    /* Scalar reference E-step (inline) */
    double ll_scalar = 0.0;
    for (size_t i = 0; i < n; i++) {
        double re_i = data[2*i], im_i = data[2*i+1];
        double lps[4], mx = -1e30;
        for (int j = 0; j < k; j++) {
            double dr = re_i - mu_re[j], di = im_i - mu_im[j];
            lps[j] = log_w[j] - log(M_PI) - log(var[j]) - (dr*dr+di*di)/var[j];
            if (lps[j] > mx) mx = lps[j];
        }
        double tot = 0;
        for (int j = 0; j < k; j++) { lps[j] = exp(lps[j]-mx); tot += lps[j]; }
        ll_scalar += mx + log(tot);
        for (int j = 0; j < k; j++) resp_s[j*n+i] = lps[j] / tot;
    }

    /* SIMD E-step */
    double ll_simd = simd_complex_circular_estep(data, n, log_w, mu_re, mu_im,
                                                 var, k, resp_v);

    ASSERT_NEAR(ll_simd, ll_scalar, 1e-6, "SIMD LL matches scalar LL");

    /* Check each responsibility */
    int mismatch = 0;
    for (int j = 0; j < k && !mismatch; j++) {
        for (size_t i = 0; i < n && !mismatch; i++) {
            double diff = fabs(resp_v[j*n+i] - resp_s[j*n+i]);
            if (diff > 1e-10) {
                printf("  FAIL at j=%d i=%zu: simd=%.12f scalar=%.12f diff=%.2e\n",
                       j, i, resp_v[j*n+i], resp_s[j*n+i], diff);
                mismatch = 1;
            }
        }
    }
    if (!mismatch) tests_passed++;
    else           tests_failed++;

    free(data); free(resp_s); free(resp_v);
}


/* ════════════════════════════════════════════════════════════════════
 * Feature 3 test: Streaming Complex EM
 * ════════════════════════════════════════════════════════════════════ */

static void test_streaming_complex(void) {
    printf("test_streaming_complex...\n");
    srand(999);

    /* Write two-component IQ data to a temp file */
    const char* tmpfile = "/tmp/test_iq_stream.txt";
    size_t n1 = 3000, n2 = 3000, n = n1 + n2;
    double mu1_re =  4.0, mu1_im =  2.0, var1 = 1.0;
    double mu2_re = -4.0, mu2_im = -2.0, var2 = 1.5;

    FILE* fp = fopen(tmpfile, "w");
    if (!fp) {
        printf("  SKIP: cannot write to %s\n", tmpfile);
        tests_passed++;   /* Skip, don't fail */
        return;
    }

    /* Generate and write component 1 */
    double s1 = sqrt(var1 / 2.0);
    for (size_t i = 0; i < n1; i++) {
        double re, im;
        box_muller(&re, &im);
        fprintf(fp, "%.8f %.8f\n", mu1_re + s1*re, mu1_im + s1*im);
    }
    /* Generate and write component 2 */
    double s2 = sqrt(var2 / 2.0);
    for (size_t i = 0; i < n2; i++) {
        double re, im;
        box_muller(&re, &im);
        fprintf(fp, "%.8f %.8f\n", mu2_re + s2*re, mu2_im + s2*im);
    }
    fclose(fp);

    ComplexStreamConfig cfg = {0};
    cfg.num_components = 2;
    cfg.chunk_size  = 500;
    cfg.max_passes  = 20;
    cfg.rtole       = 1e-5;
    cfg.verbose     = 0;
    cfg.eta_decay   = 0.6;
    cfg.type        = CGAUSS_CIRCULAR;

    CCircMixtureResult result = {0};
    int ret = UnmixComplexStreaming(tmpfile, &cfg, &result);

    ASSERT(ret == 0, "Streaming EM returns success");
    ASSERT(result.num_components == 2, "Streaming: k=2");

    /* Sort by mu_re */
    int c1 = (result.components[0].mu_re < result.components[1].mu_re) ? 0 : 1;
    int c2 = 1 - c1;

    ASSERT_NEAR(result.components[c1].mu_re, mu2_re, 0.5, "Streaming c1 mu_re");
    ASSERT_NEAR(result.components[c1].mu_im, mu2_im, 0.5, "Streaming c1 mu_im");
    ASSERT_NEAR(result.components[c2].mu_re, mu1_re, 0.5, "Streaming c2 mu_re");
    ASSERT_NEAR(result.components[c2].mu_im, mu1_im, 0.5, "Streaming c2 mu_im");

    ASSERT_NEAR(result.mixing_weights[c1], 0.5, 0.1, "Streaming weight c1");
    ASSERT_NEAR(result.mixing_weights[c2], 0.5, 0.1, "Streaming weight c2");

    ReleaseCCircResult(&result);
    (void)n;   /* suppress unused-variable warning */

    /* Clean up */
    remove(tmpfile);
}


/* ════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("=== Complex EM Tests ===\n\n");

    test_circular_pdf();
    test_single_circular();
    test_two_circular();
    test_autok_circular();
    test_noncircular_pdf();
    test_noncircular_recovery();
    test_edge_cases();
    test_qpsk_scenario();

    printf("\n--- Feature 1: Multivariate Complex Gaussian ---\n\n");
    test_mv_complex_cholesky();
    test_mv_complex_single();
    test_mv_complex_two();

    printf("\n--- Feature 2: SIMD E-step ---\n\n");
    test_simd_matches_scalar();

    printf("\n--- Feature 3: Streaming Complex EM ---\n\n");
    test_streaming_complex();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
