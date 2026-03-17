/*
 * Tests for multivariate Gaussian mixture EM.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "multivariate.h"

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define ASSERT_CLOSE(a, b, tol, msg) do { \
    tests_run++; \
    if (fabs((a)-(b)) < (tol)) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s  (got %.4f, want %.4f)\n", msg, (double)(a), (double)(b)); } \
} while(0)
#define ASSERT_TRUE(c, msg) do { \
    tests_run++; if (c) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

static double randn(double mu, double sd, unsigned* seed) {
    double u = ((*seed = *seed * 1664525 + 1013904223) % 100000 + 1) / 100001.0;
    double v = ((*seed = *seed * 1664525 + 1013904223) % 100000 + 1) / 100001.0;
    return mu + sd * sqrt(-2.0 * log(u)) * cos(2.0 * 3.14159265 * v);
}

/* ─── Test 1: 2D, k=2, full covariance ─── */
void test_2d_k2_full(void) {
    printf("Test: 2D k=2 full covariance\n");
    unsigned seed = 1234;
    int n = 2000, d = 2;
    double* data = malloc(sizeof(double) * n * d);

    /* Component 0: mean=(-4,-4), Component 1: mean=(4,4) */
    for (int i = 0; i < n/2; i++) {
        data[i*d+0] = randn(-4, 1, &seed);
        data[i*d+1] = randn(-4, 1, &seed);
    }
    for (int i = n/2; i < n; i++) {
        data[i*d+0] = randn(4, 1, &seed);
        data[i*d+1] = randn(4, 1, &seed);
    }

    MVMixtureResult r;
    int rc = UnmixMVGaussian(data, n, d, 2, COV_FULL, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "MV k=2 full succeeds");
    ASSERT_TRUE(r.num_components == 2, "k=2 preserved");

    /* Sort by x-mean of first component */
    int lo = r.components[0].mean[0] < r.components[1].mean[0] ? 0 : 1;
    int hi = 1 - lo;
    printf("    mean[0]: (%.2f, %.2f)  mean[1]: (%.2f, %.2f)\n",
           r.components[lo].mean[0], r.components[lo].mean[1],
           r.components[hi].mean[0], r.components[hi].mean[1]);

    ASSERT_CLOSE(r.components[lo].mean[0], -4.0, 1.0, "lower mean[0] near -4");
    ASSERT_CLOSE(r.components[lo].mean[1], -4.0, 1.0, "lower mean[1] near -4");
    ASSERT_CLOSE(r.components[hi].mean[0],  4.0, 1.0, "upper mean[0] near 4");
    ASSERT_CLOSE(r.components[hi].mean[1],  4.0, 1.0, "upper mean[1] near 4");
    ASSERT_CLOSE(r.mixing_weights[0], 0.5, 0.1, "weight near 0.5");
    ASSERT_TRUE(r.loglikelihood < 0 && r.loglikelihood > -1e8, "LL is finite negative");

    ReleaseMVMixtureResult(&r);
    free(data);
}

/* ─── Test 2: 3D, k=3, diagonal covariance ─── */
void test_3d_k3_diagonal(void) {
    printf("Test: 3D k=3 diagonal covariance\n");
    unsigned seed = 5678;
    int n = 3000, d = 3;
    double* data = malloc(sizeof(double) * n * d);
    double centers[3][3] = {{-8,-8,-8}, {0,0,0}, {8,8,8}};

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < n/3; i++) {
            int idx = c*(n/3) + i;
            for (int dd = 0; dd < d; dd++)
                data[idx*d+dd] = randn(centers[c][dd], 1.0, &seed);
        }
    }

    MVMixtureResult r;
    int rc = UnmixMVGaussian(data, n, d, 3, COV_DIAGONAL, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "MV 3D k=3 diagonal succeeds");
    printf("    LL=%.1f  BIC=%.1f  iterations=%d\n",
           r.loglikelihood, r.bic, r.iterations);
    ASSERT_TRUE(r.loglikelihood < -5000, "LL < -5000 (3000 points)");

    ReleaseMVMixtureResult(&r);
    free(data);
}

/* ─── Test 3: Spherical covariance ─── */
void test_2d_spherical(void) {
    printf("Test: 2D k=2 spherical covariance\n");
    unsigned seed = 9999;
    int n = 1000, d = 2;
    double* data = malloc(sizeof(double) * n * d);

    for (int i = 0; i < n/2; i++) {
        data[i*d+0] = randn(-3, 1.2, &seed);
        data[i*d+1] = randn(-3, 1.2, &seed);
    }
    for (int i = n/2; i < n; i++) {
        data[i*d+0] = randn(3, 1.2, &seed);
        data[i*d+1] = randn(3, 1.2, &seed);
    }

    MVMixtureResult r;
    int rc = UnmixMVGaussian(data, n, d, 2, COV_SPHERICAL, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "MV spherical succeeds");

    int lo = r.components[0].mean[0] < r.components[1].mean[0] ? 0 : 1;
    int hi = 1 - lo;
    ASSERT_CLOSE(r.components[lo].mean[0], -3.0, 1.0, "spherical lower mean near -3");
    ASSERT_CLOSE(r.components[hi].mean[0],  3.0, 1.0, "spherical upper mean near 3");

    ReleaseMVMixtureResult(&r);
    free(data);
}

/* ─── Test 4: PDF sanity ─── */
void test_mvgauss_pdf(void) {
    printf("Test: mvgauss_pdf sanity\n");
    int d = 2;
    MVGaussParams p;
    p.dim = d;
    p.mean = (double*)calloc(d, sizeof(double));    /* mean = (0,0) */
    p.cov = (double*)calloc(d*d, sizeof(double));   /* identity */
    p.cov_chol = (double*)calloc(d*d, sizeof(double));
    /* Identity matrix */
    p.cov[0] = p.cov[3] = 1.0;
    /* Cholesky of identity = identity */
    p.cov_chol[0] = p.cov_chol[3] = 1.0;
    p.log_det = 0.0;  /* log|I| = 0 */

    /* PDF at origin should be 1/(2π) ≈ 0.15915 */
    double x0[2] = {0.0, 0.0};
    double v0 = mvgauss_pdf(x0, &p);
    printf("    PDF at (0,0) = %.6f  (expected %.6f)\n", v0, 1.0/(2*3.14159265));
    ASSERT_CLOSE(v0, 1.0/(2*3.14159265), 0.001, "PDF at origin near 1/(2pi)");

    /* PDF at (3,3) should be much smaller */
    double x3[2] = {3.0, 3.0};
    double v3 = mvgauss_pdf(x3, &p);
    ASSERT_TRUE(v3 < 0.001, "PDF at (3,3) is tiny");

    /* Log-PDF should match */
    double lv0 = mvgauss_logpdf(x0, &p);
    ASSERT_CLOSE(lv0, log(v0), 0.001, "logpdf matches log(pdf)");

    free(p.mean); free(p.cov); free(p.cov_chol);
}

/* ─── Test 5: BIC/AIC decrease with better k ─── */
void test_bic_k_selection(void) {
    printf("Test: BIC improves with correct k\n");
    unsigned seed = 3141;
    int n = 2000, d = 2;
    double* data = malloc(sizeof(double) * n * d);

    for (int i = 0; i < n/2; i++) {
        data[i*d+0] = randn(-5, 0.8, &seed);
        data[i*d+1] = randn(-5, 0.8, &seed);
    }
    for (int i = n/2; i < n; i++) {
        data[i*d+0] = randn(5, 0.8, &seed);
        data[i*d+1] = randn(5, 0.8, &seed);
    }

    MVMixtureResult r1, r2, r3;
    UnmixMVGaussian(data, n, d, 1, COV_DIAGONAL, 200, 1e-5, 0, &r1);
    UnmixMVGaussian(data, n, d, 2, COV_DIAGONAL, 200, 1e-5, 0, &r2);
    UnmixMVGaussian(data, n, d, 3, COV_DIAGONAL, 200, 1e-5, 0, &r3);

    printf("    k=1: LL=%.1f  BIC=%.1f\n", r1.loglikelihood, r1.bic);
    printf("    k=2: LL=%.1f  BIC=%.1f\n", r2.loglikelihood, r2.bic);
    printf("    k=3: LL=%.1f  BIC=%.1f\n", r3.loglikelihood, r3.bic);

    ASSERT_TRUE(r2.loglikelihood > r1.loglikelihood, "k=2 LL > k=1 LL");
    ASSERT_TRUE(r2.bic < r1.bic, "k=2 BIC < k=1 BIC (correct k wins)");

    ReleaseMVMixtureResult(&r1);
    ReleaseMVMixtureResult(&r2);
    ReleaseMVMixtureResult(&r3);
    free(data);
}

int main(void) {
    printf("\n========================================\n");
    printf("  Multivariate Gaussian EM Tests\n");
    printf("========================================\n\n");

    test_mvgauss_pdf();
    test_2d_k2_full();
    test_2d_spherical();
    test_3d_k3_diagonal();
    test_bic_k_selection();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) printf(" (%d FAILED)", tests_failed);
    printf("\n========================================\n\n");
    return tests_failed > 0 ? 1 : 0;
}
