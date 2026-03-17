/*
 * Tests for adaptive mixed-family EM with auto-k selection
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "distributions.h"

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define ASSERT_CLOSE(a, b, tol, msg) do { \
    tests_run++; \
    if (fabs((a)-(b)) < (tol)) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s (got %.4f, want %.4f)\n", msg, (double)(a), (double)(b)); } \
} while(0)
#define ASSERT_TRUE(c, msg) do { \
    tests_run++; if (c) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

static double randn(double mu, double sd) {
    double u=((rand()%100000)+1)/100001.0, v=((rand()%100000)+1)/100001.0;
    return mu + sd*sqrt(-2*log(u))*cos(2*3.14159265*v);
}
static double randgamma(double shape, double scale) {
    if (shape < 1) return randgamma(1+shape, scale)*pow((rand()%100000+1)/100001.0, 1/shape);
    double d=shape-1.0/3, c=1/sqrt(9*d);
    while(1) {
        double x=randn(0,1), v=1+c*x;
        if (v<=0) continue;
        v=v*v*v;
        double u=(rand()%100000+1)/100001.0;
        if (u<1-0.0331*(x*x)*(x*x)) return d*v*scale;
        if (log(u)<0.5*x*x+d*(1-v+log(v))) return d*v*scale;
    }
}

/* ===== Auto-k: unimodal data → k=1 ===== */
void test_auto_k_unimodal(void) {
    printf("Test: Auto-k — unimodal data → k=1\n");
    srand(1234);
    int n = 1000;
    double* data = malloc(sizeof(double)*n);
    for (int i=0; i<n; i++) data[i] = randn(0, 1);

    AdaptiveResult r;
    int rc = UnmixAdaptive(data, n, 5, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Adaptive EM succeeds");
    ASSERT_TRUE(r.num_components == 1, "Unimodal → k=1");
    if (rc == 0) {
        ASSERT_CLOSE(r.params[0].p[0], 0.0, 0.3, "Mean near 0");
        printf("    Family: %s  mu=%.3f\n", GetDistName(r.families[0]), r.params[0].p[0]);
    }
    ReleaseAdaptiveResult(&r);
    free(data);
}

/* ===== Auto-k: bimodal Gaussian → k=2 ===== */
void test_auto_k_bimodal(void) {
    printf("Test: Auto-k — bimodal Gaussian → k=2\n");
    srand(5678);
    int n = 2000;
    double* data = malloc(sizeof(double)*n);
    for (int i=0; i<1000; i++) data[i] = randn(-5, 1);
    for (int i=1000; i<2000; i++) data[i] = randn(5, 1);

    AdaptiveResult r;
    int rc = UnmixAdaptive(data, n, 5, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Adaptive EM succeeds");
    ASSERT_TRUE(r.num_components == 2, "Bimodal → k=2");
    if (rc == 0 && r.num_components == 2) {
        double m0=r.params[0].p[0], m1=r.params[1].p[0];
        double lo=(m0<m1)?m0:m1, hi=(m0<m1)?m1:m0;
        ASSERT_CLOSE(lo, -5.0, 1.0, "Lower mean near -5");
        ASSERT_CLOSE(hi,  5.0, 1.0, "Upper mean near +5");
        printf("    Families: %s %s  means: %.2f %.2f\n",
               GetDistName(r.families[0]), GetDistName(r.families[1]), m0, m1);
    }
    ReleaseAdaptiveResult(&r);
    free(data);
}

/* ===== Mixed family: Gaussian + Gamma → discovers both ===== */
void test_mixed_family_discovery(void) {
    printf("Test: Mixed-family — Gaussian + Gamma data\n");
    srand(9999);
    int n = 2000;
    double* data = malloc(sizeof(double)*n);
    for (int i=0; i<1000; i++) data[i] = randn(-5, 1);
    for (int i=1000; i<2000; i++) data[i] = randgamma(4, 2);  /* mean=8, positive */

    AdaptiveResult r;
    int rc = UnmixAdaptive(data, n, 5, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Adaptive EM succeeds");
    ASSERT_TRUE(r.num_components >= 2, "Mixed data → k>=2");
    if (rc == 0) {
        printf("    k=%d  LL=%.2f  BIC=%.2f\n", r.num_components, r.loglikelihood, r.bic);
        for (int j=0; j<r.num_components; j++) {
            printf("    Comp %d: %s  w=%.3f  mu=%.2f\n",
                   j, GetDistName(r.families[j]),
                   r.mixing_weights[j], r.params[j].p[0]);
        }
        /* Check BIC is better than single Gaussian */
        MixtureResult single;
        UnmixGeneric(data, n, DIST_GAUSSIAN, 1, 300, 1e-5, 0, &single);
        ASSERT_TRUE(r.bic < single.bic, "Adaptive BIC better than single Gaussian");
        ReleaseMixtureResult(&single);
    }
    ReleaseAdaptiveResult(&r);
    free(data);
}

/* ===== Heavy-tailed: Student-t data is preferred over Gaussian ===== */
void test_heavy_tail_preferred(void) {
    printf("Test: Heavy-tailed data — Student-t preferred over Gaussian\n");
    srand(4242);
    int n = 2000;
    double* data = malloc(sizeof(double)*n);
    /* t(3) approximation via ratio of normals */
    for (int i=0; i<n; i++) {
        double x = randn(0,1), y = randn(0,1), z = randn(0,1);
        data[i] = x / sqrt((y*y+z*z)/2.0);
    }

    AdaptiveResult r;
    int rc = UnmixAdaptive(data, n, 3, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Adaptive EM succeeds");
    printf("    k=%d families:", r.num_components);
    int found_t = 0;
    for (int j=0; j<r.num_components; j++) {
        printf(" %s", GetDistName(r.families[j]));
        if (r.families[j] == DIST_STUDENT_T || r.families[j] == DIST_CAUCHY ||
            r.families[j] == DIST_LAPLACE) found_t = 1;
    }
    printf("\n");
    ASSERT_TRUE(found_t, "Heavy-tailed family selected (StudentT/Cauchy/Laplace)");
    ReleaseAdaptiveResult(&r);
    free(data);
}

/* ===== Trimodal → k=3 ===== */
void test_trimodal(void) {
    printf("Test: Auto-k — trimodal → k=3\n");
    srand(7777);
    int n = 3000;
    double* data = malloc(sizeof(double)*n);
    for (int i=0; i<1000; i++) data[i] = randn(-6, 0.8);
    for (int i=1000; i<2000; i++) data[i] = randn(0, 0.8);
    for (int i=2000; i<3000; i++) data[i] = randn(6, 0.8);

    AdaptiveResult r;
    int rc = UnmixAdaptive(data, n, 6, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Adaptive EM succeeds");
    ASSERT_TRUE(r.num_components == 3, "Trimodal → k=3");
    printf("    k=%d  BIC=%.1f\n", r.num_components, r.bic);
    ReleaseAdaptiveResult(&r);
    free(data);
}

/* ===== BIC improves with each split until wrong ===== */
void test_bic_stops_correctly(void) {
    printf("Test: BIC stops at correct k\n");
    srand(3141);
    int n = 1500;
    double* data = malloc(sizeof(double)*n);
    for (int i=0; i<750; i++) data[i] = randn(-4, 1);
    for (int i=750; i<1500; i++) data[i] = randn(4, 1);

    AdaptiveResult r;
    int rc = UnmixAdaptive(data, n, 8, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Adaptive EM succeeds");
    ASSERT_TRUE(r.num_components <= 4, "BIC stops at reasonable k (<=4)");
    ASSERT_TRUE(r.num_components >= 2, "BIC finds at least 2 components");
    printf("    k=%d  BIC=%.1f\n", r.num_components, r.bic);
    ReleaseAdaptiveResult(&r);
    free(data);
}

int main(void) {
    printf("\n========================================\n");
    printf("  Adaptive Mixed-Family EM Tests\n");
    printf("========================================\n\n");

    test_auto_k_unimodal();
    test_auto_k_bimodal();
    test_mixed_family_discovery();
    test_heavy_tail_preferred();
    test_trimodal();
    test_bic_stops_correctly();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) printf(" (%d FAILED)", tests_failed);
    printf("\n========================================\n\n");
    return tests_failed > 0 ? 1 : 0;
}
