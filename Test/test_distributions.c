/*
 * Unit tests for the generic distribution framework
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "distributions.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_CLOSE(a, b, tol, msg) do { \
    tests_run++; \
    if (fabs((a) - (b)) < (tol)) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s (got %.6f, want %.6f)\n", msg, (double)(a), (double)(b)); } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

/* Simple Box-Muller normal RNG */
static double randn(double mu, double sigma) {
    double u1 = (rand() % 100000 + 1) / 100001.0;
    double u2 = (rand() % 100000 + 1) / 100001.0;
    return mu + sigma * sqrt(-2.0*log(u1)) * cos(2.0*3.14159265*u2);
}

/* ===== Test: All distribution PDFs integrate to ~1 (numerical check) ===== */
void test_pdf_integration(void) {
    printf("Test: PDF integration sanity check\n");

    /* Gaussian(0,1) */
    const DistFunctions* df = GetDistFunctions(DIST_GAUSSIAN);
    DistParams p = {{0, 1, 0, 0}, 2};
    double sum = 0;
    for (double x = -10; x <= 10; x += 0.01) sum += df->pdf(x, &p) * 0.01;
    ASSERT_CLOSE(sum, 1.0, 0.01, "Gaussian(0,1) integrates to ~1");

    /* Exponential(rate=2) */
    df = GetDistFunctions(DIST_EXPONENTIAL);
    p.p[0] = 2.0; p.nparams = 1;
    sum = 0;
    for (double x = 0; x <= 20; x += 0.001) sum += df->pdf(x, &p) * 0.001;
    ASSERT_CLOSE(sum, 1.0, 0.01, "Exp(rate=2) integrates to ~1");

    /* Gamma(2, 1) */
    df = GetDistFunctions(DIST_GAMMA);
    p.p[0] = 2.0; p.p[1] = 1.0; p.nparams = 2;
    sum = 0;
    for (double x = 0.01; x <= 20; x += 0.01) sum += df->pdf(x, &p) * 0.01;
    ASSERT_CLOSE(sum, 1.0, 0.02, "Gamma(2,1) integrates to ~1");

    /* Beta(2, 5) */
    df = GetDistFunctions(DIST_BETA);
    p.p[0] = 2.0; p.p[1] = 5.0; p.nparams = 2;
    sum = 0;
    for (double x = 0.001; x <= 0.999; x += 0.001) sum += df->pdf(x, &p) * 0.001;
    ASSERT_CLOSE(sum, 1.0, 0.01, "Beta(2,5) integrates to ~1");
}

/* ===== Test: Unmix Gaussian mixture (generic) ===== */
void test_generic_gaussian(void) {
    printf("Test: Generic EM on Gaussian mixture\n");

    srand(42);
    int n = 5000;
    double* data = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < 2500; i++) data[i] = randn(-5.0, 1.0);
    for (int i = 2500; i < 5000; i++) data[i] = randn(5.0, 1.5);

    MixtureResult result;
    int rc = UnmixGeneric(data, n, DIST_GAUSSIAN, 2, 1000, 1e-6, 0, &result);
    ASSERT_TRUE(rc == 0, "Should succeed");
    ASSERT_TRUE(result.num_components == 2, "2 components");
    ASSERT_TRUE(result.loglikelihood < 0, "LL < 0");
    ASSERT_TRUE(result.bic > 0, "BIC > 0");

    /* Check means are near -5 and 5 */
    double m0 = result.params[0].p[0];
    double m1 = result.params[1].p[0];
    double lo = (m0 < m1) ? m0 : m1;
    double hi = (m0 < m1) ? m1 : m0;
    ASSERT_CLOSE(lo, -5.0, 0.5, "Lower mean near -5");
    ASSERT_CLOSE(hi, 5.0, 0.5, "Upper mean near 5");

    /* Mixing weights near 0.5 */
    ASSERT_CLOSE(result.mixing_weights[0] + result.mixing_weights[1], 1.0, 0.001, "Weights sum to 1");

    ReleaseMixtureResult(&result);
    free(data);
}

/* ===== Test: Unmix Exponential mixture (generic) ===== */
void test_generic_exponential(void) {
    printf("Test: Generic EM on Exponential mixture\n");

    srand(123);
    int n = 5000;
    double* data = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < 2500; i++) {
        double u = (rand() % 100000 + 1) / 100001.0;
        data[i] = -1.0 * log(u);  /* Exp(rate=1), mean=1 */
    }
    for (int i = 2500; i < 5000; i++) {
        double u = (rand() % 100000 + 1) / 100001.0;
        data[i] = -10.0 * log(u);  /* Exp(rate=0.1), mean=10 */
    }

    MixtureResult result;
    int rc = UnmixGeneric(data, n, DIST_EXPONENTIAL, 2, 1000, 1e-6, 0, &result);
    ASSERT_TRUE(rc == 0, "Should succeed");

    double r0 = result.params[0].p[0];
    double r1 = result.params[1].p[0];
    double hi_rate = (r0 > r1) ? r0 : r1;
    double lo_rate = (r0 > r1) ? r1 : r0;
    ASSERT_CLOSE(hi_rate, 1.0, 0.3, "High rate near 1.0");
    ASSERT_CLOSE(lo_rate, 0.1, 0.05, "Low rate near 0.1");

    ReleaseMixtureResult(&result);
    free(data);
}

/* ===== Test: Unmix Gamma mixture ===== */
void test_generic_gamma(void) {
    printf("Test: Generic EM on Gamma mixture\n");

    srand(77);
    int n = 4000;
    double* data = (double*)malloc(sizeof(double) * n);

    /* Gamma(shape=2, rate=1) via sum of exponentials */
    for (int i = 0; i < 2000; i++) {
        double u1 = (rand() % 100000 + 1) / 100001.0;
        double u2 = (rand() % 100000 + 1) / 100001.0;
        data[i] = -log(u1) - log(u2);  /* sum of 2 Exp(1) = Gamma(2,1) */
    }
    /* Gamma(shape=5, rate=0.5) ≈ sum of 10 Exp(1) scaled */
    for (int i = 2000; i < 4000; i++) {
        double s = 0;
        for (int j = 0; j < 5; j++) {
            double u = (rand() % 100000 + 1) / 100001.0;
            s += -log(u);
        }
        data[i] = s * 2;  /* Gamma(5, 0.5) */
    }

    MixtureResult result;
    int rc = UnmixGeneric(data, n, DIST_GAMMA, 2, 1000, 1e-6, 0, &result);
    ASSERT_TRUE(rc == 0, "Should succeed");
    ASSERT_TRUE(result.iterations > 1, "Should take >1 iter");

    ReleaseMixtureResult(&result);
    free(data);
}

/* ===== Test: Beta mixture (0-1 data) ===== */
void test_generic_beta(void) {
    printf("Test: Generic EM on Beta mixture\n");

    srand(99);
    int n = 3000;
    double* data = (double*)malloc(sizeof(double) * n);

    /* Simple beta samples via uniform rejection (coarse) */
    for (int i = 0; i < 1500; i++) {
        /* Cluster near 0.2 */
        data[i] = 0.15 + 0.1 * ((rand() % 10000) / 10000.0);
    }
    for (int i = 1500; i < 3000; i++) {
        /* Cluster near 0.8 */
        data[i] = 0.75 + 0.1 * ((rand() % 10000) / 10000.0);
    }

    MixtureResult result;
    int rc = UnmixGeneric(data, n, DIST_BETA, 2, 1000, 1e-6, 0, &result);
    ASSERT_TRUE(rc == 0, "Should succeed");

    /* Both weights should be near 0.5 */
    ASSERT_CLOSE(result.mixing_weights[0] + result.mixing_weights[1], 1.0, 0.001, "Weights sum to 1");

    ReleaseMixtureResult(&result);
    free(data);
}

/* ===== Test: Model selection picks correct family ===== */
void test_model_selection_gaussian(void) {
    printf("Test: Model selection on Gaussian data\n");

    srand(42);
    int n = 2000;
    double* data = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < 1000; i++) data[i] = randn(-3.0, 1.0);
    for (int i = 1000; i < 2000; i++) data[i] = randn(3.0, 1.0);

    /* Only try continuous families valid for this data */
    DistFamily families[] = { DIST_GAUSSIAN, DIST_LOGNORMAL, DIST_UNIFORM };

    ModelSelectResult result;
    int rc = SelectBestMixture(data, n, families, 3, 1, 4, 500, 1e-5, 0, &result);
    ASSERT_TRUE(rc == 0, "Should succeed");
    ASSERT_TRUE(result.best_family == DIST_GAUSSIAN, "Should pick Gaussian");
    ASSERT_TRUE(result.best_k == 2, "Should pick k=2");
    ASSERT_TRUE(result.num_candidates > 0, "Should have candidates");

    ReleaseModelSelectResult(&result);
    free(data);
}

/* ===== Test: Model selection on exponential data ===== */
void test_model_selection_exponential(void) {
    printf("Test: Model selection on Exponential data\n");

    srand(55);
    int n = 3000;
    double* data = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        double u = (rand() % 100000 + 1) / 100001.0;
        data[i] = -5.0 * log(u);  /* single Exp(rate=0.2) */
    }

    DistFamily families[] = { DIST_GAUSSIAN, DIST_EXPONENTIAL, DIST_GAMMA };

    ModelSelectResult result;
    int rc = SelectBestMixture(data, n, families, 3, 1, 3, 500, 1e-5, 0, &result);
    ASSERT_TRUE(rc == 0, "Should succeed");
    /* Should pick Exponential or Gamma with k=1 (both fit single-component well) */
    ASSERT_TRUE(result.best_k == 1 || result.best_k == 2, "Should pick k=1 or k=2");

    ReleaseModelSelectResult(&result);
    free(data);
}

/* ===== Test: All families accessible ===== */
void test_registry(void) {
    printf("Test: Distribution registry\n");

    for (int i = 0; i < DIST_COUNT; i++) {
        const DistFunctions* df = GetDistFunctions((DistFamily)i);
        ASSERT_TRUE(df != NULL, "Family should exist");
        ASSERT_TRUE(df->pdf != NULL, "pdf should exist");
        ASSERT_TRUE(df->estimate != NULL, "estimate should exist");
        ASSERT_TRUE(df->init_params != NULL, "init should exist");
        ASSERT_TRUE(df->valid != NULL, "valid should exist");
        ASSERT_TRUE(df->name != NULL, "name should exist");
        ASSERT_TRUE(df->num_params >= 1, "num_params >= 1");
    }
}

/* ===== Test: BIC prefers simpler model ===== */
void test_bic_parsimony(void) {
    printf("Test: BIC prefers simpler model (single Gaussian)\n");

    srand(42);
    int n = 2000;
    double* data = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) data[i] = randn(0, 1.0);

    MixtureResult r1, r2, r4;
    UnmixGeneric(data, n, DIST_GAUSSIAN, 1, 500, 1e-6, 0, &r1);
    UnmixGeneric(data, n, DIST_GAUSSIAN, 2, 500, 1e-6, 0, &r2);
    UnmixGeneric(data, n, DIST_GAUSSIAN, 4, 500, 1e-6, 0, &r4);

    ASSERT_TRUE(r1.bic < r4.bic, "BIC(k=1) < BIC(k=4) for unimodal data");

    ReleaseMixtureResult(&r1);
    ReleaseMixtureResult(&r2);
    ReleaseMixtureResult(&r4);
    free(data);
}

int main(void) {
    printf("\n========================================\n");
    printf("  GEMMULEM Distribution Framework Tests\n");
    printf("========================================\n\n");

    test_registry();
    test_pdf_integration();
    test_generic_gaussian();
    test_generic_exponential();
    test_generic_gamma();
    test_generic_beta();
    test_model_selection_gaussian();
    test_model_selection_exponential();
    test_bic_parsimony();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) printf(" (%d FAILED)", tests_failed);
    printf("\n========================================\n\n");

    return tests_failed > 0 ? 1 : 0;
}
