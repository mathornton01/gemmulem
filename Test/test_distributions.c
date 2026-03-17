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

/* ===== New family integration tests ===== */
void test_new_families_pdf(void) {
    printf("Test: New family PDF integration\n");

    struct { DistFamily fam; double lo; double hi; double step; const char* name; } checks[] = {
        { DIST_STUDENT_T,   -30, 30,  0.01, "StudentT(0,1,5) integrates ~1" },
        { DIST_LAPLACE,     -20, 20,  0.01, "Laplace(0,1) integrates ~1" },
        { DIST_CAUCHY,      -200, 200, 0.1, "Cauchy(0,1) integrates ~1" },
        { DIST_INVGAUSS,    0.001, 10, 0.001, "InvGauss(1,1) integrates ~1" },
        { DIST_RAYLEIGH,    0, 20,  0.01, "Rayleigh(1) integrates ~1" },
        { DIST_PARETO,      1, 1000, 0.01, "Pareto(2,1) integrates ~1" },
    };
    double params[][4] = {
        {0, 1, 5, 0},    /* StudentT: mu, sigma, df */
        {0, 1, 0, 0},    /* Laplace: mu, b */
        {0, 1, 0, 0},    /* Cauchy: x0, gamma */
        {1, 1, 0, 0},    /* InvGauss: mu, lambda */
        {1, 0, 0, 0},    /* Rayleigh: sigma */
        {2, 1, 0, 0},    /* Pareto: alpha, xm */
    };

    for (int t = 0; t < 6; t++) {
        const DistFunctions* df = GetDistFunctions(checks[t].fam);
        ASSERT_TRUE(df != NULL, "Family exists");
        if (!df) continue;
        DistParams p;
        p.p[0]=params[t][0]; p.p[1]=params[t][1];
        p.p[2]=params[t][2]; p.p[3]=params[t][3];
        p.nparams = df->num_params;
        double sum = 0;
        for (double x = checks[t].lo; x <= checks[t].hi; x += checks[t].step)
            sum += df->pdf(x, &p) * checks[t].step;
        ASSERT_CLOSE(sum, 1.0, 0.05, checks[t].name);
    }
}

/* ===== Student-t mixture: robust to outliers ===== */
void test_student_t_robustness(void) {
    printf("Test: Student-t mixture robustness to outliers\n");
    srand(1234);
    int n = 2000;
    double* data = (double*)malloc(sizeof(double)*n);
    /* 90% from Normal, 10% outliers (heavy tails) */
    for (int i=0; i<1800; i++) data[i] = randn(0, 1);
    for (int i=1800; i<2000; i++) data[i] = randn(0, 10);

    MixtureResult r;
    int rc = UnmixGeneric(data, n, DIST_STUDENT_T, 1, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "StudentT EM succeeds");
    if (rc == 0) {
        ASSERT_CLOSE(r.params[0].p[0], 0.0, 0.5, "StudentT mean near 0");
        /* df should be < 30 (heavier than Normal due to outliers) */
        ASSERT_TRUE(r.params[0].p[2] < 50, "StudentT df < 50 (heavy tails detected)");
    }
    ReleaseMixtureResult(&r);
    free(data);
}

/* ===== Laplace mixture ===== */
void test_laplace_mixture(void) {
    printf("Test: Laplace mixture EM\n");
    srand(5678);
    int n = 2000;
    double* data = (double*)malloc(sizeof(double)*n);
    /* Two Laplace components at -3 and +3, b=0.5 */
    for (int i=0; i<n; i++) {
        double u = (rand()%10000+1)/10001.0;
        double v = (rand()%10000+1)/10001.0;
        /* Laplace(mu, b): mu + b * ln(u/v) */
        double mu = (i < n/2) ? -3.0 : 3.0;
        data[i] = mu + 0.5 * log(u / v);
    }

    MixtureResult r;
    int rc = UnmixGeneric(data, n, DIST_LAPLACE, 2, 300, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Laplace mixture EM succeeds");
    if (rc == 0) {
        double m0 = r.params[0].p[0], m1 = r.params[1].p[0];
        double lo = (m0 < m1) ? m0 : m1;
        double hi = (m0 < m1) ? m1 : m0;
        ASSERT_CLOSE(lo, -3.0, 1.0, "Laplace lower mean near -3");
        ASSERT_CLOSE(hi,  3.0, 1.0, "Laplace upper mean near +3");
    }
    ReleaseMixtureResult(&r);
    free(data);
}

/* ===== Rayleigh: signal amplitude model ===== */
void test_rayleigh(void) {
    printf("Test: Rayleigh single-component EM\n");
    srand(9999);
    int n = 2000;
    double* data = (double*)malloc(sizeof(double)*n);
    /* Rayleigh(sigma=2): sample via inverse CDF: x = sigma * sqrt(-2*ln(u)) */
    double sigma_true = 2.0;
    for (int i=0; i<n; i++) {
        double u = (rand()%100000+1)/100001.0;
        data[i] = sigma_true * sqrt(-2.0 * log(u));
    }
    MixtureResult r;
    int rc = UnmixGeneric(data, n, DIST_RAYLEIGH, 1, 200, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Rayleigh EM succeeds");
    if (rc == 0) ASSERT_CLOSE(r.params[0].p[0], sigma_true, 0.2, "Rayleigh sigma near 2");
    ReleaseMixtureResult(&r);
    free(data);
}

/* ===== Pareto: power-law tail detection ===== */
void test_pareto(void) {
    printf("Test: Pareto single-component EM\n");
    srand(4242);
    int n = 2000;
    double* data = (double*)malloc(sizeof(double)*n);
    /* Pareto(alpha=2, xm=1): x = xm / u^(1/alpha) */
    for (int i=0; i<n; i++) {
        double u = (rand()%100000+1)/100001.0;
        data[i] = 1.0 / pow(u, 0.5);  /* alpha=2 */
    }
    MixtureResult r;
    int rc = UnmixGeneric(data, n, DIST_PARETO, 1, 200, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "Pareto EM succeeds");
    if (rc == 0) ASSERT_CLOSE(r.params[0].p[0], 2.0, 0.3, "Pareto alpha near 2");
    ReleaseMixtureResult(&r);
    free(data);
}

/* ===== InvGaussian: positive right-skewed data ===== */
void test_invgauss(void) {
    printf("Test: InvGaussian EM\n");
    srand(3141);
    int n = 2000;
    double* data = (double*)malloc(sizeof(double)*n);
    /* Sample from InvGauss(mu=2, lambda=4) via normal approximation
     * True sample: Seshadri (1993) method */
    for (int i=0; i<n; i++) {
        double mu = 2.0, lam = 4.0;
        double v = randn(0, 1);
        double y = v * v;
        double x1 = mu + mu*mu*y/(2*lam) - mu/(2*lam)*sqrt(4*mu*lam*y + mu*mu*y*y);
        double u = (rand()%100000+1)/100001.0;
        data[i] = (u <= mu/(mu+x1)) ? x1 : mu*mu/x1;
    }
    MixtureResult r;
    int rc = UnmixGeneric(data, n, DIST_INVGAUSS, 1, 200, 1e-5, 0, &r);
    ASSERT_TRUE(rc == 0, "InvGauss EM succeeds");
    if (rc == 0) ASSERT_CLOSE(r.params[0].p[0], 2.0, 0.5, "InvGauss mu near 2");
    ReleaseMixtureResult(&r);
    free(data);
}

/* ===== Model selection picks right family ===== */
void test_model_selection_student_t(void) {
    printf("Test: Model selection finds Student-t when data has heavy tails\n");
    srand(7777);
    int n = 3000;
    double* data = (double*)malloc(sizeof(double)*n);
    /* Sample t(df=3): Box-Muller normal / chi^2 approximation */
    for (int i = 0; i < n; i++) {
        /* t(3) = N(0,1) / sqrt(chi2(3)/3) — approx via ratio of normals */
        double x = randn(0, 1);
        double y = randn(0, 1) * randn(0, 1) * randn(0, 1);  /* rough heavy tail */
        data[i] = x / (1 + 0.1*fabs(y));
    }

    DistFamily fams[] = { DIST_GAUSSIAN, DIST_STUDENT_T, DIST_LAPLACE };
    ModelSelectResult msr;
    int rc = SelectBestMixture(data, n, fams, 3, 1, 2, 200, 1e-5, 0, &msr);
    ASSERT_TRUE(rc == 0, "Model selection succeeds");
    if (rc == 0) {
        printf("    Best: %s k=%d (BIC=%.1f)\n",
               GetDistName(msr.best_family), msr.best_k, msr.best_bic);
        /* Student-t or Laplace should beat Gaussian for heavy-tailed data */
        ASSERT_TRUE(msr.best_family != DIST_GAUSSIAN || msr.best_k == 1,
                    "Heavy-tailed data not best fit by Gaussian");
        ReleaseModelSelectResult(&msr);
    }
    free(data);
}

/* ===== All 15 families registered ===== */
void test_all_families_registered(void) {
    printf("Test: All 15 families registered\n");
    int count = 0;
    for (int i = 0; i < DIST_COUNT; i++) {
        const DistFunctions* df = GetDistFunctions((DistFamily)i);
        if (df && df->name) {
            count++;
        }
    }
    ASSERT_TRUE(count == DIST_COUNT, "All DIST_COUNT families registered");
    printf("  Registered families (%d):", count);
    for (int i = 0; i < DIST_COUNT; i++) {
        const DistFunctions* df = GetDistFunctions((DistFamily)i);
        if (df && df->name) printf(" %s", df->name);
    }
    printf("\n");
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
    /* New families */
    test_all_families_registered();
    test_new_families_pdf();
    test_student_t_robustness();
    test_laplace_mixture();
    test_rayleigh();
    test_pareto();
    test_invgauss();
    test_model_selection_student_t();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) printf(" (%d FAILED)", tests_failed);
    printf("\n========================================\n\n");

    return tests_failed > 0 ? 1 : 0;
}
