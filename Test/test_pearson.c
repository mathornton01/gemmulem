/*
 * Unit tests for the Pearson distribution system
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pearson.h"
#include "distributions.h"

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define ASSERT_CLOSE(a, b, tol, msg) do { \
    tests_run++; \
    if (fabs((a) - (b)) < (tol)) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s (got %.6f, want %.6f)\n", msg, (double)(a), (double)(b)); } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

/* Box-Muller normal RNG */
static double randn(double mu, double sigma) {
    double u1 = (rand() % 100000 + 1) / 100001.0;
    double u2 = (rand() % 100000 + 1) / 100001.0;
    return mu + sigma * sqrt(-2.0*log(u1)) * cos(2.0*3.14159265*u2);
}

/* ===== Classification tests ===== */
void test_classification(void) {
    printf("Test: Pearson type classification\n");

    ASSERT_TRUE(pearson_classify(0.0, 3.0) == PEARSON_TYPE_0, "beta1=0,beta2=3 -> Normal");
    ASSERT_TRUE(pearson_classify(0.0, 2.0) == PEARSON_TYPE_II, "beta1=0,beta2=2 -> Sym Beta");
    ASSERT_TRUE(pearson_classify(0.0, 5.0) == PEARSON_TYPE_VII, "beta1=0,beta2=5 -> Student-t");
    ASSERT_TRUE(pearson_classify(1.0, 4.5) == PEARSON_TYPE_III, "Type III line -> Gamma");

    /* Type I: below the Type III line */
    PearsonType t = pearson_classify(0.5, 3.0);
    ASSERT_TRUE(t == PEARSON_TYPE_I || t == PEARSON_TYPE_IV, "beta1=0.5,beta2=3.0 -> Type I or IV");

    /* Type IV: above the Type III line */
    t = pearson_classify(1.0, 6.0);
    ASSERT_TRUE(t != PEARSON_TYPE_UNDEFINED, "beta1=1,beta2=6 should be valid");
}

/* ===== Normal (Type 0) PDF ===== */
void test_normal_pdf(void) {
    printf("Test: Normal (Type 0) PDF\n");

    PearsonParams pp;
    int rc = pearson_from_moments(0.0, 1.0, 0.0, 3.0, &pp);
    ASSERT_TRUE(rc == 0, "Normal should succeed");
    ASSERT_TRUE(pp.type == PEARSON_TYPE_0, "Should be Type 0");
    ASSERT_TRUE(pp.valid, "Should be valid");

    /* PDF at 0 should be ~0.3989 */
    double p = pearson_pdf(0.0, &pp);
    ASSERT_CLOSE(p, 0.3989, 0.01, "N(0,1) pdf at 0");

    /* Integration */
    double sum = 0;
    for (double x = -6; x <= 6; x += 0.001) sum += pearson_pdf(x, &pp) * 0.001;
    ASSERT_CLOSE(sum, 1.0, 0.01, "N(0,1) integrates to 1");
}

/* ===== Gamma (Type III) PDF ===== */
void test_gamma_pdf(void) {
    printf("Test: Gamma (Type III) PDF\n");

    /* Gamma(shape=4, rate=1): mean=4, var=4, gamma1=1, gamma2=1.5
     * beta1 = 1.0, beta2 = 3 + 1.5 = 4.5 (on the Type III line) */
    PearsonParams pp;
    int rc = pearson_from_moments(4.0, 2.0, 1.0, 4.5, &pp);
    ASSERT_TRUE(rc == 0, "Gamma should succeed");
    ASSERT_TRUE(pp.type == PEARSON_TYPE_III, "Should be Type III");

    /* Integration */
    double sum = 0;
    for (double x = 0.01; x <= 20; x += 0.01) sum += pearson_pdf(x, &pp) * 0.01;
    ASSERT_CLOSE(sum, 1.0, 0.05, "Gamma integrates to ~1");
}

/* ===== Student-t (Type VII) PDF ===== */
void test_student_t_pdf(void) {
    printf("Test: Student-t (Type VII) PDF\n");

    /* t with df=5: beta1=0, beta2 = 3+6/(5-4) = 9 ... no
     * For Student-t(df): beta2 = 3 + 6/(df-4) for df > 4
     * df=10: beta2 = 3 + 6/6 = 4.0 */
    PearsonParams pp;
    int rc = pearson_from_moments(0.0, 1.0, 0.0, 4.0, &pp);
    ASSERT_TRUE(rc == 0, "Student-t should succeed");
    ASSERT_TRUE(pp.type == PEARSON_TYPE_VII, "Should be Type VII");
    ASSERT_TRUE(pp.valid, "Should be valid");

    double sum = 0;
    for (double x = -20; x <= 20; x += 0.01) sum += pearson_pdf(x, &pp) * 0.01;
    ASSERT_CLOSE(sum, 1.0, 0.05, "Student-t integrates to ~1");
}

/* ===== Moment estimation roundtrip ===== */
void test_moment_estimation(void) {
    printf("Test: Moment estimation from Normal data\n");

    srand(42);
    int n = 10000;
    double* data = (double*)malloc(sizeof(double) * n);
    double* weights = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        data[i] = randn(5.0, 2.0);
        weights[i] = 1.0;
    }

    PearsonParams pp;
    int rc = pearson_estimate(data, weights, n, &pp);
    ASSERT_TRUE(rc == 0, "Estimation should succeed");
    ASSERT_CLOSE(pp.mu, 5.0, 0.2, "Mean near 5.0");
    ASSERT_CLOSE(pp.sigma, 2.0, 0.2, "Sigma near 2.0");
    ASSERT_CLOSE(pp.beta1, 0.0, 0.1, "Beta1 near 0 (symmetric)");
    ASSERT_CLOSE(pp.beta2, 3.0, 0.3, "Beta2 near 3 (normal kurtosis)");
    ASSERT_TRUE(pp.type == PEARSON_TYPE_0 || pp.type == PEARSON_TYPE_VII,
                "Should classify as Normal or near-Normal");

    free(data);
    free(weights);
}

/* ===== Pearson in generic EM framework ===== */
void test_pearson_generic_em(void) {
    printf("Test: Pearson in generic mixture EM\n");

    srand(42);
    int n = 4000;
    double* data = (double*)malloc(sizeof(double) * n);

    /* Mixture: 2000 from N(-5, 1) and 2000 from N(5, 1.5) */
    for (int i = 0; i < 2000; i++) data[i] = randn(-5.0, 1.0);
    for (int i = 2000; i < 4000; i++) data[i] = randn(5.0, 1.5);

    MixtureResult result;
    int rc = UnmixGeneric(data, n, DIST_PEARSON, 2, 200, 1e-5, 0, &result);
    ASSERT_TRUE(rc == 0, "Pearson EM should succeed");
    ASSERT_TRUE(result.num_components == 2, "2 components");

    /* Check means are near -5 and 5 */
    double m0 = result.params[0].p[0];
    double m1 = result.params[1].p[0];
    double lo = (m0 < m1) ? m0 : m1;
    double hi = (m0 < m1) ? m1 : m0;
    ASSERT_CLOSE(lo, -5.0, 1.0, "Lower mean near -5");
    ASSERT_CLOSE(hi, 5.0, 1.0, "Upper mean near 5");

    /* Mixing weights should sum to 1 */
    ASSERT_CLOSE(result.mixing_weights[0] + result.mixing_weights[1], 1.0, 0.001,
                 "Weights sum to 1");

    ReleaseMixtureResult(&result);
    free(data);
}

/* ===== Mixed-family discovery ===== */
void test_mixed_family_discovery(void) {
    printf("Test: Mixed-family discovery (Normal + skewed)\n");

    srand(77);
    int n = 6000;
    double* data = (double*)malloc(sizeof(double) * n);

    /* Component 1: N(0, 1) -> should discover Type 0 (Normal) */
    for (int i = 0; i < 3000; i++) data[i] = randn(0.0, 1.0);

    /* Component 2: Exponential(rate=0.5) shifted to mean=5 -> should discover skewed type */
    for (int i = 3000; i < 6000; i++) {
        double u = (rand() % 100000 + 1) / 100001.0;
        data[i] = -2.0 * log(u) + 3.0;  /* Exp(0.5) shifted */
    }

    MixtureResult result;
    int rc = UnmixGeneric(data, n, DIST_PEARSON, 2, 300, 1e-5, 0, &result);
    ASSERT_TRUE(rc == 0, "Should succeed");

    /* Check that one component has low beta1 (symmetric) and one has higher beta1 (skewed) */
    double b1_0 = result.params[0].p[2];  /* beta1 of component 0 */
    double b1_1 = result.params[1].p[2];  /* beta1 of component 1 */
    double max_b1 = (b1_0 > b1_1) ? b1_0 : b1_1;
    double min_b1 = (b1_0 > b1_1) ? b1_1 : b1_0;

    /* The skewed component should have meaningfully higher beta1 */
    ASSERT_TRUE(max_b1 > min_b1, "Skewed component should have higher beta1");

    /* Verify the Pearson types differ */
    PearsonParams pp0, pp1;
    pearson_from_moments(result.params[0].p[0], result.params[0].p[1],
                         result.params[0].p[2], result.params[0].p[3], &pp0);
    pearson_from_moments(result.params[1].p[0], result.params[1].p[1],
                         result.params[1].p[2], result.params[1].p[3], &pp1);

    printf("    Component 0: %s (beta1=%.3f, beta2=%.3f)\n",
           pearson_type_name(pp0.type), pp0.beta1, pp0.beta2);
    printf("    Component 1: %s (beta1=%.3f, beta2=%.3f)\n",
           pearson_type_name(pp1.type), pp1.beta1, pp1.beta2);

    ReleaseMixtureResult(&result);
    free(data);
}

/* ===== Type name coverage ===== */
void test_type_names(void) {
    printf("Test: Type name strings\n");
    ASSERT_TRUE(pearson_type_name(PEARSON_TYPE_0) != NULL, "Type 0 has name");
    ASSERT_TRUE(pearson_type_name(PEARSON_TYPE_I) != NULL, "Type I has name");
    ASSERT_TRUE(pearson_type_name(PEARSON_TYPE_III) != NULL, "Type III has name");
    ASSERT_TRUE(pearson_type_name(PEARSON_TYPE_IV) != NULL, "Type IV has name");
    ASSERT_TRUE(pearson_type_name(PEARSON_TYPE_VII) != NULL, "Type VII has name");
}

int main(void) {
    printf("\n========================================\n");
    printf("  Pearson Distribution Tests\n");
    printf("========================================\n\n");

    test_classification();
    test_normal_pdf();
    test_gamma_pdf();
    test_student_t_pdf();
    test_moment_estimation();
    test_type_names();
    test_pearson_generic_em();
    test_mixed_family_discovery();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) printf(" (%d FAILED)", tests_failed);
    printf("\n========================================\n\n");

    return tests_failed > 0 ? 1 : 0;
}
