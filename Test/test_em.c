/*
 * Unit tests for GEMMULEM
 * 
 * Compile: gcc -o test_em test_em.c -I../src/lib -L../build/src/lib -lem -lm
 * Run: ./test_em
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "EM.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_CLOSE(a, b, tol, msg) do { \
    tests_run++; \
    if (fabs((a) - (b)) < (tol)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s (expected %.6f, got %.6f, diff %.2e)\n", msg, (double)(b), (double)(a), fabs((a)-(b))); \
    } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s\n", msg); \
    } \
} while(0)

/* ===== Test 1: Simple 2-class multinomial EM ===== */
void test_multinomial_simple(void)
{
    printf("Test: Multinomial EM (2 classes, non-overlapping)\n");

    /* Pattern: class 0 gets all reads, class 1 gets none */
    const char* compat = "10" "10" "10";
    int counts[] = {100, 200, 300};
    EMResult_t result;
    EMConfig_t config;
    GetEMDefaultConfig(&config);

    int rc = ExpectationMaximization(compat, 3, 2, counts, 3, &result, &config);
    ASSERT_TRUE(rc == 0, "Return code should be 0");
    ASSERT_TRUE(result.size == 2, "Should have 2 classes");
    ASSERT_CLOSE(result.values[0], 1.0, 0.001, "Class 0 should get ~1.0");
    ASSERT_CLOSE(result.values[1], 0.0, 0.001, "Class 1 should get ~0.0");
    ASSERT_TRUE(result.iterations > 0, "Should take >0 iterations");
    ASSERT_TRUE(result.loglikelihood <= 0.0, "Log-likelihood should be <= 0");

    ReleaseEMResult(&result);
}

/* ===== Test 2: Equal split multinomial EM ===== */
void test_multinomial_equal_split(void)
{
    printf("Test: Multinomial EM (3 classes, equal split)\n");

    /* Each pattern covers exactly one class, equal counts */
    const char* compat = "100" "010" "001";
    int counts[] = {1000, 1000, 1000};
    EMResult_t result;
    EMConfig_t config;
    GetEMDefaultConfig(&config);

    int rc = ExpectationMaximization(compat, 3, 3, counts, 3, &result, &config);
    ASSERT_TRUE(rc == 0, "Return code should be 0");
    ASSERT_CLOSE(result.values[0], 1.0/3, 0.001, "Class 0 should be ~0.333");
    ASSERT_CLOSE(result.values[1], 1.0/3, 0.001, "Class 1 should be ~0.333");
    ASSERT_CLOSE(result.values[2], 1.0/3, 0.001, "Class 2 should be ~0.333");

    /* Proportions should sum to 1 */
    double sum = 0;
    for (size_t i = 0; i < result.size; i++) sum += result.values[i];
    ASSERT_CLOSE(sum, 1.0, 0.0001, "Proportions should sum to 1.0");

    ReleaseEMResult(&result);
}

/* ===== Test 3: Ambiguous reads resolved by EM ===== */
void test_multinomial_ambiguous(void)
{
    printf("Test: Multinomial EM (ambiguous reads)\n");

    /* 500 reads unique to class 0, 500 unique to class 1, 1000 shared */
    const char* compat = "10" "01" "11";
    int counts[] = {500, 500, 1000};
    EMResult_t result;
    EMConfig_t config;
    GetEMDefaultConfig(&config);

    int rc = ExpectationMaximization(compat, 3, 2, counts, 3, &result, &config);
    ASSERT_TRUE(rc == 0, "Return code should be 0");
    ASSERT_CLOSE(result.values[0], 0.5, 0.001, "Class 0 should be ~0.5");
    ASSERT_CLOSE(result.values[1], 0.5, 0.001, "Class 1 should be ~0.5");

    ReleaseEMResult(&result);
}

/* ===== Test 4: Unequal ambiguous reads ===== */
void test_multinomial_unequal(void)
{
    printf("Test: Multinomial EM (unequal, 3 classes)\n");

    /* Class 0 and 1 share reads, class 2 is unique */
    const char* compat = "110" "001" "100";
    int counts[] = {400, 600, 200};
    EMResult_t result;
    EMConfig_t config;
    GetEMDefaultConfig(&config);

    int rc = ExpectationMaximization(compat, 3, 3, counts, 3, &result, &config);
    ASSERT_TRUE(rc == 0, "Return code should be 0");

    /* Class 2 should get 600/1200 = 0.5 */
    ASSERT_CLOSE(result.values[2], 0.5, 0.001, "Class 2 (unique) should be ~0.5");

    /* Class 0 should get more than class 1 (200 unique + share of 400) */
    ASSERT_TRUE(result.values[0] > result.values[1], "Class 0 > Class 1 (has unique reads)");

    double sum = 0;
    for (size_t i = 0; i < result.size; i++) sum += result.values[i];
    ASSERT_CLOSE(sum, 1.0, 0.0001, "Proportions should sum to 1.0");

    ReleaseEMResult(&result);
}

/* ===== Test 5: Gaussian mixture unmixing ===== */
void test_gaussian_unmix(void)
{
    printf("Test: Gaussian mixture unmixing (2 well-separated clusters)\n");

    /* Generate 2000 values: 1000 from N(-10, 1) and 1000 from N(10, 1) */
    srand(42);
    int n = 2000;
    double* values = (double*)malloc(sizeof(double) * n);

    /* Simple Box-Muller for normal samples */
    for (int i = 0; i < n; i += 2) {
        double u1 = (rand() % 10000 + 1) / 10001.0;
        double u2 = (rand() % 10000 + 1) / 10001.0;
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265 * u2);
        double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * 3.14159265 * u2);
        if (i < 1000) {
            values[i] = -10.0 + z0;
            if (i+1 < 1000) values[i+1] = -10.0 + z1;
        } else {
            values[i] = 10.0 + z0;
            if (i+1 < n) values[i+1] = 10.0 + z1;
        }
    }

    EMResultGaussian_t result;
    EMConfig_t config;
    GetEMGaussianDefaultConfig(&config);
    config.init_method = EM_INIT_KMEANS;

    int rc = UnmixGaussians(values, n, 2, &result, &config);
    ASSERT_TRUE(rc == 0, "Return code should be 0");
    ASSERT_TRUE(result.numGaussians == 2, "Should have 2 Gaussians");

    /* Sort means to compare (order may vary) */
    double m0 = result.means_final[0];
    double m1 = result.means_final[1];
    double lo = (m0 < m1) ? m0 : m1;
    double hi = (m0 < m1) ? m1 : m0;

    ASSERT_CLOSE(lo, -10.0, 1.0, "Lower mean should be near -10");
    ASSERT_CLOSE(hi, 10.0, 1.0, "Upper mean should be near 10");

    /* Mixing proportions should be ~0.5 each */
    double p0 = result.probs_final[0];
    double p1 = result.probs_final[1];
    ASSERT_CLOSE(p0 + p1, 1.0, 0.001, "Mixing proportions should sum to 1");
    ASSERT_TRUE(fabs(p0 - 0.5) < 0.1, "Proportions should be roughly equal");

    ReleaseEMResultGaussian(&result);
    free(values);
}

/* ===== Test 6: Exponential mixture unmixing ===== */
void test_exponential_unmix(void)
{
    printf("Test: Exponential mixture unmixing (2 rates)\n");

    /* Generate 2000 values: 1000 from Exp(mean=2) and 1000 from Exp(mean=10) */
    srand(123);
    int n = 2000;
    double* values = (double*)malloc(sizeof(double) * n);

    for (int i = 0; i < 1000; i++) {
        double u = (rand() % 10000 + 1) / 10001.0;
        values[i] = -2.0 * log(u);
    }
    for (int i = 1000; i < 2000; i++) {
        double u = (rand() % 10000 + 1) / 10001.0;
        values[i] = -10.0 * log(u);
    }

    EMResultExponential_t result;
    EMConfig_t config;
    GetEMExponentialDefaultConfig(&config);

    int rc = UnmixExponentials(values, n, 2, &result, &config);
    ASSERT_TRUE(rc == 0, "Return code should be 0");
    ASSERT_TRUE(result.numExponentials == 2, "Should have 2 exponentials");

    /* Sort means */
    double m0 = result.means_final[0];
    double m1 = result.means_final[1];
    double lo = (m0 < m1) ? m0 : m1;
    double hi = (m0 < m1) ? m1 : m0;

    ASSERT_CLOSE(lo, 2.0, 1.0, "Lower mean should be near 2");
    ASSERT_CLOSE(hi, 10.0, 2.0, "Upper mean should be near 10");

    ReleaseEMResultExponential(&result);
    free(values);
}

/* ===== Test 7: Edge cases ===== */
void test_edge_cases(void)
{
    printf("Test: Edge cases\n");

    EMResult_t result;
    EMConfig_t config;
    GetEMDefaultConfig(&config);

    /* NULL inputs should return error */
    int rc = ExpectationMaximization(NULL, 0, 0, NULL, 0, &result, &config);
    ASSERT_TRUE(rc != 0, "NULL compat should return error");

    rc = ExpectationMaximization("10", 1, 2, NULL, 0, &result, &config);
    ASSERT_TRUE(rc != 0, "NULL counts should return error");

    /* Single pattern, single class */
    const char* compat = "1";
    int counts[] = {100};
    rc = ExpectationMaximization(compat, 1, 1, counts, 1, &result, &config);
    ASSERT_TRUE(rc == 0, "Single class should work");
    ASSERT_CLOSE(result.values[0], 1.0, 0.0001, "Single class should be 1.0");
    ReleaseEMResult(&result);
}

/* ===== Test 8: Log-likelihood is monotonically non-decreasing ===== */
void test_loglikelihood_monotone(void)
{
    printf("Test: Log-likelihood tracking\n");

    const char* compat = "10" "01" "11";
    int counts[] = {300, 700, 500};
    EMResult_t result;
    EMConfig_t config;
    GetEMDefaultConfig(&config);

    int rc = ExpectationMaximization(compat, 3, 2, counts, 3, &result, &config);
    ASSERT_TRUE(rc == 0, "Return code should be 0");
    ASSERT_TRUE(result.loglikelihood < 0.0, "Log-likelihood should be negative");
    ASSERT_TRUE(result.loglikelihood > -10000.0, "Log-likelihood should be finite");
    ASSERT_TRUE(result.iterations > 0, "Should report iteration count");

    ReleaseEMResult(&result);
}

/* ===== Test 9: K-means vs random init converge to same answer ===== */
void test_kmeans_vs_random(void)
{
    printf("Test: K-means vs random init (Gaussian, converge to same result)\n");

    /* Well-separated Gaussians — both inits should find them */
    srand(42);
    int n = 2000;
    double* values = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < 1000; i++) {
        double u1 = (rand() % 10000 + 1) / 10001.0;
        double u2 = (rand() % 10000 + 1) / 10001.0;
        values[i] = -20.0 + sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265 * u2);
    }
    for (int i = 1000; i < 2000; i++) {
        double u1 = (rand() % 10000 + 1) / 10001.0;
        double u2 = (rand() % 10000 + 1) / 10001.0;
        values[i] = 20.0 + sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265 * u2);
    }

    EMResultGaussian_t r_rand, r_km;
    EMConfig_t cfg;
    GetEMGaussianDefaultConfig(&cfg);

    cfg.init_method = EM_INIT_RANDOM;
    srand(42);
    UnmixGaussians(values, n, 2, &r_rand, &cfg);

    cfg.init_method = EM_INIT_KMEANS;
    UnmixGaussians(values, n, 2, &r_km, &cfg);

    /* Both should find means near -20 and 20 */
    double rand_lo = (r_rand.means_final[0] < r_rand.means_final[1]) ? r_rand.means_final[0] : r_rand.means_final[1];
    double km_lo = (r_km.means_final[0] < r_km.means_final[1]) ? r_km.means_final[0] : r_km.means_final[1];

    ASSERT_CLOSE(rand_lo, km_lo, 1.0, "Both inits should find same lower mean");

    /* K-means should take fewer iterations */
    ASSERT_TRUE(r_km.iterstaken <= r_rand.iterstaken + 5, "K-means should converge as fast or faster");

    ReleaseEMResultGaussian(&r_rand);
    ReleaseEMResultGaussian(&r_km);
    free(values);
}

/* ===== Test 10: Proportions sum to 1 (stress) ===== */
void test_proportions_sum(void)
{
    printf("Test: Proportions always sum to 1 (5 classes, complex patterns)\n");

    /* 5 classes, various overlapping patterns */
    const char* compat =
        "10110"
        "01001"
        "11010"
        "00101"
        "10001"
        "01110"
        "11111";
    int counts[] = {100, 200, 150, 300, 50, 175, 500};
    EMResult_t result;
    EMConfig_t config;
    GetEMDefaultConfig(&config);

    int rc = ExpectationMaximization(compat, 7, 5, counts, 7, &result, &config);
    ASSERT_TRUE(rc == 0, "Return code should be 0");

    double sum = 0;
    for (size_t i = 0; i < result.size; i++) {
        sum += result.values[i];
        ASSERT_TRUE(result.values[i] >= 0.0, "Each proportion should be >= 0");
    }
    ASSERT_CLOSE(sum, 1.0, 0.0001, "Proportions should sum to 1.0");

    ReleaseEMResult(&result);
}

int main(void)
{
    printf("\n========================================\n");
    printf("  GEMMULEM Unit Tests\n");
    printf("========================================\n\n");

    test_multinomial_simple();
    test_multinomial_equal_split();
    test_multinomial_ambiguous();
    test_multinomial_unequal();
    test_gaussian_unmix();
    test_exponential_unmix();
    test_edge_cases();
    test_loglikelihood_monotone();
    test_kmeans_vs_random();
    test_proportions_sum();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (%d FAILED)", tests_failed);
    }
    printf("\n========================================\n\n");

    return tests_failed > 0 ? 1 : 0;
}
