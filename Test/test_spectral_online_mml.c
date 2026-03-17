/*
 * Tests for: Spectral EM init, Online EM, MML criterion, KDE distribution
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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

/* ===== Spectral Init: recover separated means ===== */
void test_spectral_init_separated(void) {
    printf("Test: Spectral init — well-separated Gaussians\n");
    srand(1111);
    int n = 2000;
    double* data = malloc(sizeof(double) * n);
    for (int i = 0; i < 1000; i++) data[i] = randn(-6, 0.8);
    for (int i = 1000; i < 2000; i++) data[i] = randn(6, 0.8);

    double means[2], weights[2];
    int rc = SpectralInit(data, n, 2, means, weights);
    ASSERT_TRUE(rc == 0, "SpectralInit succeeds");

    /* Means should be roughly -6 and 6 (within 3.0) */
    double lo = (means[0] < means[1]) ? means[0] : means[1];
    double hi = (means[0] < means[1]) ? means[1] : means[0];
    printf("    Spectral means: %.2f, %.2f (true: -6, 6)\n", lo, hi);
    ASSERT_CLOSE(lo, -6.0, 3.0, "Lower mean near -6");
    ASSERT_CLOSE(hi, 6.0, 3.0, "Upper mean near 6");

    /* Weights should be roughly 0.5 each */
    ASSERT_CLOSE(weights[0], 0.5, 0.3, "Weight 0 near 0.5");
    free(data);
}

/* ===== Spectral Init: 3 components ===== */
void test_spectral_init_3comp(void) {
    printf("Test: Spectral init — 3 components\n");
    srand(2222);
    int n = 3000;
    double* data = malloc(sizeof(double) * n);
    for (int i = 0; i < 1000; i++) data[i] = randn(-8, 0.5);
    for (int i = 1000; i < 2000; i++) data[i] = randn(0, 0.5);
    for (int i = 2000; i < 3000; i++) data[i] = randn(8, 0.5);

    double means[3], weights[3];
    int rc = SpectralInit(data, n, 3, means, weights);
    ASSERT_TRUE(rc == 0, "SpectralInit k=3 succeeds");
    printf("    Spectral means: %.2f, %.2f, %.2f\n", means[0], means[1], means[2]);
    free(data);
}

/* ===== Online EM: converges on 2-Gaussian ===== */
void test_online_em_2gauss(void) {
    printf("Test: Online EM — 2-Gaussian convergence\n");
    srand(3333);
    int n = 10000;
    double* data = malloc(sizeof(double) * n);
    for (int i = 0; i < 5000; i++) data[i] = randn(-4, 1);
    for (int i = 5000; i < 10000; i++) data[i] = randn(4, 1);

    MixtureResult result;
    int rc = UnmixOnline(data, n, DIST_GAUSSIAN, 2, 500, 1e-5, 256, 0, &result);
    ASSERT_TRUE(rc == 0, "Online EM succeeds");
    ASSERT_TRUE(result.num_components == 2, "k=2 preserved");

    double m0 = result.params[0].p[0], m1 = result.params[1].p[0];
    double lo = (m0 < m1) ? m0 : m1, hi = (m0 < m1) ? m1 : m0;
    printf("    Online means: %.2f, %.2f (true: -4, 4)\n", lo, hi);
    ASSERT_CLOSE(lo, -4.0, 1.5, "Lower mean near -4");
    ASSERT_CLOSE(hi, 4.0, 1.5, "Upper mean near 4");

    /* Compare with standard EM */
    MixtureResult std_result;
    UnmixGeneric(data, n, DIST_GAUSSIAN, 2, 300, 1e-5, 0, &std_result);
    printf("    Standard EM means: %.2f, %.2f\n", std_result.params[0].p[0], std_result.params[1].p[0]);
    printf("    Online LL=%.1f  Standard LL=%.1f\n", result.loglikelihood, std_result.loglikelihood);

    /* Online should get within 5% of standard EM's LL */
    double ratio = result.loglikelihood / std_result.loglikelihood;
    ASSERT_TRUE(ratio > 0.95 && ratio < 1.05, "Online LL within 5% of standard");

    ReleaseMixtureResult(&result);
    ReleaseMixtureResult(&std_result);
    free(data);
}

/* ===== Online EM: large dataset ===== */
void test_online_em_large(void) {
    printf("Test: Online EM — large dataset (50K)\n");
    srand(4444);
    int n = 50000;
    double* data = malloc(sizeof(double) * n);
    for (int i = 0; i < 25000; i++) data[i] = randn(-3, 1.5);
    for (int i = 25000; i < 50000; i++) data[i] = randn(5, 1);

    MixtureResult result;
    int rc = UnmixOnline(data, n, DIST_GAUSSIAN, 2, 1000, 1e-5, 512, 0, &result);
    ASSERT_TRUE(rc == 0, "Online EM on 50K succeeds");
    printf("    Means: %.2f, %.2f  LL=%.1f\n",
           result.params[0].p[0], result.params[1].p[0], result.loglikelihood);
    ReleaseMixtureResult(&result);
    free(data);
}

/* ===== MML: selects correct k ===== */
void test_mml_correct_k(void) {
    printf("Test: MML criterion — correct k selection\n");
    srand(5555);
    int n = 2000;
    double* data = malloc(sizeof(double) * n);
    for (int i = 0; i < 1000; i++) data[i] = randn(-6, 0.8);
    for (int i = 1000; i < 2000; i++) data[i] = randn(6, 0.8);

    AdaptiveResult r;
    int rc = UnmixAdaptiveEx(data, n, 5, 300, 1e-5, 0, KMETHOD_MML, &r);
    ASSERT_TRUE(rc == 0, "MML adaptive succeeds");
    ASSERT_TRUE(r.num_components >= 2, "MML finds at least 2 components");
    printf("    MML k=%d  BIC=%.1f\n", r.num_components, r.bic);
    ReleaseAdaptiveResult(&r);
    free(data);
}

/* ===== MML vs AIC: MML should be more conservative ===== */
void test_mml_conservative(void) {
    printf("Test: MML more conservative than AIC\n");
    srand(6666);
    int n = 1000;
    double* data = malloc(sizeof(double) * n);
    /* Unimodal data — should be k=1 */
    for (int i = 0; i < n; i++) data[i] = randn(0, 2);

    AdaptiveResult r_mml, r_aic;
    UnmixAdaptiveEx(data, n, 5, 300, 1e-5, 0, KMETHOD_MML, &r_mml);
    UnmixAdaptiveEx(data, n, 5, 300, 1e-5, 0, KMETHOD_AIC, &r_aic);
    printf("    MML k=%d  AIC k=%d\n", r_mml.num_components, r_aic.num_components);
    ASSERT_TRUE(r_mml.num_components <= r_aic.num_components,
                "MML k <= AIC k (more conservative)");
    ReleaseAdaptiveResult(&r_mml);
    ReleaseAdaptiveResult(&r_aic);
    free(data);
}

/* Helper: compute KDE PDF manually (since the C function is static) */
static double kde_pdf_manual(const double* data, size_t n, double x, double h) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double z = (x - data[i]) / h;
        sum += exp(-0.5 * z * z);
    }
    return sum / (n * h * sqrt(2 * 3.14159265));
}

/* ===== KDE: basic PDF sanity ===== */
void test_kde_basic(void) {
    printf("Test: KDE distribution — PDF sanity\n");
    srand(7777);
    int n = 500;
    double* data = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) data[i] = randn(0, 1);

    double h = 0.5;

    /* PDF at mean should be positive */
    double pdf_at_0 = kde_pdf_manual(data, n, 0.0, h);
    ASSERT_TRUE(pdf_at_0 > 0.1, "KDE PDF at mean > 0.1");

    /* PDF far from data should be near 0 */
    double pdf_far = kde_pdf_manual(data, n, 100.0, h);
    ASSERT_TRUE(pdf_far < 0.001, "KDE PDF far from data < 0.001");

    printf("    KDE(0)=%.4f  KDE(100)=%.6f\n", pdf_at_0, pdf_far);

    /* Also test via the distribution framework */
    KDE_SetData(data, n);
    const DistFunctions* kde = GetDistFunctions(DIST_KDE);
    ASSERT_TRUE(kde != NULL, "KDE registered in framework");
    if (kde) {
        DistParams p; p.p[0] = h; p.nparams = 1;
        double v = kde->pdf(0.0, &p);
        ASSERT_CLOSE(v, pdf_at_0, 0.01, "Framework KDE matches manual");
    }
    KDE_SetData(NULL, 0);
    free(data);
}

/* ===== KMethod name ===== */
void test_kmethod_names(void) {
    printf("Test: KMethod names\n");
    ASSERT_TRUE(strcmp(GetKMethodName(KMETHOD_BIC), "BIC") == 0, "BIC name");
    ASSERT_TRUE(strcmp(GetKMethodName(KMETHOD_AIC), "AIC") == 0, "AIC name");
    ASSERT_TRUE(strcmp(GetKMethodName(KMETHOD_ICL), "ICL") == 0, "ICL name");
    ASSERT_TRUE(strcmp(GetKMethodName(KMETHOD_VBEM), "VBEM") == 0, "VBEM name");
    ASSERT_TRUE(strcmp(GetKMethodName(KMETHOD_MML), "MML") == 0, "MML name");
}

int main(void) {
    printf("\n========================================\n");
    printf("  Spectral / Online / MML / KDE Tests\n");
    printf("========================================\n\n");

    test_spectral_init_separated();
    test_spectral_init_3comp();
    test_online_em_2gauss();
    test_online_em_large();
    test_mml_correct_k();
    test_mml_conservative();
    test_kde_basic();
    test_kmethod_names();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) printf(" (%d FAILED)", tests_failed);
    printf("\n========================================\n\n");
    return tests_failed > 0 ? 1 : 0;
}
