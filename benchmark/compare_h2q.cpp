/*
 * Benchmark: GEMMULEM EM vs H2Q EM
 *
 * Generates compatibility pattern data of various sizes and compares:
 *   - Output proportions (should match)
 *   - Convergence speed (iterations)
 *   - Wall clock time
 *
 * Build:
 *   g++ -O3 -o compare_h2q compare_h2q.cpp \
 *       -I../src/lib -L../build/src/lib -lem -lm
 *
 * Note: H2Q's EM is embedded inline here (extracted from quant.cpp)
 *       so we can compare without linking the full H2Q codebase.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <map>
#include <set>

extern "C" {
#include "EM.h"
}

/* ====================================================================
 * H2Q-style EM (extracted/simplified from h2q quant.cpp)
 * Uses map<set<uint>, uint> compatibility matrix like H2Q does.
 * ==================================================================== */
namespace h2q {

static const double CONV_TOL = 0.00001;
static const int MAX_ITER = 1000;

struct Result {
    std::vector<double> abundances;
    int iterations;
    double loglikelihood;
    double elapsed_us;
};

Result calculate(
    const std::vector<size_t>& tranLens,   // transcript lengths (for normalization)
    const std::map<std::set<unsigned>, unsigned>& compMat,  // compat pattern -> count
    bool length_normalize)
{
    size_t T = tranLens.size();
    std::vector<double> a(T, 1.0 / T);  // abundances
    std::vector<double> n(T, 0.0);      // expected counts

    if (compMat.empty()) {
        return {a, 0, 0.0, 0.0};
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int iter = 0;
    for (; iter < MAX_ITER; iter++) {
        // E-step
        std::fill(n.begin(), n.end(), 0.0);
        for (auto& kv : compMat) {
            const auto& pattern = kv.first;
            unsigned count = kv.second;

            double total = 0.0;
            for (unsigned t : pattern) {
                if (length_normalize && tranLens[t] > 0)
                    total += a[t] / tranLens[t];
                else
                    total += a[t];
            }
            if (total <= 0.0) continue;

            for (unsigned t : pattern) {
                double w;
                if (length_normalize && tranLens[t] > 0)
                    w = (a[t] / tranLens[t]) / total;
                else
                    w = a[t] / total;
                n[t] += w * count;
            }
        }

        // M-step
        double total_n = 0.0;
        for (size_t t = 0; t < T; t++) total_n += n[t];
        if (total_n <= 0.0) break;

        double change = 0.0;
        for (size_t t = 0; t < T; t++) {
            double new_a = n[t] / total_n;
            change += fabs(new_a - a[t]);
            a[t] = new_a;
        }

        if (change <= CONV_TOL) {
            iter++;
            break;
        }
    }

    // Log-likelihood
    double ll = 0.0;
    for (auto& kv : compMat) {
        double row_prob = 0.0;
        for (unsigned t : kv.first) {
            if (length_normalize && tranLens[t] > 0)
                row_prob += a[t] / tranLens[t];
            else
                row_prob += a[t];
        }
        if (row_prob > 0.0) ll += kv.second * log(row_prob);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;

    return {a, iter, ll, elapsed};
}

} // namespace h2q


/* ====================================================================
 * Test data generation
 * ==================================================================== */

struct TestCase {
    const char* name;
    int num_transcripts;
    int num_patterns;
    int total_reads;
};

/* Generate random compatibility patterns */
void generate_test(
    int num_trans, int num_patterns, int total_reads, unsigned seed,
    std::vector<std::string>& gem_patterns,
    std::vector<int>& gem_counts,
    std::map<std::set<unsigned>, unsigned>& h2q_compat,
    std::vector<size_t>& trans_lens)
{
    srand(seed);
    gem_patterns.clear();
    gem_counts.clear();
    h2q_compat.clear();
    trans_lens.resize(num_trans);

    // Random transcript lengths (200-5000)
    for (int i = 0; i < num_trans; i++) {
        trans_lens[i] = 200 + rand() % 4800;
    }

    // Generate random patterns
    int reads_left = total_reads;
    for (int p = 0; p < num_patterns && reads_left > 0; p++) {
        std::string pat(num_trans, '0');
        std::set<unsigned> h2q_pat;

        // Each pattern covers 1-5 transcripts
        int coverage = 1 + rand() % 5;
        if (coverage > num_trans) coverage = num_trans;

        for (int c = 0; c < coverage; c++) {
            int t = rand() % num_trans;
            pat[t] = '1';
            h2q_pat.insert(t);
        }

        // Skip all-zero patterns
        if (h2q_pat.empty()) continue;

        int count = (p == num_patterns - 1) ? reads_left : 1 + rand() % (2 * total_reads / num_patterns);
        if (count > reads_left) count = reads_left;
        reads_left -= count;

        gem_patterns.push_back(pat);
        gem_counts.push_back(count);
        h2q_compat[h2q_pat] += count;
    }
}

/* Run GEMMULEM EM and measure time */
struct GemResult {
    std::vector<double> abundances;
    int iterations;
    double loglikelihood;
    double elapsed_us;
};

GemResult run_gemmulem(const std::vector<std::string>& patterns,
                        const std::vector<int>& counts, int num_trans)
{
    // Build concatenated compatibility string
    std::string compat_str;
    for (auto& p : patterns) compat_str += p;

    EMResult_t result;
    EMConfig_t config;
    GetEMDefaultConfig(&config);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    ExpectationMaximization(
        compat_str.data(),
        patterns.size(),
        num_trans,
        counts.data(),
        counts.size(),
        &result, &config);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;

    GemResult gr;
    gr.abundances.assign(result.values, result.values + result.size);
    gr.iterations = result.iterations;
    gr.loglikelihood = result.loglikelihood;
    gr.elapsed_us = elapsed;

    ReleaseEMResult(&result);
    return gr;
}

/* Compare two abundance vectors */
double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    double mx = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        double d = fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

double pearson_r(const std::vector<double>& a, const std::vector<double>& b)
{
    size_t n = a.size();
    if (n < 2) return 0;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; i++) {
        cov += (a[i]-ma)*(b[i]-mb);
        va += (a[i]-ma)*(a[i]-ma);
        vb += (b[i]-mb)*(b[i]-mb);
    }
    return (va > 0 && vb > 0) ? cov / sqrt(va * vb) : 0;
}

int main()
{
    printf("\n");
    printf("============================================================\n");
    printf("  GEMMULEM vs H2Q EM Benchmark\n");
    printf("============================================================\n\n");

    TestCase tests[] = {
        {"Tiny (5 tx, 10 pat, 1K reads)",        5,    10,     1000},
        {"Small (20 tx, 50 pat, 10K reads)",     20,    50,    10000},
        {"Medium (100 tx, 200 pat, 100K reads)", 100,   200,  100000},
        {"Large (500 tx, 1K pat, 500K reads)",   500,  1000,  500000},
        {"XL (2000 tx, 5K pat, 1M reads)",      2000,  5000, 1000000},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    printf("%-45s  %8s  %8s  %10s  %10s  %8s  %10s\n",
           "Scenario", "GEM_iter", "H2Q_iter", "GEM_us", "H2Q_us", "Speedup", "MaxDiff");
    printf("%-45s  %8s  %8s  %10s  %10s  %8s  %10s\n",
           "--------", "--------", "--------", "------", "------", "-------", "-------");

    int all_match = 1;

    for (int t = 0; t < num_tests; t++) {
        TestCase& tc = tests[t];

        std::vector<std::string> gem_pats;
        std::vector<int> gem_counts;
        std::map<std::set<unsigned>, unsigned> h2q_compat;
        std::vector<size_t> trans_lens;

        generate_test(tc.num_transcripts, tc.num_patterns, tc.total_reads, 42 + t,
                      gem_pats, gem_counts, h2q_compat, trans_lens);

        // Run both (no length normalization for apples-to-apples)
        GemResult gem = run_gemmulem(gem_pats, gem_counts, tc.num_transcripts);
        h2q::Result h2q_r = h2q::calculate(trans_lens, h2q_compat, false);

        double diff = max_abs_diff(gem.abundances, h2q_r.abundances);
        double r = pearson_r(gem.abundances, h2q_r.abundances);
        double speedup = (gem.elapsed_us > 0) ? h2q_r.elapsed_us / gem.elapsed_us : 0;

        printf("%-45s  %8d  %8d  %10.0f  %10.0f  %7.2fx  %10.2e",
               tc.name, gem.iterations, h2q_r.iterations,
               gem.elapsed_us, h2q_r.elapsed_us, speedup, diff);

        if (diff > 0.001) {
            printf("  *** MISMATCH (r=%.4f)", r);
            all_match = 0;
        } else {
            printf("  OK (r=%.6f)", r);
        }
        printf("\n");
    }

    printf("\n");

    /* Now run WITH length normalization in H2Q to show the difference */
    printf("============================================================\n");
    printf("  Length Normalization Comparison (H2Q only)\n");
    printf("============================================================\n\n");

    printf("%-45s  %10s  %10s  %10s\n",
           "Scenario", "H2Q_noLN_LL", "H2Q_LN_LL", "LL_diff");
    printf("%-45s  %10s  %10s  %10s\n",
           "--------", "-----------", "----------", "-------");

    for (int t = 0; t < num_tests; t++) {
        TestCase& tc = tests[t];
        std::vector<std::string> gem_pats;
        std::vector<int> gem_counts;
        std::map<std::set<unsigned>, unsigned> h2q_compat;
        std::vector<size_t> trans_lens;

        generate_test(tc.num_transcripts, tc.num_patterns, tc.total_reads, 42 + t,
                      gem_pats, gem_counts, h2q_compat, trans_lens);

        h2q::Result no_ln = h2q::calculate(trans_lens, h2q_compat, false);
        h2q::Result with_ln = h2q::calculate(trans_lens, h2q_compat, true);

        printf("%-45s  %10.2f  %10.2f  %10.2f\n",
               tc.name, no_ln.loglikelihood, with_ln.loglikelihood,
               with_ln.loglikelihood - no_ln.loglikelihood);
    }

    printf("\n");
    if (all_match) {
        printf("RESULT: All scenarios match (max diff < 0.001)\n");
        printf("Both EMs converge to the same proportions.\n");
    } else {
        printf("RESULT: Some scenarios diverged — check details above.\n");
    }
    printf("\n");

    return all_match ? 0 : 1;
}
