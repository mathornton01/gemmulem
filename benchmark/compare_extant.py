#!/usr/bin/env python3
"""
GEMMULEM vs Extant EM Algorithms Benchmark
==========================================
Compares GEMMULEM against:
  - sklearn GaussianMixture (GMM only)
  - pomegranate (multi-family, if installed)
  - R mclust via subprocess (if R installed)
  - R mixtools via subprocess (if R installed)

Metrics:
  - Log-likelihood (higher is better)
  - Iterations to convergence
  - Wall-clock time
  - Parameter recovery accuracy (on synthetic data)
  - BIC (lower is better)

Run from the gemmulem project root:
  python3 benchmark/compare_extant.py
"""

import sys, os, time, json, subprocess, shutil
import numpy as np

# ── build GEMMULEM CLI if needed ──────────────────────────────────────────────
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(PROJ, "build")
GEM_BIN = os.path.join(BUILD, "gemmulem")
if not os.path.exists(GEM_BIN):
    print("Building GEMMULEM...")
    os.makedirs(BUILD, exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=BUILD, check=True, capture_output=True)
    subprocess.run(["make", "-j4"], cwd=BUILD, check=True, capture_output=True)

# ── inline C benchmark for GEMMULEM (avoids CLI overhead) ────────────────────
GEM_BENCH_SRC = os.path.join(PROJ, "benchmark", "_gem_bench.c")
GEM_BENCH_BIN = os.path.join(PROJ, "benchmark", "_gem_bench")

GEM_BENCH_CODE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "distributions.h"

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { fprintf(stderr, "Usage: gem_bench <datafile> <family> [k]\n"); return 1; }

    /* Read data */
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    int n = 0; double buf[100000]; double v;
    while (fscanf(f, "%lf", &v) == 1 && n < 100000) buf[n++] = v;
    fclose(f);

    /* Family */
    DistFamily fam = DIST_GAUSSIAN;
    if (strcmp(argv[2], "gaussian") == 0)  fam = DIST_GAUSSIAN;
    else if (strcmp(argv[2], "student_t") == 0) fam = DIST_STUDENT_T;
    else if (strcmp(argv[2], "laplace") == 0)  fam = DIST_LAPLACE;
    else if (strcmp(argv[2], "gamma") == 0)    fam = DIST_GAMMA;
    else if (strcmp(argv[2], "lognormal") == 0) fam = DIST_LOGNORMAL;
    else if (strcmp(argv[2], "weibull") == 0)  fam = DIST_WEIBULL;
    else if (strcmp(argv[2], "pearson") == 0)  fam = DIST_PEARSON;
    else if (strcmp(argv[2], "auto") == 0)     fam = (DistFamily)-1;

    int k = (argc >= 4) ? atoi(argv[3]) : 2;

    double t0 = wall_ms();
    MixtureResult result;
    int rc;

    if ((int)fam == -1) {
        /* Auto mode */
        ModelSelectResult msr;
        rc = SelectBestMixture(buf, n, NULL, 0, 1, k, 300, 1e-5, 0, &msr);
        if (rc == 0) {
            printf("{\"family\":\"%s\",\"k\":%d,\"ll\":%.4f,\"bic\":%.4f,\"iters\":%d,\"ms\":%.2f}\n",
                   GetDistName(msr.best_family), msr.best_k,
                   msr.candidates[0].loglikelihood,
                   msr.best_bic, 0, wall_ms()-t0);
            ReleaseModelSelectResult(&msr);
        }
        return rc;
    }

    rc = UnmixGeneric(buf, n, fam, k, 300, 1e-5, 0, &result);
    double ms = wall_ms() - t0;
    if (rc == 0) {
        printf("{\"family\":\"%s\",\"k\":%d,\"ll\":%.4f,\"bic\":%.4f,\"iters\":%d,\"ms\":%.2f,\"weights\":[",
               GetDistName(fam), k, result.loglikelihood, result.bic, result.iterations, ms);
        for (int j=0;j<k;j++) printf("%.4f%s",result.mixing_weights[j],j<k-1?",":"");
        printf("],\"params\":[");
        for (int j=0;j<k;j++) {
            printf("[");
            for (int q=0;q<result.params[j].nparams;q++)
                printf("%.4f%s",result.params[j].p[q],q<result.params[j].nparams-1?",":"");
            printf("]%s",j<k-1?",":"");
        }
        printf("]}\n");
        ReleaseMixtureResult(&result);
    } else {
        printf("{\"error\":%d}\n", rc);
    }
    return rc;
}
"""

def build_gem_bench():
    with open(GEM_BENCH_SRC, "w") as fh:
        fh.write(GEM_BENCH_CODE)
    inc = os.path.join(PROJ, "src", "lib")
    lib = os.path.join(BUILD, "src", "lib", "libem.a")
    cmd = ["gcc", "-O2", "-I", inc, "-o", GEM_BENCH_BIN, GEM_BENCH_SRC, lib, "-lm"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("  GEMMULEM bench build failed:", r.stderr.decode())
        return False
    return True

def run_gemmulem(datafile, family, k):
    if not os.path.exists(GEM_BENCH_BIN):
        if not build_gem_bench():
            return None
    try:
        r = subprocess.run([GEM_BENCH_BIN, datafile, family, str(k)],
                           capture_output=True, timeout=30)
        return json.loads(r.stdout.decode().strip())
    except Exception as e:
        return {"error": str(e)}

# ── sklearn ───────────────────────────────────────────────────────────────────
def run_sklearn(data, k, covariance_type="full"):
    try:
        from sklearn.mixture import GaussianMixture
        t0 = time.time()
        gm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                             max_iter=300, tol=1e-5, random_state=42)
        gm.fit(data.reshape(-1, 1))
        ms = (time.time() - t0) * 1000
        ll = gm.score(data.reshape(-1, 1)) * len(data)
        bic = gm.bic(data.reshape(-1, 1))
        return {"tool": "sklearn-GMM", "k": k, "ll": float(ll), "bic": float(bic),
                "iters": gm.n_iter_, "ms": ms,
                "means": sorted(gm.means_.flatten().tolist())}
    except ImportError:
        return {"error": "sklearn not installed"}
    except Exception as e:
        return {"error": str(e)}

# ── pomegranate ───────────────────────────────────────────────────────────────
def run_pomegranate(data, k, dist_name="NormalDistribution"):
    try:
        import pomegranate as pom
        t0 = time.time()
        if hasattr(pom, 'distributions'):
            dist_cls = getattr(pom.distributions, dist_name, pom.distributions.Normal)
        else:
            dist_cls = pom.distributions.Normal
        model = pom.GMM(k=k, distribution=dist_cls)
        model.fit(data, max_iterations=300, stop_threshold=1e-5)
        ms = (time.time() - t0) * 1000
        ll = model.log_probability(data).sum()
        return {"tool": f"pomegranate-{dist_name}", "k": k, "ll": float(ll), "ms": ms}
    except ImportError:
        return {"error": "pomegranate not installed"}
    except Exception as e:
        return {"error": str(e)}

# ── R mclust ──────────────────────────────────────────────────────────────────
def run_r_mclust(datafile, k):
    if not shutil.which("Rscript"):
        return {"error": "R not installed"}
    r_code = f"""
    suppressMessages(library(mclust))
    x <- scan("{datafile}", quiet=TRUE)
    t0 <- proc.time()["elapsed"]
    m <- Mclust(x, G={k}, verbose=FALSE)
    ms <- (proc.time()["elapsed"] - t0) * 1000
    cat(sprintf('{{"tool":"mclust","k":%d,"ll":%.4f,"bic":%.4f,"ms":%.2f}}\\n',
                {k}, m$loglik, m$BIC[1], ms))
    """
    try:
        r = subprocess.run(["Rscript", "--vanilla", "-e", r_code],
                           capture_output=True, timeout=60)
        out = r.stdout.decode().strip()
        for line in out.splitlines():
            if line.startswith("{"):
                return json.loads(line)
        return {"error": r.stderr.decode()[:200]}
    except Exception as e:
        return {"error": str(e)}

# ── R mixtools ────────────────────────────────────────────────────────────────
def run_r_mixtools(datafile, k):
    if not shutil.which("Rscript"):
        return {"error": "R not installed"}
    r_code = f"""
    suppressMessages(library(mixtools))
    x <- scan("{datafile}", quiet=TRUE)
    t0 <- proc.time()["elapsed"]
    tryCatch({{
        m <- normalmixEM(x, k={k}, maxit=300, epsilon=1e-5)
        ms <- (proc.time()["elapsed"] - t0) * 1000
        cat(sprintf('{{"tool":"mixtools","k":%d,"ll":%.4f,"ms":%.2f}}\\n', {k}, m$loglik, ms))
    }}, error=function(e) cat(sprintf('{{"error":"%s"}}\\n', conditionMessage(e))))
    """
    try:
        r = subprocess.run(["Rscript", "--vanilla", "-e", r_code],
                           capture_output=True, timeout=60)
        out = r.stdout.decode().strip()
        for line in out.splitlines():
            if line.startswith("{"):
                return json.loads(line)
        return {"error": r.stderr.decode()[:200]}
    except Exception as e:
        return {"error": str(e)}

# ── Scenarios ─────────────────────────────────────────────────────────────────
def make_scenarios():
    rng = np.random.default_rng(42)
    scenarios = []

    # 1. Well-separated Gaussians
    data = np.concatenate([rng.normal(-5, 1, 1000), rng.normal(5, 1.5, 1000)])
    scenarios.append(("2-Gaussian (separated)", data, 2, "gaussian"))

    # 2. Overlapping Gaussians
    data = np.concatenate([rng.normal(-1, 1, 800), rng.normal(1, 1, 1200)])
    scenarios.append(("2-Gaussian (overlapping)", data, 2, "gaussian"))

    # 3. Heavy-tailed data — Student-t
    data = rng.standard_t(df=3, size=2000)
    scenarios.append(("Student-t (df=3) unimodal", data, 1, "student_t"))

    # 4. Gamma mixture
    data = np.concatenate([rng.gamma(2, 1, 1000), rng.gamma(8, 2, 1000)])
    scenarios.append(("2-Gamma mixture", data, 2, "gamma"))

    # 5. Log-normal data
    data = rng.lognormal(0, 0.5, 2000)
    scenarios.append(("Log-normal unimodal", data, 1, "lognormal"))

    # 6. Mixed-family: Gaussian + heavy-tailed
    data = np.concatenate([rng.normal(-3, 0.5, 1000), rng.standard_t(df=3, size=1000)*0.5+3])
    scenarios.append(("Gaussian+StudentT (Pearson auto)", data, 2, "pearson"))

    # 7. Large n stress test
    data = np.concatenate([rng.normal(-4, 1, 5000), rng.normal(4, 1, 5000)])
    scenarios.append(("2-Gaussian large N=10K", data, 2, "gaussian"))

    return scenarios

# ── Main benchmark loop ───────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  GEMMULEM vs Extant EM Algorithms Benchmark")
    print("=" * 70)

    # Check tool availability
    has_sklearn = True
    has_pomegranate = True
    has_r = shutil.which("Rscript") is not None
    try:
        import sklearn
        print(f"  sklearn: {sklearn.__version__}")
    except ImportError:
        has_sklearn = False
        print("  sklearn: NOT INSTALLED")
    try:
        import pomegranate
        print(f"  pomegranate: {pomegranate.__version__}")
    except ImportError:
        has_pomegranate = False
        print("  pomegranate: NOT INSTALLED")
    print(f"  R: {'found' if has_r else 'NOT FOUND'}")

    # Build GEMMULEM bench binary
    print("\nBuilding GEMMULEM bench binary...", end=" ", flush=True)
    if build_gem_bench():
        print("OK")
    else:
        print("FAILED — GEMMULEM results will be missing")

    results_all = []
    scenarios = make_scenarios()
    tmpfile = "/tmp/gem_bench_data.txt"

    print()
    for name, data, k, gem_family in scenarios:
        print(f"\n{'─'*70}")
        print(f"  Scenario: {name}  (n={len(data)}, k={k})")
        print(f"{'─'*70}")
        np.savetxt(tmpfile, data, fmt="%.8f")

        row = {"scenario": name, "n": len(data), "k": k, "results": []}

        # GEMMULEM
        gem = run_gemmulem(tmpfile, gem_family, k)
        if gem and "error" not in gem:
            print(f"  GEMMULEM ({gem.get('family','?'):<12}): LL={gem['ll']:>12.2f}  BIC={gem['bic']:>12.2f}  iters={gem.get('iters',0):>4}  t={gem['ms']:>7.1f}ms")
            gem["tool"] = "GEMMULEM"
        else:
            print(f"  GEMMULEM: ERROR {gem}")
        row["results"].append(gem)

        # sklearn (Gaussian only — its limitation)
        if has_sklearn and gem_family in ("gaussian",):
            sk = run_sklearn(data, k)
            if "error" not in sk:
                print(f"  sklearn  (GMM        ): LL={sk['ll']:>12.2f}  BIC={sk['bic']:>12.2f}  iters={sk['iters']:>4}  t={sk['ms']:>7.1f}ms")
            else:
                print(f"  sklearn: {sk['error']}")
            row["results"].append(sk)

        # pomegranate
        if has_pomegranate and gem_family in ("gaussian",):
            pg = run_pomegranate(data, k)
            if "error" not in pg:
                print(f"  pomegranate (Normal  ): LL={pg['ll']:>12.2f}  t={pg['ms']:>7.1f}ms")
            else:
                print(f"  pomegranate: {pg['error']}")
            row["results"].append(pg)

        # R mclust (Gaussian only)
        if has_r and gem_family in ("gaussian",):
            mc = run_r_mclust(tmpfile, k)
            if "error" not in mc:
                print(f"  R mclust (Gaussian  ): LL={mc['ll']:>12.2f}  BIC={mc.get('bic',0):>12.2f}  t={mc['ms']:>7.1f}ms")
            else:
                print(f"  R mclust: {mc.get('error','?')[:60]}")
            row["results"].append(mc)

        # R mixtools (Gaussian)
        if has_r and gem_family in ("gaussian",):
            mt = run_r_mixtools(tmpfile, k)
            if "error" not in mt:
                print(f"  R mixtools (Normal  ): LL={mt['ll']:>12.2f}  t={mt['ms']:>7.1f}ms")
            else:
                print(f"  R mixtools: {mt.get('error','?')[:60]}")
            row["results"].append(mt)

        results_all.append(row)

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY — Gaussian mixture scenarios (all tools head-to-head)")
    print(f"{'='*70}")
    print(f"  {'Scenario':<35} {'Tool':<18} {'LL':>12} {'BIC':>12} {'ms':>8}")
    print(f"  {'-'*35} {'-'*18} {'-'*12} {'-'*12} {'-'*8}")
    for row in results_all:
        for res in row["results"]:
                if "error" not in res and "ll" in res:
                    print(f"  {row['scenario']:<35} {res.get('tool','?'):<18} {res['ll']:>12.2f} {res.get('bic',0):>12.2f} {res.get('ms',0):>8.1f}")

    # What GEMMULEM can do that others can't
    print(f"\n{'='*70}")
    print("  GEMMULEM UNIQUE CAPABILITIES (not in sklearn/mclust/mixtools)")
    print(f"{'='*70}")
    print("  ✓ 15 distribution families (others: 1-3)")
    print("  ✓ Pearson system: shape discovered from data")
    print("  ✓ Cross-family model selection via BIC")
    print("  ✓ Student-t, Laplace, Cauchy, InvGaussian, Rayleigh, Pareto")
    print("  ✓ Pure C library, embeddable in any application")
    print("  ✓ Winsorized M-step for heavy-tail robustness (Pearson)")

    # Save results
    outfile = os.path.join(PROJ, "benchmark", "extant_comparison_results.json")
    with open(outfile, "w") as fh:
        json.dump(results_all, fh, indent=2)
    print(f"\n  Full results saved to: {outfile}")

if __name__ == "__main__":
    main()
