#!/usr/bin/env python3
"""Smoke test: verify all major CLI features work end-to-end."""
import numpy as np, subprocess, os, sys

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)
passed = 0
failed = 0

def run(name, cmd, expect_in_output=None):
    global passed, failed
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"  ❌ {name}: exit code {r.returncode}")
            if r.stderr: print(f"     {r.stderr[:200]}")
            failed += 1
            return False
        if expect_in_output and expect_in_output not in r.stdout:
            print(f"  ❌ {name}: missing '{expect_in_output}' in output")
            failed += 1
            return False
        print(f"  ✅ {name}")
        passed += 1
        return True
    except subprocess.TimeoutExpired:
        print(f"  ❌ {name}: TIMEOUT")
        failed += 1
        return False

print("=" * 60)
print("  SMOKE TESTS — End-to-end CLI verification")
print("=" * 60)

# Generate test data
gauss2 = np.concatenate([rng.normal(-5, 1, 2000), rng.normal(5, 1, 2000)])
np.savetxt("/tmp/gauss2.txt", gauss2)

gauss3 = np.concatenate([rng.normal(-10, 1, 1500), rng.normal(0, 1.5, 2000), rng.normal(10, 0.8, 1500)])
np.savetxt("/tmp/gauss3.txt", gauss3)

expo2 = np.concatenate([rng.exponential(1, 2000), rng.exponential(8, 2000)])
np.savetxt("/tmp/expo2.txt", expo2)

poisson2 = np.concatenate([rng.poisson(3, 2000), rng.poisson(20, 2000)]).astype(float)
np.savetxt("/tmp/poisson2.txt", poisson2)

beta2 = np.concatenate([rng.beta(2, 8, 2000), rng.beta(8, 2, 2000)])
np.savetxt("/tmp/beta2.txt", beta2)

mv2d = np.vstack([
    rng.multivariate_normal([-5, 0], [[1, 0], [0, 1]], 1000),
    rng.multivariate_normal([5, 0], [[1, 0], [0, 1]], 1000)
])
# Use space-separated (gemmulem -g expects space/tab by default)
np.savetxt("/tmp/mv2d.txt", mv2d, delimiter=" ")

single = rng.normal(0, 1, 3000)
np.savetxt("/tmp/single.txt", single)

large = np.concatenate([rng.normal(j * 8, 1, 5000) for j in range(3)])
np.savetxt("/tmp/large.txt", large)

print("\n--- Basic Gaussian ---")
run("Gaussian k=2", [GEM, "-g", "/tmp/gauss2.txt", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("Gaussian k=3", [GEM, "-g", "/tmp/gauss3.txt", "-k", "3", "-o", "/tmp/out.csv"], "Component")
run("Gaussian k=1", [GEM, "-g", "/tmp/single.txt", "-k", "1", "-o", "/tmp/out.csv"], "Component")

print("\n--- Other families ---")
run("Exponential k=2", [GEM, "-g", "/tmp/expo2.txt", "-d", "exponential", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("Gamma k=2", [GEM, "-g", "/tmp/expo2.txt", "-d", "gamma", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("Poisson k=2", [GEM, "-g", "/tmp/poisson2.txt", "-d", "poisson", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("Beta k=2", [GEM, "-g", "/tmp/beta2.txt", "-d", "beta", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("Laplace k=2", [GEM, "-g", "/tmp/gauss2.txt", "-d", "laplace", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("Student-t k=2", [GEM, "-g", "/tmp/gauss2.txt", "-d", "studentt", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("Weibull k=2", [GEM, "-g", "/tmp/expo2.txt", "-d", "weibull", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("LogNormal k=2", [GEM, "-g", "/tmp/expo2.txt", "-d", "lognormal", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("Cauchy k=1", [GEM, "-g", "/tmp/gauss2.txt", "-d", "cauchy", "-k", "1", "-o", "/tmp/out.csv"], "Component")
run("Gumbel k=2", [GEM, "-g", "/tmp/gauss2.txt", "-d", "gumbel", "-k", "2", "-o", "/tmp/out.csv"], "Component")
run("HalfNormal k=1", [GEM, "-g", "/tmp/expo2.txt", "-d", "halfnormal", "-k", "1", "-o", "/tmp/out.csv"], "Component")
run("Rayleigh k=1", [GEM, "-g", "/tmp/expo2.txt", "-d", "rayleigh", "-k", "1", "-o", "/tmp/out.csv"], "Component")
run("InvGaussian k=1", [GEM, "-g", "/tmp/expo2.txt", "-d", "invgaussian", "-k", "1", "-o", "/tmp/out.csv"], "Component")
run("Logistic k=2", [GEM, "-g", "/tmp/gauss2.txt", "-d", "logistic", "-k", "2", "-o", "/tmp/out.csv"], "Component")

print("\n--- Adaptive EM ---")
run("Adaptive BIC", [GEM, "-g", "/tmp/gauss2.txt", "--adaptive", "--kmethod", "bic", "-o", "/tmp/out.csv"], "Adaptive Result")
run("Adaptive AIC", [GEM, "-g", "/tmp/gauss2.txt", "--adaptive", "--kmethod", "aic", "-o", "/tmp/out.csv"], "Adaptive Result")
run("Adaptive ICL", [GEM, "-g", "/tmp/gauss2.txt", "--adaptive", "--kmethod", "icl", "-o", "/tmp/out.csv"], "Adaptive Result")
run("Adaptive VBEM", [GEM, "-g", "/tmp/gauss2.txt", "--adaptive", "--kmethod", "vbem", "-o", "/tmp/out.csv"], "Adaptive Result")
run("Adaptive MML", [GEM, "-g", "/tmp/gauss2.txt", "--adaptive", "--kmethod", "mml", "-o", "/tmp/out.csv"], "Adaptive Result")

print("\n--- Streaming ---")
run("Streaming k=2", [GEM, "-g", "/tmp/gauss2.txt", "-k", "2", "--stream", "-o", "/tmp/out.csv"], "Component")

print("\n--- Online EM ---")
run("Online k=2", [GEM, "-g", "/tmp/gauss2.txt", "-k", "2", "--online", "--batch-size", "500", "-o", "/tmp/out.csv"], "Component")

print("\n--- Multivariate ---")
# Note: -m is MAXIT, not multivariate! Use --mv with -g for multivariate data
run("MV 2D full k=2", [GEM, "-g", "/tmp/mv2d.txt", "--mv", "-k", "2", "--dim", "2", "--cov", "full", "-o", "/tmp/out.csv"], "Component")
run("MV 2D diag k=2", [GEM, "-g", "/tmp/mv2d.txt", "--mv", "-k", "2", "--dim", "2", "--cov", "diagonal", "-o", "/tmp/out.csv"], "Component")
run("MV 2D spherical k=2", [GEM, "-g", "/tmp/mv2d.txt", "--mv", "-k", "2", "--dim", "2", "--cov", "spherical", "-o", "/tmp/out.csv"], "Component")
run("MV auto-k", [GEM, "-g", "/tmp/mv2d.txt", "--mv-autok", "--dim", "2", "--kmin", "1", "--kmax", "5", "-o", "/tmp/out.csv"])

print("\n--- Large data ---")
run("Large n=15K k=3", [GEM, "-g", "/tmp/large.txt", "-k", "3", "-o", "/tmp/out.csv"], "Component")

print("\n--- CSV output check ---")
run("Output file exists", ["test", "-f", "/tmp/out.csv"])
r = subprocess.run(["wc", "-l", "/tmp/out.csv"], capture_output=True, text=True)
print(f"     Output has {r.stdout.strip()} lines")

print(f"\n{'=' * 60}")
print(f"  RESULTS: {passed}/{passed + failed} passed", end="")
if failed: print(f" ({failed} FAILED)")
else: print()
print(f"{'=' * 60}")
sys.exit(1 if failed else 0)
