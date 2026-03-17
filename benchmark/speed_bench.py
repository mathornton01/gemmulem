#!/usr/bin/env python3
"""Speed benchmark for Gemmulem vs competitors."""
import numpy as np, time, subprocess, tempfile, os, sys

BUILD = os.path.dirname(os.path.abspath(__file__)) + "/../build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)

def bench_gem(data, args, label=""):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(f'{x}\n')
    f.close()
    t0 = time.perf_counter()
    r = subprocess.run([GEM, '-g', f.name] + args + ['-o', '/dev/null'],
                       capture_output=True, text=True, timeout=120)
    t = (time.perf_counter() - t0) * 1000
    os.unlink(f.name)
    return t

print("=" * 70)
print("  GEMMULEM Speed Benchmark")
print("=" * 70)

# Scaling test
print("\n--- Scaling: 2-Gaussian, k=2, standard EM ---")
for n in [1000, 5000, 10000, 50000, 100000]:
    d = np.concatenate([rng.normal(-4, 1, n//2), rng.normal(4, 1, n//2)])
    t = bench_gem(d, ['-d', 'Gaussian', '-k', '2'])
    print(f"  n={n:>7d}  time={t:>8.1f}ms  throughput={n/t*1000:>10.0f} pts/sec")

# Online vs standard
print("\n--- Online vs Standard EM: n=50000, k=2 ---")
d = np.concatenate([rng.normal(-4, 1, 25000), rng.normal(4, 1, 25000)])
t_std = bench_gem(d, ['-d', 'Gaussian', '-k', '2'])
t_onl = bench_gem(d, ['-d', 'Gaussian', '-k', '2', '--online', '--batch-size', '256'])
print(f"  Standard:  {t_std:>8.1f}ms")
print(f"  Online:    {t_onl:>8.1f}ms  ({t_std/t_onl:.1f}x speedup)")

# Family speed comparison
print("\n--- Family speed: n=10000, k=2 ---")
d = np.concatenate([rng.normal(-4, 1, 5000), rng.normal(4, 1, 5000)])
for fam in ['Gaussian', 'StudentT', 'Gamma', 'Laplace', 'Logistic', 'Cauchy', 'SkewNormal', 'Gumbel']:
    t = bench_gem(d, ['-d', fam, '-k', '2'])
    print(f"  {fam:<12s}  {t:>8.1f}ms")

# Adaptive with different k-methods
print("\n--- Adaptive k-selection: n=2000, mixed data ---")
d = np.concatenate([rng.normal(-5, 1, 1000), rng.gamma(4, 2, 1000)])
for km in ['bic', 'aic', 'icl', 'vbem', 'mml']:
    t = bench_gem(d, ['--adaptive', '--kmethod', km, '--kmax', '4'])
    print(f"  --kmethod {km:<5s}  {t:>8.1f}ms")

# Compare with sklearn if available
print("\n--- sklearn comparison (if available) ---")
try:
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    d = np.concatenate([rng.normal(-4, 1, 25000), rng.normal(4, 1, 25000)])
    
    t0 = time.perf_counter()
    gm = GaussianMixture(n_components=2, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
    t_sk = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    bgm = BayesianGaussianMixture(n_components=5, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
    t_sk_bayes = (time.perf_counter() - t0) * 1000
    
    t_gem = bench_gem(d, ['-d', 'Gaussian', '-k', '2'])
    t_gem_vbem = bench_gem(d, ['--adaptive', '--kmethod', 'vbem', '--kmax', '5'])
    
    print(f"  sklearn GMM (k=2):         {t_sk:>8.1f}ms")
    print(f"  sklearn BayesGMM (k≤5):    {t_sk_bayes:>8.1f}ms")
    print(f"  Gemmulem standard (k=2):   {t_gem:>8.1f}ms  ({t_sk/t_gem:.1f}x vs sklearn)")
    print(f"  Gemmulem VBEM (k≤5):       {t_gem_vbem:>8.1f}ms  ({t_sk_bayes/t_gem_vbem:.1f}x vs sklearn)")
except ImportError:
    print("  sklearn not installed — skipping")

# Compare with R mixtools if available
print("\n--- R mixtools comparison (if available) ---")
try:
    r = subprocess.run(['Rscript', '-e', 'library(mixtools)'], capture_output=True, timeout=10)
    if r.returncode == 0:
        d = np.concatenate([rng.normal(-4, 1, 10000), rng.normal(4, 1, 10000)])
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir='/tmp')
        for x in d: f.write(f'{x}\n')
        f.close()
        t0 = time.perf_counter()
        subprocess.run(['Rscript', '-e', f'd=scan("{f.name}",quiet=T);library(mixtools);normalmixEM(d,k=2,maxit=300,epsilon=1e-5)'],
                      capture_output=True, timeout=60)
        t_r = (time.perf_counter() - t0) * 1000
        t_gem = bench_gem(d, ['-d', 'Gaussian', '-k', '2'])
        print(f"  R mixtools (k=2):  {t_r:>8.1f}ms")
        print(f"  Gemmulem (k=2):    {t_gem:>8.1f}ms  ({t_r/t_gem:.1f}x faster)")
        os.unlink(f.name)
    else:
        print("  R mixtools not installed — skipping")
except Exception:
    print("  R not found — skipping")

print("\n" + "=" * 70)
