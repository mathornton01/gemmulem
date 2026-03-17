#!/usr/bin/env python3
"""Final comprehensive benchmark."""
import numpy as np, time, subprocess, tempfile, os
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)

def bench_gem(data, args, timeout=60):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(str(x) + '\n')
    f.close()
    try:
        t0 = time.perf_counter()
        subprocess.run([GEM, '-g', f.name] + args + ['-o', '/dev/null'],
                       capture_output=True, text=True, timeout=timeout)
        t = (time.perf_counter() - t0) * 1000
    except subprocess.TimeoutExpired:
        t = timeout * 1000
    os.unlink(f.name)
    return t

print("=" * 72)
print("  GEMMULEM v2.0 Final Benchmark (with OpenMP + OpenCL stub)")
print("=" * 72)

# Test 1: Standard EM vs sklearn
print("\n─── Gaussian k=2, varying n ───")
print(f"  {'n':>8}  {'Gemmulem':>10}  {'sklearn':>10}  {'speedup':>8}")
for n in [1000, 5000, 10000, 50000, 100000, 500000]:
    d = np.concatenate([rng.normal(-4, 1, n//2), rng.normal(4, 1, n//2)])
    tg = bench_gem(d, ['-d', 'Gaussian', '-k', '2'])
    t0 = time.perf_counter()
    GaussianMixture(2, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
    ts = (time.perf_counter() - t0) * 1000
    print(f"  {n:>8d}  {tg:>8.1f}ms  {ts:>8.1f}ms  {ts/tg:>6.1f}x")

# Test 2: high-k comparison (the fixed case)
print("\n─── High-k comparison (n=20000) ───")
print(f"  {'k':>3}  {'Gemmulem':>10}  {'sklearn':>10}  {'speedup':>8}")
for k in [2, 3, 5, 8, 10]:
    parts = [rng.normal(-10 + 20*j/(k-1) if k>1 else 0, 0.8, 20000//k) for j in range(k)]
    d = np.concatenate(parts)
    tg = bench_gem(d, ['-d', 'Gaussian', '-k', str(k)])
    t0 = time.perf_counter()
    GaussianMixture(k, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
    ts = (time.perf_counter() - t0) * 1000
    print(f"  {k:>3d}  {tg:>8.1f}ms  {ts:>8.1f}ms  {ts/tg:>6.1f}x")

# Test 3: Streaming vs standard
print("\n─── Streaming vs Standard (Gaussian k=3) ───")
for n in [10000, 100000]:
    d = np.concatenate([rng.normal(-6,1,n//3), rng.normal(0,1,n//3), rng.normal(6,1,n//3)])
    t_std = bench_gem(d, ['-d', 'Gaussian', '-k', '3'])
    t_stream = bench_gem(d, ['-d', 'Gaussian', '-k', '3', '--stream', '--chunk-size', '2000', '--passes', '5'])
    print(f"  n={n:<7d}  standard={t_std:>8.1f}ms  streaming={t_stream:>8.1f}ms")

# Test 4: Adaptive k (BIC vs VBEM)
print("\n─── Adaptive k-selection (n=5000, true k=3) ───")
d = np.concatenate([rng.normal(-8,1,1666), rng.normal(0,1,1667), rng.normal(8,1,1667)])
for km in ['bic', 'aic', 'icl', 'vbem', 'mml']:
    t = bench_gem(d, ['--adaptive', '--kmethod', km, '--kmax', '5'], timeout=30)
    print(f"  --kmethod {km:<5s}  {t:>8.1f}ms")

# Test 5: VBEM vs BayesGMM
print("\n─── VBEM vs sklearn BayesianGMM (n=5000) ───")
d = np.concatenate([rng.normal(-5,1,1666), rng.normal(0,0.5,1667), rng.normal(5,1.5,1667)])
t_vbem = bench_gem(d, ['--adaptive', '--kmethod', 'vbem', '--kmax', '6'], timeout=30)
t0 = time.perf_counter()
BayesianGaussianMixture(6, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
t_sk = (time.perf_counter() - t0) * 1000
print(f"  Gemmulem VBEM:    {t_vbem:>8.1f}ms")
print(f"  sklearn BayesGMM: {t_sk:>8.1f}ms  ({t_sk/t_vbem:.1f}x)")

# Test 6: Online EM
print("\n─── Online EM (Gaussian k=2) ───")
for n in [10000, 100000]:
    d = np.concatenate([rng.normal(-4,1,n//2), rng.normal(4,1,n//2)])
    t_std = bench_gem(d, ['-d','Gaussian','-k','2'])
    t_onl = bench_gem(d, ['-d','Gaussian','-k','2','--online','--batch-size','512'])
    print(f"  n={n:<7d}  standard={t_std:>8.1f}ms  online={t_onl:>8.1f}ms")

print("\n" + "=" * 72)
