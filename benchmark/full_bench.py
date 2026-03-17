#!/usr/bin/env python3
"""Head-to-head benchmark: Gemmulem vs sklearn."""
import numpy as np, time, subprocess, tempfile, os

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)

def bench_gem(data, args):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(f'{x}\n')
    f.close()
    t0 = time.perf_counter()
    subprocess.run([GEM, '-g', f.name] + args + ['-o', '/dev/null'],
                   capture_output=True, text=True, timeout=120)
    t = (time.perf_counter() - t0) * 1000
    os.unlink(f.name)
    return t

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

print("=" * 72)
print("  GEMMULEM vs SKLEARN — Head-to-Head Benchmark")
print("=" * 72)

# 1. Fixed k, varying n
print("\n═══ Test 1: Fixed k=2 Gaussian, varying n ═══")
print(f"  {'n':>8s}  {'Gemmulem':>10s}  {'sklearn':>10s}  {'speedup':>8s}")
print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
for n in [1000, 5000, 10000, 50000, 100000, 500000]:
    d = np.concatenate([rng.normal(-4, 1, n//2), rng.normal(4, 1, n//2)])
    
    t_gem = bench_gem(d, ['-d', 'Gaussian', '-k', '2'])
    
    t0 = time.perf_counter()
    GaussianMixture(n_components=2, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
    t_sk = (time.perf_counter() - t0) * 1000
    
    print(f"  {n:>8d}  {t_gem:>8.1f}ms  {t_sk:>8.1f}ms  {t_sk/t_gem:>6.1f}x")

# 2. Varying k
print("\n═══ Test 2: n=20000, varying k ═══")
print(f"  {'k':>3s}  {'Gemmulem':>10s}  {'sklearn':>10s}  {'speedup':>8s}")
print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*8}")
for k in [2, 3, 5, 8, 10]:
    parts = []
    for j in range(k):
        parts.append(rng.normal(-10 + 20*j/(k-1), 0.8, 20000//k))
    d = np.concatenate(parts)
    
    t_gem = bench_gem(d, ['-d', 'Gaussian', '-k', str(k)])
    
    t0 = time.perf_counter()
    GaussianMixture(n_components=k, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
    t_sk = (time.perf_counter() - t0) * 1000
    
    print(f"  {k:>3d}  {t_gem:>8.1f}ms  {t_sk:>8.1f}ms  {t_sk/t_gem:>6.1f}x")

# 3. Auto k-selection: VBEM vs BayesianGMM
print("\n═══ Test 3: Automatic k-selection (VBEM vs BayesianGMM) ═══")
print(f"  {'n':>8s}  {'Gem VBEM':>10s}  {'sk Bayes':>10s}  {'speedup':>8s}")
print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
for n in [2000, 10000, 50000]:
    d = np.concatenate([rng.normal(-5, 1, n//3), rng.normal(0, 0.5, n//3), rng.normal(5, 1.5, n//3)])
    
    t_gem = bench_gem(d, ['--adaptive', '--kmethod', 'vbem', '--kmax', '8'])
    
    t0 = time.perf_counter()
    BayesianGaussianMixture(n_components=8, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
    t_sk = (time.perf_counter() - t0) * 1000
    
    print(f"  {n:>8d}  {t_gem:>8.1f}ms  {t_sk:>8.1f}ms  {t_sk/t_gem:>6.1f}x")

# 4. Accuracy comparison
print("\n═══ Test 4: Parameter recovery accuracy (n=10000, k=2) ═══")
d = np.concatenate([rng.normal(-4, 1, 5000), rng.normal(4, 1, 5000)])

# sklearn
gm = GaussianMixture(n_components=2, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
sk_means = sorted(gm.means_.flatten())
sk_vars = [gm.covariances_.flatten()[i] for i in np.argsort(gm.means_.flatten())]
sk_weights = [gm.weights_[i] for i in np.argsort(gm.means_.flatten())]

# gemmulem (parse output)
f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
for x in d: f.write(f'{x}\n')
f.close()
r = subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', '2', '-v'],
                   capture_output=True, text=True, timeout=30)
os.unlink(f.name)

print(f"  True:    μ=[-4.000, 4.000]  σ²=[1.000, 1.000]  w=[0.500, 0.500]")
print(f"  sklearn: μ=[{sk_means[0]:.3f}, {sk_means[1]:.3f}]  σ²=[{sk_vars[0]:.3f}, {sk_vars[1]:.3f}]  w=[{sk_weights[0]:.3f}, {sk_weights[1]:.3f}]")
print(f"  Gemmulem output (verbose):")
for line in r.stdout.split('\n'):
    if 'Component' in line or 'LL=' in line or 'Converged' in line:
        print(f"    {line.strip()}")

# 5. Feature comparison table
print("\n═══ Feature Comparison ═══")
print(f"  {'Feature':<35s}  {'Gemmulem':>10s}  {'sklearn':>10s}  {'R mixtools':>10s}")
print(f"  {'─'*35}  {'─'*10}  {'─'*10}  {'─'*10}")
rows = [
    ("Distribution families",          "35",      "1",       "3"),
    ("Auto k-selection methods",       "5",       "1",       "0"),
    ("Spectral initialization",        "✓",       "✗",       "✗"),
    ("Online/stochastic EM",           "✓",       "✗",       "✗"),
    ("Nonparametric (KDE)",            "✓",       "✗",       "✗"),
    ("Cross-family adaptive EM",       "✓",       "✗",       "✗"),
    ("Pearson auto-type",              "✓",       "✗",       "✗"),
    ("Multivariate",                   "✗",       "✓",       "✗"),
    ("Language",                       "C",       "Python",  "R"),
    ("External dependencies",          "none",    "numpy",   "R core"),
]
for name, gem, sk, r in rows:
    print(f"  {name:<35s}  {gem:>10s}  {sk:>10s}  {r:>10s}")

print("\n" + "=" * 72)
