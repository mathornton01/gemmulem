#!/usr/bin/env python3
"""
Correctness validation: verify Gemmulem recovers known-truth parameters
across a wide range of scenarios. Tests both accuracy AND mathematical properties.
"""
import numpy as np, subprocess, tempfile, os, sys

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"

def run_gem(data, k, family="Gaussian"):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(str(x) + '\n')
    f.close()
    r = subprocess.run([GEM, '-g', f.name, '-d', family, '-k', str(k),
                        '-o', '/tmp/corr_out.csv'],
                       capture_output=True, text=True, timeout=60)
    os.unlink(f.name)
    params = []
    ll = None
    try:
        for line in (r.stderr + '\n' + r.stdout).split('\n'):
            if 'LL=' in line:
                try:
                    ll = float(line.split('LL=')[1].split()[0])
                except: pass
        with open('/tmp/corr_out.csv') as of:
            for line in of:
                if line.startswith('#') or line.strip() == '': continue
                p = line.strip().split(',')
                if len(p) >= 3:
                    params.append({'weight': float(p[0]), 'mean': float(p[1]), 'var': float(p[2])})
    except: pass
    return sorted(params, key=lambda p: p['mean']), ll

passed = 0
failed = 0
total = 0

def check(name, condition, msg=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {msg}")

print("=" * 70)
print("  CORRECTNESS VALIDATION")
print("=" * 70)

# Test 1: Single Gaussian (k=1) should recover true parameters
print("\n--- Test 1: Single Gaussian (k=1, μ=5, σ²=2) ---")
rng = np.random.default_rng(42)
d = rng.normal(5, np.sqrt(2), 10000)
p, ll = run_gem(d, 1)
check("Recovered mean", abs(p[0]['mean'] - 5) < 0.1, f"got {p[0]['mean']:.4f}")
check("Recovered variance", abs(p[0]['var'] - 2) < 0.2, f"got {p[0]['var']:.4f}")
check("Weight = 1.0", abs(p[0]['weight'] - 1.0) < 0.01, f"got {p[0]['weight']:.4f}")
check("LL is finite", ll is not None and np.isfinite(ll), f"got {ll}")

# Test 2: Well-separated Gaussians (k=2)
print("\n--- Test 2: Well-separated Gaussians (k=2, μ=±10) ---")
d = np.concatenate([rng.normal(-10, 1, 5000), rng.normal(10, 1, 5000)])
rng.shuffle(d)
p, ll = run_gem(d, 2)
check("Mean 1 ≈ -10", abs(p[0]['mean'] - (-10)) < 0.1, f"got {p[0]['mean']:.4f}")
check("Mean 2 ≈ 10", abs(p[1]['mean'] - 10) < 0.1, f"got {p[1]['mean']:.4f}")
check("Var ≈ 1", abs(p[0]['var'] - 1) < 0.2, f"got {p[0]['var']:.4f}")
check("Equal weights", abs(p[0]['weight'] - 0.5) < 0.02, f"got {p[0]['weight']:.4f}")

# Test 3: Weights sum to 1
print("\n--- Test 3: Weights sum to 1.0 (k=5) ---")
d = np.concatenate([rng.normal(j*5, 1, 2000) for j in range(5)])
rng.shuffle(d)
p, ll = run_gem(d, 5)
wsum = sum(c['weight'] for c in p)
check("Weights sum to 1", abs(wsum - 1.0) < 1e-6, f"got {wsum:.8f}")
check("All weights > 0", all(c['weight'] > 0 for c in p))
check("All variances > 0", all(c['var'] > 0 for c in p))
check("5 components returned", len(p) == 5, f"got {len(p)}")

# Test 4: EM monotonicity (LL never decreases)
print("\n--- Test 4: EM monotonicity ---")
d = np.concatenate([rng.normal(-3, 1.5, 3000), rng.normal(2, 0.8, 4000), rng.normal(8, 1.2, 3000)])
rng.shuffle(d)
f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
for x in d: f.write(str(x) + '\n')
f.close()
r = subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', '3', '-o', '/tmp/corr_mono.csv', '-v'],
                   capture_output=True, text=True, timeout=60)
os.unlink(f.name)
lls = []
for line in (r.stderr + '\n' + r.stdout).split('\n'):
    if 'LL=' in line and 'delta=' in line:
        try:
            ll_val = float(line.split('LL=')[1].split()[0])
            lls.append(ll_val)
        except: pass
monotonic = all(lls[i] >= lls[i-1] - 1e-6 for i in range(1, len(lls)))
check(f"LL monotonically increases ({len(lls)} iterations)", monotonic,
      f"violations: {[(i, lls[i]-lls[i-1]) for i in range(1,len(lls)) if lls[i] < lls[i-1]-1e-6]}")

# Test 5: Large n convergence (should approach true parameters)
print("\n--- Test 5: Large n convergence (n=100K) ---")
d = np.concatenate([rng.normal(-5, np.sqrt(0.5), 25000),
                    rng.normal(0, np.sqrt(2.0), 50000),
                    rng.normal(5, np.sqrt(0.3), 25000)])
rng.shuffle(d)
p, ll = run_gem(d, 3)
check("Mean 1 ≈ -5", abs(p[0]['mean'] - (-5)) < 0.05, f"got {p[0]['mean']:.4f}")
check("Mean 2 ≈ 0", abs(p[1]['mean'] - 0) < 0.05, f"got {p[1]['mean']:.4f}")
check("Mean 3 ≈ 5", abs(p[2]['mean'] - 5) < 0.05, f"got {p[2]['mean']:.4f}")
check("Var 1 ≈ 0.5", abs(p[0]['var'] - 0.5) < 0.1, f"got {p[0]['var']:.4f}")
check("Var 2 ≈ 2.0", abs(p[1]['var'] - 2.0) < 0.2, f"got {p[1]['var']:.4f}")
check("Weight 2 ≈ 0.5", abs(p[1]['weight'] - 0.5) < 0.05, f"got {p[1]['weight']:.4f}")

# Test 6: Numerical stability with extreme values
print("\n--- Test 6: Numerical stability ---")
d = np.concatenate([rng.normal(0, 0.01, 2000), rng.normal(100, 0.01, 2000)])
rng.shuffle(d)
p, ll = run_gem(d, 2)
check("Handles large separation", abs(p[0]['mean'] - 0) < 0.1 and abs(p[1]['mean'] - 100) < 0.1,
      f"got {p[0]['mean']:.4f}, {p[1]['mean']:.4f}")
check("Small variance preserved", p[0]['var'] < 0.1 and p[1]['var'] < 0.1,
      f"got {p[0]['var']:.6f}, {p[1]['var']:.6f}")
check("LL is finite", ll is not None and np.isfinite(ll), f"got {ll}")

# Test 7: k=1 should be equivalent to sample statistics
print("\n--- Test 7: k=1 matches sample statistics ---")
d = rng.normal(3.7, np.sqrt(1.3), 5000)
p, ll = run_gem(d, 1)
sample_mean = np.mean(d)
sample_var = np.var(d)  # population variance (same as MLE)
check("Mean matches np.mean", abs(p[0]['mean'] - sample_mean) < 1e-4,
      f"gem={p[0]['mean']:.6f}, np={sample_mean:.6f}")
check("Var matches np.var", abs(p[0]['var'] - sample_var) < 1e-3,
      f"gem={p[0]['var']:.6f}, np={sample_var:.6f}")

# Test 8: Reproducibility (same data → same result)
print("\n--- Test 8: Deterministic reproducibility ---")
d = np.concatenate([rng.normal(-3, 1, 2000), rng.normal(3, 1, 2000)])
p1, ll1 = run_gem(d, 2)
p2, ll2 = run_gem(d, 2)
check("Same means on repeated runs", 
      abs(p1[0]['mean'] - p2[0]['mean']) < 1e-10 and abs(p1[1]['mean'] - p2[1]['mean']) < 1e-10,
      f"run1={[c['mean'] for c in p1]}, run2={[c['mean'] for c in p2]}")
check("Same LL on repeated runs", 
      ll1 is not None and ll2 is not None and abs(ll1 - ll2) < 1e-10,
      f"ll1={ll1}, ll2={ll2}")

# Summary
print("\n" + "=" * 70)
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)
if failed == 0:
    print("  🏆 ALL CORRECTNESS CHECKS PASSED")
else:
    print(f"  ⚠️  {failed} FAILURES — INVESTIGATE BEFORE RELEASE")
sys.exit(1 if failed > 0 else 0)
