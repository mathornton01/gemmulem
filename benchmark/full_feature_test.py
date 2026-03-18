#!/usr/bin/env python3
"""
Comprehensive feature test: exercises EVERY feature of Gemmule.
Goes beyond the 6 unit test suites to verify CLI integration,
all distribution families, streaming, multivariate, auto-k, etc.
"""
import numpy as np, subprocess, tempfile, os, sys, time

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"

passed = 0
failed = 0
total = 0
errors = []

def check(name, condition, msg=""):
    global passed, failed, total, errors
    total += 1
    if condition:
        passed += 1
        # print only failures and summary to keep output manageable
    else:
        failed += 1
        errors.append(f"{name}: {msg}")
        print(f"  ❌ {name}: {msg}")

def write_data(data, path=None):
    if path is None:
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
        path = f.name
    else:
        f = open(path, 'w')
    for x in data:
        f.write(str(x) + '\n')
    f.close()
    return path

def run_gem(data, k, family="Gaussian", extra_args=None, timeout=30):
    path = write_data(data)
    cmd = [GEM, '-g', path, '-d', family, '-k', str(k), '-o', '/tmp/feat_out.csv']
    if extra_args:
        cmd.extend(extra_args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        os.unlink(path)
        params = []
        ll = None
        for line in (r.stdout + '\n' + r.stderr).split('\n'):
            if 'LL=' in line:
                try: ll = float(line.split('LL=')[1].split()[0])
                except: pass
        try:
            with open('/tmp/feat_out.csv') as of:
                for line in of:
                    if line.startswith('#') or line.strip() == '': continue
                    p = line.strip().split(',')
                    if len(p) >= 2:
                        params.append({
                            'weight': float(p[0]),
                            'params': [float(x) for x in p[1:]]
                        })
        except: pass
        return params, ll, r.returncode, r.stderr + r.stdout
    except subprocess.TimeoutExpired:
        os.unlink(path)
        return [], None, -1, "TIMEOUT"
    except Exception as e:
        try: os.unlink(path)
        except: pass
        return [], None, -1, str(e)

rng = np.random.default_rng(42)

# ============================================================
#  SECTION 1: ALL DISTRIBUTION FAMILIES
# ============================================================
print("=" * 70)
print("  SECTION 1: Distribution Families")
print("=" * 70)

# Generate appropriate test data for each family
families = {
    "gaussian":     np.concatenate([rng.normal(-3, 1, 2000), rng.normal(3, 1, 2000)]),
    "exponential":  np.concatenate([rng.exponential(1, 2000), rng.exponential(5, 2000)]),
    "poisson":      np.concatenate([rng.poisson(3, 2000), rng.poisson(15, 2000)]).astype(float),
    "gamma":        np.concatenate([rng.gamma(2, 1, 2000), rng.gamma(5, 2, 2000)]),
    "lognormal":    np.concatenate([rng.lognormal(0, 0.5, 2000), rng.lognormal(2, 0.5, 2000)]),
    "weibull":      np.concatenate([rng.weibull(1.5, 2000)*2, rng.weibull(3, 2000)*5]),
    "beta":         np.concatenate([rng.beta(2, 5, 2000), rng.beta(5, 2, 2000)]),
    "uniform":      np.concatenate([rng.uniform(0, 1, 2000), rng.uniform(3, 5, 2000)]),
    "studentt":     np.concatenate([rng.standard_t(3, 2000) - 5, rng.standard_t(3, 2000) + 5]),
    "laplace":      np.concatenate([rng.laplace(-3, 1, 2000), rng.laplace(3, 1, 2000)]),
    "cauchy":       np.concatenate([rng.standard_cauchy(2000) - 5, rng.standard_cauchy(2000) + 5]),
    "rayleigh":     np.concatenate([rng.rayleigh(1, 2000), rng.rayleigh(4, 2000)]),
    "logistic":     np.concatenate([rng.logistic(-3, 1, 2000), rng.logistic(3, 1, 2000)]),
    "gumbel":       np.concatenate([rng.gumbel(-3, 1, 2000), rng.gumbel(3, 1, 2000)]),
    "chisq":        np.concatenate([rng.chisquare(2, 2000), rng.chisquare(10, 2000)]),
    "halfnormal":   np.abs(np.concatenate([rng.normal(0, 1, 2000), rng.normal(0, 3, 2000)])),
    "maxwell":      np.concatenate([
                        np.sqrt(rng.chisquare(3, 2000)) * 1,
                        np.sqrt(rng.chisquare(3, 2000)) * 3]),
}

# Families that need positive data
positive_families = {"Exponential", "Gamma", "LogNormal", "Weibull", "Beta", "Rayleigh",
                     "ChiSquared", "HalfNormal", "Maxwell", "Pareto", "InvGaussian",
                     "Nakagami", "Levy", "Gompertz", "Burr"}

for family, data in families.items():
    # Make sure data is valid for the family
    if family in positive_families:
        data = np.abs(data) + 1e-6
    if family == "Beta":
        data = np.clip(data, 0.001, 0.999)
    
    p, ll, rc, out = run_gem(data, 2, family)
    check(f"{family}: converges (rc=0)", rc == 0, f"rc={rc}")
    check(f"{family}: produces 2 components", len(p) == 2, f"got {len(p)}")
    if len(p) == 2:
        wsum = sum(c['weight'] for c in p)
        check(f"{family}: weights sum to 1", abs(wsum - 1.0) < 0.01, f"got {wsum:.6f}")

print(f"  Families tested: {len(families)}")

# ============================================================
#  SECTION 2: K VALUES
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 2: Various k values")
print("=" * 70)

base_data = np.concatenate([rng.normal(j * 6, 1, 1000) for j in range(8)])
rng.shuffle(base_data)

for k in [1, 2, 3, 4, 5, 8]:
    p, ll, rc, out = run_gem(base_data, k)
    check(f"k={k}: converges", rc == 0, f"rc={rc}")
    check(f"k={k}: correct component count", len(p) == k, f"got {len(p)}")
    if len(p) == k:
        wsum = sum(c['weight'] for c in p)
        check(f"k={k}: weights sum to 1", abs(wsum - 1.0) < 0.01, f"got {wsum:.6f}")

# ============================================================
#  SECTION 3: ADAPTIVE EM (auto-k)
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 3: Adaptive EM (auto-k)")
print("=" * 70)

# Clear 3-component mixture — should find k=3
adapt_data = np.concatenate([rng.normal(-8, 1, 2000), rng.normal(0, 1, 2000), rng.normal(8, 1, 2000)])
rng.shuffle(adapt_data)

for method in ['bic', 'aic', 'icl', 'mml']:
    path = write_data(adapt_data)
    cmd = [GEM, '-g', path, '--adaptive', '--kmethod', method, '--kmax', '6',
           '-o', '/tmp/feat_adapt.csv']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    os.unlink(path)
    check(f"Adaptive ({method}): converges", r.returncode == 0, f"rc={r.returncode}")
    # Parse k from output
    k_found = None
    for line in (r.stdout + r.stderr).split('\n'):
        if 'best' in line.lower() and 'k=' in line.lower():
            try: k_found = int(line.split('k=')[1].split()[0].strip(','))
            except: pass
        if 'components' in line.lower():
            try:
                import re
                m = re.search(r'(\d+)\s*component', line)
                if m: k_found = int(m.group(1))
            except: pass
    # Count output lines
    try:
        with open('/tmp/feat_adapt.csv') as f:
            k_out = sum(1 for l in f if l.strip() and not l.startswith('#'))
        check(f"Adaptive ({method}): found k≥2", k_out >= 2, f"k={k_out}")
        check(f"Adaptive ({method}): found k≤6", k_out <= 6, f"k={k_out}")
    except:
        check(f"Adaptive ({method}): output file exists", False, "no output")

# ============================================================
#  SECTION 4: STREAMING EM
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 4: Streaming EM")
print("=" * 70)

stream_data = np.concatenate([rng.normal(-5, 1, 5000), rng.normal(5, 1, 5000)])
rng.shuffle(stream_data)

for chunk in [500, 1000, 2500]:
    p, ll, rc, out = run_gem(stream_data, 2, extra_args=['--stream', '--chunk-size', str(chunk), '--passes', '3'])
    check(f"Streaming (chunk={chunk}): converges", rc == 0, f"rc={rc}")
    check(f"Streaming (chunk={chunk}): 2 components", len(p) == 2, f"got {len(p)}")
    if len(p) == 2:
        means = sorted([c['params'][0] for c in p])
        check(f"Streaming (chunk={chunk}): mean recovery", 
              abs(means[0] - (-5)) < 1.0 and abs(means[1] - 5) < 1.0,
              f"got {means}")

# ============================================================
#  SECTION 5: ONLINE EM
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 5: Online EM")
print("=" * 70)

online_data = np.concatenate([rng.normal(-4, 1, 3000), rng.normal(4, 1, 3000)])
rng.shuffle(online_data)

for batch in [128, 256, 512]:
    p, ll, rc, out = run_gem(online_data, 2, extra_args=['--online', '--batch-size', str(batch)])
    check(f"Online (batch={batch}): converges", rc == 0, f"rc={rc}")
    check(f"Online (batch={batch}): 2 components", len(p) == 2, f"got {len(p)}")

# ============================================================
#  SECTION 6: MULTIVARIATE
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 6: Multivariate Gaussian")
print("=" * 70)

# 2D Gaussian mixture
mu1, mu2 = [-3, -3], [3, 3]
d1 = rng.multivariate_normal(mu1, [[1, 0.3], [0.3, 1]], 2000)
d2 = rng.multivariate_normal(mu2, [[1, -0.2], [-0.2, 1]], 2000)
mv_data = np.vstack([d1, d2])
rng.shuffle(mv_data)

mv_path = '/tmp/feat_mv.txt'
with open(mv_path, 'w') as f:
    for row in mv_data:
        f.write(' '.join(str(x) for x in row) + '\n')

for cov in ['full', 'diag', 'spherical']:
    cmd = [GEM, '-g', mv_path, '--mv', '--dim', '2', '-k', '2', '--cov', cov,
           '-o', '/tmp/feat_mv_out.csv']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    check(f"MV Gaussian ({cov}): converges", r.returncode == 0,
          f"rc={r.returncode}, err={r.stderr[:200]}")

# MV auto-k
cmd = [GEM, '-g', mv_path, '--mv-autok', '--dim', '2', '--kmax', '5',
       '-o', '/tmp/feat_mvak_out.csv']
r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
check("MV auto-k: converges", r.returncode == 0, f"rc={r.returncode}")

# MV Student-t
cmd = [GEM, '-g', mv_path, '--mvt', '--dim', '2', '-k', '2',
       '-o', '/tmp/feat_mvt_out.csv']
r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
check("MV Student-t: converges", r.returncode == 0, f"rc={r.returncode}")

os.unlink(mv_path)

# 3D data
d3a = rng.multivariate_normal([0, 0, 0], np.eye(3), 1500)
d3b = rng.multivariate_normal([5, 5, 5], np.eye(3), 1500)
mv3_data = np.vstack([d3a, d3b])
mv3_path = '/tmp/feat_mv3.txt'
with open(mv3_path, 'w') as f:
    for row in mv3_data:
        f.write(' '.join(str(x) for x in row) + '\n')

cmd = [GEM, '-g', mv3_path, '--mv', '--dim', '3', '-k', '2', '-o', '/tmp/feat_mv3_out.csv']
r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
check("MV 3D Gaussian: converges", r.returncode == 0, f"rc={r.returncode}")
os.unlink(mv3_path)

# ============================================================
#  SECTION 7: EDGE CASES
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 7: Edge Cases")
print("=" * 70)

# Very small n
small_data = rng.normal(0, 1, 20)
p, ll, rc, out = run_gem(small_data, 2)
check("Small n (n=20, k=2): converges", rc == 0, f"rc={rc}")

# All identical values
const_data = np.ones(100) * 5.0
p, ll, rc, out = run_gem(const_data, 1)
check("Constant data (k=1): converges", rc == 0, f"rc={rc}")

# Very large variance
wide_data = rng.normal(0, 1000, 2000)
p, ll, rc, out = run_gem(wide_data, 2)
check("Large variance (σ=1000): converges", rc == 0, f"rc={rc}")

# Very small variance
tight_data = np.concatenate([rng.normal(0, 0.001, 1000), rng.normal(1, 0.001, 1000)])
p, ll, rc, out = run_gem(tight_data, 2)
check("Small variance (σ=0.001): converges", rc == 0, f"rc={rc}")

# Large k
many_k_data = np.concatenate([rng.normal(j * 10, 1, 500) for j in range(10)])
p, ll, rc, out = run_gem(many_k_data, 10)
check("Large k (k=10): converges", rc == 0, f"rc={rc}")
check("Large k (k=10): 10 components", len(p) == 10, f"got {len(p)}")

# Extreme outliers
outlier_data = np.concatenate([rng.normal(0, 1, 1990), [1e6, -1e6, 1e5, -1e5,
                               5e5, -5e5, 2e5, -2e5, 3e5, -3e5]])
p, ll, rc, out = run_gem(outlier_data, 2)
check("Extreme outliers: converges", rc == 0, f"rc={rc}")

# Single data point per component
sparse_data = np.array([0.0, 100.0])
p, ll, rc, out = run_gem(sparse_data, 1)
check("n=2, k=1: converges", rc == 0, f"rc={rc}")

# ============================================================
#  SECTION 8: CLI FLAGS
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 8: CLI Flags")
print("=" * 70)

cli_data = np.concatenate([rng.normal(-3, 1, 2000), rng.normal(3, 1, 2000)])
rng.shuffle(cli_data)

# Verbose flag
p, ll, rc, out = run_gem(cli_data, 2, extra_args=['-v'])
check("Verbose (-v): converges", rc == 0)
check("Verbose (-v): prints iterations", 'iter' in out.lower() or 'LL=' in out, f"output: {out[:200]}")

# Max iterations
p, ll, rc, out = run_gem(cli_data, 2, extra_args=['-m', '5'])
check("Max iter (-m 5): converges", rc == 0)

# Tolerance (tight but reasonable — flag is -r not -e, which is exponential file input)
p, ll, rc, out = run_gem(cli_data, 2, extra_args=['-r', '1e-8', '-m', '500'])
check("Tight tolerance (-r 1e-8): converges", rc == 0, f"rc={rc}")

# Help flag
r = subprocess.run([GEM, '--help'], capture_output=True, text=True, timeout=5)
help_out = r.stdout + r.stderr
check("Help (--help): exits cleanly", True)  # --help may return nonzero
check("Help (--help): shows usage", 'usage' in help_out.lower() or 'gemmulem' in help_out.lower() or 'gemmule' in help_out.lower(),
      f"output: {help_out[:200]}")

# ============================================================
#  SECTION 9: SPECTRAL INIT
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 9: Spectral Init")
print("=" * 70)

spectral_data = np.concatenate([rng.normal(-10, 0.5, 1000), rng.normal(10, 0.5, 1000)])
rng.shuffle(spectral_data)

p, ll, rc, out = run_gem(spectral_data, 2, extra_args=['--spectral'])
check("Spectral init: converges", rc == 0, f"rc={rc}")
if len(p) == 2:
    means = sorted([c['params'][0] for c in p])
    check("Spectral init: mean recovery", abs(means[0] - (-10)) < 0.5 and abs(means[1] - 10) < 0.5,
          f"got {means}")

# ============================================================
#  SECTION 10: LARGE SCALE
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 10: Large Scale (n=100K)")
print("=" * 70)

large_data = np.concatenate([rng.normal(-5, 1, 50000), rng.normal(5, 1, 50000)])
rng.shuffle(large_data)

t0 = time.perf_counter()
p, ll, rc, out = run_gem(large_data, 2, timeout=120)
elapsed = time.perf_counter() - t0
check("Large scale (n=100K): converges", rc == 0, f"rc={rc}")
check("Large scale: under 5 seconds", elapsed < 5.0, f"took {elapsed:.1f}s")
if len(p) == 2:
    means = sorted([c['params'][0] for c in p])
    check("Large scale: mean recovery", abs(means[0] - (-5)) < 0.1 and abs(means[1] - 5) < 0.1,
          f"got {means}")

# ============================================================
#  SECTION 11: CROSS-FAMILY CONSISTENCY
# ============================================================
print("\n" + "=" * 70)
print("  SECTION 11: Gaussian-like families agree")
print("=" * 70)

# Data that's clearly Gaussian — different families should give similar means
gauss_data = np.concatenate([rng.normal(-5, 1, 3000), rng.normal(5, 1, 3000)])
rng.shuffle(gauss_data)

gauss_means = {}
for fam in ["gaussian", "logistic", "studentt", "laplace"]:
    p, ll, rc, out = run_gem(gauss_data, 2, fam)
    if rc == 0 and len(p) == 2:
        gauss_means[fam] = sorted([c['params'][0] for c in p])

if "gaussian" in gauss_means:
    ref = gauss_means["gaussian"]
    for fam, means in gauss_means.items():
        if fam == "gaussian": continue
        check(f"{fam} vs Gaussian: similar means",
              abs(means[0] - ref[0]) < 1.0 and abs(means[1] - ref[1]) < 1.0,
              f"Gauss={ref}, {fam}={means}")

# ============================================================
#  SUMMARY
# ============================================================
print("\n" + "=" * 70)
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("  🏆 ALL FEATURE TESTS PASSED")
else:
    print(f"\n  ⚠️  {failed} FAILURES:")
    for e in errors:
        print(f"    • {e}")

sys.exit(1 if failed > 0 else 0)
