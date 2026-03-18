#!/usr/bin/env python3
"""
Adaptive distribution finder accuracy benchmark.

For each scenario: generate data from a KNOWN mixture, run Gemmule's
adaptive finder, check if it correctly identifies:
  1. The number of components (k)
  2. The distribution family
  3. Parameter recovery accuracy

This is the ground-truth validation that matters most.
"""
import numpy as np, subprocess, tempfile, os, sys, time, re

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"

rng = np.random.default_rng(42)

# ============================================================
#  SCENARIO DEFINITIONS
# ============================================================
# Each scenario: name, true_family, true_k, data generator, n_per_component

def gen_gaussian_k2():
    """Well-separated Gaussians"""
    return np.concatenate([rng.normal(-5, 1, 3000), rng.normal(5, 1, 3000)])

def gen_gaussian_k3():
    """Three Gaussians, unequal weights"""
    return np.concatenate([rng.normal(-8, 1.5, 4000), rng.normal(0, 0.8, 2000), rng.normal(6, 1.2, 4000)])

def gen_gaussian_k5():
    """Five well-separated Gaussians"""
    return np.concatenate([rng.normal(j*8, 1, 2000) for j in range(5)])

def gen_exponential_k2():
    """Two exponential components"""
    return np.concatenate([rng.exponential(1, 3000), rng.exponential(8, 3000)])

def gen_exponential_k3():
    """Three exponential components"""
    return np.concatenate([rng.exponential(0.5, 2000), rng.exponential(3, 2000), rng.exponential(10, 2000)])

def gen_gamma_k2():
    """Two gamma components"""
    return np.concatenate([rng.gamma(2, 1, 3000), rng.gamma(8, 2, 3000)])

def gen_gamma_k3():
    """Three gamma components"""
    return np.concatenate([rng.gamma(1, 0.5, 2000), rng.gamma(5, 1, 2000), rng.gamma(15, 0.5, 2000)])

def gen_lognormal_k2():
    """Two lognormal components"""
    return np.concatenate([rng.lognormal(0, 0.5, 3000), rng.lognormal(3, 0.5, 3000)])

def gen_weibull_k2():
    """Two Weibull components"""
    return np.concatenate([rng.weibull(1.5, 3000) * 2, rng.weibull(4, 3000) * 8])

def gen_poisson_k2():
    """Two Poisson components"""
    return np.concatenate([rng.poisson(3, 3000), rng.poisson(20, 3000)]).astype(float)

def gen_poisson_k3():
    """Three Poisson components"""
    return np.concatenate([rng.poisson(2, 2000), rng.poisson(10, 2000), rng.poisson(30, 2000)]).astype(float)

def gen_beta_k2():
    """Two beta components (on [0,1])"""
    return np.concatenate([rng.beta(2, 8, 3000), rng.beta(8, 2, 3000)])

def gen_uniform_k2():
    """Two uniform components (well-separated)"""
    return np.concatenate([rng.uniform(0, 3, 3000), rng.uniform(7, 10, 3000)])

def gen_laplace_k2():
    """Two Laplace components"""
    return np.concatenate([rng.laplace(-5, 1, 3000), rng.laplace(5, 1, 3000)])

def gen_logistic_k2():
    """Two logistic components"""
    return np.concatenate([rng.logistic(-5, 1, 3000), rng.logistic(5, 1, 3000)])

def gen_gumbel_k2():
    """Two Gumbel components"""
    return np.concatenate([rng.gumbel(-5, 1, 3000), rng.gumbel(5, 1, 3000)])

# Related families (should we accept close relatives?)
FAMILY_GROUPS = {
    "gaussian":    {"gaussian", "logistic", "studentt", "laplace", "gengaussian", "skewnormal"},
    "exponential": {"exponential", "gamma", "weibull"},
    "gamma":       {"gamma", "exponential", "weibull", "lognormal"},
    "lognormal":   {"lognormal", "gamma", "weibull"},
    "weibull":     {"weibull", "gamma", "exponential"},
    "poisson":     {"poisson", "negbinomial", "geometric"},
    "beta":        {"beta", "kumaraswamy"},
    "uniform":     {"uniform"},
    "laplace":     {"laplace", "gaussian", "logistic", "studentt"},
    "logistic":    {"logistic", "gaussian", "laplace"},
    "gumbel":      {"gumbel", "weibull"},
}

scenarios = [
    # (name, true_family, true_k, generator, kmax)
    ("Gaussian k=2, sep=10σ",     "gaussian",    2, gen_gaussian_k2, 5),
    ("Gaussian k=3, unequal wt",  "gaussian",    3, gen_gaussian_k3, 6),
    ("Gaussian k=5, sep=8σ",      "gaussian",    5, gen_gaussian_k5, 8),
    ("Exponential k=2",           "exponential",  2, gen_exponential_k2, 5),
    ("Exponential k=3",           "exponential",  3, gen_exponential_k3, 6),
    ("Gamma k=2",                 "gamma",        2, gen_gamma_k2, 5),
    ("Gamma k=3",                 "gamma",        3, gen_gamma_k3, 6),
    ("LogNormal k=2",             "lognormal",    2, gen_lognormal_k2, 5),
    ("Weibull k=2",               "weibull",      2, gen_weibull_k2, 5),
    ("Poisson k=2",               "poisson",      2, gen_poisson_k2, 5),
    ("Poisson k=3",               "poisson",      3, gen_poisson_k3, 6),
    ("Beta k=2",                  "beta",         2, gen_beta_k2, 5),
    ("Uniform k=2",               "uniform",      2, gen_uniform_k2, 5),
    ("Laplace k=2, sep=10",       "laplace",      2, gen_laplace_k2, 5),
    ("Logistic k=2, sep=10",      "logistic",     2, gen_logistic_k2, 5),
    ("Gumbel k=2, sep=10",        "gumbel",       2, gen_gumbel_k2, 5),
]


def run_adaptive(data, kmax, method='bic', timeout=120):
    """Run adaptive mode and parse results."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(str(x) + '\n')
    f.close()
    
    cmd = [GEM, '-g', f.name, '--adaptive', '--kmethod', method, '--kmax', str(kmax),
           '-o', '/tmp/adapt_bench_out.csv', '-v']
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        os.unlink(f.name)
    except subprocess.TimeoutExpired:
        os.unlink(f.name)
        return None, None, None, "TIMEOUT"
    except Exception as e:
        try: os.unlink(f.name)
        except: pass
        return None, None, None, str(e)
    
    output = r.stdout + '\n' + r.stderr
    
    # Parse the best model from "Adaptive Result: k=N" line
    best_family = None
    best_k = None
    best_bic = None
    
    for line in output.split('\n'):
        # Parse "Adaptive Result: k=2  (method: BIC)"
        if 'adaptive result' in line.lower() and 'k=' in line.lower():
            km = re.search(r'k\s*=\s*(\d+)', line)
            if km:
                best_k = int(km.group(1))
        
        # Parse BIC from result block
        if line.strip().startswith('BIC') and '=' in line:
            bm = re.search(r'BIC\s*=\s*([\d.e+-]+)', line)
            if bm:
                try: best_bic = float(bm.group(1))
                except: pass
    
    # Parse family from CSV output: "Gaussian,0.5,-5.02,1.01"
    families_found = set()
    try:
        with open('/tmp/adapt_bench_out.csv') as of:
            for line in of:
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    fam = parts[0].strip().lower()
                    if fam and not fam[0].isdigit():
                        families_found.add(fam)
    except: pass
    
    # If all components are the same family, that's the detected family
    if len(families_found) == 1:
        best_family = families_found.pop()
    elif len(families_found) > 1:
        # Mixed families — report the majority
        best_family = max(families_found, key=lambda f: f)  # just pick one for reporting
    
    # Fallback: count k from CSV if not found in stdout
    if best_k is None:
        try:
            with open('/tmp/adapt_bench_out.csv') as of:
                best_k = sum(1 for l in of if l.strip() and not l.startswith('#'))
        except: pass
    
    return best_family, best_k, best_bic, output


# ============================================================
#  RUN ALL SCENARIOS
# ============================================================
print("=" * 80)
print("  ADAPTIVE DISTRIBUTION FINDER ACCURACY BENCHMARK")
print("  Testing: does Gemmule correctly identify the generating distribution?")
print("=" * 80)

results = []
n_seeds = 3  # Multiple seeds per scenario for statistical confidence

for sname, true_fam, true_k, gen_fn, kmax in scenarios:
    print(f"\n{'─'*80}")
    print(f"  {sname}")
    print(f"  True: {true_fam} k={true_k}")
    print(f"{'─'*80}")
    
    k_correct = 0
    fam_correct = 0
    fam_related = 0
    
    for seed in range(n_seeds):
        # Generate fresh data with different seed
        data = gen_fn()
        
        found_fam, found_k, found_bic, output = run_adaptive(data, kmax)
        
        k_match = found_k == true_k
        fam_match = found_fam == true_fam if found_fam else False
        related = False
        if found_fam and true_fam in FAMILY_GROUPS:
            related = found_fam in FAMILY_GROUPS[true_fam]
        
        if k_match: k_correct += 1
        if fam_match: fam_correct += 1
        if related: fam_related += 1
        
        status_k = "✅" if k_match else "❌"
        status_f = "✅" if fam_match else ("~" if related else "❌")
        print(f"  seed {seed}: k={found_k} {status_k}  family={found_fam} {status_f}")
    
    k_pct = k_correct / n_seeds * 100
    fam_pct = fam_correct / n_seeds * 100
    rel_pct = fam_related / n_seeds * 100
    
    results.append({
        'name': sname, 'true_fam': true_fam, 'true_k': true_k,
        'k_accuracy': k_pct, 'family_exact': fam_pct, 'family_related': rel_pct
    })
    
    print(f"  → k accuracy: {k_pct:.0f}%  |  family exact: {fam_pct:.0f}%  |  family group: {rel_pct:.0f}%")


# ============================================================
#  SUMMARY TABLE
# ============================================================
print(f"\n{'='*80}")
print(f"  SUMMARY TABLE ({n_seeds} seeds per scenario)")
print(f"{'='*80}")
print(f"  {'Scenario':<30s}  {'True':>8s}  {'k acc':>6s}  {'Fam ✓':>6s}  {'Fam ~':>6s}")
print(f"  {'─'*30}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}")

total_k = 0
total_fam = 0
total_rel = 0

for r in results:
    label = f"{r['true_fam']} k={r['true_k']}"
    print(f"  {r['name']:<30s}  {label:>8s}  {r['k_accuracy']:5.0f}%  {r['family_exact']:5.0f}%  {r['family_related']:5.0f}%")
    total_k += r['k_accuracy']
    total_fam += r['family_exact']
    total_rel += r['family_related']

n = len(results)
print(f"  {'─'*30}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}")
print(f"  {'AVERAGE':<30s}  {'':>8s}  {total_k/n:5.1f}%  {total_fam/n:5.1f}%  {total_rel/n:5.1f}%")

print(f"\n  Legend: k acc = correct number of components")
print(f"          Fam ✓ = exact family match")
print(f"          Fam ~ = correct family group (e.g. Gaussian picked for Logistic data)")

# Overall grade
avg_k = total_k / n
avg_fam = total_fam / n
avg_rel = total_rel / n

print(f"\n{'='*80}")
if avg_k >= 80 and avg_rel >= 60:
    print(f"  🏆 GRADE: EXCELLENT (k={avg_k:.0f}%, family group={avg_rel:.0f}%)")
elif avg_k >= 60 and avg_rel >= 40:
    print(f"  ✅ GRADE: GOOD (k={avg_k:.0f}%, family group={avg_rel:.0f}%)")
elif avg_k >= 40:
    print(f"  ⚠️  GRADE: NEEDS WORK (k={avg_k:.0f}%, family group={avg_rel:.0f}%)")
else:
    print(f"  ❌ GRADE: POOR (k={avg_k:.0f}%, family group={avg_rel:.0f}%)")
print(f"{'='*80}")
