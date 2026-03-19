#!/usr/bin/env python3
"""
Extended adaptive distribution finder benchmark.

Tests WEIRD and HARD scenarios beyond the standard single-family mixtures:
  - Mixed-family components (Gaussian + Exponential in same data)
  - Heavy overlap (components barely separable)
  - Extreme imbalance (1% minority component)
  - High k (many components)
  - Edge cases (near-degenerate, single point masses with noise)
  - Scale extremes (tiny and huge parameter values)
  - Multimodal within single families
"""
import numpy as np, subprocess, tempfile, os, sys, re

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"

rng = np.random.default_rng(2026)

# ============================================================
#  FAMILY GROUPS (for approximate matching)
# ============================================================
FAMILY_GROUPS = {
    "gaussian":    {"gaussian", "logistic", "studentt", "laplace", "gengaussian", "skewnormal"},
    "exponential": {"exponential", "gamma", "weibull", "lognormal"},
    "gamma":       {"gamma", "exponential", "weibull", "lognormal"},
    "lognormal":   {"lognormal", "gamma", "weibull", "invgaussian"},
    "weibull":     {"weibull", "gamma", "exponential", "lognormal"},
    "poisson":     {"poisson", "negbinomial", "geometric"},
    "beta":        {"beta", "kumaraswamy", "uniform"},
    "uniform":     {"uniform", "beta"},
    "laplace":     {"laplace", "gaussian", "logistic", "studentt"},
    "logistic":    {"logistic", "gaussian", "laplace"},
    "gumbel":      {"gumbel", "weibull", "studentt"},
    "studentt":    {"studentt", "gaussian", "cauchy", "laplace"},
    "cauchy":      {"cauchy", "studentt", "laplace"},
    "halfnormal":  {"halfnormal", "rayleigh", "exponential", "gamma"},
    "rayleigh":    {"rayleigh", "halfnormal", "gamma", "weibull"},
    "mixed":       set(),  # mixed-family scenarios have special scoring
}

# ============================================================
#  SCENARIO GENERATORS
# ============================================================

# ── MIXED FAMILY SCENARIOS ──────────────────────────────────

def gen_gauss_plus_expo():
    """Gaussian(-3,1) + Exponential(rate=0.5) shifted to positive"""
    return np.concatenate([rng.normal(-3, 1, 3000), rng.exponential(2, 3000) + 2])

def gen_gauss_plus_gamma():
    """Gaussian(0,1) + Gamma(3,2)"""
    return np.concatenate([rng.normal(0, 1, 3000), rng.gamma(3, 2, 3000)])

def gen_laplace_plus_expo():
    """Laplace(0,1) + Exponential(rate=1)"""
    return np.concatenate([rng.laplace(0, 1, 3000), rng.exponential(1, 3000) + 5])

def gen_poisson_plus_gauss():
    """Poisson(5) + Gaussian(20,3)"""
    return np.concatenate([rng.poisson(5, 3000).astype(float), rng.normal(20, 3, 3000)])

def gen_beta_plus_gauss():
    """Beta(2,5) scaled to [0,10] + Gaussian(15,2)"""
    return np.concatenate([rng.beta(2, 5, 3000) * 10, rng.normal(15, 2, 3000)])

def gen_3family_mix():
    """Gaussian(-10,1) + Exponential(2) + Laplace(10,0.5)"""
    return np.concatenate([rng.normal(-10, 1, 2000),
                           rng.exponential(2, 2000),
                           rng.laplace(10, 0.5, 2000)])

# ── HEAVY OVERLAP ──────────────────────────────────────────

def gen_gauss_overlap_2sigma():
    """Two Gaussians only 2σ apart (hard to separate)"""
    return np.concatenate([rng.normal(0, 1, 3000), rng.normal(2, 1, 3000)])

def gen_gauss_overlap_1sigma():
    """Two Gaussians only 1σ apart (very hard)"""
    return np.concatenate([rng.normal(0, 1, 3000), rng.normal(1, 1, 3000)])

def gen_gamma_overlap():
    """Two Gammas with similar shapes"""
    return np.concatenate([rng.gamma(3, 1, 3000), rng.gamma(5, 1.2, 3000)])

# ── EXTREME IMBALANCE ──────────────────────────────────────

def gen_gauss_imbalanced_95_5():
    """95% component + 5% outlier component"""
    return np.concatenate([rng.normal(0, 1, 9500), rng.normal(10, 0.5, 500)])

def gen_gauss_imbalanced_99_1():
    """99% component + 1% tiny component"""
    return np.concatenate([rng.normal(0, 1, 9900), rng.normal(8, 0.3, 100)])

def gen_expo_imbalanced():
    """90% fast decay + 10% slow decay"""
    return np.concatenate([rng.exponential(0.5, 9000), rng.exponential(10, 1000)])

# ── HIGH K ──────────────────────────────────────────────────

def gen_gauss_k8():
    """8 well-separated Gaussians"""
    return np.concatenate([rng.normal(j * 6, 0.8, 1000) for j in range(8)])

def gen_gauss_k10():
    """10 Gaussians (pushing limits)"""
    return np.concatenate([rng.normal(j * 5, 0.7, 800) for j in range(10)])

def gen_expo_k4():
    """4 exponential components"""
    rates = [0.3, 1, 3, 10]
    return np.concatenate([rng.exponential(1/r, 2000) for r in rates])

# ── SCALE EXTREMES ──────────────────────────────────────────

def gen_gauss_tiny():
    """Two Gaussians with very small values"""
    return np.concatenate([rng.normal(0.001, 0.0002, 3000), rng.normal(0.003, 0.0002, 3000)])

def gen_gauss_huge():
    """Two Gaussians with very large values"""
    return np.concatenate([rng.normal(1e6, 1e4, 3000), rng.normal(1.5e6, 1e4, 3000)])

def gen_gauss_mixed_scale():
    """Narrow + wide Gaussians at same location"""
    return np.concatenate([rng.normal(0, 0.1, 3000), rng.normal(0, 10, 3000)])

# ── EDGE CASES ──────────────────────────────────────────────

def gen_single_component():
    """Just one Gaussian (k=1 test)"""
    return rng.normal(5, 2, 6000)

def gen_nearly_uniform():
    """Very flat, nearly uniform distribution"""
    return rng.uniform(-10, 10, 6000) + rng.normal(0, 0.01, 6000)

def gen_heavy_tail():
    """Student-t with df=2 (very heavy tails)"""
    return np.concatenate([rng.standard_t(2, 3000) - 5, rng.standard_t(2, 3000) + 5])

def gen_bimodal_beta():
    """U-shaped beta (both modes at edges)"""
    return rng.beta(0.3, 0.3, 6000)

def gen_sparse_poisson():
    """Mostly zeros with a rare high-count component"""
    return np.concatenate([rng.poisson(0.5, 5000), rng.poisson(25, 1000)]).astype(float)

# ── LARGE N STRESS TEST ──────────────────────────────────────

def gen_gauss_large_n():
    """100K points, 3 Gaussians"""
    return np.concatenate([rng.normal(-5, 1, 33000), rng.normal(0, 1.5, 34000), rng.normal(7, 0.8, 33000)])


# ============================================================
#  SCENARIO DEFINITIONS
# ============================================================
scenarios = [
    # ── Mixed Family (the exciting ones!) ──
    ("Gaussian + Exponential",      "mixed", 2, gen_gauss_plus_expo, 5),
    ("Gaussian + Gamma",            "mixed", 2, gen_gauss_plus_gamma, 5),
    ("Laplace + Exponential",       "mixed", 2, gen_laplace_plus_expo, 5),
    ("Poisson + Gaussian",          "mixed", 2, gen_poisson_plus_gauss, 5),
    ("Beta(scaled) + Gaussian",     "mixed", 2, gen_beta_plus_gauss, 5),
    ("3-family mix (G+E+L)",        "mixed", 3, gen_3family_mix, 6),

    # ── Heavy Overlap ──
    ("Gaussian 2σ overlap",         "gaussian", 2, gen_gauss_overlap_2sigma, 5),
    ("Gaussian 1σ overlap",         "gaussian", 2, gen_gauss_overlap_1sigma, 5),
    ("Gamma overlap",               "gamma", 2, gen_gamma_overlap, 5),

    # ── Extreme Imbalance ──
    ("Gaussian 95/5 split",         "gaussian", 2, gen_gauss_imbalanced_95_5, 5),
    ("Gaussian 99/1 split",         "gaussian", 2, gen_gauss_imbalanced_99_1, 5),
    ("Exponential 90/10",           "exponential", 2, gen_expo_imbalanced, 5),

    # ── High k ──
    ("Gaussian k=8",                "gaussian", 8, gen_gauss_k8, 12),
    ("Gaussian k=10",               "gaussian", 10, gen_gauss_k10, 15),
    ("Exponential k=4",             "exponential", 4, gen_expo_k4, 8),

    # ── Scale Extremes ──
    ("Gaussian tiny (μ~0.001)",     "gaussian", 2, gen_gauss_tiny, 5),
    ("Gaussian huge (μ~1e6)",       "gaussian", 2, gen_gauss_huge, 5),
    ("Gaussian mixed scale (σ: 0.1 vs 10)", "gaussian", 2, gen_gauss_mixed_scale, 5),

    # ── Edge Cases ──
    ("Single component k=1",       "gaussian", 1, gen_single_component, 5),
    ("Nearly uniform",              "uniform", 1, gen_nearly_uniform, 5),
    ("Heavy-tail Student-t k=2",   "studentt", 2, gen_heavy_tail, 5),
    ("U-shaped Beta",               "beta", 1, gen_bimodal_beta, 5),
    ("Sparse Poisson + rare high", "poisson", 2, gen_sparse_poisson, 5),

    # ── Large N ──
    ("Large n=100K, k=3",          "gaussian", 3, gen_gauss_large_n, 6),
]


def run_adaptive(data, kmax, method='bic', timeout=180):
    """Run adaptive mode and parse results."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(str(x) + '\n')
    f.close()

    cmd = [GEM, '-g', f.name, '--adaptive', '--kmethod', method, '--kmax', str(kmax),
           '-o', '/tmp/adapt_ext_out.csv']

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        os.unlink(f.name)
    except subprocess.TimeoutExpired:
        os.unlink(f.name)
        return None, None, [], "TIMEOUT"
    except Exception as e:
        try: os.unlink(f.name)
        except: pass
        return None, None, [], str(e)

    output = r.stdout + '\n' + r.stderr

    best_k = None
    for line in output.split('\n'):
        if 'adaptive result' in line.lower() and 'k=' in line.lower():
            km = re.search(r'k\s*=\s*(\d+)', line)
            if km: best_k = int(km.group(1))

    # Parse families from CSV
    families_found = []
    try:
        with open('/tmp/adapt_ext_out.csv') as of:
            for line in of:
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    fam = parts[0].strip().lower()
                    if fam and not fam[0].isdigit():
                        families_found.append(fam)
    except: pass

    if best_k is None:
        best_k = len(families_found) if families_found else None

    return families_found[0] if len(set(families_found)) == 1 else (families_found if families_found else None), \
           best_k, families_found, output


# ============================================================
#  RUN ALL SCENARIOS
# ============================================================
print("=" * 80)
print("  EXTENDED ADAPTIVE BENCHMARK — Weird & Hard Scenarios")
print("  Testing mixed-family, overlap, imbalance, high-k, scale, edge cases")
print("=" * 80)

results = []
n_seeds = 3

for sname, true_fam, true_k, gen_fn, kmax in scenarios:
    print(f"\n{'─'*80}")
    print(f"  {sname}")
    print(f"  True: {true_fam} k={true_k}")
    print(f"{'─'*80}")

    k_correct = 0
    fam_scores = []

    for seed in range(n_seeds):
        data = gen_fn()
        result_fam, found_k, all_fams, output = run_adaptive(data, kmax)

        k_match = found_k == true_k
        if k_match: k_correct += 1

        # For mixed-family scenarios, check if k is right (family matching is ambiguous)
        if true_fam == "mixed":
            fam_display = ",".join(sorted(set(all_fams))) if all_fams else "None"
            status_k = "✅" if k_match else "❌"
            print(f"  seed {seed}: k={found_k} {status_k}  families=[{fam_display}]")
            fam_scores.append(1 if k_match else 0)  # k is what matters for mixed
        else:
            found = result_fam if isinstance(result_fam, str) else (all_fams[0] if all_fams else None)
            fam_match = found == true_fam if found else False
            related = False
            if found and true_fam in FAMILY_GROUPS:
                related = found in FAMILY_GROUPS[true_fam]

            status_k = "✅" if k_match else "❌"
            status_f = "✅" if fam_match else ("~" if related else "❌")
            print(f"  seed {seed}: k={found_k} {status_k}  family={found} {status_f}")
            fam_scores.append(2 if fam_match else (1 if related else 0))

    k_pct = k_correct / n_seeds * 100
    # For family: 2=exact, 1=related, 0=wrong; normalize to %
    fam_pct = sum(1 for s in fam_scores if s == 2) / n_seeds * 100
    rel_pct = sum(1 for s in fam_scores if s >= 1) / n_seeds * 100

    results.append({
        'name': sname, 'true_fam': true_fam, 'true_k': true_k,
        'k_accuracy': k_pct, 'family_exact': fam_pct, 'family_related': rel_pct
    })

    print(f"  → k accuracy: {k_pct:.0f}%  |  family exact: {fam_pct:.0f}%  |  family group: {rel_pct:.0f}%")


# ============================================================
#  SUMMARY TABLE
# ============================================================
print(f"\n{'='*80}")
print(f"  EXTENDED SUMMARY ({n_seeds} seeds per scenario)")
print(f"{'='*80}")

categories = {
    "Mixed Family":    [r for r in results if r['true_fam'] == 'mixed'],
    "Heavy Overlap":   [r for r in results if 'overlap' in r['name'].lower()],
    "Imbalance":       [r for r in results if 'split' in r['name'].lower() or '90/10' in r['name']],
    "High k":          [r for r in results if 'k=8' in r['name'] or 'k=10' in r['name'] or 'k=4' in r['name']],
    "Scale Extremes":  [r for r in results if 'tiny' in r['name'].lower() or 'huge' in r['name'].lower() or 'mixed scale' in r['name'].lower()],
    "Edge Cases":      [r for r in results if any(w in r['name'].lower() for w in ['single', 'nearly', 'heavy-tail', 'u-shaped', 'sparse'])],
    "Large N":         [r for r in results if 'large' in r['name'].lower()],
}

total_k = total_fam = total_rel = total_n = 0

for cat_name, cat_results in categories.items():
    if not cat_results: continue
    print(f"\n  ── {cat_name} ──")
    print(f"  {'Scenario':<35s}  {'True':>10s}  {'k acc':>6s}  {'Fam ✓':>6s}  {'Fam ~':>6s}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*6}")

    for r in cat_results:
        label = f"{r['true_fam']} k={r['true_k']}"
        print(f"  {r['name']:<35s}  {label:>10s}  {r['k_accuracy']:5.0f}%  {r['family_exact']:5.0f}%  {r['family_related']:5.0f}%")
        total_k += r['k_accuracy']
        total_fam += r['family_exact']
        total_rel += r['family_related']
        total_n += 1

print(f"\n  {'─'*35}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*6}")
if total_n > 0:
    print(f"  {'OVERALL AVERAGE':<35s}  {'':>10s}  {total_k/total_n:5.1f}%  {total_fam/total_n:5.1f}%  {total_rel/total_n:5.1f}%")

print(f"\n{'='*80}")
avg_k = total_k / total_n if total_n else 0
avg_rel = total_rel / total_n if total_n else 0
if avg_k >= 70 and avg_rel >= 50:
    print(f"  🏆 GRADE: EXCELLENT (k={avg_k:.0f}%, family group={avg_rel:.0f}%)")
elif avg_k >= 50 and avg_rel >= 35:
    print(f"  ✅ GRADE: GOOD (k={avg_k:.0f}%, family group={avg_rel:.0f}%)")
elif avg_k >= 35:
    print(f"  ⚠️  GRADE: NEEDS WORK (k={avg_k:.0f}%, family group={avg_rel:.0f}%)")
else:
    print(f"  ❌ GRADE: POOR (k={avg_k:.0f}%, family group={avg_rel:.0f}%)")
print(f"{'='*80}")
