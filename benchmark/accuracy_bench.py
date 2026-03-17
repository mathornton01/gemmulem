#!/usr/bin/env python3
"""
Accuracy comparison: Gemmulem vs sklearn GaussianMixture.
Tests parameter recovery (means, variances, weights) on known ground truth.
"""
import numpy as np, subprocess, tempfile, os, json
from sklearn.mixture import GaussianMixture

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)

def run_gem(data, k):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(str(x) + '\n')
    f.close()
    r = subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', str(k),
                        '-o', '/tmp/gem_out.csv'],
        capture_output=True, text=True, timeout=30)
    os.unlink(f.name)
    # Parse output CSV: weight,mean,var per line (Gemmulem output format)
    params = []
    try:
        with open('/tmp/gem_out.csv') as of:
            for line in of:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    params.append({
                        'weight': float(parts[0]),
                        'mean': float(parts[1]),
                        'var': float(parts[2])
                    })
    except:
        pass
    return params

def run_sklearn(data, k):
    gm = GaussianMixture(k, max_iter=300, tol=1e-6, n_init=1, random_state=42)
    gm.fit(data.reshape(-1, 1))
    params = []
    for j in range(k):
        params.append({
            'mean': gm.means_[j, 0],
            'var': gm.covariances_[j, 0, 0],
            'weight': gm.weights_[j]
        })
    return sorted(params, key=lambda p: p['mean'])

def match_and_score(estimated, truth_means, truth_vars, truth_weights):
    """Match estimated components to truth by nearest mean, compute errors."""
    est = sorted(estimated, key=lambda p: p['mean'])
    truth = sorted(zip(truth_means, truth_vars, truth_weights), key=lambda t: t[0])
    
    if len(est) != len(truth):
        return {'mean_mae': 999, 'var_mae': 999, 'weight_mae': 999}
    
    mean_err = np.mean([abs(e['mean'] - t[0]) for e, t in zip(est, truth)])
    var_err = np.mean([abs(e['var'] - t[1]) for e, t in zip(est, truth)])
    weight_err = np.mean([abs(e['weight'] - t[2]) for e, t in zip(est, truth)])
    
    # Relative errors
    mean_rel = np.mean([abs(e['mean'] - t[0]) / max(abs(t[0]), 0.01) for e, t in zip(est, truth)])
    var_rel = np.mean([abs(e['var'] - t[1]) / max(t[1], 0.01) for e, t in zip(est, truth)])
    
    return {
        'mean_mae': mean_err,
        'var_mae': var_err,
        'weight_mae': weight_err,
        'mean_rel': mean_rel,
        'var_rel': var_rel
    }

print("=" * 80)
print("  ACCURACY COMPARISON: Gemmulem vs sklearn")
print("=" * 80)

# ════════════════════════════════════════════════════════════════════
# Test 1: Easy — well-separated Gaussians
# ════════════════════════════════════════════════════════════════════
print("\n─── Test 1: Well-separated (k=3, n=3000, separation=8σ) ───")
true_means = [-8, 0, 8]
true_vars = [1, 1, 1]
true_weights = [0.333, 0.333, 0.334]
d = np.concatenate([rng.normal(m, np.sqrt(v), int(w*3000)) 
                     for m, v, w in zip(true_means, true_vars, true_weights)])

gem = run_gem(d, 3)
sk = run_sklearn(d, 3)
gem_score = match_and_score(gem, true_means, true_vars, true_weights)
sk_score = match_and_score(sk, true_means, true_vars, true_weights)

print(f"  {'Metric':<20} {'Gemmulem':>12} {'sklearn':>12} {'Winner':>10}")
for key in ['mean_mae', 'var_mae', 'weight_mae']:
    g, s = gem_score[key], sk_score[key]
    winner = "Gemmulem" if g <= s else "sklearn"
    print(f"  {key:<20} {g:>12.6f} {s:>12.6f} {winner:>10}")

# ════════════════════════════════════════════════════════════════════
# Test 2: Overlapping components
# ════════════════════════════════════════════════════════════════════
print("\n─── Test 2: Overlapping (k=3, n=5000, separation=2σ) ───")
true_means = [-2, 0, 2]
true_vars = [1, 1, 1]
true_weights = [0.3, 0.4, 0.3]
d = np.concatenate([rng.normal(m, np.sqrt(v), int(w*5000))
                     for m, v, w in zip(true_means, true_vars, true_weights)])

gem = run_gem(d, 3)
sk = run_sklearn(d, 3)
gem_score = match_and_score(gem, true_means, true_vars, true_weights)
sk_score = match_and_score(sk, true_means, true_vars, true_weights)

print(f"  {'Metric':<20} {'Gemmulem':>12} {'sklearn':>12} {'Winner':>10}")
for key in ['mean_mae', 'var_mae', 'weight_mae']:
    g, s = gem_score[key], sk_score[key]
    winner = "Gemmulem" if g <= s else "sklearn"
    print(f"  {key:<20} {g:>12.6f} {s:>12.6f} {winner:>10}")

# ════════════════════════════════════════════════════════════════════
# Test 3: Unequal weights
# ════════════════════════════════════════════════════════════════════
print("\n─── Test 3: Unequal weights (k=3, n=10000, weights=0.7/0.2/0.1) ───")
true_means = [-5, 0, 7]
true_vars = [0.5, 2.0, 0.3]
true_weights = [0.7, 0.2, 0.1]
d = np.concatenate([rng.normal(m, np.sqrt(v), int(w*10000))
                     for m, v, w in zip(true_means, true_vars, true_weights)])

gem = run_gem(d, 3)
sk = run_sklearn(d, 3)
gem_score = match_and_score(gem, true_means, true_vars, true_weights)
sk_score = match_and_score(sk, true_means, true_vars, true_weights)

print(f"  {'Metric':<20} {'Gemmulem':>12} {'sklearn':>12} {'Winner':>10}")
for key in ['mean_mae', 'var_mae', 'weight_mae']:
    g, s = gem_score[key], sk_score[key]
    winner = "Gemmulem" if g <= s else "sklearn"
    print(f"  {key:<20} {g:>12.6f} {s:>12.6f} {winner:>10}")

# ════════════════════════════════════════════════════════════════════
# Test 4: High-k 
# ════════════════════════════════════════════════════════════════════
print("\n─── Test 4: High-k (k=8, n=20000, separation=5σ) ───")
true_means = [j * 5.0 for j in range(8)]
true_vars = [1.0] * 8
true_weights = [1.0/8] * 8
d = np.concatenate([rng.normal(m, np.sqrt(v), int(w*20000))
                     for m, v, w in zip(true_means, true_vars, true_weights)])

gem = run_gem(d, 8)
sk = run_sklearn(d, 8)
gem_score = match_and_score(gem, true_means, true_vars, true_weights)
sk_score = match_and_score(sk, true_means, true_vars, true_weights)

print(f"  {'Metric':<20} {'Gemmulem':>12} {'sklearn':>12} {'Winner':>10}")
for key in ['mean_mae', 'var_mae', 'weight_mae']:
    g, s = gem_score[key], sk_score[key]
    winner = "Gemmulem" if g <= s else "sklearn"
    print(f"  {key:<20} {g:>12.6f} {s:>12.6f} {winner:>10}")

# ════════════════════════════════════════════════════════════════════
# Test 5: Small sample
# ════════════════════════════════════════════════════════════════════
print("\n─── Test 5: Small sample (k=2, n=100) ───")
true_means = [-3, 3]
true_vars = [1, 1]
true_weights = [0.5, 0.5]
d = np.concatenate([rng.normal(m, np.sqrt(v), int(w*100))
                     for m, v, w in zip(true_means, true_vars, true_weights)])

gem = run_gem(d, 2)
sk = run_sklearn(d, 2)
gem_score = match_and_score(gem, true_means, true_vars, true_weights)
sk_score = match_and_score(sk, true_means, true_vars, true_weights)

print(f"  {'Metric':<20} {'Gemmulem':>12} {'sklearn':>12} {'Winner':>10}")
for key in ['mean_mae', 'var_mae', 'weight_mae']:
    g, s = gem_score[key], sk_score[key]
    winner = "Gemmulem" if g <= s else "sklearn"
    print(f"  {key:<20} {g:>12.6f} {s:>12.6f} {winner:>10}")

# ════════════════════════════════════════════════════════════════════
# Test 6: Large sample convergence
# ════════════════════════════════════════════════════════════════════
print("\n─── Test 6: Large sample (k=3, n=100000) ───")
true_means = [-5, 0, 5]
true_vars = [1, 2, 0.5]
true_weights = [0.25, 0.5, 0.25]
d = np.concatenate([rng.normal(m, np.sqrt(v), int(w*100000))
                     for m, v, w in zip(true_means, true_vars, true_weights)])

gem = run_gem(d, 3)
sk = run_sklearn(d, 3)
gem_score = match_and_score(gem, true_means, true_vars, true_weights)
sk_score = match_and_score(sk, true_means, true_vars, true_weights)

print(f"  {'Metric':<20} {'Gemmulem':>12} {'sklearn':>12} {'Winner':>10}")
for key in ['mean_mae', 'var_mae', 'weight_mae']:
    g, s = gem_score[key], sk_score[key]
    winner = "Gemmulem" if g <= s else "sklearn"
    print(f"  {key:<20} {g:>12.6f} {s:>12.6f} {winner:>10}")

# ════════════════════════════════════════════════════════════════════
# Test 7: Statistical consistency — 20 random seeds
# ════════════════════════════════════════════════════════════════════
print("\n─── Test 7: Statistical consistency (k=3, n=2000, 20 seeds) ───")
true_means = [-5, 0, 5]
true_vars = [1, 1, 1]
true_weights = [0.333, 0.334, 0.333]

gem_wins = {'mean_mae': 0, 'var_mae': 0, 'weight_mae': 0}
sk_wins = {'mean_mae': 0, 'var_mae': 0, 'weight_mae': 0}
gem_total = {'mean_mae': 0, 'var_mae': 0, 'weight_mae': 0}
sk_total = {'mean_mae': 0, 'var_mae': 0, 'weight_mae': 0}

for seed in range(20):
    rng2 = np.random.default_rng(seed * 137 + 42)
    d = np.concatenate([rng2.normal(m, np.sqrt(v), int(w*2000))
                         for m, v, w in zip(true_means, true_vars, true_weights)])
    gem = run_gem(d, 3)
    sk = run_sklearn(d, 3)
    gs = match_and_score(gem, true_means, true_vars, true_weights)
    ss = match_and_score(sk, true_means, true_vars, true_weights)
    for key in ['mean_mae', 'var_mae', 'weight_mae']:
        gem_total[key] += gs[key]
        sk_total[key] += ss[key]
        if gs[key] <= ss[key]:
            gem_wins[key] += 1
        else:
            sk_wins[key] += 1

print(f"  {'Metric':<20} {'Gem avg':>12} {'SK avg':>12} {'Gem wins':>10} {'SK wins':>10}")
for key in ['mean_mae', 'var_mae', 'weight_mae']:
    ga = gem_total[key] / 20
    sa = sk_total[key] / 20
    print(f"  {key:<20} {ga:>12.6f} {sa:>12.6f} {gem_wins[key]:>10d} {sk_wins[key]:>10d}")

print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)
print("  Both use the same EM algorithm (converge to same MLE)")
print("  Differences come from initialization and convergence tolerance")
print("  Lower MAE = better parameter recovery")
print("=" * 80 + "\n")
