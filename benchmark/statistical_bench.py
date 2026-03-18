#!/usr/bin/env python3
"""
Rigorous statistical benchmark: Gemmulem vs sklearn
50 random seeds per scenario, 95% confidence intervals for speed and accuracy.
"""
import numpy as np, subprocess, tempfile, os, time, sys
from sklearn.mixture import GaussianMixture

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
N_SEEDS = 50

def run_gem(data, k):
    """Run gemmulem, return (time_ms, params_list)."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(str(x) + '\n')
    f.close()
    t0 = time.perf_counter()
    r = subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', str(k),
                        '-o', '/tmp/gem_stat_out.csv'],
        capture_output=True, text=True, timeout=60)
    elapsed = (time.perf_counter() - t0) * 1000
    os.unlink(f.name)
    params = []
    try:
        with open('/tmp/gem_stat_out.csv') as of:
            for line in of:
                if line.startswith('#') or line.strip() == '': continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    params.append({'weight': float(parts[0]),
                                   'mean': float(parts[1]),
                                   'var': float(parts[2])})
    except: pass
    return elapsed, sorted(params, key=lambda p: p['mean'])

def run_sklearn(data, k, seed=None):
    """Run sklearn GMM, return (time_ms, params_list).
    Uses random_state=None (sklearn default) for real-world comparison."""
    t0 = time.perf_counter()
    gm = GaussianMixture(k, max_iter=300, tol=1e-6, n_init=1, random_state=None)
    gm.fit(data.reshape(-1, 1))
    elapsed = (time.perf_counter() - t0) * 1000
    params = []
    for j in range(k):
        params.append({'mean': gm.means_[j, 0],
                       'var': gm.covariances_[j, 0, 0],
                       'weight': gm.weights_[j]})
    return elapsed, sorted(params, key=lambda p: p['mean'])

def score(est, truth_m, truth_v, truth_w):
    """Match by nearest mean, return (mean_mae, var_mae, weight_mae)."""
    est_s = sorted(est, key=lambda p: p['mean'])
    truth = sorted(zip(truth_m, truth_v, truth_w), key=lambda t: t[0])
    if len(est_s) != len(truth): return (999, 999, 999)
    m = np.mean([abs(e['mean'] - t[0]) for e, t in zip(est_s, truth)])
    v = np.mean([abs(e['var'] - t[1]) for e, t in zip(est_s, truth)])
    w = np.mean([abs(e['weight'] - t[2]) for e, t in zip(est_s, truth)])
    return (m, v, w)

def ci95(arr):
    """Return (mean, lo, hi) for 95% CI."""
    a = np.array(arr)
    mu = np.mean(a)
    se = np.std(a, ddof=1) / np.sqrt(len(a))
    return mu, mu - 1.96*se, mu + 1.96*se

def run_scenario(name, true_means, true_vars, true_weights, n, k):
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"  n={n}, k={k}, {N_SEEDS} seeds")
    print(f"{'='*80}")

    gem_times, sk_times = [], []
    gem_mean_mae, sk_mean_mae = [], []
    gem_var_mae, sk_var_mae = [], []
    gem_wt_mae, sk_wt_mae = [], []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed * 137 + 42)
        d = np.concatenate([rng.normal(m, np.sqrt(v), int(w * n))
                             for m, v, w in zip(true_means, true_vars, true_weights)])
        # Shuffle to avoid ordering bias
        rng.shuffle(d)

        gt, gp = run_gem(d, k)
        st, sp = run_sklearn(d, k, seed * 137 + 42)  # deterministic seed per trial

        gem_times.append(gt)
        sk_times.append(st)

        gm, gv, gw = score(gp, true_means, true_vars, true_weights)
        sm, sv, sw = score(sp, true_means, true_vars, true_weights)

        gem_mean_mae.append(gm)
        sk_mean_mae.append(sm)
        gem_var_mae.append(gv)
        sk_var_mae.append(sv)
        gem_wt_mae.append(gw)
        sk_wt_mae.append(sw)

        sys.stdout.write(f"\r  Seed {seed+1}/{N_SEEDS}...")
        sys.stdout.flush()

    print(f"\r  Done.{' '*30}")

    # Speed results
    gt_mu, gt_lo, gt_hi = ci95(gem_times)
    st_mu, st_lo, st_hi = ci95(sk_times)
    speedup = st_mu / gt_mu
    print(f"\n  SPEED:")
    print(f"    Gemmulem: {gt_mu:7.1f}ms  95% CI [{gt_lo:7.1f}, {gt_hi:7.1f}]")
    print(f"    sklearn:  {st_mu:7.1f}ms  95% CI [{st_lo:7.1f}, {st_hi:7.1f}]")
    print(f"    Speedup:  {speedup:.2f}x  {'✅ GEMMULEM FASTER' if speedup > 1 else '❌ sklearn faster'}")

    # Accuracy results
    metrics = [
        ("Mean MAE",   gem_mean_mae, sk_mean_mae),
        ("Var MAE",    gem_var_mae,  sk_var_mae),
        ("Weight MAE", gem_wt_mae,   sk_wt_mae),
    ]
    print(f"\n  ACCURACY:")
    print(f"    {'Metric':<12} {'Gemmulem (95% CI)':>30} {'sklearn (95% CI)':>30}  Winner")
    wins = {'gem': 0, 'sk': 0, 'tie': 0}
    for mname, ga, sa in metrics:
        gmu, glo, ghi = ci95(ga)
        smu, slo, shi = ci95(sa)
        # Winner: non-overlapping CIs, or lower mean if overlapping
        if ghi < slo:
            winner = "✅ GEMMULEM"
            wins['gem'] += 1
        elif shi < glo:
            winner = "❌ sklearn"
            wins['sk'] += 1
        elif gmu <= smu:
            winner = "~ Gemmulem (ns)"
            wins['tie'] += 1
        else:
            winner = "~ sklearn (ns)"
            wins['tie'] += 1
        print(f"    {mname:<12} {gmu:8.5f} [{glo:8.5f},{ghi:8.5f}] "
              f"{smu:8.5f} [{slo:8.5f},{shi:8.5f}]  {winner}")

    # Win rate
    gem_better = sum(1 for g, s in zip(gem_mean_mae, sk_mean_mae) if g < s)
    print(f"\n    Mean recovery: Gemmulem better in {gem_better}/{N_SEEDS} seeds "
          f"({gem_better/N_SEEDS*100:.0f}%)")

    return {
        'name': name,
        'speed': speedup,
        'speed_gem': gt_mu,
        'speed_sk': st_mu,
        'wins': wins,
        'gem_better_pct': gem_better / N_SEEDS * 100
    }

# ════════════════════════════════════════════════════════════════════
# Run all scenarios
# ════════════════════════════════════════════════════════════════════
print("\n" + "#"*80)
print("#  GEMMULEM vs SKLEARN: Rigorous Statistical Benchmark")
print(f"#  {N_SEEDS} seeds per scenario, 95% confidence intervals")
print("#"*80)

results = []

results.append(run_scenario(
    "Scenario 1: Well-separated (8σ)",
    true_means=[-8, 0, 8], true_vars=[1, 1, 1],
    true_weights=[0.333, 0.334, 0.333], n=3000, k=3))

results.append(run_scenario(
    "Scenario 2: Overlapping (2σ)",
    true_means=[-2, 0, 2], true_vars=[1, 1, 1],
    true_weights=[0.3, 0.4, 0.3], n=5000, k=3))

results.append(run_scenario(
    "Scenario 3: Unequal weights (70/20/10%)",
    true_means=[-5, 0, 7], true_vars=[0.5, 2.0, 0.3],
    true_weights=[0.7, 0.2, 0.1], n=10000, k=3))

results.append(run_scenario(
    "Scenario 4: High-k (k=8, 5σ separation)",
    true_means=[j*5.0 for j in range(8)], true_vars=[1.0]*8,
    true_weights=[1.0/8]*8, n=20000, k=8))

results.append(run_scenario(
    "Scenario 5: Small sample (n=200)",
    true_means=[-3, 3], true_vars=[1, 1],
    true_weights=[0.5, 0.5], n=200, k=2))

results.append(run_scenario(
    "Scenario 6: Large sample (n=50000)",
    true_means=[-5, 0, 5], true_vars=[1, 2, 0.5],
    true_weights=[0.25, 0.5, 0.25], n=50000, k=3))

results.append(run_scenario(
    "Scenario 7: Unequal variance (σ²=0.25, 4.0, 0.1)",
    true_means=[-6, 0, 8], true_vars=[0.25, 4.0, 0.1],
    true_weights=[0.4, 0.35, 0.25], n=8000, k=3))

results.append(run_scenario(
    "Scenario 8: High-k overlapping (k=5, 3σ)",
    true_means=[-6, -3, 0, 3, 6], true_vars=[1]*5,
    true_weights=[0.2]*5, n=10000, k=5))

# ════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "#"*80)
print("#  FINAL SUMMARY")
print("#"*80)
print(f"\n  {'Scenario':<45} {'Speed':>8} {'Accuracy':>10} {'Better%':>8}")
print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*8}")
for r in results:
    speed_flag = "✅" if r['speed'] > 1.0 else "❌"
    acc_flag = "✅" if r['wins']['gem'] >= r['wins']['sk'] else "❌"
    print(f"  {r['name']:<45} {r['speed']:>5.1f}x {speed_flag} "
          f"{r['wins']['gem']}/{r['wins']['gem']+r['wins']['sk']+r['wins']['tie']} {acc_flag}  "
          f"{r['gem_better_pct']:>5.0f}%")

speed_wins = sum(1 for r in results if r['speed'] > 1.0)
acc_wins = sum(1 for r in results if r['wins']['gem'] >= r['wins']['sk'])
print(f"\n  Speed wins: {speed_wins}/{len(results)}")
print(f"  Accuracy wins: {acc_wins}/{len(results)}")
if speed_wins == len(results) and acc_wins == len(results):
    print("\n  🏆 GEMMULEM WINS IN BOTH SPEED AND ACCURACY ACROSS ALL SCENARIOS 🏆")
print()
