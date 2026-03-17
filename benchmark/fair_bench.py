"""Fair comparison: same tolerance (1e-3) as sklearn default."""
import numpy as np, subprocess, tempfile, os, time
rng = np.random.default_rng(42)
from sklearn.mixture import GaussianMixture

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"

print("=" * 70)
print("  Fair Comparison: same rtol=1e-3, k-means++ init, n=20000")
print("=" * 70)
print(f"\n  {'k':>3}  {'iters':>6}  {'Gemmulem':>10}  {'sklearn':>8}  {'ratio':>6}")
for k in [2, 4, 6, 8, 10]:
    n = 20000
    parts = [rng.normal(j * 5.0, 1.0, n//k) for j in range(k)]
    d = np.concatenate(parts)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for x in d: f.write(str(x) + '\n')
    f.close()

    t0 = time.perf_counter()
    r = subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', str(k),
                        '-e', '1e-3', '-o', '/dev/null'],
        capture_output=True, text=True, timeout=30)
    tg = (time.perf_counter() - t0) * 1000
    iters = [l for l in r.stdout.splitlines() if 'Converged' in l or 'iter' in l.lower()]

    t0 = time.perf_counter()
    GaussianMixture(k, max_iter=300, tol=1e-3, random_state=42).fit(d.reshape(-1,1))
    ts = (time.perf_counter() - t0) * 1000
    ratio = ts / tg if tg > 0 else 0
    flag = " ✅" if ratio >= 1.0 else " ❌"
    print(f"  {k:>3d}  {len(iters):>6d}  {tg:>8.1f}ms  {ts:>6.1f}ms  {ratio:>5.2f}x{flag}")
    os.unlink(f.name)

print()
print("="*70)
print("  Standard comparison: rtol=1e-5, n=10000")
print("="*70)
print(f"\n  {'k':>3}  {'iters':>6}  {'Gemmulem':>10}  {'sklearn':>8}  {'ratio':>6}")
for k in [2, 4, 6, 8]:
    n = 10000
    parts = [rng.normal(j * 5.0, 1.0, n//k) for j in range(k)]
    d = np.concatenate(parts)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for x in d: f.write(str(x) + '\n')
    f.close()

    t0 = time.perf_counter()
    r = subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', str(k), '-o', '/dev/null'],
        capture_output=True, text=True, timeout=30)
    tg = (time.perf_counter() - t0) * 1000
    iters = [l for l in r.stdout.splitlines() if 'Converged' in l]

    t0 = time.perf_counter()
    GaussianMixture(k, max_iter=300, tol=1e-5, random_state=42).fit(d.reshape(-1,1))
    ts = (time.perf_counter() - t0) * 1000
    ratio = ts / tg if tg > 0 else 0
    flag = " ✅" if ratio >= 1.0 else " ❌"
    print(f"  {k:>3d}  {len(iters):>6d}  {tg:>8.1f}ms  {ts:>6.1f}ms  {ratio:>5.2f}x{flag}")
    os.unlink(f.name)
