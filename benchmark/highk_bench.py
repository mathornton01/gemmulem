#!/usr/bin/env python3
"""Profile high-k slowdown."""
import numpy as np, subprocess, time, tempfile, os

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)

def bench(data, args):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(str(x) + '\n')
    f.close()
    t0 = time.perf_counter()
    r = subprocess.run([GEM, '-g', f.name] + args + ['-o', '/dev/null'],
                       capture_output=True, text=True, timeout=120)
    t = (time.perf_counter() - t0) * 1000
    # Extract iteration count
    iters = "?"
    for line in r.stdout.split('\n'):
        if 'Converged' in line:
            iters = line.split('in')[1].split('iter')[0].strip() if 'in' in line else '?'
    os.unlink(f.name)
    return t, iters

n = 20000
print("High-k profiling: n=20000, Gaussian")
print(f"  {'k':>3}  {'time':>10}  {'iters':>6}  {'ms/iter':>8}")
print(f"  {'─'*3}  {'─'*10}  {'─'*6}  {'─'*8}")

for k in [2, 3, 4, 5, 6, 8, 10, 15]:
    parts = [rng.normal(-10 + 20*j/(k-1) if k > 1 else 0, 0.8, n//k) for j in range(k)]
    d = np.concatenate(parts)
    t, iters = bench(d, ['-d', 'Gaussian', '-k', str(k), '-v'])
    try:
        ms_per = t / int(iters)
    except:
        ms_per = 0
    print(f"  {k:>3}  {t:>8.1f}ms  {iters:>6}  {ms_per:>6.1f}ms")
