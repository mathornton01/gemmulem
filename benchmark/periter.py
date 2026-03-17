import numpy as np, subprocess, tempfile, os, time
rng = np.random.default_rng(42)

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"

for k in [2, 4, 6, 8, 10]:
    n = 20000
    parts = [rng.normal(-10 + 20*j/(max(k-1,1)), 0.8, n//k) for j in range(k)]
    d = np.concatenate(parts)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for x in d: f.write(str(x) + '\n')
    f.close()
    t0 = time.perf_counter()
    r = subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', str(k), '-v', '-o', '/dev/null'],
        capture_output=True, text=True, timeout=30)
    total = (time.perf_counter() - t0) * 1000
    iters = [l for l in r.stdout.splitlines() if 'iter' in l and 'LL=' in l]
    n_iters = len(iters)
    ms_per = total / n_iters if n_iters else 0
    print(f"k={k:2d}  n_iters={n_iters:3d}  total={total:7.1f}ms  ms/iter={ms_per:6.1f}ms")
    os.unlink(f.name)
