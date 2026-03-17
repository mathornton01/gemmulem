"""Profile k=8 to figure out what's slow."""
import numpy as np, subprocess, tempfile, os, time
rng = np.random.default_rng(42)

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"

for k in [4, 8]:
    n = 20000
    parts = [rng.normal(j * 5.0, 1.0, n//k) for j in range(k)]
    d = np.concatenate(parts)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for x in d: f.write(str(x) + '\n')
    f.close()

    # Run 3 times, take min
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        subprocess.run([BUILD + '/gemmulem', '-g', f.name, '-d', 'Gaussian',
                        '-k', str(k), '-o', '/dev/null'],
            capture_output=True, timeout=30)
        times.append((time.perf_counter() - t0) * 1000)
    t = min(times)

    # Get iter count
    r = subprocess.run([BUILD + '/gemmulem', '-g', f.name, '-d', 'Gaussian',
                        '-k', str(k), '-v', '-o', '/dev/null'],
        capture_output=True, text=True, timeout=30)
    iters = [l for l in r.stdout.splitlines() if 'iter' in l and 'LL=' in l]

    print(f"k={k}: {min(times):.1f}ms total, {len(iters)} iters, {t/len(iters):.1f}ms/iter")
    print(f"  Expected E-step ops: {n}×{k}×(mul+add+exp) = {n*k} exp calls per iter")
    print(f"  Throughput: {n*k*len(iters)/t*1000/1e6:.0f}M exp/sec")
    os.unlink(f.name)
