#!/usr/bin/env python3
"""Profile the hot path: time breakdown for overlapping clusters."""
import numpy as np, time, subprocess, tempfile, os

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)

scenarios = {
    "overlapping 2σ (n=10K,k=3)": (np.concatenate([rng.normal(-2,1,3000), rng.normal(0,1,4000), rng.normal(2,1,3000)]), 3),
    "well-sep 8σ (n=10K,k=3)":    (np.concatenate([rng.normal(-8,1,3333), rng.normal(0,1,3334), rng.normal(8,1,3333)]), 3),
    "high-k (n=20K,k=8)":         (np.concatenate([rng.normal(j*5,1,2500) for j in range(8)]), 8),
    "large-n (n=50K,k=3)":        (np.concatenate([rng.normal(j*6,1,16667) for j in range(3)]), 3),
}

for name, (data, k) in scenarios.items():
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
    for x in data: f.write(str(x) + '\n')
    f.close()
    
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', str(k), '-o', '/tmp/prof_out.csv'],
                      capture_output=True)
        times.append(time.perf_counter() - t0)
    
    os.unlink(f.name)
    print(f"{name:35s}  median={np.median(times)*1000:6.1f}ms  min={min(times)*1000:6.1f}ms")
