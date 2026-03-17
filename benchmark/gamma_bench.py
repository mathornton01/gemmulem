import subprocess, time
import numpy as np

rng = np.random.default_rng(42)
d = np.concatenate([rng.gamma(4, 2, 5000), rng.gamma(8, 1, 5000)])
with open('/tmp/gamma_bench.txt', 'w') as f:
    for x in d:
        f.write(str(x) + '\n')

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"

t0 = time.perf_counter()
subprocess.run([BUILD + '/gemmulem', '-g', '/tmp/gamma_bench.txt', '-d', 'Gamma', '-k', '2', '-o', '/dev/null'], capture_output=True)
t = (time.perf_counter() - t0) * 1000
print(f'Gamma n=10000 k=2 (after optimization): {t:.1f}ms  (was 430ms)')

# Also benchmark Gaussian for comparison
d2 = np.concatenate([rng.normal(-4, 1, 5000), rng.normal(4, 1, 5000)])
with open('/tmp/gauss_bench.txt', 'w') as f:
    for x in d2:
        f.write(str(x) + '\n')

t0 = time.perf_counter()
subprocess.run([BUILD + '/gemmulem', '-g', '/tmp/gauss_bench.txt', '-d', 'Gaussian', '-k', '2', '-o', '/dev/null'], capture_output=True)
t2 = (time.perf_counter() - t0) * 1000
print(f'Gaussian n=10000 k=2 (baseline):        {t2:.1f}ms')
print(f'Speedup ratio (Gamma/Gaussian):           {t/t2:.1f}x  (ideal: ~1.0x)')
