import numpy as np, subprocess, tempfile, os
rng = np.random.default_rng(42)
k = 6
n = 5000
parts = [rng.normal(j * 5.0, 1.0, n//k) for j in range(k)]
d = np.concatenate(parts)
f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
for x in d: f.write(str(x) + '\n')
f.close()
r = subprocess.run(
    ['/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build/gemmulem',
     '-g', f.name, '-d', 'Gaussian', '-k', str(k), '-v', '-o', '/dev/null'],
    capture_output=True, text=True, timeout=60)
lines = r.stdout.strip().splitlines()
iters = [l for l in lines if 'iter' in l and 'LL=' in l]
print(f"Total EM iterations: {len(iters)}")
print("First 5:")
for l in iters[:5]: print(l)
print("Last 5:")
for l in iters[-5:]: print(l)
os.unlink(f.name)
