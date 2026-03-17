import numpy as np, subprocess, tempfile, os
rng = np.random.default_rng(42)
k = 8
n = 20000
parts = [rng.normal(-10 + 20*j/(k-1), 0.8, n//k) for j in range(k)]
d = np.concatenate(parts)
f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
for x in d:
    f.write(str(x) + '\n')
f.close()
r = subprocess.run(
    ['/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build/gemmulem',
     '-g', f.name, '-d', 'Gaussian', '-k', '8', '-v', '-o', '/dev/null'],
    capture_output=True, text=True, timeout=30)
lines = r.stdout.strip().splitlines()
iters = [l for l in lines if 'iter' in l.lower() and 'LL=' in l]
print(f"Total EM iterations: {len(iters)}")
if iters:
    print("First:", iters[0])
    print("Last:", iters[-1])
os.unlink(f.name)
