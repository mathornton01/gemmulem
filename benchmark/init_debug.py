import numpy as np, subprocess, tempfile, os
rng = np.random.default_rng(42)

for k in [4, 6, 8]:
    n = 20000
    parts = [rng.normal(j * 5.0, 1.0, n//k) for j in range(k)]
    d = np.concatenate(parts)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for x in d: f.write(str(x) + '\n')
    f.close()
    r = subprocess.run(
        ['/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build/gemmulem',
         '-g', f.name, '-d', 'Gaussian', '-k', str(k), '-v', '-o', '/dev/null'],
        capture_output=True, text=True, timeout=60)
    lines = [l for l in r.stdout.splitlines() if 'iter' in l and 'LL=' in l]
    print(f"k={k}: {len(lines)} iters | first LL={lines[0].split('LL=')[1].split()[0] if lines else 'N/A'}")
    print(f"       delta[0]={lines[0].split('delta=')[1] if lines else 'N/A'}")
    if len(lines) >= 5:
        print(f"       delta[4]={lines[4].split('delta=')[1]}")
    os.unlink(f.name)
