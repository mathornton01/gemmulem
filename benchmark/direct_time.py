import numpy as np, subprocess, tempfile, os, time
rng = np.random.default_rng(42)
BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"

for k in [4, 6, 8]:
    n = 20000
    parts = [rng.normal(j * 5.0, 1.0, n//k) for j in range(k)]
    d = np.concatenate(parts)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for x in d: f.write(str(x)+'\n')
    f.close()

    r = subprocess.run(
        ['bash', '-c',
         f"time {BUILD}/gemmulem -g {f.name} -d Gaussian -k {k} -o /dev/null"],
        capture_output=True, text=True, timeout=30)
    timing_line = [l for l in r.stderr.splitlines() if 'real' in l or 'elapsed' in l]
    gemmulem_out = [l for l in r.stdout.splitlines() if 'Converged' in l or 'iter' in l.lower()]
    print(f"k={k}: {r.stderr.strip()[-100:]} | stdout: {r.stdout[-200:]}")
    os.unlink(f.name)
