import numpy as np, subprocess, tempfile, os
rng = np.random.default_rng(42)
d = np.concatenate([rng.normal(-8,1,500), rng.normal(0,1,500), rng.normal(8,1,500)])
f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
for x in d:
    f.write(str(x) + '\n')
f.close()
r = subprocess.run(
    ['/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build/gemmulem',
     '-g', f.name, '--adaptive', '--kmethod', 'vbem', '--kmax', '4', '-v', '-o', '/dev/null'],
    capture_output=True, text=True, timeout=15)
print(r.stdout[-2000:])
print("STDERR:", r.stderr[-200:])
os.unlink(f.name)
