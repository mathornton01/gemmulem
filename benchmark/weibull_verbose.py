#!/usr/bin/env python3
import numpy as np, subprocess, os
rng = np.random.default_rng(42)
d = np.concatenate([rng.weibull(1.5, 3000)*2, rng.weibull(4, 3000)*8])
with open("/tmp/wb.txt", "w") as f:
    for x in d: f.write(str(x)+"\n")
r = subprocess.run(["/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build/gemmulem",
    "-g", "/tmp/wb.txt", "--adaptive", "--kmethod", "bic", "--kmax", "4", "-v",
    "-o", "/tmp/wb_out.csv"], capture_output=True, text=True, timeout=120)
for line in r.stdout.split('\n'):
    if any(s in line for s in ['k=', 'Splitting', 'score', 'LL=']):
        print(line)
os.unlink("/tmp/wb.txt")
