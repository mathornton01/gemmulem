#!/usr/bin/env python3
import numpy as np, subprocess, os
BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)
d = np.concatenate([rng.weibull(1.5, 3000)*2, rng.weibull(4, 3000)*8])
with open('/tmp/wb.txt', 'w') as f:
    for x in d: f.write(str(x)+'\n')
r = subprocess.run([GEM, '-g', '/tmp/wb.txt', '--adaptive', '--kmethod', 'bic',
                    '--kmax', '4', '-o', '/tmp/wb_out.csv'],
                   capture_output=True, text=True, timeout=120)
print(r.stdout[-1500:])
os.unlink('/tmp/wb.txt')
