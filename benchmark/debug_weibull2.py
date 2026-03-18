#!/usr/bin/env python3
"""Check what BIC a 2-component Weibull model gets vs 1-component HalfNormal."""
import numpy as np, subprocess, os
BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)
d = np.concatenate([rng.weibull(1.5, 3000)*2, rng.weibull(4, 3000)*8])
with open('/tmp/wb.txt', 'w') as f:
    for x in d: f.write(str(x)+'\n')

# Direct k=2 Weibull fit
r = subprocess.run([GEM, '-g', '/tmp/wb.txt', '-d', 'weibull', '-k', '2', '-o', '/tmp/wb2_out.csv'],
                   capture_output=True, text=True, timeout=30)
print("=== 2-Weibull fit ===")
for line in r.stdout.split('\n'):
    if 'LL=' in line or 'Component' in line:
        print(line)

# Direct k=1 HalfNormal
r2 = subprocess.run([GEM, '-g', '/tmp/wb.txt', '-d', 'halfnormal', '-k', '1', '-o', '/dev/null'],
                   capture_output=True, text=True, timeout=30)
print("\n=== 1-HalfNormal fit ===")
for line in r2.stdout.split('\n'):
    if 'LL=' in line or 'Component' in line:
        print(line)

# Direct k=1 Weibull
r3 = subprocess.run([GEM, '-g', '/tmp/wb.txt', '-d', 'weibull', '-k', '1', '-o', '/dev/null'],
                   capture_output=True, text=True, timeout=30)
print("\n=== 1-Weibull fit ===")
for line in r3.stdout.split('\n'):
    if 'LL=' in line or 'Component' in line:
        print(line)

os.unlink('/tmp/wb.txt')
