#!/usr/bin/env python3
"""Quick check: what does adaptive mode actually output?"""
import numpy as np, subprocess, os

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"
rng = np.random.default_rng(42)

# Simple 2-component Gaussian
d = np.concatenate([rng.normal(-5, 1, 3000), rng.normal(5, 1, 3000)])
with open('/tmp/ad_test.txt', 'w') as f:
    for x in d: f.write(str(x) + '\n')

print("=== ADAPTIVE BIC kmax=5 (verbose) ===")
r = subprocess.run([GEM, '-g', '/tmp/ad_test.txt', '--adaptive', '--kmethod', 'bic',
                    '--kmax', '5', '-o', '/tmp/ad_out.csv', '-v'],
                   capture_output=True, text=True, timeout=120)
print("STDOUT:")
print(r.stdout)
print("STDERR:")
print(r.stderr)
print("RC:", r.returncode)
print("\n=== OUTPUT CSV ===")
try:
    with open('/tmp/ad_out.csv') as f: print(f.read())
except: print("(no file)")

os.unlink('/tmp/ad_test.txt')
