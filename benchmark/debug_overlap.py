#!/usr/bin/env python3
"""Diagnose why we lose on overlapping clusters."""
import numpy as np, subprocess, tempfile, os
from sklearn.mixture import GaussianMixture

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
GEM = BUILD + "/gemmulem"

rng = np.random.default_rng(42)
d = np.concatenate([rng.normal(-2, 1, 1500), rng.normal(0, 1, 2000), rng.normal(2, 1, 1500)])
rng.shuffle(d)
true = [(-2, 0.3), (0, 0.4), (2, 0.3)]

# Write data
f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
for x in d: f.write(str(x) + '\n')
f.close()

# Run gemmulem
r = subprocess.run([GEM, '-g', f.name, '-d', 'Gaussian', '-k', '3', '-o', '/tmp/dbg_out.csv'],
    capture_output=True, text=True)
os.unlink(f.name)

print("=== Gemmulem output ===")
for line in r.stderr.split('\n'): 
    if 'iter' in line and ('LL=' in line or 'Converged' in line): print(line)
print(r.stdout[-500:])

gem_params = []
with open('/tmp/dbg_out.csv') as of:
    for line in of:
        if line.startswith('#') or line.strip() == '': continue
        p = line.strip().split(',')
        if len(p) >= 3:
            gem_params.append((float(p[1]), float(p[0])))  # mean, weight

print(f"\nGemmulem found: {sorted(gem_params)}")
print(f"True:           {true}")
print(f"Mean MAE: {np.mean([abs(g[0]-t[0]) for g,t in zip(sorted(gem_params),true)]):.4f}")

# Run sklearn WITH the same seed as data
print("\n=== sklearn (random_state=None — fresh random) ===")
best_ll = -1e30
best_params = None
for trial in range(10):
    gm = GaussianMixture(3, max_iter=300, tol=1e-6, n_init=1, random_state=None)
    gm.fit(d.reshape(-1,1))
    ll = gm.score(d.reshape(-1,1)) * len(d)
    if ll > best_ll:
        best_ll = ll
        best_params = sorted(zip(gm.means_[:,0], gm.weights_))
    print(f"  Trial {trial}: LL={ll:.2f} means={sorted(gm.means_[:,0])}")

print(f"\nBest sklearn (10 trials): {best_params}")
print(f"Mean MAE: {np.mean([abs(g[0]-t[0]) for g,t in zip(best_params,true)]):.4f}")

# Gemmulem with verbose to see convergence
print("\n=== Gemmulem with verbose convergence ===")
f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp')
for x in d: f2.write(str(x) + '\n')
f2.close()
r2 = subprocess.run([GEM, '-g', f2.name, '-d', 'Gaussian', '-k', '3', '-o', '/tmp/dbg2_out.csv', '-v'],
    capture_output=True, text=True)
os.unlink(f2.name)
# Show first and last 5 iter lines
lines = [l for l in r2.stderr.split('\n') if 'iter' in l]
for l in lines[:5]: print(l)
print('...')
for l in lines[-5:]: print(l)
print(r2.stdout)
