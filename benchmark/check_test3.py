import numpy as np, subprocess
rng = np.random.default_rng(42)
# Reproduce the exact data from accuracy_bench.py tests 1-2 first (they consume rng state)
_ = np.concatenate([rng.normal(-8,1,999), rng.normal(0,1,999), rng.normal(8,1,1002)])
_ = np.concatenate([rng.normal(-2,1,1500), rng.normal(0,1,2000), rng.normal(2,1,1500)])

# Test 3 data
true_means = [-5, 0, 7]
true_vars = [0.5, 2.0, 0.3]
true_weights = [0.7, 0.2, 0.1]
d = np.concatenate([rng.normal(m, np.sqrt(v), int(w*10000))
                     for m, v, w in zip(true_means, true_vars, true_weights)])

with open('/tmp/test3.txt', 'w') as f:
    for x in d: f.write(str(x) + '\n')

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
r = subprocess.run([BUILD + '/gemmulem', '-g', '/tmp/test3.txt', '-d', 'Gaussian',
                    '-k', '3', '-v', '-o', '/tmp/test3_out.csv'],
    capture_output=True, text=True, timeout=15)
# Show last lines
lines = r.stdout.strip().splitlines()
for l in lines[-10:]:
    print(l)
