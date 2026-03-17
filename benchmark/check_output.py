import numpy as np, subprocess
rng = np.random.default_rng(42)
d = np.concatenate([rng.normal(-5, 1, 1000), rng.normal(0, 1, 1000), rng.normal(5, 1, 1000)])
with open('/tmp/acc_test.txt', 'w') as f:
    for x in d:
        f.write(str(x) + '\n')

BUILD = "/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
r = subprocess.run([BUILD + '/gemmulem', '-g', '/tmp/acc_test.txt', '-d', 'Gaussian',
                    '-k', '3', '-v', '-o', '/tmp/acc_out.csv'],
    capture_output=True, text=True, timeout=15)
print("STDOUT (last 500 chars):")
print(r.stdout[-500:])
print("\nOUTPUT CSV:")
with open('/tmp/acc_out.csv') as f:
    print(f.read())
