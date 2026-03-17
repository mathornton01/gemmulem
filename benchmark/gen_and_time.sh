#!/bin/bash
BUILD="/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
for k in 4 6 8; do
  python3 -c "
import numpy as np
rng = np.random.default_rng(42)
k=$k; n=20000
parts = [rng.normal(j*5.0, 1.0, n//k) for j in range(k)]
d = list(np.concatenate(parts))
with open('/tmp/k${k}_test.txt','w') as f:
    for x in d: f.write(str(x)+'\n')
"
  echo -n "k=$k: "
  { time $BUILD/gemmulem -g /tmp/k${k}_test.txt -d Gaussian -k $k -o /dev/null; } 2>&1 | grep -E 'real|Converged'
done
