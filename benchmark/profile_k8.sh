#!/bin/bash
BUILD="/mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build"
cd /tmp

python3 -c "
import numpy as np
rng = np.random.default_rng(42)
k=8; n=20000
parts = [rng.normal(j*5.0, 1.0, n//k) for j in range(k)]
d = list(np.concatenate(parts))
with open('/tmp/k8_test.txt','w') as f:
    for x in d: f.write(str(x)+'\n')
"

echo "=== gprof profile ==="
# Build with profiling
cd /mnt/c/Users/Micah/.openclaw/workspace/projects/gemmulem/build
cmake -DCMAKE_C_FLAGS="-pg -O2 -mavx2" -DCMAKE_CXX_FLAGS="-pg -O2 -mavx2" .. -DCMAKE_BUILD_TYPE=RelWithDebInfo > /dev/null 2>&1
cmake --build . > /dev/null 2>&1

$BUILD/gemmulem -g /tmp/k8_test.txt -d Gaussian -k 8 -o /dev/null > /dev/null 2>&1
gprof $BUILD/gemmulem gmon.out 2>/dev/null | head -40

# Reset to normal build
cmake -DCMAKE_C_FLAGS="-O3 -mavx2" -DCMAKE_CXX_FLAGS="-O3 -mavx2" .. > /dev/null 2>&1
cmake --build . > /dev/null 2>&1
