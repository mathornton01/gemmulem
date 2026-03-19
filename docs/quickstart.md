---
layout: default
title: Quick Start
nav_order: 2
description: "Get Gemmule running in under a minute."
---

# Quick Start
{: .fs-9 }

From zero to mixture decomposition in 60 seconds.
{: .fs-6 .fw-300 }

---

## Install

### Option 1: One-liner (Linux/macOS)

```bash
curl -sSL https://raw.githubusercontent.com/mathornton01/gemmulem/master/install.sh | bash
```

### Option 2: Build from source

```bash
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### Option 3: Simple Makefile

```bash
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem
make          # auto-detects OpenMP, AVX2
sudo make install
```

### Verify

```bash
gemmulem --help
```

---

## Your First Fit

### 1. Create some test data

```bash
python3 -c "
import numpy as np
rng = np.random.default_rng(42)
data = np.concatenate([rng.normal(-5, 1, 3000), rng.normal(5, 1, 3000)])
np.savetxt('test_data.txt', data, fmt='%.6f')
print(f'Generated {len(data)} points')
"
```

### 2. Fit a 2-component Gaussian mixture

```bash
gemmulem -g test_data.txt -k 2 -o result.csv
```

Output:
```
INFO: LL=-8547.31  BIC=17120.06  AIC=17104.62

Component 0: weight=0.500  mu=-5.01  sigma=1.00
Component 1: weight=0.500  mu= 5.01  sigma=0.99
```

### 3. Read the results

```bash
cat result.csv
# Gaussian,0.500,-5.01,1.002
# Gaussian,0.500, 5.01,0.994
```

Each row: `family, weight, param1, param2, ...`

---

## Try Different Families

```bash
# Exponential mixture
gemmulem -g positive_data.txt -d exponential -k 2 -o result.csv

# Gamma mixture  
gemmulem -g positive_data.txt -d gamma -k 3 -o result.csv

# Poisson (integer data)
gemmulem -g count_data.txt -d poisson -k 2 -o result.csv

# Beta (data in [0,1])
gemmulem -g proportion_data.txt -d beta -k 2 -o result.csv
```

All 35 families available via `-d <name>`. See [CLI Reference](cli-reference) for the full list.

---

## Don't Know k or the Family?

Let Gemmule figure it out:

```bash
# Adaptive: discovers both k AND the best family
gemmulem -g mystery_data.txt --adaptive --kmethod bic -o result.csv
```

This will:
1. Start with k=1 and try 9 candidate families
2. Split-merge to find the optimal k
3. Grid-search 16 families at the best k
4. Return the model with lowest BIC

Other k-selection methods:

```bash
--kmethod aic    # Less conservative than BIC
--kmethod icl    # Penalizes fuzzy cluster assignments
--kmethod vbem   # Variational Bayes (auto-prunes components)
--kmethod mml    # Minimum Message Length
```

---

## Multivariate Data

```bash
# 2D Gaussian mixture, k=3, full covariance
gemmulem -m data_2d.csv -k 3 --cov full -o result.csv

# Diagonal covariance (faster)
gemmulem -m data_2d.csv -k 3 --cov diagonal -o result.csv

# Auto-detect k with BIC
gemmulem -m data_2d.csv --mv-auto-k --kmin 1 --kmax 8 -o result.csv
```

---

## Streaming (Large Files)

For files too large to fit in RAM:

```bash
gemmulem -g huge_data.txt -k 5 --stream --chunk-size 10000 -o result.csv
```

---

## Online EM (Mini-batch)

For datasets arriving in a stream:

```bash
gemmulem -g data.txt -k 3 --online --batch-size 500 -o result.csv
```

---

## Use as a C Library

```c
#include "distributions.h"

int main() {
    double data[] = {1.2, 5.3, 1.1, 5.5, 0.9, 5.1, ...};
    size_t n = sizeof(data) / sizeof(data[0]);
    
    MixtureResult result;
    int rc = UnmixGeneric(data, n, DIST_GAUSSIAN, 2, 
                          200,    /* max iterations */
                          1e-5,   /* tolerance */
                          0,      /* verbose */
                          &result);
    
    if (rc == 0) {
        for (int j = 0; j < result.num_components; j++) {
            printf("Component %d: weight=%.3f mu=%.3f sigma=%.3f\n",
                   j, result.mixing_weights[j],
                   result.params[j].p[0],
                   sqrt(result.params[j].p[1]));
        }
        ReleaseMixtureResult(&result);
    }
    return 0;
}
```

Compile:
```bash
gcc my_program.c -lgemmulem -lm -o my_program
```

---

## Next Steps

- [CLI Reference](cli-reference) — every flag and option
- [C API Reference](api-reference) — embed in your code
- [Distributions](theory/distributions) — all 35 families explained
- [Benchmarks](benchmarks) — speed and accuracy data
