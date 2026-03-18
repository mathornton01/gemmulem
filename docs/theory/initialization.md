---
layout: default
title: Initialization
parent: Theory
nav_order: 2
math: true
---

# Initialization: K-means++ and Beyond
{: .no_toc }

How Gemmulem starts the EM algorithm — and why it matters enormously.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Why Initialization Matters

The EM algorithm is **guaranteed to converge** — but only to a *local* maximum of the likelihood. The mixture model log-likelihood is generally non-concave and has many local optima. The quality of the final solution depends critically on where EM starts.

### A Motivating Example

Consider fitting a 3-component Gaussian mixture. With **bad initialization** (all components starting at the mean):

- All responsibilities are initially equal: $\gamma_{ij} = 1/k$ for all $i, j$
- The M-step updates all components to approximately the same values
- EM finds the trivial local optimum: $k$ identical components

With **good initialization** (components spread across the data):

- Responsibilities are differentiated from iteration 1
- EM quickly finds distinct, meaningful components

{: .important }
> In practice, initialization quality can change the final log-likelihood by 5–20% and the number of EM iterations needed by 2–10×.

---

## K-means++ Algorithm

Gemmulem uses the **K-means++ initialization** of Arthur & Vassilvitskii (2007), which selects $k$ well-separated initial centers via **D²-weighted sampling**.

### Algorithm

**Input:** data $\mathbf{x} = \{x_1, \ldots, x_n\}$, number of components $k$

1. Choose the first center $c_1$ uniformly at random from $\{x_1, \ldots, x_n\}$

2. For $\ell = 2, 3, \ldots, k$:
   - For each point $x_i$, compute its squared distance to the nearest existing center:
   $$D_i = \min_{j < \ell} \|x_i - c_j\|^2$$
   - Sample the next center $c_\ell = x_i$ with probability proportional to $D_i$:
   $$P(\text{choose } x_i) = \frac{D_i}{\sum_{m=1}^{n} D_m}$$

3. Return centers $\{c_1, \ldots, c_k\}$

**Output:** Initial means; variances set to the average within-cluster variance; weights set to cluster fractions.

### Why D² Sampling Works

The D²-weighted sampling rule encodes a powerful geometric intuition: **far-away points are better initial centers** because they help partition the data space more evenly.

**Theorem (Arthur & Vassilvitskii, 2007):** Let $\text{OPT}$ be the optimal k-means cost. K-means++ produces an initial solution with expected cost at most $O(\log k) \cdot \text{OPT}$.

**Proof sketch:**

Let $C^*$ be the optimal centers. At step $\ell$, suppose we have chosen $\ell - 1$ centers and consider a specific optimal cluster $A^*$ (with center $c^*$) that contains no chosen center yet.

By the D²-sampling rule, the probability that the next chosen center falls inside $A^*$ is at least:

$$P(\text{next center} \in A^*) \geq \frac{\sum_{x \in A^*} D(x)^2}{\sum_{x} D(x)^2}$$

Using geometric series arguments over the $k$ clusters, the total expected cost satisfies:

$$\mathbb{E}[\text{K-means++ cost}] \leq 8(\ln k + 2) \cdot \text{OPT}$$

In practice, the constant is much smaller than the theoretical bound, typically achieving within 2–3× of OPT in one pass.

---

## D²-Weighted Sampling: Implementation

Efficient D²-weighted sampling requires:
1. Computing distances to all current centers: $O(n \cdot \ell)$
2. Sampling from the resulting distribution: $O(n)$ with inverse CDF

### Inverse CDF Sampling

Compute the cumulative distribution:

$$F_i = \sum_{m=1}^{i} D_m \bigg/ \sum_{m=1}^{n} D_m$$

Draw $u \sim \text{Uniform}[0, 1)$ and find the smallest $i$ such that $F_i \geq u$ via binary search in $O(\log n)$.

```c
// Gemmulem D²-sampling kernel
static size_t sample_d2(const double *distances, size_t n, uint64_t *rng_state) {
    double total = 0.0;
    for (size_t i = 0; i < n; i++) total += distances[i];

    double u = xorshift128plus(rng_state) * (1.0 / (double)UINT64_MAX) * total;
    double cumsum = 0.0;
    for (size_t i = 0; i < n; i++) {
        cumsum += distances[i];
        if (cumsum >= u) return i;
    }
    return n - 1;  // numerical edge case
}
```

---

## xorshift128+ PRNG

The quality of D²-weighted sampling depends on the quality of the pseudo-random numbers used. Gemmulem uses the **xorshift128+** generator (Vigna, 2014), which offers:

- **Period**: $2^{128} - 1$
- **Speed**: ~1.4 ns/number on modern CPUs (faster than Mersenne Twister)
- **Statistical quality**: Passes BigCrush and PractRand test suites
- **Simplicity**: Only 3 XOR-shift operations

### Why This Matters for D² Sampling

Poor-quality PRNGs (like `rand()` with period $2^{31}$) can introduce subtle biases:
- Points with very small $D_i^2$ may be selected more often than their weight indicates
- Multiple restarts with the same underlying seed sequence may converge to the same local optima
- For $n > 10^6$, LCG generators show visible correlation patterns in the CDF sampling

xorshift128+ avoids these issues: it has excellent equidistribution properties up to 64 dimensions.

### Implementation

```c
// xorshift128+ state (two 64-bit integers)
typedef struct { uint64_t s[2]; } XorShift128State;

static uint64_t xorshift128plus(XorShift128State *state) {
    uint64_t s1 = state->s[0];
    const uint64_t s0 = state->s[1];
    state->s[0] = s0;
    s1 ^= s1 << 23;                              // a
    state->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);  // b, c
    return state->s[1] + s0;
}

// Seeding from user-provided seed
static void seed_xorshift128(XorShift128State *state, uint64_t seed) {
    // Use splitmix64 to expand the seed (avoids zero-state)
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    state->s[0] = z ^ (z >> 31);
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    state->s[1] = z ^ (z >> 31);
}
```

---

## Cluster-Fraction Weight Initialization

After K-means++ selects initial centers $\{c_1, \ldots, c_k\}$, Gemmulem assigns each observation $x_i$ to its nearest center and uses the resulting cluster sizes to initialize component weights:

$$\pi_j^{(0)} = \frac{|C_j|}{n}$$

where $|C_j|$ is the number of points assigned to cluster $j$.

This is preferable to uniform initialization ($\pi_j = 1/k$) because:
- It reflects any inherent imbalance in the data
- Components immediately represent their likely true proportions
- Fewer EM iterations are needed before weights stabilize

For **variance initialization**, Gemmulem uses the within-cluster variance:

$$\sigma_j^{(0)} = \sqrt{\frac{\sum_{i \in C_j} (x_i - c_j)^2}{|C_j|}}$$

with a minimum floor of $10^{-4}$ times the data range to prevent degenerate starting points.

---

## Multiple Restarts and Inertia-Based Selection

Because K-means++ still has randomness (the first center is chosen uniformly, and subsequent choices depend on the PRNG), running multiple restarts and keeping the best provides additional protection against poor local optima.

### Inertia Metric

For each restart, compute the **within-cluster sum of squares (inertia)**:

$$W = \sum_{j=1}^{k} \sum_{i \in C_j} (x_i - c_j)^2$$

Lower inertia indicates tighter clusters and a better initialization. The restart with lowest inertia is selected to seed EM.

### Restart Schedule

By default, Gemmulem runs `max(1, min(10, k²))` restarts. For $k = 3$, this is 9 restarts. You can override with `--restarts N`.

```bash
gemmulem -g data.txt -k 5 --restarts 50 -o results.csv
```

**Cost:** Each restart requires $O(n \cdot k \cdot k_{\text{init}})$ operations for K-means++ (where $k_{\text{init}} \leq 20$ Lloyd iterations are run). With 50 restarts and $n = 10^5$, $k = 5$, this takes about 0.5 seconds — still much cheaper than running full EM from a bad start.

---

## Comparison: Initialization Strategies

| Strategy | Convergence quality | Cost | Recommended for |
|---|---|---|---|
| Random (uniform) | Poor | $O(k)$ | Never (unless debugging) |
| Random from data | Fair | $O(k)$ | Small, well-separated clusters |
| K-means++ (1 restart) | Good | $O(nk)$ | Default; most cases |
| K-means++ (10 restarts) | Very good | $O(10nk)$ | Moderate k, important fits |
| K-means++ (50 restarts) | Excellent | $O(50nk)$ | High-stakes inference |
| Grid search (1D only) | Excellent | $O(nk^2)$ | Univariate, small n |

---

## References

- Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035.
- Vigna, S. (2014). Further scramblings of Marsaglia's xorshift generators. *arXiv:1404.0390*.
- Celebi, M.E., Kingravi, H.A., & Vela, P.A. (2013). A comparative study of efficient initialization methods for the k-means clustering algorithm. *Expert Systems with Applications*, 40(1), 200–210.
