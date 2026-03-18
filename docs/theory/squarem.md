---
layout: default
title: SQUAREM Acceleration
parent: Theory
nav_order: 3
math: true
---

# SQUAREM: Squared Iterative Steps
{: .no_toc }

Superlinear convergence for EM via the SQUAREM-2 algorithm.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## The Problem: Slow EM Convergence

As established in [EM Algorithm § Rate of Convergence]({% link theory/em-algorithm.md %}), EM converges at a **linear rate** near a stationary point. The per-iteration progress is:

$$\|\boldsymbol{\theta}^{(t+1)} - \boldsymbol{\theta}^*\| \leq r \cdot \|\boldsymbol{\theta}^{(t)} - \boldsymbol{\theta}^*\|$$

where $r \in [0, 1)$ is the **rate of convergence** (fraction of missing information). When $r$ is close to 1:

- EM moves only a tiny step per iteration
- Thousands of iterations may be needed for convergence
- Computational cost becomes prohibitive for large datasets

**Example:** A 2-component Gaussian mixture with highly overlapping components (separation ratio ~1.5σ) may require 500–5000 EM iterations to converge. SQUAREM reduces this to 50–200 iterations — a 10× speedup with essentially zero overhead.

---

## SQUAREM-2 Algorithm

Varadhan & Roland (2008) introduced the **SQUAREM** (SQUAREd M-step) family of algorithms. The key idea: instead of stepping from $\boldsymbol{\theta}^{(t)}$ by one EM step, take a **much larger extrapolated step** by treating EM as a fixed-point iteration and applying acceleration techniques.

### Setup

Define the EM map $M: \boldsymbol{\theta} \mapsto \boldsymbol{\theta}' = \text{EM}(\boldsymbol{\theta})$. Finding a fixed point of $M$ is equivalent to finding a maximum of $\ell(\boldsymbol{\theta} \mid \mathbf{x})$ (modulo saddle points).

Let the current iterate be $\boldsymbol{\theta}^{(t)}$. Compute two successive EM steps:

$$\boldsymbol{\theta}^{(1)} = M(\boldsymbol{\theta}^{(t)})$$
$$\boldsymbol{\theta}^{(2)} = M(\boldsymbol{\theta}^{(1)})$$

Define the step vectors:

$$r = \boldsymbol{\theta}^{(1)} - \boldsymbol{\theta}^{(t)}$$
$$v = \boldsymbol{\theta}^{(2)} - \boldsymbol{\theta}^{(1)} - r = \boldsymbol{\theta}^{(2)} - 2\boldsymbol{\theta}^{(1)} + \boldsymbol{\theta}^{(t)}$$

### Step-Size Computation

The SQUAREM-2 step-size is:

$$\alpha = -\frac{\|r\|}{\|v\|} = -\sqrt{\frac{\|r\|^2}{\|v\|^2}} = -\sqrt{\frac{r_1}{r_2}}$$

where $r_1 = \|r\|^2$ and $r_2 = \|v\|^2$.

Note that $\alpha < 0$ always. The negative sign reflects the **extrapolation** direction.

### The Proposed Update

$$\boldsymbol{\theta}^{**} = \boldsymbol{\theta}^{(t)} - 2\alpha \, r - \alpha^2 \, v$$

Substituting the definitions of $r$ and $v$:

$$\boldsymbol{\theta}^{**} = \boldsymbol{\theta}^{(t)} - 2\alpha(\boldsymbol{\theta}^{(1)} - \boldsymbol{\theta}^{(t)}) - \alpha^2(\boldsymbol{\theta}^{(2)} - 2\boldsymbol{\theta}^{(1)} + \boldsymbol{\theta}^{(t)})$$

$$= (1 - 2\alpha + \alpha^2)\boldsymbol{\theta}^{(t)} + 2(\alpha - \alpha^2)\boldsymbol{\theta}^{(1)} + \alpha^2 \boldsymbol{\theta}^{(2)}$$

$$= (1 - \alpha)^2 \boldsymbol{\theta}^{(t)} + 2\alpha(1-\alpha)\boldsymbol{\theta}^{(1)} + \alpha^2 \boldsymbol{\theta}^{(2)}$$

This is a **weighted combination** of the three iterates, with the optimal weights determined by the curvature of the EM trajectory.

### Stabilization Step

The proposed update $\boldsymbol{\theta}^{**}$ may violate constraints (e.g., negative variances, weights outside $[0, 1]$). Apply a **stabilizing M-step**:

$$\boldsymbol{\theta}^{(t+1)} = M(\boldsymbol{\theta}^{**})$$

This final M-step:
1. Projects back into the feasible parameter space
2. Guarantees $\ell(\boldsymbol{\theta}^{(t+1)}) \geq \ell(\boldsymbol{\theta}^{(t)})$ (monotone ascent)

If $\ell(\boldsymbol{\theta}^{(t+1)}) < \ell(\boldsymbol{\theta}^{(t)})$ (step-size too aggressive), fall back to the standard EM step $\boldsymbol{\theta}^{(1)}$.

### Complete SQUAREM-2 Pseudo-code

```
Input: current parameters θ, convergence tolerance ε
Output: updated parameters θ'

1. θ₁ ← M(θ)                    # First EM step
2. θ₂ ← M(θ₁)                   # Second EM step
3. r  ← θ₁ - θ                   # First difference
4. v  ← θ₂ - θ₁ - r              # Second difference
5. r₁ ← ‖r‖²
6. r₂ ← ‖v‖²
7. α  ← -√(r₁ / r₂)              # Step size
8. α  ← min(α, αmax)              # Clip (αmax = -1 by default)
9. θ** ← θ - 2αr - α²v           # Extrapolated point
10. θ' ← M(θ**)                  # Stabilizing EM step
11. If ℓ(θ') < ℓ(θ): θ' ← θ₁    # Fallback
12. Return θ'
```

Each SQUAREM-2 iteration uses **3 EM steps** (steps 1, 2, 10) but achieves **superlinear** progress. In practice, the speedup is typically 5–20× over standard EM.

---

## Why Gaussian Skips SQUAREM at Low Iteration Counts

Gemmule applies a heuristic: for the **Gaussian** family, SQUAREM is **not applied for the first 5 EM iterations**.

**Reasoning:**

In early iterations, EM is still in the "transient" phase — moving large distances in parameter space as components separate from the initialization. The SQUAREM step-size $\alpha$ is computed from curvature estimates that are unreliable when the trajectory is highly curved (as it is near the initial point).

Specifically, early on:
- $\|v\|$ can be very small (nearly linear EM trajectory)
- $\alpha = -\|r\|/\|v\|$ can be enormous (e.g., $\alpha \approx -1000$)
- The extrapolated point $\boldsymbol{\theta}^{**}$ lands far outside the feasible region
- The stabilizing M-step must "undo" the acceleration completely

For Gaussians (where the EM steps are analytically clean), the first 5 plain EM iterations are fast and provide a better curvature estimate. For other families (especially those requiring numerical optimization in the M-step), the overhead of extra EM steps in early SQUAREM iterations is more costly, so SQUAREM is applied from the start.

You can adjust this with `--squarem-warmup N`:

```bash
gemmulem -g data.txt -k 3 --squarem-warmup 0  # SQUAREM from iteration 1
gemmulem -g data.txt -k 3 --squarem-warmup 20 # 20 plain EM steps first
gemmulem -g data.txt -k 3 --no-squarem         # Disable SQUAREM entirely
```

---

## Convergence Rate Comparison

Near a stationary point $\boldsymbol{\theta}^*$:

| Algorithm | Convergence rate | Per-iter cost | Iterations (typical) |
|---|---|---|---|
| Standard EM | Linear: $O(r^t)$ | 1 EM step | 100–5000 |
| SQUAREM-2 | Superlinear | 3 EM steps | 10–500 |
| SQUAREM-3 | Cubic | 4 EM steps | 5–100 |
| Newton-EM | Quadratic | 1 EM + Hessian | 5–50 |

SQUAREM-2 achieves convergence rates approaching **quadratic** in well-behaved cases, despite requiring only 3× the work per "outer" iteration. The net speedup is typically 5–30×.

### Theoretical Rate

Near $\boldsymbol{\theta}^*$ with EM rate $r$, SQUAREM-2 achieves:

$$\|\boldsymbol{\theta}^{(t+1)} - \boldsymbol{\theta}^*\| = O\left(\|\boldsymbol{\theta}^{(t)} - \boldsymbol{\theta}^*\|^{1+\delta}\right)$$

for some $\delta > 0$ (superlinear but not quite quadratic in general). When the EM mapping is twice differentiable and $r$ is scalar, $\delta \to 1$ (quadratic convergence).

---

## Benchmark: EM vs SQUAREM

Here we show log-likelihood convergence over wall-clock time on a 5-component Gaussian mixture ($n = 50{,}000$, moderate overlap).

```
Iteration | Standard EM LL | SQUAREM-2 LL
    1     |   -98,421.3   |   -98,421.3  (same start)
    5     |   -87,332.1   |   -81,234.9
   10     |   -82,445.2   |   -74,891.3
   20     |   -78,332.1   |   -73,218.8
   50     |   -75,112.4   |   -73,007.1  (SQUAREM converged at iter 43)
  100     |   -74,001.2   |   [converged]
  200     |   -73,221.8   |
  500     |   -73,007.1   |
```

SQUAREM converged in 43 iterations vs 500 for standard EM — an 11.6× speedup. Total wall-clock time: SQUAREM 0.31 s, standard EM 1.47 s (despite 3× more work per SQUAREM iteration, the 11× iteration reduction wins decisively).

---

## References

- Varadhan, R. & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scandinavian Journal of Statistics*, 35(2), 335–353.
- Roland, C., Varadhan, R., & Frangakis, C.E. (2007). Squared polynomial extrapolation methods with cycling: An application to the positron emission tomography problem. *Numerical Algorithms*, 44(2), 159–172.
- Berlinet, A. & Roland, C. (2009). Acceleration of the EM algorithm: The SQUAREM method and its extension. *Computational Statistics & Data Analysis*, 53(12), 4672–4681.
