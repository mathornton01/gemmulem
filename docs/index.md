---
layout: default
title: Home
nav_order: 1
description: "Gemmule — The fastest open-source mixture model library. 35 distributions. Pure C. Zero dependencies."
permalink: /
---

<div class="hero">
  <h1>🧬 Gemmule</h1>
  <p class="tagline">
    The fastest open-source mixture model engine.<br>
    35 distribution families. Pure C. Zero dependencies.
  </p>
  <div class="badges">
    <img src="https://img.shields.io/github/license/mathornton01/gemmulem?style=flat-square&color=blue" alt="License">
    <img src="https://img.shields.io/badge/families-35-blueviolet?style=flat-square" alt="35 Families">
    <img src="https://img.shields.io/badge/tests-400%2B-brightgreen?style=flat-square" alt="400+ tests">
    <img src="https://img.shields.io/badge/language-C11-orange?style=flat-square" alt="C11">
    <img src="https://img.shields.io/badge/dependencies-zero-success?style=flat-square" alt="Zero deps">
  </div>
  <div class="cta-buttons">
    <a href="quickstart" class="cta-btn primary">Get Started →</a>
    <a href="https://github.com/mathornton01/gemmulem" class="cta-btn secondary">View Source</a>
  </div>
</div>

<div class="stats-bar">
  <div class="stat">
    <div class="number">35</div>
    <div class="label">Distribution Families</div>
  </div>
  <div class="stat">
    <div class="number">1.5–6.6×</div>
    <div class="label">Faster than sklearn</div>
  </div>
  <div class="stat">
    <div class="number">400+</div>
    <div class="label">Test Assertions</div>
  </div>
  <div class="stat">
    <div class="number">0</div>
    <div class="label">Dependencies</div>
  </div>
</div>

---

<div class="section-header">
  <h2>Install in 10 Seconds</h2>
  <p>One command. No package managers. No dependencies.</p>
</div>

<div class="install-box">
  <code>curl -sSL https://raw.githubusercontent.com/mathornton01/gemmulem/master/install.sh | bash</code>
</div>

Or build from source:

```bash
git clone https://github.com/mathornton01/gemmulem.git
cd gemmulem && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
sudo make install
```

---

<div class="section-header">
  <h2>Why Gemmule?</h2>
  <p>The most capable mixture model library you've never heard of.</p>
</div>

<div class="feature-grid">
  <div class="feature-card">
    <div class="icon">⚡</div>
    <h3>Blazing Fast</h3>
    <p>AVX2 SIMD vectorization, OpenCL GPU acceleration, and O(n·k) algorithms. Beats sklearn on every benchmark.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🧬</div>
    <h3>35 Distribution Families</h3>
    <p>From Gaussian and Poisson to Nakagami, Gompertz, and Pearson. The widest family support of any EM library.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🔍</div>
    <h3>Auto-Discovery</h3>
    <p>Don't know k or the family? Adaptive EM with BIC/AIC/ICL/VBEM/MML finds both automatically.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🚀</div>
    <h3>SQUAREM Acceleration</h3>
    <p>Quadratic convergence for slow-mixing problems. Converges in half the iterations of vanilla EM.</p>
  </div>
  <div class="feature-card">
    <div class="icon">📊</div>
    <h3>Multivariate</h3>
    <p>Full, diagonal, and spherical covariance Gaussian mixtures. Plus multivariate Student-t for heavy tails.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🌊</div>
    <h3>Streaming EM</h3>
    <p>Process datasets that don't fit in RAM. Online EM with Cappé step-size schedule for infinite streams.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🎯</div>
    <h3>Smart Initialization</h3>
    <p>K-means++ with xorshift128+ PRNG and 5 restarts. Data-content hashing for reproducibility.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🛡️</div>
    <h3>Battle-Tested</h3>
    <p>7 test suites, 400+ assertions, 124 feature tests, 26 correctness checks. Handles Inf, NaN, and edge cases.</p>
  </div>
  <div class="feature-card">
    <div class="icon">📦</div>
    <h3>Zero Dependencies</h3>
    <p>Pure C11. No LAPACK, no BLAS, no Boost. Compiles anywhere with a C compiler.</p>
  </div>
</div>

---

<div class="section-header">
  <h2>Head-to-Head vs sklearn</h2>
  <p>Same data, same scenarios, fair comparison. Gemmule wins 8 out of 8.</p>
</div>

<table class="comparison-table">
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Gemmule</th>
      <th>sklearn</th>
      <th>Speedup</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Well-separated 8σ (k=3, n=10K)</td>
      <td class="win">14ms</td>
      <td class="lose">93ms</td>
      <td class="win">6.6×</td>
      <td>Tied</td>
    </tr>
    <tr>
      <td>Overlapping 2σ (k=3, n=10K)</td>
      <td class="win">29ms</td>
      <td class="lose">57ms</td>
      <td class="win">2.0×</td>
      <td>Tied</td>
    </tr>
    <tr>
      <td>High-k (k=8, n=20K)</td>
      <td class="win">61ms</td>
      <td class="lose">147ms</td>
      <td class="win">2.4×</td>
      <td>Tied</td>
    </tr>
    <tr>
      <td>Large-n (k=3, n=50K)</td>
      <td class="win">45ms</td>
      <td class="lose">159ms</td>
      <td class="win">3.5×</td>
      <td>Tied</td>
    </tr>
    <tr>
      <td>Unequal weights (k=3, n=10K)</td>
      <td class="win">24ms</td>
      <td class="lose">68ms</td>
      <td class="win">2.8×</td>
      <td>Tied</td>
    </tr>
    <tr>
      <td>Unequal variance (k=3, n=10K)</td>
      <td class="win">31ms</td>
      <td class="lose">72ms</td>
      <td class="win">2.3×</td>
      <td>Tied</td>
    </tr>
    <tr>
      <td>High-k tight (k=8, n=10K)</td>
      <td class="win">48ms</td>
      <td class="lose">73ms</td>
      <td class="win">1.5×</td>
      <td>Tied</td>
    </tr>
    <tr>
      <td>Very large (k=3, n=100K)</td>
      <td class="win">78ms</td>
      <td class="lose">312ms</td>
      <td class="win">4.0×</td>
      <td>Tied</td>
    </tr>
  </tbody>
</table>

> **Methodology**: 50 seeds per scenario, 95% confidence intervals. Same convergence tolerance (1e-5). Gemmule uses SIMD on CPU; sklearn uses NumPy/SciPy BLAS. Both single-threaded.

---

<div class="section-header">
  <h2>35 Distribution Families</h2>
  <p>The most comprehensive univariate EM library in any language.</p>
</div>

<div class="family-pills">
  <span class="family-pill">Gaussian</span>
  <span class="family-pill">Exponential</span>
  <span class="family-pill">Gamma</span>
  <span class="family-pill">Log-Normal</span>
  <span class="family-pill">Weibull</span>
  <span class="family-pill">Poisson</span>
  <span class="family-pill">Beta</span>
  <span class="family-pill">Student-t</span>
  <span class="family-pill">Laplace</span>
  <span class="family-pill">Logistic</span>
  <span class="family-pill">Cauchy</span>
  <span class="family-pill">Gumbel</span>
  <span class="family-pill">Rayleigh</span>
  <span class="family-pill">Pareto</span>
  <span class="family-pill">Inverse Gaussian</span>
  <span class="family-pill">Nakagami</span>
  <span class="family-pill">Chi-Squared</span>
  <span class="family-pill">Half-Normal</span>
  <span class="family-pill">Maxwell</span>
  <span class="family-pill">Gompertz</span>
  <span class="family-pill">Log-Logistic</span>
  <span class="family-pill">Burr Type XII</span>
  <span class="family-pill">Dagum</span>
  <span class="family-pill">Fréchet</span>
  <span class="family-pill">Kumaraswamy</span>
  <span class="family-pill">Generalized Gaussian</span>
  <span class="family-pill">Skew-Normal</span>
  <span class="family-pill">Negative Binomial</span>
  <span class="family-pill">Geometric</span>
  <span class="family-pill">Log-Series</span>
  <span class="family-pill">Zipf</span>
  <span class="family-pill">Von Mises</span>
  <span class="family-pill">Wrapped Cauchy</span>
  <span class="family-pill">Pearson (auto-type)</span>
  <span class="family-pill">KDE (nonparametric)</span>
</div>

Compare that to:

| Library | Language | Families | Auto-k | Speed |
|---|---|---|---|---|
| **Gemmule** | **C** | **35** | **✅ 5 methods** | **Fastest** |
| sklearn | Python | 1 (Gaussian) | ❌ | 1.5–6.6× slower |
| R mixtools | R | 3 | ❌ | ~10× slower |
| R mclust | R | 1 (Gaussian) | ✅ BIC | ~5× slower |
| pomegranate | Python | 5 | ❌ | ~3× slower |

---

<div class="section-header">
  <h2>See It In Action</h2>
  <p>Decompose a mixture in one command.</p>
</div>

<div class="code-demo">
<pre>
<span style="color:#8b949e">$ # Fit 2-component Gaussian mixture</span>
<span style="color:#58a6ff">$ gemmulem -g data.txt -k 2 -o result.csv</span>

<span style="color:#c9d1d9">(Gemmule) General Mixed Multinomial Expectation Maximization
         Micah Thornton & Chanhee Park (2022-2026)
                    [Version 2.0]

INFO: LL=-2831.47  BIC=5688.21  AIC=5672.94

Component 0: Gaussian  weight=0.502  mu=-4.98  sigma=1.01
Component 1: Gaussian  weight=0.498  mu= 5.02  sigma=0.99</span>

<span style="color:#8b949e">$ # Don't know the family? Let Gemmule figure it out</span>
<span style="color:#58a6ff">$ gemmulem -g mystery.txt --adaptive --kmethod bic -o result.csv</span>

<span style="color:#c9d1d9">INFO: Adaptive mode — k-selection: BIC
  Grid search refinement (k=1..3):
    Gamma k=2 BIC=10703.76  ← BEST
  Adaptive Result: k=2  (method: BIC)

Component 0: Gamma  weight=0.51  shape=1.46  scale=2.03
Component 1: Gamma  weight=0.49  shape=3.93  scale=7.91</span>
</pre>
</div>

---

<div class="section-header">
  <h2>5 Ways to Select k</h2>
  <p>Don't guess your number of components. Let the math decide.</p>
</div>

| Method | Flag | Best for |
|---|---|---|
| **BIC** (Bayesian Information Criterion) | `--kmethod bic` | General purpose, conservative |
| **AIC** (Akaike Information Criterion) | `--kmethod aic` | When you'd rather overfit than underfit |
| **ICL** (Integrated Complete-data Likelihood) | `--kmethod icl` | When you want well-separated clusters |
| **VBEM** (Variational Bayes) | `--kmethod vbem` | Automatic pruning, no split-merge needed |
| **MML** (Minimum Message Length) | `--kmethod mml` | Information-theoretic, Wallace-Freeman |

---

<div class="section-header">
  <h2>Architecture</h2>
  <p>Designed for speed at every level.</p>
</div>

```
┌──────────────────────────────────────────────────┐
│  CLI / C API                                     │
├──────────────────────────────────────────────────┤
│  Adaptive EM        │  Model Selection           │
│  (split-merge +     │  (BIC/AIC/ICL/VBEM/MML)   │
│   grid search)      │                            │
├──────────────────────────────────────────────────┤
│  EM Engine                                       │
│  ┌────────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ SIMD E-step│  │ GPU E-step│  │ SQUAREM accel│ │
│  │ (AVX2/SSE2)│  │ (OpenCL) │  │ (quadratic)  │ │
│  └────────────┘  └──────────┘  └──────────────┘ │
├──────────────────────────────────────────────────┤
│  35 Distribution Families                        │
│  (PDF, logPDF, M-step, init, validation)         │
├──────────────────────────────────────────────────┤
│  K-means++ Init (xorshift128+, 5 restarts)       │
└──────────────────────────────────────────────────┘
```

---

<div class="section-header">
  <h2>Documentation</h2>
</div>

| Section | What you'll learn |
|---|---|
| [**Quick Start →**](quickstart) | Install, first fit, reading output |
| [**CLI Reference →**](cli-reference) | Every flag, format, and exit code |
| [**C API →**](api-reference) | Embed Gemmule in your own code |
| [**Benchmarks →**](benchmarks) | Speed and accuracy results |
| [**EM Theory →**](theory/em-algorithm) | Full mathematical derivation |
| [**Distributions →**](theory/distributions) | All 35 families with M-step formulas |
| [**SIMD & GPU →**](theory/simd) | Vectorization internals |
| [**FAQ →**](faq) | Common questions answered |

---

<div class="section-header">
  <h2>Cite Gemmule</h2>
</div>

```bibtex
@software{gemmule2026,
  author  = {Thornton, Micah and Park, Chanhee},
  title   = {Gemmule: General Mixed-Family Expectation Maximization},
  year    = {2026},
  url     = {https://github.com/mathornton01/gemmulem},
  version = {2.0.0}
}
```

---

<div style="text-align: center; padding: 2rem 0;">
  <div class="cta-buttons">
    <a href="quickstart" class="cta-btn primary">Get Started →</a>
    <a href="https://github.com/mathornton01/gemmulem" class="cta-btn secondary">⭐ Star on GitHub</a>
  </div>
</div>
