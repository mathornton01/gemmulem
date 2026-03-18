---
layout: default
title: Theory
nav_order: 3
has_children: true
---

# Theory
{: .no_toc }

Mathematical foundations of Gemmulem's algorithms.
{: .fs-6 .fw-300 }

This section covers the theory behind Gemmulem in sufficient depth that a graduate student could implement the algorithms from scratch. Topics are organized from foundational (EM algorithm) to advanced (SIMD, streaming).

| Page | Summary |
|---|---|
| [EM Algorithm]({% link theory/em-algorithm.md %}) | Complete derivation of EM, convergence proof, numerical stability |
| [Initialization]({% link theory/initialization.md %}) | K-means++, xorshift128+, multiple restarts |
| [SQUAREM]({% link theory/squarem.md %}) | Squared iterative steps for superlinear convergence |
| [Model Selection]({% link theory/model-selection.md %}) | BIC, AIC, ICL, VBEM, MML criteria |
| [Distributions]({% link theory/distributions.md %}) | All 35 families: PDF, M-step, use cases |
| [Multivariate]({% link theory/multivariate.md %}) | Multivariate Gaussian and Student-t mixtures |
| [SIMD & GPU]({% link theory/simd.md %}) | AVX2 vectorization and OpenCL acceleration |
| [Streaming EM]({% link theory/streaming.md %}) | Online EM for large datasets |
