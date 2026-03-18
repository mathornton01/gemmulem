# Changelog

## v2.0.0 (2026-03-17)

### New Features
- **35 distribution families** — Added Pearson (auto-classified I-VII), Student-t, Cauchy, Laplace, InvGaussian, Rayleigh, Pareto, Logistic, Gumbel, SkewNormal, GenGaussian, ChiSquared, F, LogLogistic, Nakagami, Levy, Gompertz, Burr, HalfNormal, Maxwell, Kumaraswamy, Triangular, Binomial, NegBinomial, Geometric, Zipf, KDE
- **SIMD-accelerated E-step** — AVX2 with 128-row cache tiling, SSE2/scalar fallback
- **GPU OpenCL E-step** — Auto-detects AMD/NVIDIA, activates for Gaussian n≥50K
- **Streaming EM** — File-based chunked processing for datasets too large for memory (CLI: `--stream --chunk-size N --passes M`)
- **Multivariate Gaussian EM** — Full, diagonal, spherical covariance with Cholesky factorization
- **Multivariate Student-t** — u-weights with Newton fixed-point for degrees of freedom
- **Multivariate auto-k** — BIC-driven k=1..k_max with early stopping
- **5 model selection criteria** — BIC, AIC, ICL, VBEM, MML
- **Spectral initialization** — Gap-detection on sorted data
- **Online EM** — Cappé & Moulines (2009) stochastic EM with mini-batches
- **SQUAREM acceleration** — Varadhan & Roland (2008) for slow-converging families

### Performance
- **xorshift128+ PRNG** — Replaces LCG for k-means++ D²-sampling (passes BigCrush)
- **Full-data k-means++ with 5 restarts** — Matches sklearn's initialization quality
- **K-means cluster-fraction weight init** — Initial mixing weights from cluster sizes
- **SQUAREM for Gaussian at iter≥30** — Accelerates convergence on overlapping clusters
- **1.5–6.6× faster than sklearn** on all 8 benchmark scenarios (50 seeds, 95% CIs)

### Bug Fixes
- Fixed buffer overflow for k > 64 (dynamic allocation replaces stack arrays)
- Fixed memory leak in EM polish mode (conditional allocation)
- Added NULL checks for all malloc/calloc calls with cascading cleanup
- Fixed streaming.c file pointer leak on early return
- Removed dead code (unused sorted array allocation)

### Developer Experience
- Comprehensive CLI help text with all flags documented
- Detailed code comments for key algorithms
- Updated README with benchmark results and Quick Start guide
- CI: 6 test suites + smoke tests on Ubuntu 22.04 and macOS

## v1.1.0 (2024)

- Initial release with Gaussian, Exponential, Poisson, Gamma, LogNormal, Weibull, Beta, Uniform mixture EM
- Basic CLI interface
- R and Python extensions (basic wrappers)
