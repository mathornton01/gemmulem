# Packaging Guide — CRAN & PyPI

## R Package (CRAN)

The R extension exists at `extension/R/rgemmulem/` but needs updating for v2.0.

### Building the R package

```bash
cd extension/R
R CMD build rgemmulem
R CMD check rgemmulem_*.tar.gz
```

### Updates needed for v2.0

1. **Add new distributions** — Update `NAMESPACE` and R wrappers for:
   - All 35 distribution families
   - Multivariate Gaussian (`rmvgauss_unmix`)
   - Adaptive EM (`radaptive_unmix`)
   - Online EM (`ronline_unmix`)
   - Spectral init (`rspectral_init`)

2. **Update DESCRIPTION**
   ```
   Package: rgemmulem
   Version: 2.0.0
   Title: General Mixed-Family Expectation Maximization
   Description: Univariate and multivariate mixture model EM with 35 distribution
       families, automatic model selection (BIC/AIC/ICL/VBEM/MML), spectral
       initialization, online EM, and adaptive per-component family selection.
   Authors@R: c(
       person("Micah", "Thornton", email="Micah.Thornton@UTSouthwestern.edu", role=c("aut","cre")),
       person("Chanhee", "Park", email="Chanhee.Park@UTSouthwestern.edu", role="aut"))
   License: GPL-3
   Depends: R (>= 3.5.0)
   ```

3. **Add documentation** (`man/` directory)
   - `unmix_adaptive.Rd` for adaptive EM
   - `unmix_online.Rd` for stochastic EM
   - `unmix_mvgaussian.Rd` for multivariate
   - Update existing `.Rd` files with new parameters

4. **Add tests** (`tests/`)
   - `testthat/test-adaptive.R`
   - `testthat/test-multivariate.R`
   - `testthat/test-families.R` (test all 35 families)

5. **Submit to CRAN**
   ```bash
   R CMD build rgemmulem
   R CMD check --as-cran rgemmulem_2.0.0.tar.gz
   # Fix any warnings/notes
   # Submit via https://cran.r-project.org/submit.html
   ```

---

## Python Package (PyPI)

The Python extension exists at `extension/python/` but needs major updates.

### Building the Python package

Current build uses `setuptools`:
```bash
cd extension/python
pip install build
python -m build
```

### Modern approach: scikit-build-core + nanobind

For better C integration and performance, migrate to `nanobind` (modern pybind11 alternative):

**`pyproject.toml`:**
```toml
[build-system]
requires = ["scikit-build-core>=0.3.3", "nanobind>=1.3.2"]
build-backend = "scikit_build_core.build"

[project]
name = "gemmulem"
version = "2.0.0"
description = "General Mixed-Family Expectation Maximization"
readme = "README.md"
license = {text = "GPL-3.0"}
authors = [
    {name = "Micah Thornton", email = "Micah.Thornton@UTSouthwestern.edu"},
    {name = "Chanhee Park", email = "Chanhee.Park@UTSouthwestern.edu"}
]
requires-python = ">=3.8"
dependencies = ["numpy>=1.19.0"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python :: 3",
    "Programming Language :: C",
    "Topic :: Scientific/Engineering :: Mathematics"
]

[project.urls]
Homepage = "https://github.com/mathornton01/gemmulem"
Documentation = "https://gemmulem.readthedocs.io"
Repository = "https://github.com/mathornton01/gemmulem"

[tool.scikit-build]
wheel.expand-macos-universal-tags = true

[tool.cibuildwheel]
build = ["cp38-*", "cp39-*", "cp310-*", "cp311-*", "cp312-*"]
skip = ["*-musllinux_*"]
```

**CMakeLists.txt for Python extension:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(pygemmulem)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    _gemmulem
    NB_STATIC
    python/bindings.cpp
    src/lib/EM.c
    src/lib/distributions.c
    src/lib/pearson.c
    src/lib/multivariate.c
)

target_include_directories(_gemmulem PRIVATE src/lib)
install(TARGETS _gemmulem LIBRARY DESTINATION gemmulem)
```

**Python bindings** (`python/bindings.cpp`):
```cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

extern "C" {
#include "distributions.h"
#include "multivariate.h"
}

namespace nb = nanobind;

NB_MODULE(_gemmulem, m) {
    m.def("unmix_generic", [](nb::ndarray<double, nb::shape<-1>> data,
                               int family, int k, int maxiter, double rtole) {
        MixtureResult result;
        UnmixGeneric(data.data(), data.size(), (DistFamily)family, k,
                     maxiter, rtole, 0, &result);
        
        auto weights = nb::ndarray<nb::numpy, double>({result.num_components});
        auto means = nb::ndarray<nb::numpy, double>({result.num_components});
        
        for (int j = 0; j < result.num_components; j++) {
            weights.data()[j] = result.mixing_weights[j];
            means.data()[j] = result.params[j].p[0];
        }
        
        ReleaseMixtureResult(&result);
        return std::make_tuple(weights, means);
    });

    m.def("unmix_adaptive", [](nb::ndarray<double, nb::shape<-1>> data,
                                int kmax, int kmethod, int maxiter, double rtole) {
        AdaptiveResult result;
        UnmixAdaptiveEx(data.data(), data.size(), kmax, maxiter, rtole, 0,
                        (KMethod)kmethod, &result);
        
        nb::dict out;
        out["k"] = result.num_components;
        out["loglikelihood"] = result.loglikelihood;
        out["bic"] = result.bic;
        
        ReleaseAdaptiveResult(&result);
        return out;
    });

    m.def("unmix_mvgaussian", [](nb::ndarray<double, nb::shape<-1, -1>> data,
                                  int k, int cov_type, int maxiter, double rtole) {
        size_t n = data.shape(0), d = data.shape(1);
        MVMixtureResult result;
        
        UnmixMVGaussian(data.data(), n, d, k, (CovType)cov_type,
                        maxiter, rtole, 0, &result);
        
        nb::dict out;
        out["k"] = result.num_components;
        out["loglikelihood"] = result.loglikelihood;
        out["bic"] = result.bic;
        
        ReleaseMVMixtureResult(&result);
        return out;
    });
}
```

### Wheels & CI

Use **cibuildwheel** for multi-platform wheels:

**`.github/workflows/wheels.yml`:**
```yaml
name: Build wheels

on: [push, pull_request]

jobs:
  build_wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
      - uses: actions/checkout@v3
      
      - name: Build wheels
        uses: pypa/cibuildwheel@v2.16.5
        
      - uses: actions/upload-artifact@v3
        with:
          path: ./wheelhouse/*.whl
```

### Publishing to PyPI

```bash
# Test PyPI first
python -m build
twine upload --repository testpypi dist/*

# Production
twine upload dist/*
```

---

## Distribution Checklist

### Pre-release
- [ ] All 6 test suites passing (unit_tests, distribution_tests, pearson_tests, adaptive_tests, spectral_online_mml_tests, multivariate_tests)
- [ ] Benchmark results documented
- [ ] Paper figures generated
- [ ] README.md updated
- [ ] CHANGELOG.md written
- [ ] Version bumped to 2.0.0
- [ ] Git tag created: `v2.0.0`

### R Package
- [ ] Update DESCRIPTION, NAMESPACE
- [ ] Add man pages for new functions
- [ ] Run `R CMD check --as-cran` (0 errors, 0 warnings)
- [ ] Submit to CRAN

### Python Package  
- [ ] Migrate to scikit-build-core + nanobind
- [ ] Add type stubs (`.pyi` files)
- [ ] Documentation on ReadTheDocs
- [ ] Publish wheels for Linux/Windows/macOS (Python 3.8-3.12)
- [ ] Submit to PyPI

### Publication
- [ ] arXiv preprint
- [ ] Submit to JOSS (Journal of Open Source Software)
- [ ] Zenodo DOI

---

## Conda-forge (optional)

After PyPI publication, submit a recipe to conda-forge:

**`meta.yaml`:**
```yaml
package:
  name: gemmulem
  version: 2.0.0

source:
  url: https://pypi.io/packages/source/g/gemmulem/gemmulem-2.0.0.tar.gz
  sha256: <hash>

build:
  number: 0
  script: {{ PYTHON }} -m pip install . -vv

requirements:
  build:
    - {{ compiler('c') }}
    - {{ compiler('cxx') }}
    - cmake
  host:
    - python
    - pip
    - scikit-build-core
    - nanobind
  run:
    - python
    - numpy

test:
  imports:
    - gemmulem
  commands:
    - pytest tests/

about:
  home: https://github.com/mathornton01/gemmulem
  license: GPL-3.0
  summary: General Mixed-Family Expectation Maximization
```

---

## Current Status

✅ C library complete (v2.0)  
✅ CLI tool complete  
⚠️ R extension needs v2.0 updates  
⚠️ Python extension needs rewrite (nanobind)  
❌ Conda-forge recipe not submitted  
❌ CRAN submission pending  
❌ PyPI publication pending
