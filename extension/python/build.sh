#!/bin/bash

PYGEM_DIR=$(readlink -f .)
GEMMULEM_DIR=$(readlink -f ../../)

mkdir -p ${PYGEM_DIR}/mk-build
cd ${PYGEM_DIR}/mk-build
cmake ${GEMMULEM_DIR}
cmake --build . --config Release
cmake --install . --config Release --prefix ${PYGEM_DIR}/mk-build

cd ${PYGEM_DIR}
INCLUDE_DIR="${PYGEM_DIR}/mk-build/include" LIB_DIR="${PYGEM_DIR}/mk-build/lib" python -m build
pip install dist/*.whl
