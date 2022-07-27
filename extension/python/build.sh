#!/bin/bash

PYGEM_DIR=$(readlink -f .)
GEMMULEM_DIR=$(readlink -f ../../)

mkdir -p ${PYGEM_DIR}/mk-build
cd ${PYGEM_DIR}/mk-build
cmake ${GEMMULEM_DIR}
cmake --build . --config Release
cmake --install . --config Release --prefix ${PYGEM_DIR}/mk-build

cd ${PYGEM_DIR}
CPPFLAGS="-I ${PYGEM_DIR}/mk-build/include" LDFLAGS="-L ${PYGEM_DIR}/mk-build/lib" python -m build
pip install dist/*.whl
