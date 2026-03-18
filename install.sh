#!/bin/bash
# Gemmulem one-line installer
# Usage: curl -sSL https://raw.githubusercontent.com/mathornton01/gemmulem/master/install.sh | bash
#    or: wget -qO- https://raw.githubusercontent.com/mathornton01/gemmulem/master/install.sh | bash

set -e

PREFIX="${PREFIX:-/usr/local}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}"

echo "╔══════════════════════════════════════════╗"
echo "║  Gemmulem — Mixture Model EM             ║"
echo "║  https://github.com/mathornton01/gemmulem║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check dependencies
for cmd in git cmake make; do
    if ! command -v $cmd &>/dev/null; then
        echo "❌ Missing: $cmd"
        echo "   Install with: sudo apt install $cmd  (or brew install $cmd)"
        exit 1
    fi
done

CC="${CC:-$(command -v gcc || command -v cc)}"
echo "✓ git, cmake, make found"
echo "✓ C compiler: $CC"
echo "✓ Install prefix: $PREFIX"
echo "✓ Parallel jobs: $JOBS"
echo ""

# Clone or update
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "→ Cloning repository..."
git clone --depth 1 https://github.com/mathornton01/gemmulem.git "$TMPDIR/gemmulem" 2>&1 | tail -1

echo "→ Building..."
cd "$TMPDIR/gemmulem"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" 2>&1 | grep -E "^--|enabled|found" || true
cmake --build . -j"$JOBS" 2>&1 | tail -3

echo "→ Running tests..."
if ctest --output-on-failure -q 2>&1 | tail -2 | grep -q "100%"; then
    echo "✓ All tests passed"
else
    echo "⚠ Some tests failed (installing anyway)"
fi

echo "→ Installing to $PREFIX..."
if [ -w "$PREFIX/bin" ] 2>/dev/null; then
    cmake --install . 2>&1 | head -5
else
    sudo cmake --install . 2>&1 | head -5
fi

echo ""
echo "✅ Gemmulem installed!"
echo "   Run:  gemmulem --help"
echo "   Docs: https://mathornton01.github.io/gemmulem/"
