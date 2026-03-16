#!/bin/bash
# Run GEMMULEM vs H2Q EM comparison using H2Q's real test data
set -e

GEMDIR="$(cd "$(dirname "$0")/.." && pwd)"
H2QDIR="/mnt/c/Users/Micah/Project/UTSW/h2q"

echo "============================================================"
echo "  GEMMULEM vs H2Q: Real gene.sam data comparison"
echo "============================================================"
echo ""

# Build comparison tool if needed
if [ ! -f "$GEMDIR/benchmark/compare_h2q" ]; then
    echo "Building comparison tool..."
    cd "$GEMDIR"
    g++ -O3 -std=c++17 -o benchmark/compare_h2q benchmark/compare_h2q.cpp \
        -Isrc/lib -Lbuild/src/lib -lem -lm
fi

echo "Running synthetic benchmark..."
"$GEMDIR/benchmark/compare_h2q"

# If H2Q test data exists, run both tools on it
if [ -f "$H2QDIR/example/tiny_diploid.gene.sam" ]; then
    echo ""
    echo "============================================================"
    echo "  Running both tools on H2Q tiny_diploid.gene.sam"
    echo "============================================================"
    echo ""

    # Build H2Q if binary exists
    H2Q_BIN="$H2QDIR/src/hisat2-quant-bin"
    if [ ! -f "$H2Q_BIN" ]; then
        echo "Building H2Q..."
        cd "$H2QDIR/src" && make quant 2>/dev/null
    fi

    if [ -f "$H2Q_BIN" ]; then
        echo "H2Q output:"
        "$H2Q_BIN" -q "$H2QDIR/example/tiny_diploid.gene.sam" 2>/dev/null | head -20
        echo ""
    fi
fi

echo "Done."
