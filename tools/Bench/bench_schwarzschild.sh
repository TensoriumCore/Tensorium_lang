#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT_DIR/build/tools/driver/Tensorium_cc"
OUT_BASE="$ROOT_DIR/tools/Bench/out"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_BASE/$STAMP"
mkdir -p "$OUT"

FIXTURES=(
  "$ROOT_DIR/tests/fixtures/gr/schwarzschild_2d.tn"
  "$ROOT_DIR/tests/fixtures/gr/schwarzschild_3d.tn"
)

for f in "${FIXTURES[@]}"; do
  base=$(basename "$f" .tn)
  log="$OUT/${base}.log"
  {
    echo "== $f =="
    echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "-- semantic validate --"
    /usr/bin/time -p "$BIN" --validate "$f" >/dev/null
    echo "-- canonical backend IR dump --"
    /usr/bin/time -p "$BIN" --dump-backend-expr "$f" >/dev/null
    echo "-- mlir codegen --"
    /usr/bin/time -p "$BIN" --tensorium-einstein-lower --tensorium-index-analyze --tensorium-einstein-analyze-einsum --tensorium-einstein-canonicalize --tensorium-einstein-validate --dump-mlir "$f" >/dev/null
  } 2>&1 | tee "$log"
  echo "wrote $log"
done

echo "Benchmark logs written in $OUT"
