#!/usr/bin/env bash
set -euo pipefail

BIN=./build/tools/driver/Tensorium_cc
OUT=/tmp/tensorium_bench
mkdir -p "$OUT"

FIXTURES=(
  tests/fixtures/gr/schwarzschild_2d.tn
  tests/fixtures/gr/schwarzschild_3d.tn
)

for f in "${FIXTURES[@]}"; do
  base=$(basename "$f" .tn)
  log="$OUT/${base}.log"
  {
    echo "== $f =="
    echo "-- validate --"
    /usr/bin/time -p "$BIN" --validate "$f" >/dev/null
    echo "-- codegen --"
    /usr/bin/time -p "$BIN" --tensorium-einstein-lower --tensorium-index-analyze --tensorium-einstein-analyze-einsum --tensorium-einstein-canonicalize --tensorium-einstein-validate --dump-mlir "$f" >/dev/null
  } 2>&1 | tee "$log"
  echo "wrote $log"
done
