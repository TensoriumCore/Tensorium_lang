#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
FIXTURE="$ROOT_DIR/tests/fixtures/elliptic/poisson_relax_3d.tn"

LL_PATH="/tmp/tensorium_parallel_residual_grid.ll"
MLIR_PATH="/tmp/tensorium_parallel_residual_grid.mlir"
HOST_HEADER="/tmp/tensorium_parallel_residual_grid_host.h"

if [[ ! -x "$DRIVER" ]]; then
  echo "error: missing driver binary: $DRIVER" >&2
  exit 2
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "error: missing fixture: $FIXTURE" >&2
  exit 2
fi

echo "[parallel-residual-grid] generating MLIR, LLVM IR, and host header"
"$DRIVER" \
  --tensorium-rhs-grid-parallel-lower \
  --tensorium-strip-source-funcs \
  --emit-mlir "$MLIR_PATH" \
  --emit-llvm "$LL_PATH" \
  --emit-host-header "$HOST_HEADER" \
  "$FIXTURE" >/dev/null

if [[ ! -s "$MLIR_PATH" || ! -s "$LL_PATH" || ! -s "$HOST_HEADER" ]]; then
  echo "error: expected MLIR, LLVM IR, and host header outputs" >&2
  exit 2
fi
if ! grep -q "scf.parallel" "$MLIR_PATH"; then
  echo "error: expected scf.parallel in parallel grid MLIR" >&2
  exit 2
fi
if ! grep -q "tensorium_residual_grid_parallel" "$HOST_HEADER"; then
  echo "error: expected residual parallel grid kernel in host header" >&2
  exit 2
fi
if ! grep -q "__kmpc_fork_call" "$LL_PATH"; then
  echo "error: expected OpenMP fork call in parallel LLVM IR" >&2
  exit 2
fi

echo "[parallel-residual-grid] OpenMP lowering smoke OK"
