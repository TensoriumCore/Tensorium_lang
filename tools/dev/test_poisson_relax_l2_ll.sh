#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER_SRC="$ROOT_DIR/tools/dev/runtime_poisson_relax_l2.cpp"
FIXTURE="$ROOT_DIR/tests/fixtures/elliptic/poisson_relax_3d.tn"

if [[ -n "${CLANG:-}" ]]; then
  CLANG_BIN="$CLANG"
elif [[ -x /opt/llvm-20/bin/clang ]]; then
  CLANG_BIN="/opt/llvm-20/bin/clang"
else
  CLANG_BIN="clang"
fi
if [[ -n "${LLC:-}" ]]; then
  LLC_BIN="$LLC"
elif [[ -x /opt/llvm-20/bin/llc ]]; then
  LLC_BIN="/opt/llvm-20/bin/llc"
else
  LLC_BIN="llc"
fi
source "$ROOT_DIR/tools/dev/openmp_flags.sh"

LL_PATH="/tmp/tensorium_poisson_relax_l2.ll"
OBJ_PATH="/tmp/tensorium_poisson_relax_l2.o"
EXE_PATH="/tmp/tensorium_poisson_relax_l2_runner"
HOST_HEADER="/tmp/tensorium_poisson_relax_l2_host.h"

if [[ ! -x "$DRIVER" ]]; then
  echo "error: missing driver binary: $DRIVER" >&2
  exit 2
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "error: missing fixture: $FIXTURE" >&2
  exit 2
fi
if [[ ! -f "$RUNNER_SRC" ]]; then
  echo "error: missing runner source: $RUNNER_SRC" >&2
  exit 2
fi

echo "[poisson-relax-l2] generating LLVM IR and host header: $LL_PATH"
"$DRIVER" \
  --tensorium-rhs-grid-affine-lower \
  --tensorium-rhs-grid-parallel-lower \
  --tensorium-strip-source-funcs \
  --emit-llvm "$LL_PATH" \
  --emit-host-header "$HOST_HEADER" \
  "$FIXTURE" >/dev/null

if [[ ! -s "$LL_PATH" ]]; then
  echo "error: LLVM IR file is missing or empty: $LL_PATH" >&2
  exit 2
fi
if [[ ! -s "$HOST_HEADER" ]]; then
  echo "error: generated host header is missing or empty: $HOST_HEADER" >&2
  exit 2
fi
if ! grep -q "tensorium_residual_grid_affine" "$HOST_HEADER"; then
  echo "error: expected generated residual grid kernel in host header" >&2
  exit 2
fi
if ! grep -q "tensorium_residual_grid_parallel" "$HOST_HEADER"; then
  echo "error: expected generated residual parallel grid kernel in host header" >&2
  exit 2
fi
if ! grep -q "__kmpc_fork_call" "$LL_PATH"; then
  echo "error: expected OpenMP fork call in generated LLVM IR" >&2
  exit 2
fi

echo "[poisson-relax-l2] compiling LLVM object"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[poisson-relax-l2] compiling runtime L2 runner"
"$CXX_BIN" -O0 -std=c++20 "${OPENMP_CXXFLAGS[@]}" -I "$ROOT_DIR/include" \
  -include "$HOST_HEADER" "$RUNNER_SRC" "$OBJ_PATH" -lm \
  "${OPENMP_LDFLAGS[@]}" -o "$EXE_PATH"

echo "[poisson-relax-l2] running runtime L2 executable"
"$EXE_PATH"
