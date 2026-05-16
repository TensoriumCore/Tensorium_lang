#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER_SRC="$ROOT_DIR/tools/dev/ll_rhs_runner_schwarzschild_bssn_constraints.c"
FIXTURE="$ROOT_DIR/tests/fixtures/gr/schwarzschild_bssn_constraints_analytic_3d.tn"

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
CC_BIN="${CC:-cc}"

LL_PATH="/tmp/tensorium_schwarzschild_bssn_constraints.ll"
OBJ_PATH="/tmp/tensorium_schwarzschild_bssn_constraints.o"
EXE_PATH="/tmp/tensorium_schwarzschild_bssn_constraints_runner"
HOST_HEADER="/tmp/tensorium_schwarzschild_bssn_constraints_host.h"

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

echo "[analytic-schwarzschild-bssn] generating LLVM IR and host header: $LL_PATH"
"$DRIVER" \
  --tensorium-metric-lower \
  --tensorium-init-std-lower \
  --tensorium-init-grid-affine-lower \
  --tensorium-rhs-grid-affine-lower \
  --tensorium-stencil-lower \
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

echo "[analytic-schwarzschild-bssn] compiling LLVM object"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[analytic-schwarzschild-bssn] compiling complete BSSN constraints runner"
"$CC_BIN" -O0 -std=c11 -include "$HOST_HEADER" "$RUNNER_SRC" "$OBJ_PATH" \
  -lm -o "$EXE_PATH"

echo "[analytic-schwarzschild-bssn] running complete BSSN constraints executable"
"$EXE_PATH"
