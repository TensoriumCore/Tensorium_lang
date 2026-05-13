#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER_SRC="$ROOT_DIR/tools/dev/ll_rhs_runner_z4c_kasner.c"
FIXTURE="$ROOT_DIR/tests/fixtures/gr/z4c_kasner_reduction_3d.tn"

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

LL_PATH="/tmp/tensorium_z4c_kasner.ll"
OBJ_PATH="/tmp/tensorium_z4c_kasner.o"
EXE_PATH="/tmp/tensorium_z4c_kasner_runner"
HOST_HEADER="/tmp/tensorium_z4c_kasner_host.h"

if [[ ! -x "$DRIVER" ]]; then
  echo "error: missing driver binary: $DRIVER" >&2
  exit 2
fi

echo "[ll-smoke] generating LLVM IR and host header (z4c kasner): $LL_PATH"
"$DRIVER" \
  --tensorium-rhs-grid-affine-lower \
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

echo "[ll-smoke] compiling LLVM IR object (z4c kasner)"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[ll-smoke] compiling Z4c Kasner C runner + linking"
"$CC_BIN" -O0 -std=c11 -include "$HOST_HEADER" "$RUNNER_SRC" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[ll-smoke] running Z4c Kasner executable"
"$EXE_PATH"
