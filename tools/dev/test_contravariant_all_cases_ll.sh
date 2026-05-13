#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER_SRC="$ROOT_DIR/tools/dev/ll_rhs_runner_contravariant_all_cases.c"
FIXTURE="$ROOT_DIR/tests/fixtures/gr/contravariant_all_cases_3d.tn"

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

LL_PATH="/tmp/tensorium_contravariant_all_cases.ll"
OBJ_PATH="/tmp/tensorium_contravariant_all_cases.o"
EXE_PATH="/tmp/tensorium_contravariant_all_cases_runner"

if [[ ! -x "$DRIVER" ]]; then
  echo "error: missing driver binary: $DRIVER" >&2
  exit 2
fi

echo "[ll-smoke] generating LLVM IR (contravariant nabla all-cases): $LL_PATH"
"$DRIVER" \
  --tensorium-rhs-grid-affine-lower \
  --tensorium-stencil-lower \
  --tensorium-einstein-lower \
  --tensorium-einstein-analyze-einsum \
  --tensorium-einstein-canonicalize \
  --tensorium-einstein-validate \
  --tensorium-strip-source-funcs \
  --emit-llvm "$LL_PATH" \
  "$FIXTURE" >/dev/null

if [[ ! -s "$LL_PATH" ]]; then
  echo "error: LLVM IR file is missing or empty: $LL_PATH" >&2
  exit 2
fi

echo "[ll-smoke] compiling LLVM IR object (contravariant nabla all-cases)"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[ll-smoke] compiling contravariant all-cases C runner + linking"
"$CC_BIN" -O0 -std=c11 "$RUNNER_SRC" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[ll-smoke] running contravariant all-cases executable"
"$EXE_PATH"
