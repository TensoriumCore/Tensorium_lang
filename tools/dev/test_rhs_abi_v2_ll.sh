#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER_SRC="$ROOT_DIR/tools/dev/ll_rhs_runner_abi_v2.c"
FIXTURE="$ROOT_DIR/tests/07_bssn_reduced.tn"

if [[ -n "${CLANG:-}" ]]; then
  CLANG_BIN="$CLANG"
elif [[ -x /opt/llvm-20/bin/clang ]]; then
  CLANG_BIN="/opt/llvm-20/bin/clang"
else
  CLANG_BIN="clang"
fi
if [[ -n "${LLVM_AS:-}" ]]; then
  LLVM_AS_BIN="$LLVM_AS"
elif [[ -x /opt/llvm-20/bin/llvm-as ]]; then
  LLVM_AS_BIN="/opt/llvm-20/bin/llvm-as"
else
  LLVM_AS_BIN="llvm-as"
fi
if [[ -n "${OPT:-}" ]]; then
  OPT_BIN="$OPT"
elif [[ -x /opt/llvm-20/bin/opt ]]; then
  OPT_BIN="/opt/llvm-20/bin/opt"
else
  OPT_BIN="opt"
fi

LL_PATH="/tmp/tensorium_rhs_abi_v2.ll"
BC_PATH="/tmp/tensorium_rhs_abi_v2.bc"

if [[ ! -x "$DRIVER" ]]; then
  echo "error: missing driver binary: $DRIVER" >&2
  exit 2
fi

echo "[ll-smoke] generating ABI v2 LLVM IR"
RAW_OUT="$(mktemp)"
"$DRIVER" --dump-llvm-ir "$FIXTURE" >"$RAW_OUT"
awk '
  /^\[Tensorium\]/ {exit}
  {print}
' "$RAW_OUT" >"$LL_PATH"
rm -f "$RAW_OUT"

"$LLVM_AS_BIN" "$LL_PATH" -o "$BC_PATH"
"$OPT_BIN" -passes=verify -disable-output "$BC_PATH"

for level in 0 2; do
  OBJ_PATH="/tmp/tensorium_rhs_abi_v2_O${level}.o"
  EXE_PATH="/tmp/tensorium_rhs_abi_v2_O${level}"
  echo "[ll-smoke] compiling and running ABI v2 at -O${level}"
  "$CLANG_BIN" "-O${level}" -c "$LL_PATH" -o "$OBJ_PATH"
  "$CLANG_BIN" "-O${level}" -std=c11 "$RUNNER_SRC" "$OBJ_PATH" -lm \
    -o "$EXE_PATH"
  "$EXE_PATH"
done
