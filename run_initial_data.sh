#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER="$ROOT_DIR/tools/runtime/generated_initial_data_main.cpp"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 problem.tn [output.csv]" >&2
  exit 2
fi

INPUT_PATH="$1"
OUTPUT_PATH="${2:-/tmp/tensorium_initial_data.csv}"
SLICE_N="${TENSORIUM_SLICE_N:-129}"
HALF_WIDTH="${TENSORIUM_HALF_WIDTH:-8.0}"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "initial-data source is missing: $INPUT_PATH" >&2
  exit 2
fi
if [[ ! -x "$DRIVER" ]]; then
  echo "Tensorium_cc is missing; build the project first:" >&2
  echo "  cmake --build $ROOT_DIR/build -j2" >&2
  exit 2
fi

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
CXX_BIN="${CXX:-c++}"

WORK_DIR="$(mktemp -d /tmp/tensorium-initial-data.XXXXXX)"
trap 'rm -rf -- "$WORK_DIR"' EXIT
LL_PATH="$WORK_DIR/problem.ll"
OBJ_PATH="$WORK_DIR/problem.o"
HOST_HEADER="$WORK_DIR/generated_host.h"
EXE_PATH="$WORK_DIR/initial_data"

echo "[initial_data] compiling $INPUT_PATH"
"$DRIVER" \
  --tensorium-rhs-grid-affine-lower \
  --tensorium-strip-source-funcs \
  --emit-llvm "$LL_PATH" \
  --emit-host-header "$HOST_HEADER" \
  "$INPUT_PATH" >/dev/null

if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

"$CXX_BIN" -O2 -std=c++20 -I "$ROOT_DIR/include" -include "$HOST_HEADER" \
  "$RUNNER" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[initial_data] solving and exporting $OUTPUT_PATH"
"$EXE_PATH" "$OUTPUT_PATH" "$SLICE_N" "$HALF_WIDTH"
