#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER_SRC="$ROOT_DIR/tools/dev/ll_init_runner_schwarzschild.c"
FIXTURE="$ROOT_DIR/tests/fixtures/gr/schwarzschild_3d.tn"

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
LL_PATH="/tmp/tensorium_schw3d.ll"
EXE_PATH="/tmp/tensorium_schw3d_ll_runner"
OBJ_PATH="/tmp/tensorium_schw3d_ll.o"
GENERATE_LL=1

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--ll <path>] [--no-generate]

Options:
  --ll <path>      Path to LLVM IR file (default: /tmp/tensorium_schw3d.ll)
  --no-generate    Use existing .ll file instead of regenerating from .tn
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ll)
      [[ $# -ge 2 ]] || { echo "error: --ll expects a path" >&2; exit 2; }
      LL_PATH="$2"
      shift 2
      ;;
    --no-generate)
      GENERATE_LL=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v "$CC_BIN" >/dev/null 2>&1; then
  echo "error: C compiler not found (set CC=/path/to/cc)" >&2
  exit 2
fi

if [[ ! -x "$DRIVER" ]]; then
  echo "error: missing driver binary: $DRIVER" >&2
  exit 2
fi

if [[ $GENERATE_LL -eq 1 ]]; then
  echo "[ll-smoke] generating LLVM IR: $LL_PATH"
  RAW_OUT="$(mktemp)"
  "$DRIVER" \
    --tensorium-metric-lower \
    --tensorium-init-std-lower \
    --tensorium-init-grid-affine-lower \
    --tensorium-rhs-grid-affine-lower \
    --tensorium-strip-source-funcs \
    --dump-llvm-ir \
    "$FIXTURE" >"$RAW_OUT"

  awk '
    /^\[Tensorium\]/ {exit}
    {print}
  ' "$RAW_OUT" >"$LL_PATH"
  rm -f "$RAW_OUT"
fi

if [[ ! -s "$LL_PATH" ]]; then
  echo "error: LLVM IR file is missing or empty: $LL_PATH" >&2
  exit 2
fi

echo "[ll-smoke] compiling LLVM IR object"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[ll-smoke] compiling C runner + linking"
"$CC_BIN" -O0 -std=c11 "$RUNNER_SRC" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[ll-smoke] running executable"
"$EXE_PATH"
