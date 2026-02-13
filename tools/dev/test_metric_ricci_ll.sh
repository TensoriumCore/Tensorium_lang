#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"

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

FIXTURE=""
RUNNER_SRC=""
CASE_NAME=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --case <name> --fixture <path> --runner <path>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      CASE_NAME="${2:-}"
      shift 2
      ;;
    --fixture)
      FIXTURE="${2:-}"
      shift 2
      ;;
    --runner)
      RUNNER_SRC="${2:-}"
      shift 2
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

if [[ -z "$CASE_NAME" || -z "$FIXTURE" || -z "$RUNNER_SRC" ]]; then
  echo "error: --case, --fixture and --runner are required" >&2
  usage
  exit 2
fi

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

LL_PATH="/tmp/tensorium_${CASE_NAME}_ricci.ll"
OBJ_PATH="/tmp/tensorium_${CASE_NAME}_ricci_ll.o"
EXE_PATH="/tmp/tensorium_${CASE_NAME}_ricci_ll_runner"

echo "[ll-smoke] generating Ricci LLVM IR ($CASE_NAME): $LL_PATH"
RAW_OUT="$(mktemp)"
"$DRIVER" \
  --tensorium-metric-lower \
  --tensorium-init-std-lower \
  --tensorium-init-grid-affine-lower \
  --tensorium-rhs-grid-affine-lower \
  --tensorium-stencil-lower \
  --tensorium-einstein-lower \
  --tensorium-einstein-analyze-einsum \
  --tensorium-einstein-canonicalize \
  --tensorium-einstein-validate \
  --tensorium-strip-source-funcs \
  --dump-llvm-ir \
  "$FIXTURE" >"$RAW_OUT"

awk '
  /^\[Tensorium\]/ {exit}
  {print}
' "$RAW_OUT" >"$LL_PATH"
rm -f "$RAW_OUT"

if [[ ! -s "$LL_PATH" ]]; then
  echo "error: LLVM IR file is missing or empty: $LL_PATH" >&2
  exit 2
fi

echo "[ll-smoke] compiling Ricci LLVM IR object ($CASE_NAME)"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[ll-smoke] compiling Ricci C runner + linking ($CASE_NAME)"
"$CC_BIN" -O0 -std=c11 "$RUNNER_SRC" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[ll-smoke] running Ricci executable ($CASE_NAME)"
"$EXE_PATH"
