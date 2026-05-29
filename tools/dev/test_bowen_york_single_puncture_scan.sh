#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER_SRC="$ROOT_DIR/tools/dev/runtime_bowen_york_single_puncture_relax_l2.cpp"
FIXTURE="$ROOT_DIR/tests/fixtures/elliptic/bowen_york_single_puncture_relax_3d.tn"

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

TMP_BASE="${TMPDIR:-/tmp}/tensorium_bowen_york_single_puncture_scan"
LL_PATH="$TMP_BASE.ll"
OBJ_PATH="$TMP_BASE.o"
EXE_PATH="${TMP_BASE}_runner"
HOST_HEADER="${TMP_BASE}_host.h"

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

echo "[bowen-york-single-puncture-scan] generating LLVM IR and host header: $LL_PATH" >&2
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

echo "[bowen-york-single-puncture-scan] compiling LLVM object" >&2
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[bowen-york-single-puncture-scan] compiling runtime L2 runner" >&2
"$CXX_BIN" -O0 -std=c++20 "${OPENMP_CXXFLAGS[@]}" -I "$ROOT_DIR/include" \
  -include "$HOST_HEADER" "$RUNNER_SRC" "$OBJ_PATH" -lm \
  "${OPENMP_LDFLAGS[@]}" -o "$EXE_PATH"

: "${BY_DT:=0.005}"
: "${BY_STEPS:=1600}"

echo "eta,c,initial_H,final_H,ratio,max_u,status"
failed=0
for eta in 0.5 1.0 2.0; do
  for c in 0.25 0.5 1.0; do
    row=""
    if row=$(BY_ETA="$eta" BY_C="$c" BY_DT="$BY_DT" BY_STEPS="$BY_STEPS" \
      BY_OUTPUT=csv BY_CHECKPOINTS=0 BY_FAIL_ON_WEAK=0 "$EXE_PATH"); then
      echo "$row"
    else
      code=$?
      if [[ -z "$row" ]]; then
        row="$eta,$c,nan,nan,nan,nan,runner_failed_$code"
      fi
      echo "$row"
      failed=1
    fi

    status="${row##*,}"
    if [[ "$status" == invalid* || "$status" == zero_fail ||
          "$status" == runner_failed* ]]; then
      failed=1
    fi
  done
done

if [[ "$failed" -ne 0 ]]; then
  echo "[bowen-york-single-puncture-scan] one or more runs failed hard" >&2
  exit 3
fi
