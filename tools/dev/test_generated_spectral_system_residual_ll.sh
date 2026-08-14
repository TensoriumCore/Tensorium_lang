#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
FIXTURE="$ROOT_DIR/tests/fixtures/elliptic/spectral_two_field_system_3d.tn"
RUNNER_SRC="$ROOT_DIR/tools/dev/runtime_generated_spectral_system_residual.cpp"

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

LL_PATH="/tmp/tensorium_generated_spectral_system_residual.ll"
OBJ_PATH="/tmp/tensorium_generated_spectral_system_residual.o"
HOST_HEADER="/tmp/tensorium_generated_spectral_system_residual_host.h"
EXE_PATH="/tmp/tensorium_generated_spectral_system_residual_runner"

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

echo "[generated-spectral-system] generating LLVM IR and host header"
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
if ! grep -q "tensorium_spectral_residual_Hu" "$HOST_HEADER"; then
  echo "error: expected Hu spectral residual symbol" >&2
  exit 2
fi
if ! grep -q "tensorium_spectral_residual_Hv" "$HOST_HEADER"; then
  echo "error: expected Hv spectral residual symbol" >&2
  exit 2
fi
if ! grep -q "tensorium_spectral_residual_jvp_Hu" "$HOST_HEADER"; then
  echo "error: expected Hu spectral JVP symbol" >&2
  exit 2
fi
if ! grep -q "tensorium_spectral_residual_jvp_Hv" "$HOST_HEADER"; then
  echo "error: expected Hv spectral JVP symbol" >&2
  exit 2
fi
if ! grep -q "tensorium_spectral_residual_derivative_fields" "$HOST_HEADER"; then
  echo "error: expected spectral auxiliary derivative-field ABI" >&2
  exit 2
fi
if ! grep -q "point->aux_derivatives\[1\].d11" "$HOST_HEADER"; then
  echo "error: expected generated point wrapper to forward auxiliary Hessians" >&2
  exit 2
fi
if ! grep -q "direction->aux_derivatives\[1\].d1" "$HOST_HEADER"; then
  echo "error: expected generated JVP wrapper to forward auxiliary gradients" >&2
  exit 2
fi
if ! grep -q "tensorium_spectral_residual_systems" "$HOST_HEADER"; then
  echo "error: expected spectral residual system descriptor table" >&2
  exit 2
fi
if ! grep -q "TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT 1" "$HOST_HEADER"; then
  echo "error: expected one generated spectral residual system" >&2
  exit 2
fi
if ! grep -q "tensorium_spectral_residual_grid_Hu" "$LL_PATH"; then
  echo "error: expected Hu spectral grid residual LLVM definition" >&2
  exit 2
fi
if ! grep -q "tensorium_spectral_residual_grid_Hv" "$LL_PATH"; then
  echo "error: expected Hv spectral grid residual LLVM definition" >&2
  exit 2
fi
if ! grep -q "define double @tensorium_spectral_residual_jvp_Hu" "$LL_PATH"; then
  echo "error: expected Hu spectral JVP LLVM definition" >&2
  exit 2
fi
if ! grep -q "define double @tensorium_spectral_residual_jvp_Hv" "$LL_PATH"; then
  echo "error: expected Hv spectral JVP LLVM definition" >&2
  exit 2
fi

echo "[generated-spectral-system] compiling LLVM object"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[generated-spectral-system] compiling runtime runner"
"$CXX_BIN" -O0 -std=c++20 -I "$ROOT_DIR/include" -include "$HOST_HEADER" \
  "$RUNNER_SRC" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[generated-spectral-system] running runtime executable"
"$EXE_PATH"
