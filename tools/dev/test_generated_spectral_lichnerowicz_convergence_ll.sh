#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
FIXTURE="$ROOT_DIR/tests/fixtures/elliptic/spectral_lichnerowicz_manufactured_3d.tn"
RUNNER_SRC="$ROOT_DIR/tools/dev/runtime_generated_spectral_lichnerowicz_convergence.cpp"

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

LL_PATH="/tmp/tensorium_generated_spectral_lichnerowicz_convergence.ll"
OBJ_PATH="/tmp/tensorium_generated_spectral_lichnerowicz_convergence.o"
HOST_HEADER="/tmp/tensorium_generated_spectral_lichnerowicz_convergence_host.h"
EXE_PATH="/tmp/tensorium_generated_spectral_lichnerowicz_convergence_runner"

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

echo "[generated-spectral-lichnerowicz-convergence] generating LLVM IR and host header"
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
if ! grep -q "SpectralLichnerowiczManufactured3D" "$HOST_HEADER"; then
  echo "error: expected Lichnerowicz spectral system name" >&2
  exit 2
fi
if ! grep -q "define void @tensorium_spectral_residual_grid_H" "$LL_PATH"; then
  echo "error: expected Lichnerowicz spectral grid residual LLVM definition" >&2
  exit 2
fi

echo "[generated-spectral-lichnerowicz-convergence] compiling LLVM object"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

echo "[generated-spectral-lichnerowicz-convergence] compiling runtime runner"
"$CXX_BIN" -O0 -std=c++20 -I "$ROOT_DIR/include" -include "$HOST_HEADER" \
  "$RUNNER_SRC" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[generated-spectral-lichnerowicz-convergence] running runtime executable"
"$EXE_PATH"
