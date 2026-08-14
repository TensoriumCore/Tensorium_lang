#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
FIXTURE="$ROOT_DIR/tests/fixtures/elliptic/spectral_two_puncture_hamiltonian_3d.tn"
RUNNER_SRC="$ROOT_DIR/tools/dev/runtime_generated_two_puncture_hamiltonian_solve.cpp"

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

LL_PATH="/tmp/tensorium_generated_two_puncture_hamiltonian.ll"
OBJ_PATH="/tmp/tensorium_generated_two_puncture_hamiltonian.o"
HOST_HEADER="/tmp/tensorium_generated_two_puncture_hamiltonian_host.h"
EXE_PATH="/tmp/tensorium_generated_two_puncture_hamiltonian_runner"

echo "[two-puncture-hamiltonian] generating LLVM IR and host header"
"$DRIVER" \
  --tensorium-rhs-grid-affine-lower \
  --tensorium-strip-source-funcs \
  --emit-llvm "$LL_PATH" \
  --emit-host-header "$HOST_HEADER" \
  "$FIXTURE" >/dev/null

if ! grep -q "SpectralTwoPunctureHamiltonian3D" "$HOST_HEADER"; then
  echo "error: expected generated two-puncture residual system" >&2
  exit 2
fi
if ! grep -q "define void @tensorium_spectral_residual_grid_H" "$LL_PATH"; then
  echo "error: expected generated two-puncture LLVM grid kernel" >&2
  exit 2
fi
if ! grep -q "define double @tensorium_spectral_residual_jvp_H" "$LL_PATH"; then
  echo "error: expected generated two-puncture LLVM JVP kernel" >&2
  exit 2
fi

echo "[two-puncture-hamiltonian] compiling generated LLVM and runner"
if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi
"$CXX_BIN" -O0 -std=c++20 -I "$ROOT_DIR/include" -include "$HOST_HEADER" \
  "$RUNNER_SRC" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[two-puncture-hamiltonian] running physical residual solve"
"$EXE_PATH"
