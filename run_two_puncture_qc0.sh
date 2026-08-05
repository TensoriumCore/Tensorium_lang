#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
FIXTURE="$ROOT_DIR/tests/fixtures/elliptic/spectral_two_puncture_hamiltonian_3d.tn"
RUNNER="$ROOT_DIR/tools/dev/runtime_generated_two_puncture_qc0_export.cpp"

OUTPUT_PATH="${1:-/tmp/tensorium_qc0_bssn_slice.csv}"
N_A="${TP_NA:-10}"
N_B="${TP_NB:-10}"
N_PHI="${TP_NPHI:-16}"
SLICE_N="${TP_SLICE_N:-129}"
HALF_WIDTH="${TP_HALF_WIDTH:-8.0}"

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

LL_PATH="/tmp/tensorium_qc0_hamiltonian.ll"
OBJ_PATH="/tmp/tensorium_qc0_hamiltonian.o"
HOST_HEADER="/tmp/tensorium_qc0_hamiltonian_host.h"
EXE_PATH="/tmp/tensorium_qc0_export"

echo "[qc0] compiling the DSL residual"
"$DRIVER" \
  --tensorium-rhs-grid-affine-lower \
  --tensorium-strip-source-funcs \
  --emit-llvm "$LL_PATH" \
  --emit-host-header "$HOST_HEADER" \
  "$FIXTURE" >/dev/null

if command -v "$LLC_BIN" >/dev/null 2>&1; then
  "$LLC_BIN" -filetype=obj "$LL_PATH" -o "$OBJ_PATH"
else
  "$CLANG_BIN" -c "$LL_PATH" -o "$OBJ_PATH"
fi

"$CXX_BIN" -O2 -std=c++20 -I "$ROOT_DIR/include" -include "$HOST_HEADER" \
  "$RUNNER" "$OBJ_PATH" -lm -o "$EXE_PATH"

echo "[qc0] solving and exporting Cartesian BSSN data"
"$EXE_PATH" "$OUTPUT_PATH" "$N_A" "$N_B" "$N_PHI" "$SLICE_N" \
  "$HALF_WIDTH"
