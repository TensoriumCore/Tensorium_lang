#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

mkdir -p "$ROOT_DIR/build/exports"

export TENSORIUM_BY_CONTINUATION_STAGES="$ROOT_DIR/tools/dev/bowen_york_puncture_continuation_stages.txt"
export TENSORIUM_BY_GRID_N="${TENSORIUM_BY_GRID_N:-16}"
export TENSORIUM_BY_SLICE_CSV="${TENSORIUM_BY_SLICE_CSV:-$ROOT_DIR/build/exports/bowen_york_puncture_z_slice_n${TENSORIUM_BY_GRID_N}.csv}"
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  export OMP_NUM_THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
fi

echo "[generated-spectral-bowen-york-puncture-slice-export] grid_n = $TENSORIUM_BY_GRID_N omp_threads = $OMP_NUM_THREADS"

tensorium_generated_spectral_smoke \
  "generated-spectral-bowen-york-puncture-slice-export" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_bowen_york_regularized_puncture_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_bowen_york_puncture_continuation.cpp" \
  "tensorium_generated_spectral_bowen_york_puncture_slice_export" \
  header "tensorium_spectral_residual_H" \
  "error: expected Bowen-York spectral residual symbol" \
  header "tensorium_spectral_residual_grid_H" \
  "error: expected Bowen-York spectral grid residual symbol" \
  header "SpectralBowenYorkRegularizedPuncture3D" \
  "error: expected Bowen-York puncture spectral system name" \
  header "radial" \
  "error: expected Bowen-York radial Robin boundary descriptor" \
  header "radius" \
  "error: expected Bowen-York radius Robin coefficient descriptor"

if [[ ! -s "$TENSORIUM_BY_SLICE_CSV" ]]; then
  echo "error: missing Bowen-York slice CSV: $TENSORIUM_BY_SLICE_CSV" >&2
  exit 3
fi

expected_lines=$((TENSORIUM_BY_GRID_N * TENSORIUM_BY_GRID_N + 1))
actual_lines="$(wc -l < "$TENSORIUM_BY_SLICE_CSV" | tr -d ' ')"
if [[ "$actual_lines" != "$expected_lines" ]]; then
  echo "error: expected $expected_lines CSV lines, got $actual_lines" >&2
  exit 3
fi

read -r csv_header < "$TENSORIUM_BY_SLICE_CSV"
if [[ "$csv_header" != "i,j,k,x,y,z,u,psi_singular,psi,residual,r_puncture" ]]; then
  echo "error: unexpected Bowen-York slice CSV header: $csv_header" >&2
  exit 3
fi

echo "[generated-spectral-bowen-york-puncture-slice-export] csv = $TENSORIUM_BY_SLICE_CSV"
