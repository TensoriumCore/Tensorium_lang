#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE="$ROOT_DIR/tests/fixtures/elliptic/spectral_two_puncture_hamiltonian_3d.tn"
OUTPUT_PATH="${1:-/tmp/tensorium_qc0_bssn_slice.csv}"

TENSORIUM_SLICE_N="${TP_SLICE_N:-${TENSORIUM_SLICE_N:-129}}" \
TENSORIUM_HALF_WIDTH="${TP_HALF_WIDTH:-${TENSORIUM_HALF_WIDTH:-8.0}}" \
  "$ROOT_DIR/run_initial_data.sh" "$FIXTURE" "$OUTPUT_PATH"
