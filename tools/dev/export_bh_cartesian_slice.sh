#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_SH="$ROOT_DIR/tools/dev/test_metric_ricci_ll.sh"
FIXTURE="$ROOT_DIR/tests/fixtures/gr/schwarzschild_isotropic_cartesian_ricci_3d.tn"
RUNNER="$ROOT_DIR/tools/dev/ll_rhs_runner_export_bh_cartesian_slice.c"
PLOT="$ROOT_DIR/tools/dev/plot_bh_slice_regular.py"

GRID_N="${BH_GRID_N:-64}"
HALF_WIDTH="${BH_HALF_WIDTH:-8.0}"
PLOT_DPI="${BH_PLOT_DPI:-240}"
CASE_NAME="bh_cartesian_slice${GRID_N}"

CSV_PATH="${1:-/tmp/bh_cartesian_slice${GRID_N}.csv}"
ALPHA_PNG="${2:-/tmp/bh_alpha_slice_${GRID_N}.png}"
RICCI_PNG="${3:-/tmp/bh_ricci_trace_slice_${GRID_N}.png}"

"$TEST_SH" \
  --case "$CASE_NAME" \
  --fixture "$FIXTURE" \
  --runner "$RUNNER"

"/tmp/tensorium_${CASE_NAME}_ricci_ll_runner" "$CSV_PATH" "$GRID_N" "$HALF_WIDTH"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcfg}" \
  python3 "$PLOT" --csv "$CSV_PATH" --alpha-png "$ALPHA_PNG" \
    --ricci-png "$RICCI_PNG" --dpi "$PLOT_DPI"

echo "CSV: $CSV_PATH"
echo "Alpha PNG: $ALPHA_PNG"
echo "Ricci PNG: $RICCI_PNG"
