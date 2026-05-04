#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_SH="$ROOT_DIR/tools/dev/test_metric_ricci_ll.sh"
FIXTURE="$ROOT_DIR/tests/fixtures/gr/schwarzschild_isotropic_cartesian_ricci_3d.tn"
RUNNER="$ROOT_DIR/tools/dev/ll_rhs_runner_export_bh_cartesian_slice.c"
PLOT="$ROOT_DIR/tools/dev/plot_bh_slice_regular.py"

CSV_PATH="${1:-/tmp/bh_cartesian_slice64.csv}"
ALPHA_PNG="${2:-/tmp/bh_alpha_slice_regular.png}"
RICCI_PNG="${3:-/tmp/bh_ricci_trace_slice_regular.png}"

"$TEST_SH" \
  --case bh_cartesian_slice64 \
  --fixture "$FIXTURE" \
  --runner "$RUNNER"

/tmp/tensorium_bh_cartesian_slice64_ricci_ll_runner "$CSV_PATH"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcfg}" \
  python3 "$PLOT" --csv "$CSV_PATH" --alpha-png "$ALPHA_PNG" --ricci-png "$RICCI_PNG"

echo "CSV: $CSV_PATH"
echo "Alpha PNG: $ALPHA_PNG"
echo "Ricci PNG: $RICCI_PNG"
