#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER="$ROOT_DIR/build/tools/driver/Tensorium_cc"
RUNNER_SRC="$ROOT_DIR/tools/dev/ll_init_runner_standard_metrics.c"

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

if [[ ! -x "$DRIVER" ]]; then
  echo "error: missing driver binary: $DRIVER" >&2
  exit 2
fi
if [[ ! -f "$RUNNER_SRC" ]]; then
  echo "error: missing runner source: $RUNNER_SRC" >&2
  exit 2
fi

run_case() {
  local name="$1"
  local fixture="$2"
  local macro="$3"
  local ll_path="/tmp/tensorium_${name}_analytic_init.ll"
  local obj_path="/tmp/tensorium_${name}_analytic_init.o"
  local exe_path="/tmp/tensorium_${name}_analytic_init_runner"
  local host_header="/tmp/tensorium_${name}_analytic_init_host.h"

  echo "[analytic-init] generating LLVM IR and host header ($name)"
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
    --emit-llvm "$ll_path" \
    --emit-host-header "$host_header" \
    "$fixture" >/dev/null

  if [[ ! -s "$ll_path" ]]; then
    echo "error: LLVM IR file is missing or empty: $ll_path" >&2
    exit 2
  fi
  if [[ ! -s "$host_header" ]]; then
    echo "error: generated host header is missing or empty: $host_header" >&2
    exit 2
  fi

  echo "[analytic-init] compiling LLVM object ($name)"
  if command -v "$LLC_BIN" >/dev/null 2>&1; then
    "$LLC_BIN" -filetype=obj "$ll_path" -o "$obj_path"
  else
    "$CLANG_BIN" -c "$ll_path" -o "$obj_path"
  fi

  echo "[analytic-init] compiling analytic runner ($name)"
  "$CC_BIN" -O0 -std=c11 -D"TENSORIUM_STANDARD_METRIC_CASE=$macro" \
    -include "$host_header" "$RUNNER_SRC" "$obj_path" -lm -o "$exe_path"

  echo "[analytic-init] running analytic comparison ($name)"
  "$exe_path"
}

run_case \
  "minkowski" \
  "$ROOT_DIR/tests/fixtures/gr/minkowski_ricci_3d.tn" \
  "TENSORIUM_STANDARD_METRIC_MINKOWSKI"

run_case \
  "schwarzschild" \
  "$ROOT_DIR/tests/fixtures/gr/schwarzschild_3d.tn" \
  "TENSORIUM_STANDARD_METRIC_SCHWARZSCHILD"

run_case \
  "reissner_nordstrom" \
  "$ROOT_DIR/tests/fixtures/gr/reissner_nordstrom_3d.tn" \
  "TENSORIUM_STANDARD_METRIC_REISSNER_NORDSTROM"

run_case \
  "kerr_like" \
  "$ROOT_DIR/tests/fixtures/gr/kerr_like_3d.tn" \
  "TENSORIUM_STANDARD_METRIC_KERR_LIKE"

run_case \
  "spatial_offdiag" \
  "$ROOT_DIR/tests/fixtures/gr/spatial_offdiag_3d.tn" \
  "TENSORIUM_STANDARD_METRIC_SPATIAL_OFFDIAG"

run_case \
  "schwarzschild_isotropic" \
  "$ROOT_DIR/tests/fixtures/gr/schwarzschild_isotropic_cartesian_ricci_3d.tn" \
  "TENSORIUM_STANDARD_METRIC_SCHWARZSCHILD_ISOTROPIC"
