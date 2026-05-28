#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${BY_STEPS:=500}"
: "${BY_ZERO_TOL:=1.0e-12}"

export BY_PX=0.0
export BY_EXPECT_ZERO=1
export BY_STEPS
export BY_ZERO_TOL

echo "[bowen-york-single-puncture-p0] running zero-momentum source check"
exec "$ROOT_DIR/tools/dev/test_bowen_york_single_puncture_relax_l2_ll.sh"
