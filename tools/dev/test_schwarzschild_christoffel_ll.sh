#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/tools/dev/test_metric_christoffel_ll.sh" \
  --case "schwarzschild" \
  --fixture "$ROOT_DIR/tests/fixtures/gr/schwarzschild_christoffel_3d.tn" \
  --runner "$ROOT_DIR/tools/dev/ll_rhs_runner_schwarzschild_christoffel.c"
