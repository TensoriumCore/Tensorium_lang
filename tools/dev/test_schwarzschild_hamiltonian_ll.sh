#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/tools/dev/test_metric_ricci_ll.sh" \
  --case "schwarzschild_hamiltonian" \
  --fixture "$ROOT_DIR/tests/fixtures/gr/schwarzschild_hamiltonian_3d.tn" \
  --runner "$ROOT_DIR/tools/dev/ll_rhs_runner_schwarzschild_hamiltonian.c"
