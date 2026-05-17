#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT_DIR/build/tools/runtime/Tensorium_abi_probe"
FIXTURE="$ROOT_DIR/tests/fixtures/gr/schwarzschild_bssn_constraints_analytic_3d.tn"
OUT="/tmp/tensorium_abi_contract_probe_schwarzschild_bssn.txt"

if [[ ! -x "$PROBE" ]]; then
  echo "error: missing ABI probe binary: $PROBE" >&2
  exit 2
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "error: missing fixture: $FIXTURE" >&2
  exit 2
fi

echo "[abi-probe] probing complete Schwarzschild BSSN ABI contract"
"$PROBE" "$FIXTURE" > "$OUT"

grep -q "ABI contract OK" "$OUT"
grep -q "kernel tensorium_rhs_grid_affine kind=rhs_grid_affine" "$OUT"
grep -q "stencil_radius=1" "$OUT"
grep -q "data_arena_allocations=1" "$OUT"
grep -q "gammatilde c_name=gammatilde" "$OUT"
grep -q "Atilde c_name=Atilde" "$OUT"
grep -q "dAtilde c_name=dAtilde .* access=write" "$OUT"
grep -q "DAtilde c_name=DAtilde .* components=27 scalars=3375" "$OUT"
grep -q "Hamiltonian c_name=Hamiltonian .* access=write" "$OUT"
grep -q "Momentum c_name=Momentum .* variance=(0,1)" "$OUT"
grep -q "uniform_grid nx=5 ny=5 nz=5 n_points=125 ghost_required=1" "$OUT"

echo "[abi-probe] ABI contract smoke passed"
