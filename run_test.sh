
#!/usr/bin/env bash
set -euo pipefail

BIN=./build/tools/driver/Tensorium_cc
OUT=/tmp/tensorium_tests
mkdir -p "$OUT"

PIPELINE_BASE=(
  --tensorium-stencil-lower
  --tensorium-einstein-lower
  --tensorium-einstein-analyze-einsum
  --tensorium-einstein-canonicalize
  --tensorium-einstein-validate
)

PIPELINE_DISS=(
  --tensorium-stencil-lower
  --tensorium-dissipation
  --tensorium-einstein-lower
  --tensorium-einstein-analyze-einsum
  --tensorium-einstein-canonicalize
  --tensorium-einstein-validate
)

VALID_TESTS=(
  tests/00_simple_op.tns
  tests/01_scalar_minimal.tn
  tests/02_scalar_with_parameter.tn
  tests/03_spacial_derivatives.tn
  tests/04_metric_simple.tn
  tests/05_tensor_contraction.tn
  tests/06_nested_contraction.tn
  tests/07_bssn_reduced.tn
  tests/11_test_tensorium_types.tn
  tests/12_test_non_canonical_index.tn
  tests/16_valid_permutation.tn
  tests/18_valid_scalar_contraction.tn
  tests/20_valid_heavy_contraction.tn
  tests/21_test_full.tn
  tests/22_BSSN_minimal.tn
  tests/23_bssn_like_with_riemann_contract.tn
  tests/24_Ricci_conformal_flat.tn
  tests/25_deriv_stencil.tn
)

ERROR_TESTS=(
  tests/08_error_invalid_index.tn
  tests/09_error_bad_resolution.tn
  tests/10_no_simulation.tn
  tests/13__error_free_index_unassigned.tn
  tests/14_error_duplicate_free_index.tn
  tests/15__error_self_contraction.tn
  tests/17_error_hidden_index.tn
  tests/19_error_unused_index.tn
)

echo "=============================="
echo " RUN VALID TESTS"
echo "=============================="

for f in "${VALID_TESTS[@]}"; do
  echo "[OK EXPECTED] $f"
  "$BIN" "${PIPELINE_DISS[@]}" --dump-mlir "$f" \
    > "$OUT/$(basename "$f").mlir"
done

echo
echo "=============================="
echo " RUN ERROR TESTS"
echo "=============================="

for f in "${ERROR_TESTS[@]}"; do
  echo "[FAIL EXPECTED] $f"
  if "$BIN" "${PIPELINE_BASE[@]}" "$f" > /dev/null 2>&1; then
    echo "ERROR: $f was expected to fail but passed"
    exit 1
  fi
done

echo
echo "=============================="
echo " ALL TESTS PASSED"
echo " MLIR outputs in $OUT"
echo "=============================="
