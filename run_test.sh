
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
  --dump-mlir
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
  tests/31_temp_valid_scalar.tn
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
  tests/26_error_unsupported_emit.tn
  tests/29_executable_unknown_scalar_call_error.tn
  tests/35_executable_extern_missing.tn
  tests/32_temp_invalid_tensor_rhs.tn
  tests/33_temp_use_before_def.tn
)

SYMBOLIC_VALID_TESTS=(
  tests/28_symbolic_unknown_scalar_call_ok.tn
  tests/36_symbolic_unknown_scalar_call_ok.tn
)

SYMBOLIC_ERROR_TESTS=(
  tests/30_symbolic_call_nonscalar_arg_error.tn
)

echo "=============================="
echo " RUN VALID TESTS"
echo "=============================="

for f in "${VALID_TESTS[@]}"; do
  echo "[OK EXPECTED] $f"
  "$BIN" "${PIPELINE_DISS[@]}" --dump-mlir "$f" \
    > "$OUT/$(basename "$f").mlir"
done

EXTERN_TEST=tests/34_executable_extern_declared.tn

echo
echo "=============================="
echo " RUN EXTERN DECL TESTS"
echo "=============================="

echo "[OK EXPECTED - NO MLIR] $EXTERN_TEST"
"$BIN" "$EXTERN_TEST" > /dev/null

echo "[FAIL EXPECTED - MLIR EXTERN LOWERING] $EXTERN_TEST"
TMP_ERR=$(mktemp)
if "$BIN" "${PIPELINE_BASE[@]}" "$EXTERN_TEST" > "$TMP_ERR" 2>&1; then
  echo "ERROR: $EXTERN_TEST was expected to fail during MLIR emission"
  rm -f "$TMP_ERR"
  exit 1
fi
if ! grep -q "extern function 'foo' lowering is not implemented yet" "$TMP_ERR"; then
  echo "ERROR: expected extern lowering error message, got:"
  cat "$TMP_ERR"
  rm -f "$TMP_ERR"
  exit 1
fi
rm -f "$TMP_ERR"

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
echo " RUN SYMBOLIC TESTS"
echo "=============================="

for f in "${SYMBOLIC_VALID_TESTS[@]}"; do
  echo "[SYMBOLIC OK EXPECTED] $f"
  "$BIN" --symbolic "$f" > /dev/null
done

for f in "${SYMBOLIC_ERROR_TESTS[@]}"; do
  echo "[SYMBOLIC FAIL EXPECTED] $f"
  if "$BIN" --symbolic "$f" > /dev/null 2>&1; then
    echo "ERROR: $f was expected to fail but passed"
    exit 1
  fi
done

echo
echo "=============================="
echo " ALL TESTS PASSED"
echo " MLIR outputs in $OUT"
echo "=============================="
