
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
  tests/50_large_tensor_mix.tn
  tests/51_large_tensor_flux.tn
  tests/56_metric_decl_ok.tn
  tests/31_temp_valid_scalar.tn
)

EXTERN_TESTS=(
  tests/34_executable_extern_declared.tn
  tests/39_valid_extern_call.tn
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
  tests/38_error_extern_mixedtensor_duplicate_attr.tn
  tests/40_error_extern_arity.tn
  tests/41_error_extern_variance.tn
  tests/43_error_extern_tensor_return_exec.tn
  tests/52_error_tensor_add_variance.tn
  tests/53_error_contract_free_index.tn
  tests/54_error_dt_assign_rank.tn
  tests/55_error_implicit_contraction.tn
  tests/57_error_metric_rank.tn
)

SYMBOLIC_VALID_TESTS=(
  tests/28_symbolic_unknown_scalar_call_ok.tn
  tests/36_symbolic_unknown_scalar_call_ok.tn
  tests/37_valid_extern_tensor_type.tn
  tests/42_symbolic_extern_tensor_return.tn
)

SYMBOLIC_ERROR_TESTS=(
  tests/30_symbolic_call_nonscalar_arg_error.tn
)

SYMBOLIC_MLIR_TESTS=()
SYMBOLIC_TENSOR_FAIL_TESTS=(
  tests/44_symbolic_extern_tensor_mlir.tn
)

echo "=============================="
echo " RUN VALID TESTS"
echo "=============================="

for f in "${VALID_TESTS[@]}"; do
  echo "[OK EXPECTED] $f"
  "$BIN" "${PIPELINE_DISS[@]}" --dump-mlir "$f" \
    > "$OUT/$(basename "$f").mlir"
done

PRIMARY_MLIR="$OUT/$(basename ${VALID_TESTS[0]}).mlir"
if ! grep -q "tensorium.field" "$PRIMARY_MLIR"; then
  echo "ERROR: expected tensorium.field types in $PRIMARY_MLIR"
  exit 1
fi

echo
echo "=============================="
echo " RUN EXTERN DECL TESTS"
echo "=============================="

for f in "${EXTERN_TESTS[@]}"; do
  echo "[OK EXPECTED - NO MLIR] $f"
  "$BIN" "$f" > /dev/null
  echo "[FAIL EXPECTED - MLIR EXTERN LOWERING] $f"
  TMP_ERR=$(mktemp)
  if "$BIN" "${PIPELINE_BASE[@]}" "$f" > "$TMP_ERR" 2>&1; then
    echo "ERROR: $f was expected to fail during MLIR emission"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  if ! grep -q "extern function" "$TMP_ERR"; then
    echo "ERROR: expected extern lowering error message"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
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
echo " RUN SYMBOLIC TESTS"
echo "=============================="

for f in "${SYMBOLIC_VALID_TESTS[@]}"; do
  echo "[SYMBOLIC OK EXPECTED] $f"
  "$BIN" --symbolic "$f" > /dev/null
done

echo
echo "=============================="
echo " RUN TYPE ANNOTATION TEST"
echo "=============================="

TYPE_TEST=tests/45_type_annotation.tn
TYPE_OUT="$OUT/type_annotation.log"
"$BIN" --symbolic --dump-indexed --dump-backend-expr "$TYPE_TEST" \
  > "$TYPE_OUT"
if ! grep -F -q "v[field;i][u=1,d=0]" "$TYPE_OUT"; then
  echo "ERROR: missing indexed type annotation in $TYPE_TEST"
  exit 1
fi
if ! grep -F -q "phi[field][u=0,d=0]" "$TYPE_OUT"; then
  echo "ERROR: missing scalar type annotation in $TYPE_TEST"
  exit 1
fi

for f in "${SYMBOLIC_ERROR_TESTS[@]}"; do
  echo "[SYMBOLIC FAIL EXPECTED] $f"
  if "$BIN" --symbolic "$f" > /dev/null 2>&1; then
    echo "ERROR: $f was expected to fail but passed"
    exit 1
  fi
done

if ((${#SYMBOLIC_MLIR_TESTS[@]})); then
  echo
  echo "=============================="
  echo " RUN SYMBOLIC MLIR TESTS"
  echo "=============================="

  for f in "${SYMBOLIC_MLIR_TESTS[@]}"; do
    echo "[SYMBOLIC MLIR EXPECTED] $f"
    "$BIN" --symbolic --dump-mlir "$f" \
      > "$OUT/$(basename "$f").symbolic.mlir"
  done
fi

echo
echo "=============================="
echo " RUN SYMBOLIC TENSOR FAIL TESTS"
echo "=============================="

for f in "${SYMBOLIC_TENSOR_FAIL_TESTS[@]}"; do
  echo "[SYMBOLIC MLIR FAIL EXPECTED] $f"
  TMP_ERR=$(mktemp)
  if "$BIN" --symbolic --dump-mlir "$f" > "$TMP_ERR" 2>&1; then
    echo "ERROR: $f was expected to fail during MLIR emission"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  if ! grep -q "extern function" "$TMP_ERR"; then
    echo "ERROR: expected extern tensor lowering error, got:"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
done

echo
echo "=============================="
echo " ALL TESTS PASSED"
echo " MLIR outputs in $OUT"
echo "=============================="
