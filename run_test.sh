
#!/usr/bin/env bash
set -euo pipefail

BIN=./build/tools/driver/Tensorium_cc
UNIT_BIN=./build/tools/Tester/Tensorium_unittests
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

SEMANTIC_EINSTEIN_VALID_TESTS=(
  tests/semantic/einstein/01_valid_contraction.tn
  tests/semantic/einstein/04_valid_two_sums.tn
)

SEMANTIC_EINSTEIN_ERROR_TESTS=(
  "tests/semantic/einstein/02_invalid_variance_contraction.tn|requires explicit metric or inverse metric"
  "tests/semantic/einstein/03_collision_trace.tn|Implicit trace"
  "tests/semantic/einstein/05_capture_requires_rename.tn|Index collision: symbol"
)

SEMANTIC_DIFF_VALID_TESTS=(
  tests/semantic/diff/01_partial_scalar.tn
  tests/semantic/diff/02_partial_vector_rank_plus_one.tn
  tests/semantic/diff/03_covariant_with_gamma.tn
)

SEMANTIC_DIFF_ERROR_TESTS=(
  "tests/semantic/diff/04_covariant_without_gamma_error.tn|Covariant derivative requires connection tensor Gamma"
)

INITIAL_DATA_ERROR_TESTS=(
  "tests/semantic/initial_data/01_invalid_spherical_coord.tn|uses coordinate 'x' incompatible with simulation coordinates"
  "tests/semantic/initial_data/02_symmetry_violation.tn|metric4 symmetry violation"
)

INITIAL_DATA_MLIR_ERROR_TESTS=(
  "tests/semantic/initial_data/03_missing_gammau_binding.tn|split_3p1 does not bind gammaU"
  "tests/semantic/initial_data/04_nonsymmetric_metric_not_supported.tn|decompose3p1_from_metric requires symmetric metric components"
)

INITIAL_DATA_VALID_TESTS=(
  tests/semantic/initial_data/offdiag_metric.tn
)

GR_FIXTURES=(
  tests/fixtures/gr/schwarzschild_2d.tn
  tests/fixtures/gr/schwarzschild_3d.tn
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
echo " RUN INTERNAL UNIT TESTS"
echo "=============================="
"$UNIT_BIN"
echo

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
echo " RUN SEMANTIC EINSTEIN TESTS"
echo "=============================="

for f in "${SEMANTIC_EINSTEIN_VALID_TESTS[@]}"; do
  echo "[SEMANTIC OK EXPECTED] $f"
  "$BIN" --validate "$f" > /dev/null
done

for entry in "${SEMANTIC_EINSTEIN_ERROR_TESTS[@]}"; do
  f=${entry%%|*}
  msg=${entry#*|}
  echo "[SEMANTIC FAIL EXPECTED] $f"
  TMP_ERR=$(mktemp)
  if "$BIN" --validate "$f" > "$TMP_ERR" 2>&1; then
    echo "ERROR: $f was expected to fail but passed"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  if ! grep -q "$msg" "$TMP_ERR"; then
    echo "ERROR: expected semantic diagnostic '$msg' in $f"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
done

echo
echo "=============================="
echo " RUN SEMANTIC DIFF TESTS"
echo "=============================="

for f in "${SEMANTIC_DIFF_VALID_TESTS[@]}"; do
  echo "[SEMANTIC OK EXPECTED] $f"
  "$BIN" --validate "$f" > /dev/null
done

for entry in "${SEMANTIC_DIFF_ERROR_TESTS[@]}"; do
  f=${entry%%|*}
  msg=${entry#*|}
  echo "[SEMANTIC FAIL EXPECTED] $f"
  TMP_ERR=$(mktemp)
  if "$BIN" --validate "$f" > "$TMP_ERR" 2>&1; then
    echo "ERROR: $f was expected to fail but passed"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  if ! grep -q "$msg" "$TMP_ERR"; then
    echo "ERROR: expected diff diagnostic '$msg' in $f"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
done

echo
echo "=============================="
echo " RUN GR FIXTURE CHECKS"
echo "=============================="

for f in "${GR_FIXTURES[@]}"; do
  echo "[GR OK EXPECTED] $f"
  "$BIN" --validate "$f" > /dev/null
  OUT_FILE="$OUT/$(basename "$f").backend.txt"
  "$BIN" --dump-backend-expr "$f" > "$OUT_FILE"
  if ! grep -q "contraction(" "$OUT_FILE"; then
    echo "ERROR: expected explicit contraction op in backend IR for $f"
    exit 1
  fi
  if ! grep -q "partial_" "$OUT_FILE"; then
    echo "ERROR: expected explicit partial derivative op in backend IR for $f"
    exit 1
  fi
done

echo
echo "=============================="
echo " RUN INITIAL DATA VALID TESTS"
echo "=============================="

for f in "${INITIAL_DATA_VALID_TESTS[@]}"; do
  echo "[INITIAL_DATA OK EXPECTED] $f"
  "$BIN" "${PIPELINE_BASE[@]}" "$f" > /dev/null
done

echo
echo "=============================="
echo " RUN INITIAL DATA ERROR TESTS"
echo "=============================="

for entry in "${INITIAL_DATA_ERROR_TESTS[@]}"; do
  f=${entry%%|*}
  msg=${entry#*|}
  echo "[INITIAL_DATA FAIL EXPECTED] $f"
  TMP_ERR=$(mktemp)
  if "$BIN" --validate "$f" > "$TMP_ERR" 2>&1; then
    echo "ERROR: $f was expected to fail but passed"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  if ! grep -q "$msg" "$TMP_ERR"; then
    echo "ERROR: expected initial_data diagnostic '$msg' in $f"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
done

echo
echo "=============================="
echo " RUN INITIAL DATA MLIR ERROR TESTS"
echo "=============================="

for entry in "${INITIAL_DATA_MLIR_ERROR_TESTS[@]}"; do
  f=${entry%%|*}
  msg=${entry#*|}
  echo "[INITIAL_DATA MLIR FAIL EXPECTED] $f"
  TMP_ERR=$(mktemp)
  if "$BIN" "${PIPELINE_BASE[@]}" "$f" > "$TMP_ERR" 2>&1; then
    echo "ERROR: $f was expected to fail during MLIR emission"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  if ! grep -q "$msg" "$TMP_ERR"; then
    echo "ERROR: expected MLIR initial_data diagnostic '$msg' in $f"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
done

echo
echo "=============================="
echo " RUN SCHWARZSCHILD 3+1 MLIR CHECK"
echo "=============================="

SPLIT_FIXTURE=tests/fixtures/gr/schwarzschild_3d.tn
SPLIT_OUT="$OUT/schwarzschild_3d_split3p1.mlir"
"$BIN" "${PIPELINE_BASE[@]}" "$SPLIT_FIXTURE" > "$SPLIT_OUT"

if rg -q "tensorium\\.split3p1" "$SPLIT_OUT"; then
  echo "ERROR: unexpected legacy tensorium.split3p1 op in $SPLIT_FIXTURE"
  exit 1
fi
if ! rg -q "func\\.func @tensorium_init" "$SPLIT_OUT"; then
  echo "ERROR: expected @tensorium_init function in $SPLIT_FIXTURE output"
  exit 1
fi
if ! rg -q "func\\.func @tensorium_rhs" "$SPLIT_OUT"; then
  echo "ERROR: expected @tensorium_rhs function in $SPLIT_FIXTURE output"
  exit 1
fi
if ! rg -q "call @tensorium_init" "$SPLIT_OUT"; then
  echo "ERROR: expected @tensorium_entry to call @tensorium_init"
  exit 1
fi
if ! rg -q "call @tensorium_rhs" "$SPLIT_OUT"; then
  echo "ERROR: expected @tensorium_entry to call @tensorium_rhs"
  exit 1
fi

INIT_FUNC_OUT="$OUT/schwarzschild_3d_init.func.mlir"
RHS_FUNC_OUT="$OUT/schwarzschild_3d_rhs.func.mlir"
ENTRY_FUNC_OUT="$OUT/schwarzschild_3d_entry.func.mlir"
awk '
  /^[[:space:]]*func\.func @tensorium_init\(/ {in_fn=1}
  in_fn {print}
  in_fn && /^[[:space:]]*}/ {exit}
' "$SPLIT_OUT" > "$INIT_FUNC_OUT"
awk '
  /^[[:space:]]*func\.func @tensorium_rhs\(/ {in_fn=1}
  in_fn {print}
  in_fn && /^[[:space:]]*}/ {exit}
' "$SPLIT_OUT" > "$RHS_FUNC_OUT"
awk '
  /^[[:space:]]*func\.func @tensorium_entry\(/ {in_fn=1}
  in_fn {print}
  in_fn && /^[[:space:]]*}/ {exit}
' "$SPLIT_OUT" > "$ENTRY_FUNC_OUT"

INIT_SIG_LINE=$(rg -m1 "^[[:space:]]*func\\.func @tensorium_init\\(" "$SPLIT_OUT" || true)
RHS_SIG_LINE=$(rg -m1 "^[[:space:]]*func\\.func @tensorium_rhs\\(" "$SPLIT_OUT" || true)
if [[ -z "$INIT_SIG_LINE" || -z "$RHS_SIG_LINE" ]]; then
  echo "ERROR: could not extract init/rhs function signatures"
  exit 1
fi
INIT_ARG_LIST=$(echo "$INIT_SIG_LINE" | sed -E 's/^[^(]*\(([^)]*)\).*/\1/')
RHS_ARG_LIST=$(echo "$RHS_SIG_LINE" | sed -E 's/^[^(]*\(([^)]*)\).*/\1/')
INIT_ARGC=$(echo "$INIT_ARG_LIST" | rg -o "%arg[0-9]+:" | wc -l | tr -d ' ')
RHS_ARGC=$(echo "$RHS_ARG_LIST" | rg -o "%arg[0-9]+:" | wc -l | tr -d ' ')
if [[ "$INIT_ARGC" -ne 3 ]]; then
  echo "ERROR: expected @tensorium_init to take 3 args (alpha/gamma/gammaU), got $INIT_ARGC"
  echo "$INIT_SIG_LINE"
  exit 1
fi
if [[ "$RHS_ARGC" -ne 6 ]]; then
  echo "ERROR: expected @tensorium_rhs to take 6 args (phi/alpha/gamma/gammaU/H/K), got $RHS_ARGC"
  echo "$RHS_SIG_LINE"
  exit 1
fi

ENTRY_CALL_LINES=$(rg "^[[:space:]]*call @" "$ENTRY_FUNC_OUT" || true)
ENTRY_CALL_COUNT=$(echo "$ENTRY_CALL_LINES" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')
if [[ "$ENTRY_CALL_COUNT" -ne 2 ]]; then
  echo "ERROR: @tensorium_entry must contain exactly 2 calls (init then rhs)"
  cat "$ENTRY_FUNC_OUT"
  exit 1
fi
ENTRY_CALL_1=$(echo "$ENTRY_CALL_LINES" | sed -n '1p')
ENTRY_CALL_2=$(echo "$ENTRY_CALL_LINES" | sed -n '2p')
if [[ "$ENTRY_CALL_1" != *"call @tensorium_init"* || "$ENTRY_CALL_2" != *"call @tensorium_rhs"* ]]; then
  echo "ERROR: @tensorium_entry call order must be init then rhs"
  cat "$ENTRY_FUNC_OUT"
  exit 1
fi
if rg -q "tensorium\\.metric4|tensorium\\.decompose3p1_from_metric|tensorium\\.init3p1|tensorium\\.assign|tensorium\\.dt_assign" "$ENTRY_FUNC_OUT"; then
  echo "ERROR: @tensorium_entry must not contain tensorium compute ops"
  exit 1
fi

if ! rg -q "tensorium\\.metric4" "$INIT_FUNC_OUT"; then
  echo "ERROR: expected tensorium.metric4 op in @tensorium_init"
  exit 1
fi
if ! rg -q "tensorium\\.decompose3p1_from_metric" "$INIT_FUNC_OUT"; then
  echo "ERROR: expected tensorium.decompose3p1_from_metric op in @tensorium_init"
  exit 1
fi
if ! rg -q "tensorium\\.init3p1" "$INIT_FUNC_OUT"; then
  echo "ERROR: expected tensorium.init3p1 op in @tensorium_init"
  exit 1
fi
if ! rg -q "tensorium\\.assign" "$INIT_FUNC_OUT"; then
  echo "ERROR: expected tensorium.assign stores in @tensorium_init"
  exit 1
fi
INIT_ASSIGN_COUNT=$(rg -c "\"tensorium\\.assign\"\\(" "$INIT_FUNC_OUT")
if [[ "$INIT_ASSIGN_COUNT" -ne 3 ]]; then
  echo "ERROR: expected exactly 3 tensorium.assign stores in @tensorium_init, got $INIT_ASSIGN_COUNT"
  exit 1
fi
if rg -q "tensorium\\.dt_assign" "$INIT_FUNC_OUT"; then
  echo "ERROR: @tensorium_init must not contain tensorium.dt_assign"
  exit 1
fi
if ! rg -q "tensorium\\.coord" "$INIT_FUNC_OUT"; then
  echo "ERROR: expected coordinate ops in @tensorium_init"
  exit 1
fi
if ! rg -q "tensorium\\.param" "$INIT_FUNC_OUT"; then
  echo "ERROR: expected parameter ops in @tensorium_init"
  exit 1
fi
if ! rg -q "\"tensorium.sin\"" "$INIT_FUNC_OUT"; then
  echo "ERROR: expected sin op in Schwarzschild metric components"
  exit 1
fi
if rg -q "alpha_expr|gamma_diag|components = \\[" "$SPLIT_OUT"; then
  echo "ERROR: detected legacy string-encoded initial_data attrs in MLIR output"
  exit 1
fi
if rg -q "tensorium\\.metric4|tensorium\\.decompose3p1_from_metric|tensorium\\.init3p1" "$RHS_FUNC_OUT"; then
  echo "ERROR: @tensorium_rhs must not contain metric/decompose/init ops"
  exit 1
fi
if rg -q "tensorium\\.assign" "$RHS_FUNC_OUT"; then
  echo "ERROR: @tensorium_rhs must not contain tensorium.assign"
  exit 1
fi
if ! rg -q "tensorium\\.dt_assign" "$RHS_FUNC_OUT"; then
  echo "ERROR: expected tensorium.dt_assign ops in @tensorium_rhs"
  exit 1
fi
RHS_DT_COUNT=$(rg -c "\"tensorium\\.dt_assign\"\\(" "$RHS_FUNC_OUT")
RHS_DT_SCALAR_COUNT=$(rg -c "\"tensorium\\.dt_assign\"\\([^\\n]*indices = \\[\\]" "$RHS_FUNC_OUT")
RHS_DT_TENSOR_COUNT=$(rg -c "\"tensorium\\.dt_assign\"\\([^\\n]*indices = \\[\"i\", \"j\"\\]" "$RHS_FUNC_OUT")
if [[ "$RHS_DT_COUNT" -ne 2 || "$RHS_DT_SCALAR_COUNT" -ne 1 || "$RHS_DT_TENSOR_COUNT" -ne 1 ]]; then
  echo "ERROR: expected @tensorium_rhs dt_assign only for H (scalar) and K (i,j); got total=$RHS_DT_COUNT scalar=$RHS_DT_SCALAR_COUNT tensor=$RHS_DT_TENSOR_COUNT"
  exit 1
fi

REF_LINE=$(rg -m1 "tensorium\\.ref[[:space:]]+%arg[0-9]+[[:space:]]+\\{indices = \\[\"i\", \"j\"\\], kind = \"field\"\\}.*!tensorium\\.field<f64, 2, 0>[[:space:]]*->[[:space:]]*!tensorium\\.field<f64, 2, 0>|\"tensorium\\.ref\"\\(%arg[0-9]+\\).*!tensorium\\.field<f64, 2, 0>[[:space:]]*->[[:space:]]*!tensorium\\.field<f64, 2, 0>" "$RHS_FUNC_OUT" || true)
if [[ -z "$REF_LINE" ]]; then
  echo "ERROR: expected a tensorium.ref from gammaU field in @tensorium_rhs"
  exit 1
fi
REF_GAMMAU=$(echo "$REF_LINE" | sed -E 's/^([[:space:]]*%[A-Za-z0-9_$.]+).*/\1/' | xargs)
if [[ -z "$REF_GAMMAU" ]]; then
  echo "ERROR: could not extract ref result for gammaU use-def chain in @tensorium_rhs"
  echo "$REF_LINE"
  exit 1
fi
if ! rg -q "\"tensorium\\.mul\"\\(${REF_GAMMAU},|tensorium\\.mul[[:space:]]+${REF_GAMMAU}," "$RHS_FUNC_OUT"; then
  echo "ERROR: expected RHS contraction to use gammaU field ref"
  echo "ref gammaU value: $REF_GAMMAU"
  exit 1
fi

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
