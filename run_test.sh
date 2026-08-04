
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
  tests/semantic/diff/05_laplacian_executable_not_supported.tn
  tests/50_large_tensor_mix.tn
  tests/51_large_tensor_flux.tn
  tests/56_metric_decl_ok.tn
  tests/58_nabla_scalar.tn
  tests/59_nabla_expand.tn
  tests/31_temp_valid_scalar.tn
  tests/semantic/robustness/04_explicit_parameter_declaration.tn
  tests/semantic/robustness/07_cpp_line_comment.tn
  tests/60_valid_index_offset.tn
  tests/61_valid_field_mixed_tensor.tn
  tests/62_valid_trace_builtin.tn
  tests/63_valid_laplacian_builtin.tn
  tests/64_valid_nabla_contravariant_scalar.tn
  tests/68_valid_nabla_covector.tn
  tests/69_valid_nabla_mixed_tensor.tn
  tests/70_valid_nabla_contravariant_vector.tn
  tests/73_valid_nabla_contravariant_covector.tn
  tests/74_valid_nabla_contravariant_mixed_tensor.tn
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
  tests/58_error_non_dt_field_assign.tn
  tests/65_error_nabla_contravariant_missing_inverse_metric.tn
  tests/66_error_dimension_non_integer.tn
  tests/67_error_spatial_order_non_integer.tn
  tests/71_error_nabla_tensor_missing_metric.tn
  tests/72_error_nabla_tensor_requires_indices.tn
)

SEMANTIC_EINSTEIN_VALID_TESTS=(
  tests/semantic/einstein/01_valid_contraction.tn
  tests/semantic/einstein/04_valid_two_sums.tn
  tests/semantic/einstein/canon/01_contract_ij.tn
  tests/semantic/einstein/canon/02_contract_mn.tn
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
  tests/semantic/diff/05_laplacian_executable_not_supported.tn
)

SEMANTIC_DIFF_ERROR_TESTS=(
  "tests/semantic/diff/04_covariant_without_gamma_error.tn|nabla on non-scalar tensor requires either"
)

SEMANTIC_SIMULATION_ERROR_TESTS=(
  "tests/semantic/simulation/01_missing_block.tn|E1001: missing simulation block in executable mode"
  "tests/semantic/simulation/02_missing_time_block.tn|simulation block requires 'time { dt = ... integrator = ... }'"
  "tests/semantic/simulation/03_missing_spatial_block.tn|simulation block requires 'spatial { scheme = ... derivative = ... order = ... }'"
  "tests/semantic/simulation/04_time_missing_integrator.tn|time block requires 'integrator = euler|rk3|rk4'"
  "tests/semantic/simulation/05_duplicate_dimension_entry.tn|duplicate 'dimension' entry in simulation block"
)

SEMANTIC_SIMULATION_SYMBOLIC_WARN_TESTS=(
  "tests/semantic/simulation/01_missing_block.tn|W1001: missing simulation block in symbolic mode"
)

SEMANTIC_ROBUSTNESS_ERROR_TESTS=(
  "tests/semantic/robustness/01_unknown_identifier_strict.tn|Unknown identifier: alph"
  "tests/semantic/robustness/02_evolution_scope_isolated.tn|Unknown identifier: tmp"
  "tests/semantic/robustness/03_field_metric_name_collision.tn|Name collision: field 'g' conflicts with metric 'g'"
  "tests/semantic/robustness/05_initial_data_unknown_parameter.tn|uses unknown identifier 'M'"
  "tests/semantic/robustness/06_temp_use_before_def_validate.tn|temporary 'K' referenced before definition"
)

INITIAL_DATA_ERROR_TESTS=(
  "tests/semantic/initial_data/01_invalid_spherical_coord.tn|uses coordinate 'x' incompatible with simulation coordinates"
  "tests/semantic/initial_data/02_symmetry_violation.tn|metric4 symmetry violation"
  "tests/semantic/initial_data/06_unsupported_builtin.tn|uses unsupported scalar function 'cos'"
)

INITIAL_DATA_MLIR_ERROR_TESTS=(
  "tests/semantic/initial_data/03_missing_gammau_binding.tn|split_3p1 does not bind gammaU"
  "tests/semantic/initial_data/04_nonsymmetric_metric_not_supported.tn|decompose3p1_from_metric requires symmetric metric components"
)

INITIAL_DATA_VALID_TESTS=(
  tests/semantic/initial_data/offdiag_metric.tn
  tests/semantic/initial_data/05_shift_metric_not_supported.tn
)

CONSTRAINT_INITIAL_DATA_VALID_TESTS=(
  tests/fixtures/gr/brill_lindquist_constraints.tn
)

CONSTRAINT_SOLVE_TESTS=(
  "tests/fixtures/gr/brill_lindquist_radial_solve.tn|49|1|1|mass=1"
  "tests/fixtures/gr/brill_lindquist_multidomain_solve.tn|42|2|1|mass=1"
  "tests/fixtures/gr/coupled_nonlinear_radial_solve.tn|22|2|5|mass=1"
  "tests/fixtures/gr/scalar_vector_radial_solve.tn|22|2|3|mass=1"
  "tests/fixtures/gr/tensor_contraction_radial_solve.tn|7|1|2|mass=1"
  "tests/fixtures/gr/covariant_geometry_radial_solve.tn|7|1|1|mass=1"
  "tests/fixtures/gr/ctt_radial_vacuum_solve.tn|50|2|4|amplitude=0.2"
)

GR_FIXTURES=(
  tests/fixtures/gr/schwarzschild_2d.tn
  tests/fixtures/gr/schwarzschild_3d.tn
  tests/fixtures/gr/schwarzschild_christoffel_3d.tn
  tests/fixtures/gr/schwarzschild_ricci_3d.tn
  tests/fixtures/gr/schwarzschild_hamiltonian_3d.tn
  tests/fixtures/gr/minkowski_ricci_3d.tn
  tests/fixtures/gr/reissner_nordstrom_3d.tn
  tests/fixtures/gr/reissner_nordstrom_christoffel_3d.tn
  tests/fixtures/gr/spatial_offdiag_3d.tn
  tests/fixtures/gr/kerr_like_3d.tn
  tests/fixtures/gr/kerr_like_christoffel_3d.tn
  tests/fixtures/gr/hartle_thorne_slow_rotation.tn
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
  "$BIN" --mlir-best-effort "${PIPELINE_DISS[@]}" --dump-mlir "$f" \
    > "$OUT/$(basename "$f").mlir"
done

PRIMARY_MLIR="$OUT/$(basename ${VALID_TESTS[0]}).mlir"
if ! grep -q "tensorium.field" "$PRIMARY_MLIR"; then
  echo "ERROR: expected tensorium.field types in $PRIMARY_MLIR"
  exit 1
fi

LAPLACIAN_MLIR="$OUT/$(basename tests/semantic/diff/05_laplacian_executable_not_supported.tn).mlir"
if ! grep -q "tensorium.contract" "$LAPLACIAN_MLIR"; then
  echo "ERROR: expected tensorium.contract in laplacian lowering output"
  exit 1
fi

echo
echo "=============================="
echo " TEST EMIT ARTIFACT FLAGS"
echo "=============================="
EMIT_MLIR_OUT="$OUT/emit_scalar.mlir"
EMIT_LLVM_OUT="$OUT/emit_scalar.ll"
EMIT_O3_LLVM_OUT="$OUT/emit_o3_schwarzschild.ll"
EMIT_O3_HOST_HEADER_OUT="$OUT/emit_o3_schwarzschild_host.h"
EMIT_O3_PASS_OPTS_LLVM_OUT="$OUT/emit_o3_pass_options_schwarzschild.ll"
"$BIN" --emit-mlir "$EMIT_MLIR_OUT" tests/01_scalar_minimal.tn > /dev/null
if [[ ! -s "$EMIT_MLIR_OUT" ]]; then
  echo "ERROR: --emit-mlir did not produce output file"
  exit 1
fi
if ! grep -q "module" "$EMIT_MLIR_OUT"; then
  echo "ERROR: --emit-mlir output does not look like MLIR module"
  exit 1
fi
"$BIN" --emit-llvm "$EMIT_LLVM_OUT" tests/01_scalar_minimal.tn > /dev/null
if [[ ! -s "$EMIT_LLVM_OUT" ]]; then
  echo "ERROR: --emit-llvm did not produce output file"
  exit 1
fi
if ! grep -q "define" "$EMIT_LLVM_OUT"; then
  echo "ERROR: --emit-llvm output does not look like LLVM IR"
  exit 1
fi
"$BIN" -O3 --emit-llvm "$EMIT_O3_LLVM_OUT" tests/fixtures/gr/schwarzschild_3d.tn > /dev/null
if [[ ! -s "$EMIT_O3_LLVM_OUT" ]]; then
  echo "ERROR: -O3 --emit-llvm did not produce output file"
  exit 1
fi
if ! grep -q "tensorium_init_grid_affine" "$EMIT_O3_LLVM_OUT"; then
  echo "ERROR: -O3 --emit-llvm did not apply final grid lowering preset"
  exit 1
fi
"$BIN" -O3 --emit-host-header "$EMIT_O3_HOST_HEADER_OUT" tests/fixtures/gr/schwarzschild_3d.tn > /dev/null
if [[ ! -s "$EMIT_O3_HOST_HEADER_OUT" ]]; then
  echo "ERROR: -O3 --emit-host-header did not produce output file"
  exit 1
fi
if ! grep -q "tensorium_call_init_grid_affine" "$EMIT_O3_HOST_HEADER_OUT"; then
  echo "ERROR: --emit-host-header output does not expose host wrappers"
  exit 1
fi
"$BIN" -O3 --tensorium-dx 0.25 --tensorium-stencil-order 4 \
  --tensorium-dissipation --tensorium-dissipation-strength 0.05 \
  --emit-llvm "$EMIT_O3_PASS_OPTS_LLVM_OUT" \
  tests/fixtures/gr/schwarzschild_3d.tn > /dev/null
if [[ ! -s "$EMIT_O3_PASS_OPTS_LLVM_OUT" ]]; then
  echo "ERROR: -O3 with pass options did not produce LLVM IR"
  exit 1
fi

echo
echo "=============================="
echo " RUN EXTERN DECL TESTS"
echo "=============================="

for f in "${EXTERN_TESTS[@]}"; do
  echo "[OK EXPECTED] $f"
  "$BIN" "$f" > /dev/null
  EXTERN_MLIR="$OUT/$(basename "$f").extern.mlir"
  "$BIN" "${PIPELINE_BASE[@]}" "$f" > "$EXTERN_MLIR"
  if ! grep -q "tensorium.extern_call" "$EXTERN_MLIR"; then
    echo "ERROR: expected tensorium.extern_call in extern MLIR lowering"
    exit 1
  fi
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
echo " RUN SEMANTIC SIMULATION TESTS"
echo "=============================="

for entry in "${SEMANTIC_SIMULATION_ERROR_TESTS[@]}"; do
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
    echo "ERROR: expected simulation diagnostic '$msg' in $f"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
done

for entry in "${SEMANTIC_SIMULATION_SYMBOLIC_WARN_TESTS[@]}"; do
  f=${entry%%|*}
  msg=${entry#*|}
  echo "[SYMBOLIC WARN EXPECTED] $f"
  TMP_ERR=$(mktemp)
  if ! "$BIN" --symbolic "$f" > "$TMP_ERR" 2>&1; then
    echo "ERROR: symbolic mode unexpectedly failed for $f"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  if ! grep -q "$msg" "$TMP_ERR"; then
    echo "ERROR: expected symbolic warning '$msg' in $f"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
done

echo
echo "=============================="
echo " RUN SEMANTIC ROBUSTNESS TESTS"
echo "=============================="

for entry in "${SEMANTIC_ROBUSTNESS_ERROR_TESTS[@]}"; do
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
    echo "ERROR: expected robustness diagnostic '$msg' in $f"
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
echo " RUN CONSTRAINT INITIAL DATA TESTS"
echo "=============================="

for f in "${CONSTRAINT_INITIAL_DATA_VALID_TESTS[@]}"; do
  echo "[CONSTRAINT INITIAL_DATA OK EXPECTED] $f"
  "$BIN" --validate "$f" > /dev/null
  OUT_FILE="$OUT/$(basename "$f").constraint.backend.txt"
  "$BIN" --dump-backend-expr "$f" > "$OUT_FILE"
  if ! grep -q "ConstraintProblem BrillLindquist" "$OUT_FILE"; then
    echo "ERROR: expected ConstraintProblemIR in backend dump for $f"
    exit 1
  fi
  if ! grep -q "psi\[unknown\]" "$OUT_FILE"; then
    echo "ERROR: expected typed constraint unknown in backend dump for $f"
    exit 1
  fi
  TMP_ERR=$(mktemp)
  if "$BIN" --dump-mlir "$f" > "$TMP_ERR" 2>&1; then
    echo "ERROR: constraint MLIR lowering was expected to fail explicitly"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  if ! grep -q "constraint problem MLIR lowering is not implemented" "$TMP_ERR"; then
    echo "ERROR: missing explicit constraint MLIR diagnostic"
    cat "$TMP_ERR"
    rm -f "$TMP_ERR"
    exit 1
  fi
  rm -f "$TMP_ERR"
done

for spec in "${CONSTRAINT_SOLVE_TESTS[@]}"; do
  IFS='|' read -r f expected_points expected_domains expected_iterations parameter <<< "$spec"
  echo "[CONSTRAINT SOLVE OK EXPECTED] $f"
  OUT_FILE="$OUT/$(basename "$f").solve.txt"
  "$BIN" --solve-constraints --param "$parameter" "$f" > "$OUT_FILE"
  if ! grep -q "converged=true iterations=$expected_iterations" "$OUT_FILE"; then
    echo "ERROR: expected $expected_iterations Newton iterations for $f"
    cat "$OUT_FILE"
    exit 1
  fi
  if ! grep -q "domains=$expected_domains" "$OUT_FILE"; then
    echo "ERROR: expected $expected_domains-domain solution for $f"
    cat "$OUT_FILE"
    exit 1
  fi
  if ! grep -q "unknown=psi points=$expected_points" "$OUT_FILE"; then
    echo "ERROR: expected $expected_points-point psi solution for $f"
    cat "$OUT_FILE"
    exit 1
  fi
  if [[ "$f" == *"scalar_vector_radial_solve.tn"* ]] &&
     ! grep -q "unknown=W points=22 components=3 values=66" "$OUT_FILE"; then
    echo "ERROR: expected three-component vector solution for $f"
    cat "$OUT_FILE"
    exit 1
  fi
  if [[ "$f" == *"tensor_contraction_radial_solve.tn"* ]] &&
     { ! grep -q "unknown=A points=7 components=6 values=42" "$OUT_FILE" ||
       ! grep -q "unknown=B points=7 components=6 values=42" "$OUT_FILE" ||
       ! grep -q "unknown=C points=7 components=9 values=63" "$OUT_FILE"; }; then
    echo "ERROR: expected symmetric and mixed tensor solutions for $f"
    cat "$OUT_FILE"
    exit 1
  fi
  if [[ "$f" == *"covariant_geometry_radial_solve.tn"* ]] &&
     { ! grep -q "unknown=T points=7 components=6 values=42" "$OUT_FILE" ||
       ! grep -q "unknown=V points=7 components=3 values=21" "$OUT_FILE"; }; then
    echo "ERROR: expected covariant geometry tensor solutions for $f"
    cat "$OUT_FILE"
    exit 1
  fi
  if [[ "$f" == *"ctt_radial_vacuum_solve.tn"* ]] &&
     ! grep -q "physical_ctt basis=flat_spherical_orthonormal_coframe points=50" "$OUT_FILE"; then
    echo "ERROR: expected reconstructed CTT physical fields for $f"
    cat "$OUT_FILE"
    exit 1
  fi
done

CTT_CSV="$OUT/ctt_radial_physical.csv"
"$BIN" --export-constraint-csv "$CTT_CSV" --param amplitude=0.2 \
  tests/fixtures/gr/ctt_radial_vacuum_solve.tn > /dev/null
if [[ $(wc -l < "$CTT_CSV") -ne 51 ]]; then
  echo "ERROR: expected header plus 50 CTT physical CSV rows"
  exit 1
fi
if ! grep -q '^domain,r,conformal_factor,radial_vector,mean_curvature,gamma_radial,gamma_tangential,k_radial,k_tangential$' "$CTT_CSV"; then
  echo "ERROR: unexpected CTT physical CSV schema"
  exit 1
fi

CTT_BSSN_LL="$OUT/ctt_bssn_handoff.ll"
"$BIN" --solve-constraints --param amplitude=0.2 \
  --emit-llvm "$CTT_BSSN_LL" \
  tests/fixtures/gr/ctt_bssn_handoff.tn > /dev/null
if ! grep -q 'define void @tensorium_rhs_grid_affine' "$CTT_BSSN_LL"; then
  echo "ERROR: expected generated BSSN grid RHS after the CTT solve"
  exit 1
fi

echo
echo "=============================="
echo " RUN INITIAL DATA PARAM LOWERING CHECK"
echo "=============================="

RN_INIT_FIXTURE=tests/fixtures/gr/reissner_nordstrom_3d.tn
RN_INIT_OUT="$OUT/reissner_nordstrom_init.mlir"
"$BIN" --tensorium-metric-lower --tensorium-init-std-lower --dump-mlir \
  "$RN_INIT_FIXTURE" > "$RN_INIT_OUT"
if ! grep -q "tensorium_init_point" "$RN_INIT_OUT"; then
  echo "ERROR: expected tensorium_init_point after init-to-std lowering"
  exit 1
fi
if ! grep -q "tensorium.init.param_names" "$RN_INIT_OUT"; then
  echo "ERROR: expected param metadata on tensorium_init_point"
  exit 1
fi
if ! grep -q "\"Q\"" "$RN_INIT_OUT"; then
  echo "ERROR: expected Q runtime parameter in init lowering metadata"
  exit 1
fi

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
echo " RUN SCHWARZSCHILD 3+1 MLIR SMOKE"
echo "=============================="

SPLIT_FIXTURE=tests/fixtures/gr/schwarzschild_3d.tn
"$BIN" "${PIPELINE_BASE[@]}" "$SPLIT_FIXTURE" > /dev/null

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
echo " RUN BSSN DEFAULT LLVM PIPELINE"
echo "=============================="

BSSN_DEFAULT_LL="$OUT/07_bssn_reduced.default.ll"
RAW_BSSN_LL="$(mktemp)"
"$BIN" --dump-llvm-ir tests/07_bssn_reduced.tn > "$RAW_BSSN_LL"
awk '
  /^\[Tensorium\]/ {exit}
  {print}
' "$RAW_BSSN_LL" > "$BSSN_DEFAULT_LL"
rm -f "$RAW_BSSN_LL"
if ! grep -q "define void @tensorium_rhs_grid_affine" "$BSSN_DEFAULT_LL"; then
  echo "ERROR: expected default LLVM pipeline to lower BSSN RHS grid kernel"
  exit 1
fi

echo
echo "=============================="
echo " RUN BSSN RHS LLVM SMOKE"
echo "=============================="

bash tools/dev/test_bssn_reduced_ll.sh

echo
echo "=============================="
echo " RUN BSSN RK2 LLVM SMOKE"
echo "=============================="

bash tools/dev/test_bssn_reduced_rk2_ll.sh

echo
echo "=============================="
echo " RUN BSSN MINIMAL LLVM SMOKE"
echo "=============================="

bash tools/dev/test_bssn_minimal_ll.sh

echo
echo "=============================="
echo " RUN COMPLETE BSSN KASNER ANALYTIC LLVM SMOKE"
echo "=============================="

bash tools/dev/test_bssn_kasner_full_ll.sh

echo
echo "=============================="
echo " RUN Z4C KASNER LLVM SMOKE"
echo "=============================="

bash tools/dev/test_z4c_kasner_ll.sh

echo
echo "=============================="
echo " RUN LLVM IR COMPILE+RUN SMOKE"
echo "=============================="

bash tools/dev/test_abi_contract_probe.sh
bash tools/dev/test_standard_metrics_init_analytic_ll.sh
bash tools/dev/test_hartle_thorne_metric_init_ll.sh
bash tools/dev/test_schwarzschild_ll.sh
bash tools/dev/test_schwarzschild_christoffel_ll.sh
bash tools/dev/test_reissner_christoffel_ll.sh
bash tools/dev/test_kerr_like_christoffel_ll.sh
bash tools/dev/test_minkowski_ricci_ll.sh
bash tools/dev/test_schwarzschild_ricci_ll.sh
bash tools/dev/test_schwarzschild_hamiltonian_ll.sh
bash tools/dev/test_schwarzschild_bssn_constraints_ll.sh
bash tools/dev/test_runtime_uniform_schwarzschild_bssn.sh
bash tools/dev/test_runtime_bssn_kasner_euler_iteration.sh
bash tools/dev/test_covariant_rank1_ll.sh
bash tools/dev/test_contravariant_all_cases_ll.sh
bash tools/dev/test_extern_scalar_ll.sh

echo
echo "=============================="
echo " RUN ELLIPTIC INITIAL-DATA SOLVER LLVM SMOKES"
echo "=============================="

bash tools/dev/test_poisson_relax_l2_ll.sh
bash tools/dev/test_poisson_source_relax_l2_ll.sh
bash tools/dev/test_hamiltonian_toy_relax_l2_ll.sh
bash tools/dev/test_bowen_york_single_puncture_relax_l2_ll.sh
bash tools/dev/test_bowen_york_single_puncture_p0_ll.sh
bash tools/dev/test_bowen_york_single_puncture_scan.sh
bash tools/dev/test_runtime_spectral_initial_data.sh
bash tools/dev/test_runtime_spectral_global_residual.sh
bash tools/dev/test_generated_spectral_residual_ll.sh
bash tools/dev/test_generated_spectral_aux_residual_ll.sh
bash tools/dev/test_generated_spectral_global_residual_ll.sh
bash tools/dev/test_generated_spectral_system_residual_ll.sh
bash tools/dev/test_generated_spectral_newton_solve_ll.sh
bash tools/dev/test_parallel_residual_grid_ll.sh

echo
echo "=============================="
echo " ALL TESTS PASSED"
echo " MLIR outputs in $OUT"
echo "=============================="
