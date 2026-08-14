#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-/tmp/tensorium-qc0-convergence}"
RESOLUTIONS="${TP_CONVERGENCE_RESOLUTIONS:-14x14x8 16x16x8 20x20x10 24x24x12}"
SLICE_N="${TP_CONVERGENCE_SLICE_N:-17}"
SUMMARY_PATH="$OUTPUT_DIR/summary.csv"

json_number() {
  local key="$1"
  local path="$2"
  sed -nE "s/^[[:space:]]*\"${key}\":[[:space:]]*([^,]+),?$/\1/p" "$path"
}

json_array() {
  local key="$1"
  local path="$2"
  sed -nE "s/^[[:space:]]*\"${key}\":[[:space:]]*\[([^]]+)\],?$/\1/p" "$path"
}

mkdir -p "$OUTPUT_DIR"
printf '%s\n' \
  'resolution,points,status,newton_steps,linear_iterations,solve_wall_seconds,projected_residual_l2,projected_residual_max,raw_residual_l2,raw_residual_max,projected_out_residual_l2,projected_out_residual_max,projected_out_max_A,projected_out_max_B,projected_out_max_phi,adm_energy,puncture_mass_plus,puncture_mass_minus,axis_regularity_error' \
  >"$SUMMARY_PATH"

failed=0
continuation=""
for requested in $RESOLUTIONS; do
  resolution="${requested//X/x}"
  if [[ ! "$resolution" =~ ^[0-9]+x[0-9]+x[0-9]+$ ]]; then
    echo "invalid convergence resolution '$requested'; expected N1xN2xN3" >&2
    exit 2
  fi
  IFS=x read -r n1 n2 n3 <<<"$resolution"
  points=$((n1 * n2 * n3))
  stem="$OUTPUT_DIR/qc0_${resolution}"
  csv_path="$stem.csv"
  log_path="$stem.log"

  echo "[convergence] running QC0 at $resolution ($points points)"
  if [[ -n "$continuation" ]]; then
    continuation="$continuation;$resolution"
  else
    continuation="$resolution"
  fi
  if TP_CONTINUATION="$continuation" TP_SLICE_N="$SLICE_N" \
      "$ROOT_DIR/run_two_puncture_qc0.sh" "$csv_path" \
      2>&1 | tee "$log_path"; then
    metadata_path="$csv_path.json"
    masses="$(json_array puncture_adm_masses "$metadata_path")"
    mass_plus="${masses%%,*}"
    mass_minus="${masses#*,}"
    mass_plus="${mass_plus//[[:space:]]/}"
    mass_minus="${mass_minus//[[:space:]]/}"
    rejected_location="$(json_array projected_out_residual_max_logical "$metadata_path")"
    IFS=, read -r rejected_a rejected_b rejected_phi \
      <<<"$rejected_location"
    rejected_a="${rejected_a//[[:space:]]/}"
    rejected_b="${rejected_b//[[:space:]]/}"
    rejected_phi="${rejected_phi//[[:space:]]/}"
    printf '%s\n' \
      "$resolution,$points,converged,$(json_number newton_steps "$metadata_path"),$(json_number linear_iterations "$metadata_path"),$(json_number solve_wall_seconds "$metadata_path"),$(json_number projected_residual_l2 "$metadata_path"),$(json_number projected_residual_max "$metadata_path"),$(json_number raw_residual_l2 "$metadata_path"),$(json_number raw_residual_max "$metadata_path"),$(json_number projected_out_residual_l2 "$metadata_path"),$(json_number projected_out_residual_max "$metadata_path"),$rejected_a,$rejected_b,$rejected_phi,$(json_number adm_energy "$metadata_path"),$mass_plus,$mass_minus,$(json_number axis_regularity_error "$metadata_path")" \
      >>"$SUMMARY_PATH"
  else
    failed=1
    printf '%s\n' "$resolution,$points,failed,,,,,,,,,,,,,,,," \
      >>"$SUMMARY_PATH"
  fi
done

echo "[convergence] summary = $SUMMARY_PATH"
if command -v column >/dev/null 2>&1; then
  column -s, -t "$SUMMARY_PATH"
else
  sed -n '1,$p' "$SUMMARY_PATH"
fi
exit "$failed"
