#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

tensorium_generated_spectral_smoke \
  "generated-spectral-york-lichnerowicz" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_york_lichnerowicz_constraint_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_york_lichnerowicz_solve.cpp" \
  "tensorium_generated_spectral_york_lichnerowicz_solve" \
  header "tensorium_spectral_residual_H" \
  "error: expected York Lichnerowicz spectral residual symbol" \
  header "tensorium_spectral_residual_grid_H" \
  "error: expected York Lichnerowicz spectral grid residual symbol" \
  header "TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT 1" \
  "error: expected one generated spectral residual system" \
  header "SpectralYorkLichnerowiczConstraint3D" \
  "error: expected York Lichnerowicz spectral system name" \
  header "matter_coeff" \
  "error: expected York Lichnerowicz matter coefficient parameter" \
  header "modified_coeff" \
  "error: expected York Lichnerowicz modified coefficient parameter" \
  header "modSource" \
  "error: expected York Lichnerowicz modified source auxiliary" \
  header "radial" \
  "error: expected York Lichnerowicz radial Robin boundary descriptor" \
  header "radius" \
  "error: expected York Lichnerowicz radius Robin coefficient descriptor" \
  llvm "define void @tensorium_spectral_residual_grid_H" \
  "error: expected York Lichnerowicz spectral grid residual LLVM definition"
