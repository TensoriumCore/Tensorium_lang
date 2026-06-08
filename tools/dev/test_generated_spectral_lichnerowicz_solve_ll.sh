#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

tensorium_generated_spectral_smoke \
  "generated-spectral-lichnerowicz" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_lichnerowicz_manufactured_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_lichnerowicz_solve.cpp" \
  "tensorium_generated_spectral_lichnerowicz_solve" \
  header "tensorium_spectral_residual_H" \
  "error: expected Lichnerowicz spectral residual symbol" \
  header "tensorium_spectral_residual_grid_H" \
  "error: expected Lichnerowicz spectral grid residual symbol" \
  header "TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT 1" \
  "error: expected one generated spectral residual system" \
  header "SpectralLichnerowiczManufactured3D" \
  "error: expected Lichnerowicz spectral system name" \
  llvm "define void @tensorium_spectral_residual_grid_H" \
  "error: expected Lichnerowicz spectral grid residual LLVM definition"
