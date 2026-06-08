#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

tensorium_generated_spectral_smoke \
  "generated-spectral-system" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_two_field_system_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_system_residual.cpp" \
  "tensorium_generated_spectral_system_residual" \
  header "tensorium_spectral_residual_Hu" \
  "error: expected Hu spectral residual symbol" \
  header "tensorium_spectral_residual_Hv" \
  "error: expected Hv spectral residual symbol" \
  header "tensorium_spectral_residual_systems" \
  "error: expected spectral residual system descriptor table" \
  header "TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT 1" \
  "error: expected one generated spectral residual system" \
  llvm "tensorium_spectral_residual_grid_Hu" \
  "error: expected Hu spectral grid residual LLVM definition" \
  llvm "tensorium_spectral_residual_grid_Hv" \
  "error: expected Hv spectral grid residual LLVM definition"
