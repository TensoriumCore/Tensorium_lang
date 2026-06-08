#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

export TENSORIUM_BY_CONTINUATION_STAGES="$ROOT_DIR/tools/dev/bowen_york_puncture_continuation_stages.txt"

tensorium_generated_spectral_smoke \
  "generated-spectral-bowen-york-puncture-continuation" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_bowen_york_regularized_puncture_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_bowen_york_puncture_continuation.cpp" \
  "tensorium_generated_spectral_bowen_york_puncture_continuation" \
  header "tensorium_spectral_residual_H" \
  "error: expected Bowen-York spectral residual symbol" \
  header "tensorium_spectral_residual_grid_H" \
  "error: expected Bowen-York spectral grid residual symbol" \
  header "TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT 1" \
  "error: expected one generated spectral residual system" \
  header "SpectralBowenYorkRegularizedPuncture3D" \
  "error: expected Bowen-York puncture spectral system name" \
  header "radial" \
  "error: expected Bowen-York radial Robin boundary descriptor" \
  header "radius" \
  "error: expected Bowen-York radius Robin coefficient descriptor" \
  llvm "define void @tensorium_spectral_residual_grid_H" \
  "error: expected Bowen-York spectral grid residual LLVM definition"
