#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

tensorium_generated_spectral_smoke \
  "generated-spectral-poisson-dirichlet" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_poisson_dirichlet_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_poisson_dirichlet_solve.cpp" \
  "tensorium_generated_spectral_poisson_dirichlet_solve" \
  header "tensorium_spectral_residual_H" \
  "error: expected Poisson spectral residual symbol" \
  header "tensorium_spectral_residual_grid_H" \
  "error: expected Poisson spectral grid residual symbol" \
  header "TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT 1" \
  "error: expected one generated spectral residual system" \
  header "SpectralPoissonDirichlet3D" \
  "error: expected Poisson Dirichlet spectral system name" \
  header "tensorium_spectral_boundary_condition_desc" \
  "error: expected spectral boundary descriptor type" \
  header "lower_x1" \
  "error: expected generated lower_x1 boundary metadata" \
  llvm "define void @tensorium_spectral_residual_grid_H" \
  "error: expected Poisson spectral grid residual LLVM definition"
