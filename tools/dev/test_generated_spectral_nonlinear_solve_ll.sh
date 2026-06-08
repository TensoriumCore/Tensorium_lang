#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

tensorium_generated_spectral_smoke \
  "generated-spectral-nonlinear" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_hamiltonian_toy_nonlinear_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_nonlinear_solve.cpp" \
  "tensorium_generated_spectral_nonlinear_solve" \
  header "tensorium_spectral_residual_H" \
  "error: expected nonlinear spectral residual symbol" \
  header "tensorium_spectral_residual_grid_H" \
  "error: expected nonlinear spectral grid residual symbol" \
  header "TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT 1" \
  "error: expected one generated spectral residual system" \
  header "SpectralHamiltonianToyNonlinear3D" \
  "error: expected nonlinear spectral system name" \
  llvm "define void @tensorium_spectral_residual_grid_H" \
  "error: expected nonlinear spectral grid residual LLVM definition"
