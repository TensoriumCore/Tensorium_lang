#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

tensorium_generated_spectral_smoke \
  "generated-spectral-york-momentum" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_york_momentum_split_constraint_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_york_momentum_split_solve.cpp" \
  "tensorium_generated_spectral_york_momentum_split_solve" \
  header "tensorium_spectral_residual_H" \
  "error: expected York momentum Hamiltonian spectral residual symbol" \
  header "tensorium_spectral_residual_M1" \
  "error: expected York momentum M1 spectral residual symbol" \
  header "tensorium_spectral_residual_M2" \
  "error: expected York momentum M2 spectral residual symbol" \
  header "tensorium_spectral_residual_M3" \
  "error: expected York momentum M3 spectral residual symbol" \
  header "TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT 1" \
  "error: expected one generated York momentum spectral system" \
  header "SpectralYorkMomentumSplitConstraint3D" \
  "error: expected York momentum spectral system name" \
  header "momentum_psi_coupling" \
  "error: expected York momentum coupling parameter" \
  header "vector_mass" \
  "error: expected York momentum vector mass parameter" \
  llvm "define void @tensorium_spectral_residual_grid_H" \
  "error: expected York momentum Hamiltonian spectral grid LLVM definition" \
  llvm "define void @tensorium_spectral_residual_grid_M1" \
  "error: expected York momentum M1 spectral grid LLVM definition" \
  llvm "define void @tensorium_spectral_residual_grid_M2" \
  "error: expected York momentum M2 spectral grid LLVM definition" \
  llvm "define void @tensorium_spectral_residual_grid_M3" \
  "error: expected York momentum M3 spectral grid LLVM definition"
