#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tools/dev/generated_spectral_smoke_common.sh"

tensorium_generated_spectral_smoke \
  "generated-spectral-york-vector" \
  "$ROOT_DIR/tests/fixtures/elliptic/spectral_york_momentum_vector_constraint_3d.tn" \
  "$ROOT_DIR/tools/dev/runtime_generated_spectral_york_momentum_vector_solve.cpp" \
  "tensorium_generated_spectral_york_momentum_vector_solve" \
  header "tensorium_spectral_residual_H" \
  "error: expected York vector Hamiltonian spectral residual symbol" \
  header "tensorium_spectral_residual_M1" \
  "error: expected York vector M1 spectral residual symbol" \
  header "tensorium_spectral_residual_M2" \
  "error: expected York vector M2 spectral residual symbol" \
  header "tensorium_spectral_residual_M3" \
  "error: expected York vector M3 spectral residual symbol" \
  header "SpectralYorkMomentumVectorConstraint3D" \
  "error: expected York vector spectral system name" \
  header "W1" \
  "error: expected lowered W1 unknown descriptor" \
  header "J1" \
  "error: expected lowered J1 auxiliary descriptor" \
  llvm "define void @tensorium_spectral_residual_grid_H" \
  "error: expected York vector Hamiltonian spectral grid LLVM definition" \
  llvm "define void @tensorium_spectral_residual_grid_M1" \
  "error: expected York vector M1 spectral grid LLVM definition" \
  llvm "define void @tensorium_spectral_residual_grid_M2" \
  "error: expected York vector M2 spectral grid LLVM definition" \
  llvm "define void @tensorium_spectral_residual_grid_M3" \
  "error: expected York vector M3 spectral grid LLVM definition"
