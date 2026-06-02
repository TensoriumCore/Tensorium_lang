#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER_SRC="$ROOT_DIR/tools/dev/runtime_spectral_initial_data_residual.cpp"
CXX_BIN="${CXX:-c++}"

EXE_PATH="/tmp/tensorium_runtime_spectral_initial_data_runner"

if [[ ! -f "$RUNNER_SRC" ]]; then
  echo "error: missing runner source: $RUNNER_SRC" >&2
  exit 2
fi

echo "[spectral-initial-data] compiling runtime spectral runner"
"$CXX_BIN" -O0 -std=c++20 -I "$ROOT_DIR/include" "$RUNNER_SRC" -lm \
  -o "$EXE_PATH"

echo "[spectral-initial-data] running runtime spectral executable"
"$EXE_PATH"
