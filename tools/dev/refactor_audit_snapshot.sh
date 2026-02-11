#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "== Tensorium Refactor Snapshot =="
echo "repo: $ROOT_DIR"
echo "commit: $(git rev-parse --short HEAD)"
echo

echo "== CMake Targets (from existing build dir) =="
if [[ -d build ]]; then
  cmake --build build --target help | sed -n '1,200p'
else
  echo "build/ does not exist. Run cmake -S . -B build first."
fi
echo

echo "== Declared CMake Targets =="
rg -n "add_library\\(|add_executable\\(|add_public_tablegen_target\\(" \
  CMakeLists.txt lib/CMakeLists.txt tools/CMakeLists.txt \
  tools/driver/CMakeLists.txt tools/Tester/CMakeLists.txt
echo

echo "== Source Files Not Present In lib/CMakeLists.txt =="
comm -23 \
  <(find lib -name '*.cpp' | sort) \
  <(rg -o "[A-Za-z0-9_./-]+\\.cpp" lib/CMakeLists.txt | sort | sed 's#^#lib/#')
