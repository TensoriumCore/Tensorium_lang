#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

failed=0

check_forbidden_in_path() {
  local search_path="$1"
  local pattern="$2"
  local reason="$3"

  if rg -n "$pattern" "$search_path" >/tmp/tensorium_layering_check.out 2>&1; then
    echo "LAYERING ERROR: $reason"
    cat /tmp/tensorium_layering_check.out
    failed=1
  fi
}

check_forbidden_in_path \
  "include/tensorium/Backend/DomainIR.hpp" \
  "^#include\\s+\"tensorium/AST/" \
  "DomainIR must not include AST headers"

check_forbidden_in_path \
  "lib/Runtime" \
  "^#include\\s+\"tensorium/(Parse|Sema)/" \
  "Runtime layer must not depend on Parse/Sema"

check_forbidden_in_path \
  "lib/Parse" \
  "^#include\\s+\"(tensorium/(Runtime|Backend)/|tensorium_mlir/)" \
  "Parse layer must not depend on Runtime/Backend/MLIR"

rm -f /tmp/tensorium_layering_check.out

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "Layering check: OK"
