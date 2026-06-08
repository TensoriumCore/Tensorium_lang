#!/usr/bin/env bash

tensorium_dev_root_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

tensorium_generated_spectral_smoke() {
  if [[ $# -lt 4 ]]; then
    echo "error: tensorium_generated_spectral_smoke expects label fixture runner stem" >&2
    return 2
  fi

  local label="$1"
  local fixture="$2"
  local runner_src="$3"
  local stem="$4"
  shift 4

  local root_dir="${TENSORIUM_ROOT_DIR:-$(tensorium_dev_root_dir)}"
  local driver="$root_dir/build/tools/driver/Tensorium_cc"
  local clang_bin
  local llc_bin
  local cxx_bin="${CXX:-c++}"
  local openmp_cxxflags=()
  local openmp_ldflags=()

  if [[ -n "${CLANG:-}" ]]; then
    clang_bin="$CLANG"
  elif [[ -x /opt/llvm-20/bin/clang ]]; then
    clang_bin="/opt/llvm-20/bin/clang"
  else
    clang_bin="clang"
  fi

  if [[ -n "${LLC:-}" ]]; then
    llc_bin="$LLC"
  elif [[ -x /opt/llvm-20/bin/llc ]]; then
    llc_bin="/opt/llvm-20/bin/llc"
  else
    llc_bin="llc"
  fi

  if [[ "${TENSORIUM_ENABLE_OPENMP:-1}" != "0" ]]; then
    # shellcheck source=/dev/null
    source "$root_dir/tools/dev/openmp_flags.sh"
    cxx_bin="$CXX_BIN"
    openmp_cxxflags=("${OPENMP_CXXFLAGS[@]}")
    openmp_ldflags=("${OPENMP_LDFLAGS[@]}")
  fi

  local ll_path="/tmp/${stem}.ll"
  local obj_path="/tmp/${stem}.o"
  local host_header="/tmp/${stem}_host.h"
  local exe_path="/tmp/${stem}_runner"

  if [[ ! -x "$driver" ]]; then
    echo "error: missing driver binary: $driver" >&2
    return 2
  fi
  if [[ ! -f "$fixture" ]]; then
    echo "error: missing fixture: $fixture" >&2
    return 2
  fi
  if [[ ! -f "$runner_src" ]]; then
    echo "error: missing runner source: $runner_src" >&2
    return 2
  fi

  echo "[$label] generating LLVM IR and host header"
  "$driver" \
    --tensorium-rhs-grid-affine-lower \
    --tensorium-strip-source-funcs \
    --emit-llvm "$ll_path" \
    --emit-host-header "$host_header" \
    "$fixture" >/dev/null

  if [[ ! -s "$ll_path" ]]; then
    echo "error: LLVM IR file is missing or empty: $ll_path" >&2
    return 2
  fi
  if [[ ! -s "$host_header" ]]; then
    echo "error: generated host header is missing or empty: $host_header" >&2
    return 2
  fi

  while [[ $# -gt 0 ]]; do
    if [[ $# -lt 3 ]]; then
      echo "error: malformed generated spectral smoke check" >&2
      return 2
    fi
    local target_kind="$1"
    local pattern="$2"
    local message="$3"
    shift 3

    local target_path
    case "$target_kind" in
    header)
      target_path="$host_header"
      ;;
    llvm)
      target_path="$ll_path"
      ;;
    *)
      echo "error: unknown generated spectral smoke check target: $target_kind" >&2
      return 2
      ;;
    esac

    if ! grep -q "$pattern" "$target_path"; then
      echo "$message" >&2
      return 2
    fi
  done

  echo "[$label] compiling LLVM object"
  if command -v "$llc_bin" >/dev/null 2>&1; then
    "$llc_bin" -filetype=obj "$ll_path" -o "$obj_path"
  else
    "$clang_bin" -c "$ll_path" -o "$obj_path"
  fi

  echo "[$label] compiling runtime runner"
  "$cxx_bin" -O2 -std=c++20 "${openmp_cxxflags[@]}" \
    -I "$root_dir/include" -include "$host_header" \
    "$runner_src" "$obj_path" -lm "${openmp_ldflags[@]}" -o "$exe_path"

  echo "[$label] running runtime executable"
  "$exe_path"
}
