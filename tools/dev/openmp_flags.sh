if [[ -n "${CXX:-}" ]]; then
  CXX_BIN="$CXX"
elif command -v llvm-config >/dev/null 2>&1 &&
  [[ -x "$(llvm-config --bindir)/clang++" ]]; then
  CXX_BIN="$(llvm-config --bindir)/clang++"
elif [[ -x /opt/llvm-20/bin/clang++ ]]; then
  CXX_BIN="/opt/llvm-20/bin/clang++"
elif command -v clang++ >/dev/null 2>&1; then
  CXX_BIN="clang++"
else
  CXX_BIN="c++"
fi

OPENMP_CXXFLAGS=(${TENSORIUM_OPENMP_FLAGS:--fopenmp})
OPENMP_LDFLAGS=()

if [[ -z "${TENSORIUM_OPENMP_FLAGS:-}" ]]; then
  for libdir in /opt/local/lib/libomp /opt/homebrew/opt/libomp/lib \
    /usr/local/opt/libomp/lib; do
    if [[ -f "$libdir/libomp.dylib" || -f "$libdir/libomp.a" ]]; then
      OPENMP_LDFLAGS+=("-L$libdir" "-Wl,-rpath,$libdir")
      break
    fi
  done
fi

OPENMP_LDFLAGS+=("${OPENMP_CXXFLAGS[@]}")
