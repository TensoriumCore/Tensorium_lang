
# Tensorium_lang

Tensorium_lang is a domain-specific language and compiler frontend designed
to express tensorial equations arising in numerical relativity
(Einstein equations, BSSN-like formulations, tensor contractions).

The compiler is implemented in C++20 and built on top of LLVM/MLIR 20.
It performs semantic analysis of tensor indices (variance, contractions),
followed by a custom MLIR lowering pipeline.

---

## Build

Tensorium_lang requires a custom installation of LLVM/MLIR 20.

```bash
cmake -S . -B build \
  -DLLVM_DIR=/opt/local/libexec/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/opt/local/libexec/llvm-20/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-O0 -g"

cmake --build build -j
```

# Driver 

The compiler Driver is:

```bash
build/tools/driver/Tensorium_cc
```

# Running the compiler

To compile and analyze a Tensorium source file (.tn):

```bash
./build/tools/driver/Tensorium_cc \
  --tensorium-stencil-lower \
  --tensorium-dissipation \
  --tensorium-einstein-lower \
  --tensorium-einstein-analyze-einsum \
  --tensorium-einstein-canonicalize \
  --tensorium-einstein-validate \
  --dump-mlir \
  tests/01_scalar_minimal.tn
```

This pipeline perform:

- This pipeline performs:
- stencil lowering
- artificial dissipation pass
- Einstein index lowering
- einsum analysis
- canonicalization
- Einstein validity checking
- MLIR dump

# Test

All test programs are located in tests/.
- Files starting with valid_ or without error must compile successfully
- Files containing error are expected to fail semantic or Einstein validation

```bash
./build/tools/driver/Tensorium_cc --dump-mlir tests/22_BSSN_minimal.tn

```

# Internal architecture (high level)

- Frontend: Lexer / Parser / AST / IndexedAST
- Semantic analysis: tensor variance, free and dummy indices
- MLIR dialect: tensorium
- Transformation passes: Einstein lowering, index analysis, stencil lowering
- Target: MLIR (future LLVM / Linalg lowering)

Tensor rules are explicit and must never be inferred implicitly.
