![444221947-5f75f1f9-999d-410b-971e-ba3bd5e8b5e9 (1)](https://github.com/user-attachments/assets/84311f32-c30f-4dcd-ae58-f438475d76b8)

# Tensorium_lang

Tensorium_lang is a domain-specific language and compiler frontend designed
to express tensorial equations arising in numerical relativity
(Einstein equations, BSSN-like/Z4/Z4C formulations and 3+1 decomposition, tensor contractions, differeciations etc..).

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

The driver also exposes clang-like presets:

```bash
./build/tools/driver/Tensorium_cc -O3 --emit-llvm out.ll tests/fixtures/gr/schwarzschild_3d.tn
```

`-O3` enables the deterministic analysis/lowering pipeline through final IR-ready passes. Individual `--tensorium-*` flags remain additive when a preset is used, and pass parameters can be overridden:

```bash
./build/tools/driver/Tensorium_cc \
  -O3 \
  --tensorium-dx 0.25 \
  --tensorium-stencil-order 4 \
  --tensorium-dissipation \
  --tensorium-dissipation-strength 0.05 \
  --emit-llvm out.ll \
  tests/fixtures/gr/schwarzschild_3d.tn
```

This pipeline perform:

- stencil lowering
- artificial dissipation pass
- Einstein index lowering
- einsum analysis
- canonicalization
- Einstein validity checking
- MLIR dump

# Initial-data residual kernels

Initial-data solver residuals can be written with an explicit `constraints`
block:

```tn
constraints PoissonInitialData {
  Hlocal = laplacian(u)
  residual H = Hlocal
}
```

When grid lowering is enabled, these blocks expose a host-callable
`tensorium_residual_grid_affine` kernel alongside the existing RHS-compatible
grid path. The residual fields represent `F(u)`; standalone solver runners can
then drive `F(u) -> 0`.

For CPU parallel grid experiments, `--tensorium-rhs-grid-parallel-lower`
emits `tensorium_rhs_grid_parallel` and, for `constraints` blocks,
`tensorium_residual_grid_parallel`. LLVM emission lowers the parallel grid loop
through OpenMP runtime calls.

## Solved QC0 two-puncture data

The repository includes an end-to-end binary-black-hole initial-data case. It
compiles the generated Hamiltonian residual, solves the QC0 Bowen--York puncture
configuration, and exports a Cartesian BSSN slice plus JSON metadata:

```bash
./run_two_puncture_qc0.sh /tmp/tensorium_qc0_bssn_slice.csv
```

See `docs/two_puncture_initial_data.md` for parameters, output fields,
resolution controls, validation results, and current production limitations.

# Test

All test programs are located in tests/.
- Files starting with valid_ or without error must compile successfully
- Files containing error are expected to fail semantic or Einstein validation

```bash
ctest --test-dir build --output-on-failure
bash run_test.sh
./build/tools/driver/Tensorium_cc --dump-mlir tests/22_BSSN_minimal.tn

```

# Internal architecture (high level)

- Frontend: Lexer / Parser / AST / IndexedAST
- Semantic analysis: tensor variance, free and dummy indices
- MLIR dialect: tensorium
- Transformation passes: Einstein lowering, index analysis, stencil lowering
- Target: MLIR (future LLVM / Linalg lowering)

Tensor rules are explicit and must never be inferred implicitly.

## Generated Kernel ABI

The frozen C/C++ ABI contract for generated `tensorium_init*` and
`tensorium_rhs*` functions is documented in:

- `docs/generated_kernel_abi.md`

The current compact binary-black-hole initial-data milestone and the remaining
path to a physical TwoPunctures-style solve are documented in:

- `docs/two_puncture_initial_data.md`
