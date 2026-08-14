![Nouveau projet](https://github.com/user-attachments/assets/5f75f1f9-999d-410b-971e-ba3bd5e8b5e9)

# Tensorium_lang

Tensorium is an experimental domain-specific language and compiler for
**numerical-relativity initial data**. Its primary goal is to let a researcher
describe the geometry, tensor equations, elliptic constraints, domain, solver
policy, and desired reconstruction in a compact source file, then generate
validated numerical kernels and data that can be handed to an external
evolution code.

The project is written in C++20 and built on LLVM/MLIR 20. It includes tensor
index semantics, a custom MLIR dialect and lowering pipeline, elliptic solver
runtime components, and initial-data reconstruction utilities.

> **Status:** research prototype. Tensorium can already solve and export real
> low-resolution binary-black-hole puncture data, but it is not yet a
> production replacement for TwoPunctures, KADATH, or an evolution code.

## Project scope

Tensorium is primarily an **initial-data generator**, not a complete spacetime
evolution framework. The intended workflow is:

```text
Tensorium source
  -> tensor/index and initial-data semantic checks
  -> ConstraintProblemIR or generated MLIR/LLVM residual kernels
  -> elliptic solve
  -> physical diagnostics and BSSN/ADM reconstruction
  -> external numerical-relativity evolution code
```

The language also has `simulation`/RHS syntax and lowering for BSSN-like and
Z4C-like experiments. That infrastructure is useful for compiler validation
and kernel generation, but a production time evolution stack is not the
current product target.

Long term, an initial-data source should be able to declare:

- tensor fields, parameters, metrics, index variance, and contractions;
- Hamiltonian and momentum constraints in a chosen conformal decomposition;
- domains, coordinate maps, bases, regularity, boundaries, and matching;
- scalar, vector, and tensor elliptic unknowns, including matter sources;
- solver/backend policy without embedding a physical case in C++;
- reconstruction and export into standard NR variables on the consumer's grid.

## What works today

| Area | Current state |
| --- | --- |
| Tensor DSL frontend | Lexer, parser, typed AST, explicit covariant/contravariant indices, free/dummy-index validation, contractions, partial/covariant derivatives, Laplacians, and metric rules. |
| Compiler | Custom Tensorium MLIR dialect, Einstein canonicalization/validation, stencil and grid lowering, LLVM emission, and generated host ABI metadata. |
| Radial initial data | Host-side multidomain Chebyshev solver with compactified exteriors, matching, regular scalar-ball support, coupled scalar/vector/tensor layouts, and implemented subsets of CTT and electrostatic Einstein-Maxwell problems. |
| Multidimensional initial data | Compiled scalar residual systems on tensor-product Chebyshev/Fourier grids, Newton solves, dense or matrix-free FGMRES linear solves, and reusable preconditioners/maps. |
| Binary black holes | A physical Bowen-York two-puncture Hamiltonian solve on a compactified two-centre domain, with ADM diagnostics, puncture-mass calibration, regularity checks, and low-resolution published-data guards. |
| Handoff | Cartesian BSSN reconstruction into caller-owned structure-of-arrays buffers, plus a generic runner that writes a diagnostic CSV and JSON metadata. |

Two initial-data paths currently coexist:

1. The radial backend executes `initial_data` problems directly from
   `ConstraintProblemIR`.
2. The multidimensional backend compiles `constraints` blocks to LLVM residual
   kernels and consumes a declarative `initial_data spectral` descriptor in a
   generic runtime.

This split is intentional while the generated multidimensional path matures.

## Current milestone: declarative two-puncture data

The latest development series completed an end-to-end proof of concept for
binary-black-hole puncture data:

- compactified two-puncture coordinates and Cartesian derivative transforms;
- the physical Bowen-York/Lichnerowicz Hamiltonian residual compiled from DSL;
- bounded-memory matrix-free Newton-FGMRES with mapped sparse one-level and
  experimental geometric two-grid preconditioners;
- refinement, ADM, puncture-mass, symmetry, and axis-regularity diagnostics;
- an unequal-mass published-data comparison and Cartesian BSSN handoff;
- a fully declarative spectral case: physical parameters, resolution, maps,
  solver settings, and reconstruction live in the `.tn` source;
- cached spectral operators, optional OpenMP line parallelism, and corrected
  flexible-GMRES updates.

The provided QC0 case converges on its validated `10 x 10 x 16` spectral grid
with the geometric two-grid preconditioner and exports a Cartesian `z=0` BSSN
slice. This is a genuine nonlinear physical solve, not a manufactured Poisson
example. The exported CSV is nevertheless a diagnostic artifact, not a
production 3D checkpoint.

### What is not production-ready

- QC0 is not yet reliable at the next tested `12 x 12 x 20` resolution with
  either the one-level relaxation preconditioner or the first two-grid
  prototype.
- High-resolution convergence, Fourier-mode regularity at axes/punctures, and
  spinning/unequal-mass validation are incomplete.
- Apparent-horizon masses and independent surface-integral charge checks are
  not implemented.
- The nonradial handoff has a C++ SoA API, but no stable versioned C ABI or
  concrete adapter for an external evolution code yet.
- The compiled spectral path is strongest for one scalar unknown per equation;
  general tensor-valued multidimensional elliptic systems need more lowering
  and runtime work.
- Only a small registry of coordinate maps, unknown maps, projectors, and
  reconstructions is currently available.

See [the two-puncture roadmap](docs/two_puncture_initial_data.md) for measured
residuals, validation cases, and detailed limitations.

## A minimal initial-data program

This complete example declares a spectral grid, solver policy, and a residual
equation. The zero seed already solves `Delta U = 0`:

```tn
field scalar U
field scalar H

initial_data SpectralIdentity {
  spectral {
    system = IdentityResidual
    coordinate_map = identity
    resolution = [3, 3, 4]
    basis = [chebyshev, chebyshev, fourier]
    coordinate_parameters = []
    unknown_map = identity
    unknown_map_parameters = []
    field_projector = none
    reconstruction = none

    solve {
      nonlinear = newton
      linear = direct
      tolerance = 1e-12
      max_iterations = 4
    }
  }
}

constraints IdentityResidual {
  residual H = laplacian(U)
}
```

The physical two-puncture source is in
[`tests/fixtures/elliptic/spectral_two_puncture_hamiltonian_3d.tn`](tests/fixtures/elliptic/spectral_two_puncture_hamiltonian_3d.tn).

## Build

Requirements:

- CMake 3.20 or newer;
- a C++20 compiler;
- LLVM and MLIR 20, including `clang` and preferably `llc` for generated
  initial-data executables.

Configure the LLVM/MLIR paths for your installation:

```bash
cmake -S . -B build \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j2
```

The compiler driver is `build/tools/driver/Tensorium_cc`. If `clang` or `llc`
is not discoverable by name, set `CLANG=/path/to/clang` and
`LLC=/path/to/llc` when running the generated initial-data workflow.

## Run initial-data examples

Run the minimal standalone spectral problem:

```bash
./run_initial_data.sh \
  tests/fixtures/elliptic/spectral_identity_initial_data_3d.tn \
  /tmp/tensorium_identity.csv
```

Run the declarative QC0 binary-black-hole case:

```bash
./run_two_puncture_qc0.sh /tmp/tensorium_qc0_bssn_slice.csv
./plot_constraint_slice.py /tmp/tensorium_qc0_bssn_slice.csv chi
```

`run_initial_data.sh` compiles the DSL residual to LLVM, builds a temporary
generic host runner, solves the declared system, reconstructs the requested
fields, and writes the output plus `<output>.json` metadata. Slice sampling can
be changed independently of the spectral solve:

```bash
TENSORIUM_SLICE_N=257 TENSORIUM_HALF_WIDTH=12 \
  ./run_two_puncture_qc0.sh /tmp/qc0_257.csv
```

For larger experiments, `TENSORIUM_NATIVE=1` enables host-specific compiler
optimization and `TENSORIUM_OPENMP=1 OMP_NUM_THREADS=<n>` enables spectral-line
parallelism. These options improve runtime mechanics but do not remove the
current high-resolution solver limitation.

## Inspect compiler output

```bash
./build/tools/driver/Tensorium_cc \
  --dump-mlir tests/22_BSSN_minimal.tn

./build/tools/driver/Tensorium_cc \
  -O3 --emit-llvm /tmp/schwarzschild.ll \
  tests/fixtures/gr/schwarzschild_3d.tn
```

Individual `--tensorium-*` transformation flags remain available for pass
development and debugging.

## Tests

```bash
ctest --test-dir build --output-on-failure
bash run_test.sh
```

The suite covers frontend success/error cases, Einstein-index semantics,
lowering, runtime units, generated LLVM kernels, radial constraint solves,
spectral solves, physical benchmarks, and handoff checks. Files containing
`error` are generally negative tests expected to fail validation.

## Roadmap

The next priorities are:

1. Make the three-dimensional spectral solve robust at production-oriented
   resolutions with stronger preconditioning and convergence studies.
2. Add independent physical validation: production-resolution unequal-mass
   and spinning cases, apparent horizons, and surface-integral diagnostics.
3. Define a versioned C ABI and validate one real external evolution-code
   adapter on a full 3D consumer grid.
4. Generalize multidimensional coupled and tensor-valued unknowns, matter
   sources, regularity policies, coordinate maps, and reconstructions.
5. Continue separating declarative physics from interchangeable solver and
   export backends.

## Documentation

- [Two-puncture implementation and validation](docs/two_puncture_initial_data.md)
- [Initial-data constraint DSL](docs/constraint_solver_dsl.md)
- [Language, MLIR, ABI, and runtime architecture](docs/language_mlir_abi_architecture.md)
- [Generated kernel ABI](docs/generated_kernel_abi.md)
- [Tensor typing and frontend semantics](docs/tensor_typing.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
