# Front Next Steps: Post Init-Only Milestone

## Context
- The Schwarzschild init-only front milestone is now materially complete.
- The numeric init ABI is documented in `docs/front_abi_init_only.md`.
- The codebase already has:
  - init evaluator coverage for Schwarzschild, Reissner-Nordstrom-like,
    spatial off-diagonal, and Kerr-like fixtures,
  - explicit `@tensorium_init_point` lowering for the metric/init path,
  - executable-mode `initial_data` builtin alignment on `sin` and `sqrt`,
  - generated host wrappers for LLVM smoke tests.
- The language/MLIR/ABI/runtime boundary is documented in
  `docs/language_mlir_abi_architecture.md`.

## Current State
The front pipeline is no longer blocked on init-only semantics. The useful
remaining work is now about:
1. diagnosing and preventing codegen performance regressions,
2. closing lowering gaps that block richer GR/BSSN programs,
3. tightening the runtime/lowering boundary beyond the init-only ABI,
4. reducing duplicated dev-script machinery.

The first initial-data solver surface is now explicit: `constraints` blocks
lower residual equations to generated `tensorium_residual_grid_*` kernels. The
existing Poisson, Hamiltonian-toy, and Bowen-York single-puncture relaxation
fixtures use that surface and are wired into the full test suite.

RHS/residual grid kernels also have an opt-in parallel lowering path via
`--tensorium-rhs-grid-parallel-lower`. It exposes
`tensorium_rhs_grid_parallel` / `tensorium_residual_grid_parallel`, emits
`scf.parallel` in Tensorium MLIR, and lowers to OpenMP runtime calls during
LLVM emission.

## Prioritized Work
### 1. Pass-level performance observability
- Use the opt-in `--mlir-pass-timing` mode in `Tensorium_cc`.
- Profile Tensorium and LLVM lowering passes on Ricci, Christoffel, and BSSN
  fixtures.
- Keep `tools/Bench` useful by recording comparable timings instead of relying
  on whole-command wall time only.

Why first:
- Recent Ricci smoke work showed a single no-op pass could dominate runtime.
- Without pass timings, follow-up performance work is guesswork.

### 2. Remaining high-value lowering gap
- Keep scalar external function lowering covered in both high-level MLIR and
  LLVM link smokes.
- Defer tensor, buffer, grid, and runtime-aware extern forms until the ABI
  descriptor can model them explicitly.
- Keep `nabla^` coverage in the LLVM smoke suite so inverse-metric raising does
  not regress while extern work proceeds.

### 3. Runtime/lowering contract beyond init-only
- The init-only ABI is stable enough for its milestone, but the general field
  lowering contract is still broader than that document:
  - shape/stride ownership conventions,
  - explicit treatment of runtime buffers outside the generated host wrappers,
  - consistent metadata expectations for init and RHS paths.
- Host C/C++ generation should remain ABI glue only; optimized computation
  stays in MLIR/LLVM kernels, with AMReX as an optional host/runtime target.

This should be handled before widening the backend/JIT surface too far.

### 4. Dev-script and smoke-test consolidation
- The `tools/dev/test_*_ll.sh` scripts still duplicate:
  - driver invocation,
  - IR/header temp paths,
  - compile/link logic.
- Consolidate the repeated shell logic or move more of it into the driver.

## Recommended Next Sequence
1. Establish pass-timing baselines from the updated bench workflow.
2. Profile the real GR/BSSN fixtures and record the largest offenders.
3. Link generated parallel-grid objects in benchmark runners with OpenMP and
   compare serial affine versus parallel kernels on Poisson/Bowen-York grids.
4. Expand host wrapper generation from the reusable ABI descriptor where it
   reduces duplicated glue.
5. Revisit runtime ABI generalization only after the current lowering surface is
   measured.
6. Design point or tile kernel ABI variants before adding AMReX wrappers.

## Completed Since The Earlier Roadmap
- Official init-only ABI documentation.
- Front numeric init evaluator.
- Schwarzschild reference and edge-case numeric tests.
- Off-diagonal and Kerr-like init coverage.
- Executable-mode builtin consistency for `initial_data`.
- Generated host-wrapper path for LLVM smoke runners.
- Opt-in `--mlir-pass-timing` reporting, wired into the Schwarzschild bench.
- Contravariant covariant derivative notation (`nabla^`) covered by a dedicated
  LLVM smoke over covector, vector, and mixed-tensor cases.
- Scalar extern calls lower through Tensorium MLIR to linkable `f64` C ABI
  calls in RHS grid kernels.
