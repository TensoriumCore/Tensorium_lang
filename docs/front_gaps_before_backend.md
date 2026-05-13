# Front Status Before Wider Backend/JIT Work

## Scope
- This document supersedes the earlier "init-only milestone" gap audit.
- The Schwarzschild init-only contract is now implemented and documented in
  `docs/front_abi_init_only.md`.
- The remaining questions are about scaling the front/lowering stack toward
  broader executable programs, not about making basic init evaluation exist.

## What Is Already In Place
### Init/runtime contract
- A documented SoA init ABI for:
  - parameters,
  - coordinate arrays,
  - `alpha`, `gamma`, and `gammaU` outputs.
- A front init evaluator over emitted MLIR.
- A lowered point-kernel bridge:
  - `@tensorium_init_point`,
  - generated from metric/init lowering,
  - free of Tensorium custom ops after lowering.

### Numeric front coverage
- Schwarzschild reference point.
- Schwarzschild edge cases:
  - `theta = 0`,
  - `r = 2M` with explicit IEEE behavior.
- Reissner-Nordstrom-like fixture.
- Symmetric spatial off-diagonal metric fixture.
- Kerr-like metric with non-zero shift component.

### Contract consistency
- Executable-mode `initial_data` call builtins are aligned between Sema and
  MLIRGen on:
  - `sin`,
  - `sqrt`.
- Unsupported init call forms are rejected before they become backend surprises.

## Remaining Front/Lowering Gaps
### A. Pass-level performance visibility
Current state:
- whole-command timings remain useful,
- `Tensorium_cc --mlir-pass-timing` now exposes per-pass timing for the
  Tensorium MLIR and LLVM lowering pipelines,
- the Schwarzschild bench script records those timings.

Why it matters:
- recent Ricci smoke work exposed a no-op lowering path that consumed almost all
  execution time,
- future regressions in canonicalization, Einstein lowering, or LLVM lowering
  will be hard to localize without explicit pass timings.

Needed next:
- run the timing mode on representative Ricci, Christoffel, and BSSN fixtures,
- keep a comparable baseline of the largest pass-level offenders.

### B. External function lowering
Current state:
- extern calls are typed and diagnosed,
- MLIR lowering still emits:
  - `"extern function '<name>' lowering is not implemented yet"`.

Impact:
- the language can describe externally-backed computations that cannot yet
  cross into the executable MLIR/LLVM path.

Primary file:
- `lib/tensorium_mlir/Target/MLIRGen/MLIRGenShared.cpp`

### C. Runtime buffer contract beyond init-only
The init-only ABI is concrete, but the general executable contract is wider:
- field layout expectations across init and RHS,
- stable shape/stride metadata conventions,
- how generated host wrappers map onto runtime-owned buffers.

This should be clarified before broadening backend/JIT work beyond the current
smoke runners.

### D. Smoke/bench script drift
Current state:
- multiple `tools/dev/test_*_ll.sh` scripts duplicate generation and compile
  flow,
- some use generated host headers while others use bespoke runners,
- temporary-file conventions differ.

Impact:
- performance fixes and pipeline changes are easy to apply unevenly,
- benchmarks and smokes can quietly diverge.

## Suggested Order Of Work
1. Profile Ricci, Christoffel, and BSSN fixtures to establish a real codegen
   baseline.
2. Implement extern lowering for language/runtime breadth.
3. Consolidate shared LL smoke mechanics once the pipeline surface settles.

## Kept As Explicit Non-Gaps
These were blocking items in the older audit but are no longer open:
- init-only ABI definition,
- front numeric init evaluator,
- Schwarzschild numeric reference coverage,
- horizon/axis edge-case policy,
- `initial_data` builtin mismatch,
- init decomposition scope for off-diagonal and shift-aware metrics.
- contravariant covariant derivative notation (`nabla^`), which is expanded via
  inverse-metric raising and covered by
  `tools/dev/test_contravariant_all_cases_ll.sh`.
