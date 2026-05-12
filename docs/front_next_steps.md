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

## Current State
The front pipeline is no longer blocked on init-only semantics. The useful
remaining work is now about:
1. diagnosing and preventing codegen performance regressions,
2. closing lowering gaps that block richer GR/BSSN programs,
3. tightening the runtime/lowering boundary beyond the init-only ABI,
4. reducing duplicated dev-script machinery.

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

### 2. High-value lowering gaps
- Implement or deliberately scope:
  - contravariant covariant derivative lowering (`nabla^`) in
    `MLIRGenExpr.cpp`,
  - external function lowering in `MLIRGenShared.cpp`.

Suggested order:
- prioritize `nabla^` if the immediate target is richer GR/BSSN coverage;
- prioritize `extern` if the language/runtime integration surface matters more.

### 3. Runtime/lowering contract beyond init-only
- The init-only ABI is stable enough for its milestone, but the general field
  lowering contract is still broader than that document:
  - shape/stride ownership conventions,
  - explicit treatment of runtime buffers outside the generated host wrappers,
  - consistent metadata expectations for init and RHS paths.

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
3. Pick one lowering gap (`nabla^` or `extern`) with tests.
4. Revisit runtime ABI generalization only after the next lowering gap is
   closed and measured.

## Completed Since The Earlier Roadmap
- Official init-only ABI documentation.
- Front numeric init evaluator.
- Schwarzschild reference and edge-case numeric tests.
- Off-diagonal and Kerr-like init coverage.
- Executable-mode builtin consistency for `initial_data`.
- Generated host-wrapper path for LLVM smoke runners.
- Opt-in `--mlir-pass-timing` reporting, wired into the Schwarzschild bench.
