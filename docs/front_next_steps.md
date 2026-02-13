# Front Next Steps: Schwarzschild Init-Only JIT-Ready

## Context
- Branch: `refactor/architecture-cleanup`
- Goal: make front-end execution of `@tensorium_init` numerically meaningful before full backend/JIT lowering.
- Reference baseline: `docs/front_gaps_before_backend.md`.

## Prioritized Blocking Gaps
1. Missing executable init contract (ABI + buffers)
- There is no single formal contract for passing `M`, coordinates, and output buffers to an init execution path.
- Impact: no stable hand-off from front to future JIT runtime.

2. Missing minimal executable semantics for init ops
- `metric4`, `decompose3p1_from_metric`, `init3p1`, `assign` are structurally verified but not operationally executed in a numeric front path.
- Impact: cannot validate numeric Schwarzschild values from front alone.

3. `!tensorium.field` does not carry memory layout semantics
- Current type carries element + variance only.
- Impact: lowering contract for store/load is under-specified.

4. Sema/MLIRGen builtin mismatch in `initial_data`
- Sema currently accepts more scalar functions than MLIRGen init emission supports.
- Impact: programs can pass semantic validation and fail later in MLIRGen.

5. Missing numeric non-regression tests
- Existing tests validate structure and use-def invariants but not init numeric values at reference points.
- Impact: no signal that front semantics are numerically correct.

## Design Decisions (for this milestone)
1. Execution strategy
- Choose **Option 1**: implement a compact C++ init evaluator over the emitted MLIR `@tensorium_init`.
- Rationale: smallest dependency surface, deterministic behavior, and reuses current front pipeline.

2. ABI shape
- Use SoA contract for outputs:
  - `alpha[n]`
  - `gamma[9][n]`
  - `gammaU[9][n]`
- Inputs:
  - scalar parameter map (`M` required for Schwarzschild),
  - coordinate arrays (`r`, `theta`, `phi` optional).

3. Decomposition scope (`decompose3p1_from_metric`)
- Minimum executable scope:
  - symmetric metric,
  - non-zero `g_ti` (shift/beta) supported,
  - diagonal spatial metric supported (with IEEE behavior on singular values),
  - symmetric spatial off-diagonal support allowed in evaluator via 3x3 inverse fallback.

4. Builtin contract unification
- Short-term decision: align Sema with MLIRGen-supported init call builtins for executable mode.
- Keep operator `^` support as already handled by MLIRGen.

## Definition of Done (front milestone)
The front is "Schwarzschild init-only JIT-ready" when all are true:
- `@tensorium_init` can be numerically executed for one or more points by front code only.
- ABI contract is documented and implemented in code-facing descriptors.
- Numeric test passes at reference point:
  - `M=1, r=10, theta=pi/2`
  - `alpha = sqrt(0.8)`,
  - `gamma diag = (1.25, 100, 100)`,
  - `gammaU diag = (0.8, 0.01, 0.01)`.
- Edge-case tests pass:
  - `theta=0` gives finite-safe behavior for expected components (`g_phiphi=0`) with no NaN in tested outputs.
  - `r=2M` follows documented IEEE contract (allow inf/0, no front rejection).
- Sema/MLIRGen builtin support is consistent and tested.

## Recent Progress
- Added optional metric-lowered init bridge to standard MLIR:
  - `--tensorium-metric-lower --tensorium-init-std-lower`
  - emits `@tensorium_init_point` with scalar args (`M,r,theta,phi`) and
    `memref` outputs (`alpha`, `gamma`, `gammaU`).
- Added regression test ensuring `@tensorium_init_point` is generated with
  no Tensorium ops and explicit `memref.store` writes.

## Plan (max 3 phases)
### Phase 1: Contract and consistency
- Deliver ABI doc and descriptor structs.
- Resolve Sema/MLIRGen builtin mismatch and add regression test.
- Keep architecture/init-rhs invariants unchanged.

### Phase 2: Front numeric init execution
- Implement minimal init evaluator for `@tensorium_init`.
- Support required op subset:
  - `const`, `param`, `coord`, `add/sub/mul/div`, `sin`, `sqrt`,
  - `metric4`, `decompose3p1_from_metric`, `init3p1`, `assign`.
- Provide clear diagnostics on unsupported init forms.

### Phase 3: Numeric tests and hardening
- Add Schwarzschild point numeric test and two edge-case tests.
- Validate no regression in existing structural tests and scripts.
- Keep backend/JIT lowering work explicitly out of scope.

## Sema vs MLIRGen mismatch (explicit)
Current mismatch:
- Sema `initial_data` accepts scalar call builtins: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `pow`.
- MLIRGen init currently lowers call builtins: `sin`, `sqrt` only.

Proposed correction:
- Restrict Sema executable-mode accepted init call builtins to `sin` and `sqrt` (matching MLIRGen), keep `^` operator behavior unchanged.
- Add a semantic regression test for a rejected unsupported init builtin call (e.g., `cos(...)`).
- Document this contract in `docs/front_abi_init_only.md`.
