# Front Gaps Before Backend (Schwarzschild Init-Only Milestone)

## Scope
- Audited branch: `refactor/architecture-cleanup`.
- Goal: lock down what is still missing in the front-end (`Tensorium_lang` + IR + MLIRGen) before backend/JIT work.
- Scope: **Schwarzschild init only** (`@tensorium_init`), without full backend lowering.
- Technical ground truth used: current code + MLIR dump of `tests/fixtures/gr/schwarzschild_3d.tn` (`/tmp/schw3d_initrhs.mlir`).

## Definition Of Done (front-view) for the JIT milestone
The "Schwarzschild numeric init-only" milestone is considered front-end ready when:
- the executable target function is `@tensorium_init` (not `@tensorium_entry`),
- minimal runtime inputs are explicit and stable:
  - parameters: `M`,
  - per-point coordinates: `r`, `theta`, `phi` (`phi` optional depending on dimension/config),
  - grid description: `N` points + index-to-coordinate mapping,
- outputs are explicit:
  - recommended minimum: `alpha`, `gamma_ij`, `gammaU^ij`,
  - optional: `g_{mu,nu}` if raw metric output is retained,
- a memory contract is defined (SoA format recommended, see section D),
- expected Schwarzschild numeric values are checked at least on one reference point,
- MLIR semantics for init ops are defined (not only "op is present").

## Minimal Schwarzschild numeric example
Requested reference point: `M=1`, `r=10`, `theta=pi/2`.

Formulas:
- `f = 1 - 2*M/r = 0.8`
- `g_tt = -f = -0.8`
- `g_rr = 1/f = 1.25`
- `g_thetatheta = r^2 = 100`
- `g_phiphi = r^2 * sin^2(theta) = 100`
- `alpha = sqrt(f) = 0.8944271909999159`
- `gamma_ij = diag(1.25, 100, 100)`
- `gammaU^ij = diag(0.8, 0.01, 0.01)`

These values should become non-regression assertions in the backend/JIT milestone.

## A) Gap audit: Semantics / AST / DSL
- `initial_data`/`metric4`/`split_3p1` is correctly supported at parse/AST level:
  - `lib/Parse/Parser.cpp:400`, `lib/Parse/Parser.cpp:511`, `include/tensorium/AST/AST.hpp:178`.
- Structural Sema checks are already in place:
  - dimensions/symmetry/coordinates: `lib/Sema/Sema.cpp:378`, `lib/Sema/Sema.cpp:404`, `lib/Sema/Sema.cpp:333`.
- Parameter `M` is implicit (no dedicated DSL declaration):
  - unknown `VarExpr` is turned into `IndexedVarKind::Parameter` (`lib/Sema/Sema.cpp:225`).
  - Consequence: no explicit "required params" contract, no nominal parameter typing.
- Potential front-end blocking inconsistency (S1):
  - Sema accepts `sin/cos/tan/exp/log/sqrt/pow` (`lib/Sema/Sema.cpp:349`),
  - MLIRGen init currently implements only `sin` and `sqrt` (and `^` limited to 0..4) (`lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp:522`),
  - so some programs can pass Sema and then fail in MLIRGen.
- Coordinates:
  - Sema validates coordinate-name compatibility with `simulation.coordinates` (`lib/Sema/Sema.cpp:65`),
  - but MLIR `coord` currently carries only a name string, without explicit runtime binding (see sections C/D).

## B) Gap audit: Tensorium IR / Domain IR
- Init IR exists (`InitExprIR`, `Metric4InitIR`, `InitialDataIR`) in `include/tensorium/IR/IRBase.hpp:135`.
- Main gap:
  - `InitSymbolIR` is still a string symbol (`include/tensorium/IR/IRBase.hpp:147`),
  - strict `param vs coord vs field` classification is currently materialized only in MLIRGen (`emitInitExpr`).
- `metric4` is currently a front-end builder op, not a standalone point-wise executable representation.
- No explicit IR model for field memory layout in numeric output mode.
- No dedicated front IR stage for "evaluate init over grid points".

## C) Gap audit: Tensorium MLIR dialect (init path)
Ops currently present in Schwarzschild `@tensorium_init`:
- `tensorium.const`, `tensorium.param`, `tensorium.coord`,
- `tensorium.add/sub/mul/div/sin`,
- `tensorium.metric4`,
- `tensorium.decompose3p1_from_metric`,
- `tensorium.init3p1`,
- `tensorium.assign`.

Current status by op:
- Verifier: yes for these ops (`lib/tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.cpp:141`, `:150`, `:261`, `:271`, `:298`, `:321`).
- Op-specific canonicalization/folding: no (no dedicated patterns for `metric4/decompose/init3p1/assign`).
- Executable semantics (backend/JIT view): incomplete for `param/coord/metric4/decompose/init3p1/assign`.

Important observation:
- Current Tensorium passes mostly rewrite around `dt_assign`/RHS (`lib/tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinLoweringPass.cpp:77`),
- there is no dedicated operational pass for `metric4 -> decompose3p1_from_metric -> init3p1`.

## D) Minimal runtime contract (front ABI proposal)
### Proposed minimal ABI (C-like)
SoA option (recommended):
- scalar/global inputs:
  - `double M;`
  - `size_t n_points;`
- coordinates (length `n_points`):
  - `const double *r;`
  - `const double *theta;`
  - `const double *phi;` (nullable if unused)
- outputs:
  - `double *alpha;` (1 component),
  - `double *gamma[9];` (dense covariant 3x3),
  - `double *gammaU[9];` (dense contravariant 3x3),
  - optional `double *g4[16];`.

### Why SoA
- simple for vectorization and loops,
- tensor components are directly addressable,
- aligns with field-oriented `tensorium.assign`.

### Type-system gaps to make `!tensorium.field` lowerable
Current `FieldType` only carries `elementType/up/down` (`include/tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h:37`).
Missing information:
- base pointer / ownership,
- spatial shape (`n_points` or dims),
- strides/layout,
- optionally memory space/alignment.

Without this information, a stable memory lowering contract for `assign/ref` cannot be defined.

## E) Missing front-end tests before backend
### Numeric expectations (without LLVM backend)
- Add a front "init evaluator" test layer (interpreting `InitExprIR` + minimal diag/beta=0 `decompose3p1_from_metric` rule),
  or a local "MLIR interpreter shim" limited to init ops.
- Minimum required case:
  - Schwarzschild point test `(M=1,r=10,theta=pi/2)` with numeric assertions listed above.

### Negative/edge-case tests
- `r = 2M`:
  - decide and document explicit contract (recommended: **allow** and accept IEEE `inf/0`, not reject at front level).
- `theta = 0`:
  - verify `g_phiphi = 0` without accidental NaNs.
- Dimension:
  - verify 2D axisymmetric vs 3D spherical consistency on shared components.

### What is already well covered
- Structural init/rhs and use-def invariants are already tested in C++:
  - `tools/Tester/UnitTests.cpp:849`, `:914`, `:1075`, `:1252`.
- `initial_data` safety diagnostics already exist in `run_test.sh`:
  - `tests/semantic/initial_data/*` (`run_test.sh:108`, `:113`).

## MLIR op -> minimal executable semantics -> backend dependency
| MLIR op | Minimal executable semantics (init-only) | Backend dependency |
|---|---|---|
| `tensorium.const` | produce an `f64` scalar | basic arithmetic |
| `tensorium.param(name)` | read a runtime parameter (e.g. `M`) | params table / ABI |
| `tensorium.coord(name)` | read current-point coordinate (`r/theta/phi`) | grid / coord provider |
| `tensorium.add/sub/mul/div` | point-wise `f64` arithmetic | scalar arithmetic lowering |
| `tensorium.sin/sqrt` | point-wise `f64` math | math runtime / libm |
| `tensorium.metric4(16 comps)` | build covariant 4x4 `g_{mu,nu}` at point | structured 4x4 value support |
| `tensorium.decompose3p1_from_metric(g)` | compute `alpha,beta,gamma,gammaU` (at least diag + beta=0) | decomposition / inversion algorithm |
| `tensorium.init3p1(a,b,g,gU)` | typed binding/no-op (or normalization hook) | pipeline convention |
| `tensorium.assign(field, rhs)` | store into destination field buffer | field memory ABI |
| `tensorium.ref(field,...)` | load from field buffer (useful for rhs verification) | field memory ABI |

## Recommendation: front -> backend plan (max 3 phases)
### Phase 1 — Executable init-only contract first
- Goal:
  - freeze minimal runtime ABI (`param + coord arrays + output buffers`),
  - freeze operational semantics of the init ops listed above.
- Files expected:
  - contract docs: `docs/front_gaps_before_backend.md` (this file),
  - front ABI/types: `include/tensorium_mlir/...` (new contract),
  - optional MLIRGen adjustments to materialize interfaces.
- Risks:
  - wrong memory-layout choice can block later phases.
- Tests to lock:
  - existing init/rhs structural tests + new ABI contract tests.

### Phase 2 — Make init chain directly lowerable
- Goal:
  - make `metric4/decompose/init3p1/assign` explicitly lowerable.
- Options:
  - keep ops and implement dedicated lowering,
  - or expand earlier into arithmetic SSA (if needed).
- Files expected:
  - `lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp`,
  - dedicated init rewrites/passes.
- Risks:
  - semantic drift if decomposition is partially rewritten without clear invariants.
- Tests:
  - Schwarzschild pointwise numeric checks, while preserving init/rhs invariants.

### Phase 3 — Backend/JIT only after contract stabilization
- Goal:
  - connect LLVM/JIT lowering on top of a stable init IR contract.
- Risks:
  - backend bugs and front ambiguities become indistinguishable if phases 1/2 are incomplete.
- Tests:
  - reuse same fixtures + numeric comparison at reference points.

## Action checklist
- [ ] Define the official init-only ABI (params, coords, outputs, layout).
- [ ] Add Schwarzschild pointwise numeric test (`M=1,r=10,theta=pi/2`).
- [ ] Decide and document horizon behavior at `r=2M` (IEEE-allow vs reject).
- [ ] Align Sema vs MLIRGen whitelist for `initial_data` functions.
- [ ] Define executable semantics scope for `decompose3p1_from_metric`.
- [ ] Add 2D/3D consistency tests on shared metric components.
- [ ] Document mapping from `!tensorium.field` to runtime buffers (shape/strides).
