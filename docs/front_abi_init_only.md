# Front ABI: Init-Only Numeric Contract

## Scope
This document defines the front-end executable contract for numeric execution of
`@tensorium_init` only, without full LLVM/JIT backend lowering.

## ABI Goals
- Provide deterministic passing of parameters and coordinates.
- Provide explicit output buffers for `alpha`, `gamma`, `gammaU`.
- Keep memory model simple and backend-friendly (SoA).

## Runtime Inputs
Required:
- `n_points`: number of evaluation points.
- Parameter table: currently requires at least `M` for Schwarzschild fixtures.
- Coordinate arrays:
  - `r[n_points]`
  - `theta[n_points]`
  - `phi[n_points]` (optional if not required by expression set)

## Runtime Outputs (SoA)
Required:
- `alpha[n_points]`
- `gamma[9][n_points]` (row-major spatial 3x3)
- `gammaU[9][n_points]` (row-major spatial 3x3)

Optional:
- `g4[16][n_points]` (raw covariant metric components), currently not required
  by the milestone.

## Memory/Layout Convention
- SoA indexing convention:
  - component-major then point index.
  - Example:
    - `gamma[0][p] = gamma_rr(p)`
    - `gamma[4][p] = gamma_thetatheta(p)`
    - `gamma[8][p] = gamma_phiphi(p)`
- All output component arrays must be writable and at least `n_points` long.

## MLIR Init Semantics Covered by This ABI
Supported init op subset:
- scalar ops: `tensorium.const`, `tensorium.param`, `tensorium.coord`,
  `tensorium.add/sub/mul/div`, `tensorium.sin`, `tensorium.sqrt`.
- init assembly ops: `tensorium.metric4`,
  `tensorium.decompose3p1_from_metric`,
  `tensorium.init3p1`, `tensorium.assign`.

Unsupported init ops should produce explicit diagnostics in the front evaluator.

## Optional Lowered Point Kernel (Front-Only Bridge)
When MLIR is emitted with:
- `--tensorium-metric-lower`
- `--tensorium-init-std-lower`

the module also contains:
- `func.func @tensorium_init_point(...)`

with signature:
- `(f64 M, f64 r, f64 theta, f64 phi, memref<1xf64> alpha, memref<9xf64> gamma, memref<9xf64> gammaU) -> ()`

Contract:
- this function is generated from `@tensorium_init` only;
- it contains only `arith`/`math`/`memref`/`func` ops (no Tensorium custom ops);
- it writes `alpha`, `gamma`, `gammaU` through `memref.store`.

## `decompose3p1_from_metric` Minimal Contract
For this milestone:
- metric must be symmetric;
- supported operational scope:
  - non-zero time-space terms `g_ti` (shift/beta) are supported,
  - diagonal spatial metrics (primary path),
  - symmetric spatial off-diagonal terms (fallback inverse path).

Computed outputs:
- `gamma_ij = g_ij` (`i,j in {1,2,3}`)
- `beta_i = g_{0i}`
- `gammaU = inverse(gamma)`:
  - diagonal fast path uses component-wise reciprocal (IEEE behavior preserved),
  - otherwise use 3x3 inverse for symmetric matrix
- `alpha = sqrt(beta_i beta^i - g_tt)` with `beta^i = gammaU^{ij} beta_j`

## IEEE Behavior Policy for Coordinate Singularities
This front milestone **does not reject** singular coordinate points if arithmetic
can proceed with IEEE values:
- `r = 2M` may produce `inf`/`0` in intermediate or tensor components.
- `theta = 0` may produce `g_phiphi = 0`.

Only explicit unsupported semantic forms should raise errors.

## Numeric Coverage Matrix (current)
- Supported numeric init tests:
  - Schwarzschild reference point (`M=1, r=10, theta=pi/2`),
  - Schwarzschild edge cases (`theta=0`, `r=2M`),
  - Reissner-Nordstrom-like diagonal metric (`M,Q` parameters),
  - Symmetric spatial off-diagonal metric (3x3 inverse fallback path),
  - Kerr-like metric with non-zero `g_tphi` (shift-aware alpha path).

## Sema/MLIRGen Builtin Contract (initial_data call expressions)
Executable-mode call builtins in `initial_data` are constrained to:
- `sin`, `sqrt`

Notes:
- This aligns Sema with current MLIRGen implementation.
- Exponent operator `^` remains supported through existing MLIRGen handling.

## Front Data Descriptor (code-facing)
The executable front API should expose descriptors equivalent to:
- init input descriptor:
  - `n_points`
  - parameter map
  - optional coordinate spans (`r`, `theta`, `phi`)
- init output descriptor:
  - writable spans for `alpha`
  - writable spans for `gamma` 3x3 components
  - writable spans for `gammaU` 3x3 components

The descriptor shape is part of this ABI contract and is covered by unit tests.
