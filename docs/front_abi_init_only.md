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

## `decompose3p1_from_metric` Minimal Contract
For this milestone:
- metric must be symmetric;
- time-space terms must satisfy `g_ti = 0` (beta unsupported);
- supported operational scope:
  - diagonal spatial metrics (primary path),
  - symmetric spatial off-diagonal terms (fallback inverse path).

Computed outputs:
- `gamma_ij = g_ij` (`i,j in {1,2,3}`)
- `beta_i = 0` for supported `g_ti=0` scope
- `gammaU = inverse(gamma)`:
  - diagonal fast path uses component-wise reciprocal (IEEE behavior preserved),
  - otherwise use 3x3 inverse for symmetric matrix
- `alpha = sqrt(-g_tt)` for `g_ti=0` block-diagonal supported scope

## IEEE Behavior Policy for Coordinate Singularities
This front milestone **does not reject** singular coordinate points if arithmetic
can proceed with IEEE values:
- `r = 2M` may produce `inf`/`0` in intermediate or tensor components.
- `theta = 0` may produce `g_phiphi = 0`.

Only explicit unsupported semantic forms should raise errors.

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

