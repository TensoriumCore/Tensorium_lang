# Tensorium Front-End Semantics

This document specifies the source-level semantics that must hold before any
lowering or MLIR generation occurs. It serves as the contract for the AST,
semantic analyzer, and validator.

## AST Overview

Tensorium expressions are composed from the following nodes (after parsing,
without sugar):

* `NumberExpr(double)` – literal scalar constants.
* `VarExpr(name)` – references to scalar variables, fields, temporaries, or
  coordinates (depends on semantic binding).
* `BinaryExpr(lhs, op, rhs)` – infix mathematics with `op` in `{'+','-','*','/'}`.
* `CallExpr(callee, args)` – builtins (`d_i`, `contract`, `laplacian`, `trace`) or
  extern calls by name.
* `IndexedVarExpr(base, indices)` – tensor field access with explicit indices,
  e.g. `gamma[i,j]`.

Top-level constructs:

* `FieldDecl(kind, name, indices, up, down)` – defines a tensor field. `kind`
  maps to `(up, down)` (e.g. covector = `(0,1)`); explicit index names specify
  the expected index arity for evolution equations.
* `EvolutionDecl` – a collection of right-hand-side expressions `dt field[idxs] = rhs`.
* `Assignment` – used for temporaries and metric entries; LHS is a `TensorAccess`.
* `ExternDecl` – declares external scalar/tensor functions with explicit tensor
  type signatures.
* `SimulationConfig` – dimension, resolution, and scheme metadata required to
  establish coordinate indices and derivative availability.

## Tensor Types

Every AST expression carries a **tensor type** `(up, down)` representing the
number of contravariant and covariant indices. Scalars are `(0,0)`. Field
kinds shorten to this pair:

| Kind              | `(up, down)` |
|-------------------|--------------|
| Scalar            | `(0, 0)`     |
| Vector            | `(1, 0)`     |
| Covector          | `(0, 1)`     |
| CovTensor2        | `(0, 2)`     |
| ConTensor2        | `(2, 0)`     |
| MixedTensor       | `(up, down)` specified explicitly |

These types are propagated through the semantic analyzer and **must match** the
index annotations written by the user. Any mismatch (e.g. `gamma[i,j]` with `i`
undefined or wrong count) is rejected before lowering.

## Index Scoping and Lifetime

Tensor indices are names such as `i`, `j`, `k`, `l`, `m`, or `n`. They follow
these rules:

1. **Scope** – Indices are scoped to the RHS of an evolution equation or
   assignment. Reusing the same index name across different equations is fine,
   but within a single RHS, each free index must be declared on the LHS.
2. **LHS/Lifetime** – LHS index list fixes the free indices that may appear in
   the RHS. An index present on the LHS must occur exactly once in each additive
   term of the RHS. If an index appears on the RHS but not on the LHS, it must
   be paired (contracted) exactly twice (Einstein summation) or be part of an
   explicit `contract()` call.
3. **Coordinates vs. Offsets** – When referencing fields or temporaries via
   `beta[i + 1]` in the source, the AST stores explicit offsets as integers per
   index; these offsets carry no semantic meaning beyond ensuring consistency of
   spatial shifts.
4. **Temporaries** – Temporaries declared via assignments are scalar unless
   explicitly indexed; they follow the same index rules when they appear on RHS.

## Einstein Summation Rules

Tensorium enforces Einstein rules statically:

* **Addition/Subtraction** – Both operands must have identical `(up, down)`; no
  implicit promotion is allowed.
* **Multiplication** – Ranks add: `(u1, d1) * (u2, d2) = (u1 + u2, d1 + d2)`.
* **Division** – Denominator must be scalar `(0,0)`; numerator determines the
  result rank.
* **Implicit contractions** – Occur only when the same index name appears twice
  within a term and the LHS does not mention it. The analyzer counts occurrences
  per term, ensuring that any index not on the LHS appears exactly twice or is
  fully contracted by `contract()`.
* **Explicit contract()** – `contract(expr)` requires the argument to contain at
  least one repeated index. Free indices in the argument become the result’s
  indices, and the resulting `(up, down)` is computed by removing a pair per
  contracted index.

Violations produce specific diagnostics (e.g. “tensor addition requires
identical variance” or “contract() expects at least one repeated index”).

## Operators and Builtins

### Derivative `d_i(expr)`
* Requires exactly one argument.
* Adds one covariant index to the tensor rank: `(up, down + 1)`.
* The derivative index `i` is treated as free unless contracted later. The
  result type must match the LHS indices (e.g. `dt v[i] = d_i(phi)`).

### Tensor Product
* Binary `*` multiplies tensor ranks as described above. Index annotations are
  tracked so that contractions or additions know which indices are present.

### Contract
* `contract(expr)` may only be used when `expr` contains repeated indices.
* Each index appears either zero, one, or two times in any additive term. Values
  `>=3` are rejected as ambiguous.
* Result rank is derived by removing one contravariant and one covariant slot
  per contracted index, or by matching the remaining free indices if all are of
  the same variance.

### Laplacian / Externs
* `laplacian(expr)` demands a scalar argument and produces a scalar.
* Extern calls must match the declared tensor signature exactly; both argument
  variance and return type are checked before lowering.

## Valid Program Requirements

A program is considered **valid** when:

1. Every field declared in `EvolutionDecl` has matching index arity between LHS
   and declaration (`FieldDecl`).
2. All expressions obey tensor typing rules (addition/multiplication/derivative
   rules above).
3. Each evolution RHS satisfies Einstein index constraints: LHS indices appear
   exactly once in every additive term; any other indices are contracted.
4. Temporaries are defined before use and follow the same index rules.
5. Externs are only called in executable mode if implementations exist and their
   tensor signatures match.

Examples:

```tn
field covector beta[i]
field scalar phi

evolution Valid {
  dt beta[i] = d_i(phi) + beta[i] * phi
}
```

```tn
field cov_tensor2 gamma[i,j]

evolution RicciPiece {
  dt gamma[i,j] = contract(gamma[i,k] * gamma[k,j])
}
```

Both pass semantic checks: derivatives add the appropriate index, additions
respect variance, and implicit contractions remove the repeated `k`.

## Invalid Program Patterns

1. **Rank mismatch in addition**:
   ```tn
   dt v[i] = phi + v[i]   // scalar + covector ⇒ error
   ```

2. **Invalid contraction**:
   ```tn
   dt beta[i] = contract(beta[i])  // no repeated index ⇒ error
   ```

3. **Assignment mismatch**:
   ```tn
   dt beta[i] = phi  // RHS scalar, LHS covector ⇒ error
   ```

These cases are covered by `tests/52_error_tensor_add_variance.tn`,
`tests/53_error_contract_free_index.tn`, and `tests/54_error_dt_assign_rank.tn`.

## Invariants Before MLIRGen

After semantic analysis (and before MLIR generation), the following invariants
hold:

* Every expression `ExprIR` carries a concrete `(up, down)` tensor type.
* All indices are either bound on the LHS or explicitly contracted; there are
  no dangling free indices.
* Temporaries and fields have consistent index offsets and variance.
* Binary operations, derivatives, and calls respect their typing rules:
  - `+`/`-` operate only on identical types.
  - `*` concatenates ranks, `/` divides by scalars.
  - `d_i` adds one covariant slot.
  - `contract` removes repeated indices and fails if none exist.
* Extern calls have matching argument variance and return types.

These guarantees mean MLIRGen can rely entirely on the recorded tensor types
without rechecking variance, allowing the backend to focus on emitting
`tensorium.field` values and inserting `tensorium.promote` only where the source
semantics already permit scalar-to-tensor assignments.
