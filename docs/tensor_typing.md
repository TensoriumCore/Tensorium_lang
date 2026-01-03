# Tensor Typing Design Notes

Tensorium lowers every intermediate to `tensorium.field<f64, up, down>`. The two
integer parameters encode the number of contravariant (up) and covariant (down)
indices carried by the SSA value. This single type is shared by all custom ops
and makes it obvious when a value changes rank.

## Scalar Weights First

Stencil and KO dissipation builders work with numerical weights that are purely
scalar. This keeps the coefficient tables simple and allows us to re-use the
same constant op for every field. Only after multiplying the sampled tensor
value with the scalar weight do we upgrade the result to the tensor rank needed
by the derivative or dissipation accumulator.

## Promotions

`tensorium.promote` is used in exactly two situations:

1. When MLIRGen assigns a scalar RHS to a tensor field (e.g. because the source
   program intentionally writes a scalar expression into a covector slot).
2. Inside passes when a scalar stencil term must feed a tensor accumulator.

In both cases the expression being promoted is rank-0, and the target type is
known from the destination tensor. No other implicit variance changes are
allowed.

## Accumulator Seeding

All lowering passes follow the same rule when building tensor-valued sums:

* Start the accumulator with the first computed tensor term.
* Use `tensorium.add` only for subsequent terms.
* Never create `tensorium.const … : !tensorium.field<…>` zero values.

This guarantees the accumulator always has the correct tensor type (because it
originates from a real term) and avoids spurious promotions.
