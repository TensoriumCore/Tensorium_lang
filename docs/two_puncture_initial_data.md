# Two-Puncture Initial-Data Roadmap

## Goal

Tensorium is intended to generate elliptic initial-data problems from the DSL,
solve them before evolution, and export the resulting fields to an external
numerical-relativity evolution code. The immediate binary-black-hole target is
the puncture construction used by TwoPunctures. KADATH remains a broader
architectural reference for multi-domain spectral solvers, not the algorithm
implemented by this milestone.

## TP-1: Executable compact geometry

TP-1 is implemented in `TwoPunctureMap.h`. The two Chebyshev coordinates are
`A,B in (-1,1)` and the periodic coordinate is `phi in [0,2*pi)`. Given the
coordinate half-separation `b > 0`, the runtime evaluates

```text
a = (A + 1) / 2
X = 2 atanh(a)
R = pi/2 + 2 atan(B)
x + i rho = b cosh(X + i R)
y = rho cos(phi)
z = rho sin(phi)
```

The limiting points `A -> -1, B -> -1` and
`A -> -1, B -> +1` are the punctures at `x=+b` and `x=-b`. Spatial infinity is
compactified at `A -> +1`. Chebyshev-zero collocation excludes these singular
endpoints.

`SpectralResidualProblem` now accepts a `SpectralDerivativeMap`. Residual
assembly first differentiates with respect to `(A,B,phi)`, then transforms the
complete gradient and symmetric Hessian to Cartesian `(x,y,z)` derivatives.
Both generated point kernels and generated grid kernels therefore receive
physical derivatives. This is important beyond the scalar Laplacian: future
CTT equations can consume individual and mixed Cartesian derivatives without a
special-case operator.

The half-separation is supplied as the single entry in `coordinateParams` and
is shared by the coordinate and derivative maps:

```cpp
problem.coordinateMap = makeTwoPunctureCoordinateMap();
problem.derivativeMap = makeTwoPunctureDerivativeMap();
problem.coordinateParams = std::array<double, 1>{halfSeparation};
```

The regression suite checks:

- puncture limits, reflection symmetry, cylindrical radius, and compactified
  infinity;
- all Cartesian first and second derivatives against an independent local
  inverse-map finite-difference oracle;
- a generated DSL Poisson residual on the mapped domain;
- a dense Newton solve for a manufactured compactified correction
  `U=(A-1)v`, including generated LLVM grid-kernel execution.

Run the vertical test directly with:

```bash
bash tools/dev/test_generated_two_puncture_solve_ll.sh
```

## TP-2: Generated physical Hamiltonian residual

TP-2 adds a physical equation without adding a TwoPunctures compiler mode.
`spectral_two_puncture_hamiltonian_3d.tn` expresses in the DSL:

- both Cartesian distances `r1` and `r2`;
- arbitrary linear-momentum and spin Bowen-York tensors for both punctures;
- the full symmetric contraction `Atilde_ij Atilde^ij`;
- the singular conformal background and nonlinear vacuum Hamiltonian equation.

```text
psi = 1 + m1/(2 r1) + m2/(2 r2) + U
Delta U + (1/8) Atilde_ij Atilde^ij psi^(-7) = 0.
```

The solver variable does not have to equal the residual variable. The generic
`SpectralUnknownMap` transforms a solver field and its complete derivative
bundle before coordinate transformation and generated residual evaluation.
For this problem it is configured as

```cpp
const std::array<double, 3> unknownMapParams = {0.0, 1.0, 1.0};
problem.unknownMap = makeLinearBoundaryFactorUnknownMap();
problem.unknownMapParams = unknownMapParams; // U=(A-1)*v
```

The map is not puncture-specific: it represents any scalar unknown multiplied
by a linear boundary factor along a selected logical axis. Other initial-data
equations can omit it or choose another representation independently of the
generated kernel.

The TP-2 regression validates the generated Bowen-York tensor against the
closed single-puncture contractions

```text
A_P^2 = 9/(2 r^4) * (P^2 + 2(P.n)^2)
A_S^2 = 18/r^6 * (S^2 - (S.n)^2),
```

obtains exactly zero residual for time-symmetric Brill-Lindquist data, and
solves an equal-mass binary with equal-and-opposite tangential momenta. On the
small regression grid the nonlinear residual decreases from about `1.54e-3`
to `4.4e-15` while the conformal factor remains positive.

Run it directly with:

```bash
bash tools/dev/test_generated_two_puncture_hamiltonian_solve_ll.sh
```

## TP-3: Refinement probes and asymptotic diagnostics

TP-3 adds tensor-product interpolation to the generic spectral grid. A scalar
collocation field can now be evaluated at arbitrary logical coordinates,
including the compactified endpoint `A=1`, by Chebyshev barycentric and Fourier
interpolation. This is runtime infrastructure shared by all generated spectral
initial-data systems.

For `U=(A-1)v`, the asymptotic TwoPunctures relation is

```text
M_ADM = m1 + m2 - 4 b v(A=1,B=0,phi=0).
```

`TwoPunctureDiagnostics.h` evaluates this energy together with total ADM linear
momentum and angular momentum, including the orbital terms `C_a cross P_a`.

The physical regression now solves the same equal-mass boosted binary on
`3x3x4`, `4x4x6`, and `5x5x8` grids. It compares the correction at one fixed
logical probe and the ADM energy:

```text
probe change:  9.73e-4 -> 4.95e-4
ADM change:    4.28e-3 -> 1.79e-3
fine ADM mass: 1.1052842937945357
```

The decreasing changes are the first refinement guard, not yet a production
spectral-convergence claim. The fine solution also satisfies the binary's
half-turn symmetry to `3.1e-17`, has zero total linear momentum, and has the
expected orbital angular momentum `Jz=2*b*P`.

## What remains before production TwoPunctures

The TP-2/TP-3 path is a genuine physical residual and nonlinear solve, but it
is still a small dense regression. It does not yet provide:

- puncture and axis regularity enforced through basis/parity rules;
- higher-resolution convergence studies against published TwoPunctures data;
- puncture-local ADM masses or independent surface-integral charge checks;
- the nonlinear bare-mass search needed to match requested physical masses;
- a mapped-domain preconditioner suitable for production resolutions;
- interpolation and metadata export to an external evolution solver;
- automatic selection of coordinate and unknown maps from DSL/module metadata.

The compiled spectral residual path currently differentiates one scalar
unknown per equation. General coupled scalar systems are supported through
auxiliary-unknown mappings, but arbitrary tensor-valued multidimensional
elliptic unknowns still require additional lowering and runtime work.

## Next milestone: TP-4 regularity and scaling

The production path is now:

1. implement regularity/parity conditions at the coordinate degeneracies;
2. build a mapped-domain preconditioner and move beyond dense Jacobians;
3. add puncture-mass diagnostics and the physical-parameter root search;
4. validate higher-resolution unequal-mass, boosted, and spinning cases against
   TwoPunctures cases;
5. interpolate the solved fields onto the external evolution grid and verify
   the Hamiltonian and momentum constraints after handoff.

## References

- M. Ansorg, B. Bruegmann, and W. Tichy, *A single-domain spectral method for
  black hole puncture data*, Phys. Rev. D 70, 064011 (2004),
  https://doi.org/10.1103/PhysRevD.70.064011.
- Einstein Toolkit, *TwoPunctures thorn documentation*,
  https://einsteintoolkit.org/thornguide/EinsteinInitialData/TwoPunctures/documentation.html.
- P. Grandclement, *KADATH: a spectral solver for theoretical physics*,
  J. Comput. Phys. 229, 3334-3357 (2010),
  https://doi.org/10.1016/j.jcp.2010.01.005.
