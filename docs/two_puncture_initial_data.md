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

## What TP-1 does not solve

TP-1 validates geometry and mapped elliptic machinery. It is not yet a physical
binary-black-hole initial-data solver. In particular, it does not yet provide:

- the two-centre Bowen-York conformal extrinsic curvature for arbitrary linear
  momenta and spins;
- the nonlinear puncture Hamiltonian equation;
- puncture and axis regularity conditions enforced by basis/parity rules;
- an outer-boundary equation or an automatic `U=(A-1)v` unknown
  reparameterization in the DSL;
- ADM mass and momentum diagnostics or the nonlinear parameter search used to
  match requested physical masses;
- a mapped-domain preconditioner suitable for production resolutions;
- interpolation and metadata export to an external evolution solver.

The current manufactured Newton test uses a small dense Jacobian. It proves
that the transformed generated residual is solvable, not that the runtime is
already scalable or physically complete.

## Next milestone: TP-2 physical Hamiltonian residual

The next implementation should construct the two-puncture Bowen-York tensor
and solve the vacuum conformal Hamiltonian equation

```text
psi = 1 + m1/(2 r1) + m2/(2 r2) + U
Delta U + (1/8) Atilde_ij Atilde^ij psi^(-7) = 0
```

with `r1` and `r2` measured from the two punctures and `U=(A-1)v`. The first
physical regression should use the time-symmetric limit (`P1=P2=S1=S2=0`),
where `U=0` and the Brill-Lindquist conformal factor is exact. The next test
should enable equal-and-opposite momenta and compare the Hamiltonian residual,
convergence with spectral order, and ADM diagnostics against published
TwoPunctures results.

After TP-2, the production path is:

1. implement regularity/parity and mapped-domain preconditioning;
2. add spin, unequal masses, and the physical-parameter root search;
3. validate convergence and global charges against TwoPunctures reference
   cases;
4. interpolate the solved fields onto the external evolution grid and verify
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
