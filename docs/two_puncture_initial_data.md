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

## TP-4: Matrix-free solve and an explicit symmetry subspace

The physical regression now forbids dense Jacobian assembly and uses the
generic Newton--GMRES path with a 64-vector Krylov limit. The GMRES least-
squares problem is updated by Givens rotations instead of normal equations;
this avoids squaring the condition number of the Arnoldi Hessenberg matrix.
The previous normal-equation implementation stalled near `1e-7` on the mapped
Hamiltonian operator even after using the complete 96-dimensional Krylov
space.

The selected `DiagonalJVP` preconditioner is map-aware without containing
TwoPunctures physics: its diagonal is sampled from the composed residual JVP,
after the unknown map, coordinate map, derivative map, and generated kernel.
It stores `O(N)` diagonal entries and does not construct an `N x N` Jacobian.
It is an effective regression preconditioner, but its `N` setup JVPs are not
yet the multilevel or sparse operator needed for production resolutions.

For the equal-mass, equal-and-opposite tangential-momentum case, the runtime
also installs an optional field projector onto

```text
v(A,B,phi) = v(A,-B,phi+pi).
```

This is even parity under Cartesian inversion for the current coordinate map.
The Newton state, Krylov directions, preconditioned corrections, and line-
search candidates are all projected. The projector is a runtime policy on an
individual residual problem; unequal binaries and unrelated generated DSL
problems do not inherit it.

The `3x3x4`, `4x4x6`, and `5x5x8` physical solves all converge through this
matrix-free path. Representative current results are:

```text
grid 4x4x6: Newton steps 2, cumulative GMRES iterations 38
grid 5x5x8: Newton steps 2, cumulative GMRES iterations 87
fine final linear residual L2: 4.1e-11
fine inversion-parity error: exactly zero after projection
```

## TP-5: Local puncture masses and bare-mass calibration

TP-5 implements the local ADM mass definition from Eq. (83) of the original
single-domain puncture paper. With puncture coordinate distance `D=2*b`, it is

```text
M_plus  = (1 + u_plus)  m_plus  + m_plus*m_minus/(4*b)
M_minus = (1 + u_minus) m_minus + m_plus*m_minus/(4*b).
```

Here `u_plus` and `u_minus` are the regular physical correction at the two
puncture ends, not the solver variable `v`. For the current unknown map
`u=(A-1)*v`, the runtime extrapolates `-2*v` to `A=-1`, `B=-1` and `B=+1`.
It averages the endpoint trace over `phi` to extract the regular scalar value
and separately reports the maximum angular variation. The latter is a direct
diagnostic of unresolved puncture regularity.

`calibrateTwoPunctureBareMasses` is independent of the elliptic implementation.
It receives a callback which, for a proposed pair of bare masses, solves the
problem and returns `u_plus,u_minus`. The calibrator analytically inverts the
two local-mass equations while holding the latest correction fixed, then asks
the callback to re-solve until the requested masses are reached. The callback
can therefore be backed by Tensorium's current generated residual solver or by
an external elliptic backend later.

The physical regression targets the puncture masses obtained from a reference
solve with `m_plus=m_minus=0.55`, then restarts the calibration from
`m_plus=m_minus=0.50`:

```text
target local masses:       0.6053070980728634, 0.6053070980728634
fine-grid local masses:    0.6052960934027002, 0.6052960934027002
recovered bare masses:     0.5499999644991266, 0.5499999644991266
backend solves:            4
final local-mass error:    2.2e-16
puncture phi variation:    3.1e-6
mass refinement change:    8.36e-4 -> 1.10e-5
```

This is a same-grid fixed-point regression, not yet a comparison with apparent-
horizon masses or a published production-resolution parameter set.

## TP-6: Behavioral regularity and bounded-memory scaling

The three logical boundary families `A=-1`, `B=-1`, and `B=+1` map to the
Cartesian `x` axis. A continuous scalar must have a unique value there,
independent of `phi`. `TwoPunctureRegularity.h` now measures the maximum
non-axisymmetric endpoint trace on all three families. It also provides an
idempotent transfinite projector which removes those traces while preserving
the Fourier average. A deliberately irregular manufactured field is reduced
from a boundary error of `1.48` to `4.2e-16`.

The physical solve does not impose this projector as an extra boundary
condition. The original method uses interior Chebyshev-zero collocation and
obtains regularity as a behavioral consequence of the elliptic equation.
Imposing endpoint traces as hard constraints at the current low orders makes
the discrete collocation equations over-constrained. Tensorium therefore uses
the projector as a testable policy and reports the unmodified physical trace;
on the current `7x7x12` solution its maximum variation is `4.2e-6`.

The linear solver uses restarted flexible GMRES. `gmresMaxIterations` is the
total iteration budget and `gmresRestart` bounds each Arnoldi basis. Each
preconditioned Arnoldi direction is retained and used directly in the solution
update, so the relaxation preconditioner may vary between iterations without
invalidating the Krylov correction. The physical regression uses FGMRES(24),
so Krylov storage is `O(24*N)` rather than growing with the full iteration
budget.

`MappedFiniteDifferenceLaplacianShift` is a generic sparse preconditioner. At
each collocation point it constructs nonuniform three-point first- and second-
derivative stencils, passes their derivative bundles through the configured
unknown and coordinate derivative maps, and retains the mapped physical
Laplacian plus an optional shift. The result has at most seven entries per row,
is built in `O(N)` work and storage, and is approximately inverted by symmetric
relaxation. Mixed logical derivatives produced by a fully non-orthogonal map
are deliberately omitted by this seven-point approximation; a wider stencil is
required to represent them. No TwoPunctures residual term is hard-coded into
this path.

`MappedFiniteDifferenceMultigrid` is the first geometric two-grid extension of
that operator. It coarsens every Chebyshev/Fourier axis, transfers fields with
the corresponding tensor-product spectral interpolation, applies symmetric
pre- and post-relaxation, and solves the Galerkin coarse correction `R A P`.
The dense coarse LU is built once with the preconditioner and reused by every
FGMRES application.

With the sparse mapped preconditioner, the physical regression now includes a
`7x7x12` grid with 588 unknowns:

```text
GMRES restart length:                    24
coarse cumulative linear iterations:    24
fine cumulative linear iterations:      69
two-grid cumulative linear iterations:  54
coarse nonlinear residual L2:            5.1e-11
fine final linear residual L2:           2.5e-11
two-grid nonlinear residual L2:          6.4e-11
fine scalar-axis regularity error:       4.2e-6
puncture-mass change:                    8.36e-4 -> 2.49e-4
```

This removes dense Jacobian storage and the `N` residual-JVP setup calls of the
previous diagonal preconditioner. The two-grid prototype reduces Krylov work
by about 22% on this case, but it is not recursive and does not yet unlock the
`12x12x20` QC0 solve. Production-scale distributed or algebraic multigrid
remains an external-backend concern.

## TP-7: Published unequal-mass case and Cartesian BSSN handoff

The physical regression now includes the `m_-/m_+=0.1` member of the test-mass
sequence in Table 1 of Ansorg, Bruegmann, and Tichy. With `m_+=1`, the published
parameters are

```text
D = 2*b = 5/2 + sqrt(6)
v_- = 4*sqrt(3)/(5 + 2*sqrt(6))
P_- = -P_+ = -m_-*v_-
S_- = S_+ = 0.
```

The paper places the holes and orbital momentum on different Cartesian axes;
the regression applies a rigid rotation and translation to use Tensorium's
punctures at `x=+b` and `x=-b` and tangential momentum along `y`. The scalar
observables are unchanged.

The paper reports the regular correction at the light puncture, a scaled value
at the heavy puncture, and the scaled asymptotic coefficient. Tensorium's
`10x10x16` regression compares all three:

```text
observable                         paper       Tensorium
u at the light puncture            0.03417     0.03283750
2*D*u at the heavy puncture/m_-    0.2011      0.20933912
lim(2*r*u)/m_-                     0.1688      0.16714861
maximum relative difference                    4.10%
```

The paper used `100x100x50` points for the nonzero mass-ratio rows. This is
therefore a low-resolution published-data guard, not a reproduction of the
paper's reported four-digit accuracy. Unlike the equal-mass regression, this
solve installs no inversion-symmetry projector.

`TwoPunctureHandoff.h` provides the first nonradial handoff path. It analytically
inverts the prolate TwoPunctures map for each Cartesian target point, evaluates
the solved regular correction with tensor-product spectral interpolation, and
writes caller-owned structure-of-arrays BSSN buffers:

```text
chi             = psi^(-4)
gamma_tilde_ij  = delta_ij
gamma_tilde^ij  = delta^ij
A_tilde_ij      = psi^(-6) Abar_ij
K               = 0
Gamma_tilde^i   = 0.
```

Here `Abar_ij` is reconstructed from both punctures' Bowen--York momenta and
spins. Exact puncture points are supported without evaluating the singular
tensor: the finite BSSN limits `chi=0` and `A_tilde_ij=0` are written, while the
optional diagnostic `psi` output is infinite. Lapse and shift outputs receive
explicit caller-supplied gauge seeds; they are not presented as constraint
solutions.

The regression checks inverse-map round trips, interpolation, unit determinant,
inverse metrics, symmetry and trace freedom, finite puncture limits, and the
SoA gauge output. It also re-evaluates the Hamiltonian and momentum constraints
after Cartesian interpolation with an independent centered finite-difference
probe. At the current low resolution this gives approximately
`|H|=5.0e-5` and `max|M_i|=1.2e-8`. These are handoff-grid diagnostics; the
generated collocation residual remains below its `2e-8` nonlinear tolerance.

The API deliberately accepts the generic `SpectralResidualProblem`, so the
solver-field-to-physical-field transform is obtained from its configured
unknown map rather than hard-coding `U=(A-1)v`. Generated initial-data metadata
now binds the TwoPunctures masses, momenta, spins, and half-separation directly
from the DSL declaration.

### Executable QC0 data set

`run_two_puncture_qc0.sh` is a thin wrapper around the generic
`run_initial_data.sh` command. The physical and numerical configuration is
declared in `spectral_two_puncture_hamiltonian_3d.tn`; no case-specific C++
runner contains these values. The source uses the standard QC0 parameters

```text
b = 1.168642873
m_+ = m_- = 0.453
P_+ = (0, +0.3331917498, 0)
P_- = (0, -0.3331917498, 0)
S_+ = S_- = 0.
```

From the repository root, run

```bash
./run_two_puncture_qc0.sh /tmp/tensorium_qc0_bssn_slice.csv
./plot_constraint_slice.py /tmp/tensorium_qc0_bssn_slice.csv chi
```

The generic command compiles the physical DSL residual to LLVM, reads its
generated initial-data descriptor, solves the nonlinear Hamiltonian constraint,
performs the Cartesian BSSN handoff, and writes both the requested CSV and
`<csv>.json` metadata. The default `10x10x16` spectral solve currently reports:

```text
Newton steps:                  4
cumulative FGMRES iterations: 67
Hamiltonian residual L2:       6.69e-9
Hamiltonian residual max:      7.57e-8
ADM energy:                    1.00792631
ADM angular momentum Jz:       0.77876433
puncture ADM masses:           0.51685532, 0.51685532
axis regularity error:         6.66e-6
BSSN trace error:              8.33e-17
```

An `8x8x12 -> 10x10x16` check changes the ADM energy by `1.98e-4`, each
puncture mass by `6.47e-5`, and improves the axis regularity diagnostic from
`4.63e-5` to `6.66e-6`.

The exported Cartesian `z=0` slice contains `u`, `psi`, `chi`, the
pre-collapsed gauge seed `alpha=psi^-2`, the six independent components of
`gamma_tilde_ij` and `A_tilde_ij`, `K`, all three `Gamma_tilde^i`, and the
zero shift seed. The JSON records physical parameters, resolutions, charges,
residuals, gauge choice, fields, spacing, and layout. The Cartesian sampling
resolution and slice extent can be changed without editing source:

```bash
TENSORIUM_SLICE_N=257 TENSORIUM_HALF_WIDTH=12 \
  ./run_initial_data.sh \
    tests/fixtures/elliptic/spectral_two_puncture_hamiltonian_3d.tn \
    /tmp/qc0_high.csv
```

`run_two_puncture_qc0.sh` selects `mapped_fd_multigrid` by default. For an
explicit comparison without changing the DSL, set `TP_PRECONDITIONER`; the
generic runner accepts the equivalent
`TENSORIUM_INITIAL_DATA_PRECONDITIONER` override. Sweep counts can likewise be
overridden with `TP_PRECONDITIONER_SWEEPS` or
`TENSORIUM_INITIAL_DATA_PRECONDITIONER_SWEEPS`:

```bash
TP_PRECONDITIONER=mapped_fd_laplacian_shift \
TP_PRECONDITIONER_SWEEPS=12 \
  ./run_two_puncture_qc0.sh /tmp/qc0_one_level.csv
```

This is a real solved data set and a reproducible handoff example. The CSV is
intentionally a diagnostic slice, not a production 3D evolution checkpoint;
an external solver should call the SoA handoff API on its own full Cartesian
grid. The spectral resolution and solver policy are part of the DSL `spectral`
block, so changing a physical case never requires editing or rebuilding a C++
runner. The `10x10x16` configuration is the validated default; neither the
one-level relaxation preconditioner nor the first two-grid prototype converges
reliably for QC0 at `12x12x20`, which is why production-resolution scaling
remains an explicit open item rather than an implied capability.

## Runtime performance controls

The spectral runtime caches the Chebyshev differentiation matrices, Fourier
phase tables, and Fourier derivative factors once per axis. Tensor-product
derivatives reuse one set of line buffers per worker instead of allocating for
every line. The generated runner reports solve and export wall times separately
so solver scaling can be measured without counting DSL compilation or CSV I/O.

`run_initial_data.sh` compiles the host runtime with `-O3 -DNDEBUG`. Two
optional controls are available for production experiments:

```bash
TENSORIUM_NATIVE=1 \
TENSORIUM_OPENMP=1 OMP_NUM_THREADS=8 \
  ./run_initial_data.sh problem.tn /tmp/initial_data.csv
```

`TENSORIUM_NATIVE=1` enables host-specific code generation and therefore makes
the executable non-portable. `TENSORIUM_OPENMP=1` parallelizes independent
tensor-product spectral lines; grids below 32,768 collocation points remain
sequential because thread-launch overhead dominates at that size. On macOS the
script detects MacPorts or Homebrew `libomp`; on other platforms it uses
`-fopenmp`. `TENSORIUM_CXXFLAGS` and `TENSORIUM_LDFLAGS` can append toolchain-
specific flags.

On the QC0 `10x10x16` regression, the current two-grid default completes four
Newton steps in about `0.17 s` with 67 cumulative Krylov iterations, compared
with 90 iterations for the previous one-level default. These are
single-machine engineering measurements, not a production-scaling claim.
Higher resolutions still require a stronger hierarchy and operator
approximation; the first two-grid prototype still fails the `12x12x20` QC0
acceptance criterion.

## What remains before production TwoPunctures

The TP-2 through TP-7 path is a genuine physical residual and nonlinear solve,
but it is still a small collocation regression. It does not yet provide:

- production-resolution convergence of every Fourier regularity mode at the
  axes and puncture corners;
- production-resolution convergence of the published unequal-mass case and
  published full-solve spinning comparisons;
- apparent-horizon masses or independent surface-integral charge checks;
- algebraic multigrid, distributed sparse solves, or multidomain
  preconditioning;
- a versioned C ABI or a concrete external-evolution-code adapter for the
  nonradial spectral solution;
- additional coordinate-map, projector, and reconstruction registry entries
  beyond the currently implemented identity and TwoPunctures policies.

The compiled spectral residual path currently differentiates one scalar
unknown per equation. General coupled scalar systems are supported through
auxiliary-unknown mappings, but arbitrary tensor-valued multidimensional
elliptic unknowns still require additional lowering and runtime work.

The scalar endpoint projector is not a complete tensor regularity system. In
particular, the expected `rho^|m|` behavior of each Fourier mode and the parity
of vector/tensor components still need production-resolution validation.

## Next milestone: TP-8 external adapter and production validation

The production path is now:

1. validate the published unequal-mass case at production resolution and add a
   published spinning full-solve comparison;
2. add apparent-horizon or independent surface-integral diagnostics;
3. expose the nonradial solution through a versioned C ABI and validate one
   concrete external evolution-code grid adapter.

## References

- M. Ansorg, B. Bruegmann, and W. Tichy, *A single-domain spectral method for
  black hole puncture data*, Phys. Rev. D 70, 064011 (2004),
  https://doi.org/10.1103/PhysRevD.70.064011.
- Einstein Toolkit, *TwoPunctures thorn documentation*,
  https://einsteintoolkit.org/thornguide/EinsteinInitialData/TwoPunctures/documentation.html.
- E. Bentivegna, *Solving the Einstein constraints in periodic spaces with a
  multigrid approach*, Class. Quantum Grav. 31, 035004 (2014), QC0 comparison,
  https://doi.org/10.1088/0264-9381/31/3/035004.
- P. Grandclement, *KADATH: a spectral solver for theoretical physics*,
  J. Comput. Phys. 229, 3334-3357 (2010),
  https://doi.org/10.1016/j.jcp.2010.01.005.
