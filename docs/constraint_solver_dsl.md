# Initial-data constraint problems

Tensorium now distinguishes two forms of initial data:

- the existing analytic data (`metric4` or `alpha/beta/gamma`), evaluated
  directly;
- an **elliptic constraint problem**, described in the DSL and preserved in
  `ConstraintProblemIR` for spectral solver backends.

The executable backend now solves scalar radial problems across multiple
Chebyshev domains, including a compactified exterior domain. The full coupled
tensor system remains under development.

## Minimal example

```tensorium
params { mass }

initial_data BrillLindquist {
  domain near {
    coordinates = spherical
    topology = shell
    resolution = [25]
    basis = chebyshev
    bounds = [1, 4]
  }

  domain infinity {
    coordinates = spherical
    topology = compactified
    resolution = [17]
    basis = chebyshev
    bounds = [4]
  }

  interface near -> infinity

  unknown scalar psi
  equation scalar hamiltonian = laplacian(psi)

  boundary inner {
    psi = 1 + mass / (2 * r)
  }

  boundary outer {
    psi = 1
  }

  seed psi = 1

  solve {
    nonlinear = newton
    linear = direct
    tolerance = 1e-10
    max_iterations = 30
  }
}
```

An `equation` defines a **zero-valued residual**. In the example,
`hamiltonian = laplacian(psi)` therefore means `laplacian(psi) = 0` throughout
the domains.

## Enforced invariants

- A problem has at least one domain, unknown, equation, boundary, and `solve`
  block.
- Domain, unknown, equation, and boundary names are unique.
- An interface uses `interface inner_domain -> outer_domain`. Both domains
  must exist, and a domain can have at most one neighbor on either side.
- Unknowns are typed symbols distinct from evolution fields, parameters, and
  metrics.
- The existing tensor analysis checks the variance and index count of
  residuals, boundary conditions, and seeds.
- Boundary and seed assignments may only target declared unknowns.
- Domains currently accept `cartesian`, `spherical`, and `cylindrical`
  coordinates; `ball`, `shell`, `compactified`, `bispherical`, and
  `rectilinear` topologies; and `chebyshev`, `legendre`, `fourier`,
  `chebyshev_fourier`, and `legendre_fourier` bases.
- The exposed nonlinear strategy is `newton`, with either a `direct` or
  `gmres` linear solver.

A constraint-only problem does not require a `simulation` block: its domains
carry their own geometry and resolution. Time evolution still requires
`simulation`.

Finite mapped domains provide their two physical bounds:

```tensorium
domain exterior {
  coordinates = spherical
  topology = shell
  resolution = [49]
  basis = chebyshev
  bounds = [1, 20]
}
```

A compactified radial domain provides only its finite inner radius:

```tensorium
domain infinity {
  coordinates = spherical
  topology = compactified
  resolution = [17]
  basis = chebyshev
  bounds = [4]
}
```

For this domain the spectral coordinate `x in [-1,1]` is mapped with
`r = 2 R / (1 - x)`. Its first point is `r = R`, while its final collocation
point is exactly `r = infinity`.

## Executable radial backend

The current numerical backend supports:

- one or more spherical shell domains in declaration order;
- an optional compactified final domain;
- scalar and rank-one unknowns, paired with residuals of identical variance;
- three spatial components for each rank-one unknown;
- Chebyshev-Lobatto collocation;
- the radial spherical Laplacian
  `d2/dr2 + (2/r) d/dr`;
- `radial_derivative(f) = df/dr`;
- the flat conformal vector Laplacian for a radial vector amplitude;
- algebraic nonlinearities, powers, and radial derivatives;
- global `inner` and `outer` Dirichlet conditions;
- automatic `C0` and `C1` matching at every declared interface;
- forward-mode automatic differentiation of the complete coupled residual,
  including off-diagonal Jacobian blocks;
- damped Newton iteration and a pivoted dense linear solve.

Run the Brill-Lindquist radial validation problem from `build`:

```sh
./tools/driver/Tensorium_cc \
  --solve-constraints --param mass=1 \
  ../tests/fixtures/gr/brill_lindquist_radial_solve.tn
```

The fixture starts from `psi = 1`, solves `laplacian(psi) = 0` on
`r in [1,20]`, and applies the analytic values of
`psi = 1 + mass/(2*r)` at both boundaries. With 49 collocation points it
converges in one Newton update with a residual near machine precision and a
maximum solution error below `1e-9`.

Run the multidomain compactified problem:

```sh
./tools/driver/Tensorium_cc \
  --solve-constraints --param mass=1 \
  ../tests/fixtures/gr/brill_lindquist_multidomain_solve.tn
```

This problem joins `[1,4]` to `[4,infinity]`. It imposes both value and radial
derivative continuity at `r = 4`, applies `psi(infinity) = 1`, and reproduces
`psi = 1 + mass/(2*r)` with a maximum error below `1e-9`.

For a coupled system, unknowns and equations are paired by declaration order.
Every global boundary must constrain every unknown. For example, the coupled
regression problem contains

```tensorium
unknown scalar psi
unknown scalar w

equation scalar hamiltonian = laplacian(psi) + psi * w - 1
equation scalar momentum = laplacian(w) + psi^2 - 1
```

Run it with:

```sh
./tools/driver/Tensorium_cc --solve-constraints \
  ../tests/fixtures/gr/coupled_nonlinear_radial_solve.tn
```

This fixture solves both unknowns simultaneously across a finite shell and a
compactified exterior domain. It converges from unequal seeds to the exact
constant solution `psi = w = 1` and exercises the cross-unknown Jacobian
entries.

## Physical radial CTT system

The backend now executes a coupled vacuum conformal transverse-traceless
(CTT) system, rather than only equations shaped like constraints. For a flat
conformal metric, no freely specified transverse-traceless tensor, and a
spherically symmetric vector potential

```text
V^i = w(r) n^i,
```

the radial amplitude of the conformal vector Laplacian is

```text
(Delta_L V)^i = (4/3) [w'' + 2 w'/r - 2 w/r^2] n^i.
```

Tensorium exposes this full geometric operator, including the `4/3` factor,
as `radial_conformal_vector_laplacian(w)`. The longitudinal tensor also obeys

```text
(L V)_ij (L V)^ij = (8/3) (w' - w/r)^2.
```

Consequently the conformally flat vacuum CTT equations reduce to

```text
laplacian(psi)
  + (1/3) (w' - w/r)^2 psi^(-7)
  - (1/12) K^2 psi^5 = 0

radial_conformal_vector_laplacian(w)
  - (2/3) psi^6 K' = 0.
```

These are the spherical reduction of the standard CTT Hamiltonian and
momentum equations. The coefficient and sign conventions follow equations
(10)--(12) of [Assumpcao et al., *NRPyElliptic: A fast numerical-relativity
elliptic solver*](https://doi.org/10.1103/PhysRevD.105.104037).

The regression fixture prescribes
`K(r) = 3 * amplitude * r^(-3/2)`. It has the exact coupled solution

```text
psi(r) = 1
w(r) = amplitude * r^(-1/2),
```

because both the Hamiltonian terms and the momentum terms cancel pairwise.
It is solved on two matched shells from deliberately perturbed seeds:

```sh
./tools/driver/Tensorium_cc \
  --solve-constraints --param amplitude=0.2 \
  ../tests/fixtures/gr/ctt_radial_vacuum_solve.tn
```

This is a genuine coupled vacuum Einstein constraint solve on a bounded radial
interval under spherical symmetry and a conformally flat ansatz. It is not yet
a complete asymptotically flat data set or generic CTT/XCTS: the conformal
metric is fixed, `w` is a radial vector amplitude rather than three independent
angular fields, and the solver does not yet reconstruct and export the
physical `gamma_ij` and `K_ij` tensors.

## Rank-one component layout

Rank-one equations preserve their free index through semantic analysis and
IR lowering. The radial solver instantiates that index for components `0`,
`1`, and `2`. Scalar boundary or seed expressions are broadcast to every
component, which makes homogeneous vector data concise:

```tensorium
unknown scalar psi
unknown vector W[i]

equation scalar hamiltonian = laplacian(psi) + psi - 1
equation vector momentum[i] = laplacian(W[i]) + psi * W[i]

boundary outer {
  psi = 1
  W[i] = 0
}
```

Run the component-layout regression with:

```sh
./tools/driver/Tensorium_cc --solve-constraints \
  ../tests/fixtures/gr/scalar_vector_radial_solve.tn
```

The solution buffer for a rank-one unknown is component-major: all radial
points for component `0`, followed by components `1` and `2`. The CLI reports
both the point count per component and the total number of stored values.

At this stage `laplacian(W[i])` applies the scalar radial Laplacian to each
component independently. It is not yet the covariant vector Laplacian in a
spherical basis; connection terms and tensor contractions are the next
geometric lowering step.

## Pipeline status

```text
constrained initial_data DSL
  -> typed AST
  -> unknown/residual/boundary/seed analysis
  -> ConstraintProblemIR
  -> multidomain radial maps and Chebyshev-Lobatto collocation [implemented]
  -> compactified exterior and C0/C1 matching [implemented]
  -> coupled scalar/rank-one component layout and Jacobian [implemented]
  -> radial vacuum CTT Hamiltonian-momentum system [implemented subset]
  -> damped Newton and dense linear solve [implemented subset]
  -> [next] reconstruct and export physical gamma_ij and K_ij
  -> [next] generic covariant contractions and rank-two unknowns
```

The physical target includes the 3+1 constraint equations

```text
R + K^2 - K_ij K^ij - 16 pi rho = 0
D_j (K^ij - gamma^ij K) - 8 pi S^i = 0
```

together with conformal decompositions, multiple domains, compactified
infinity, matching conditions, excision, and punctures. The architecture is
inspired by KADATH's approach while remaining an independent Tensorium IR.

## Inspection

From the `build` directory:

```sh
./tools/driver/Tensorium_cc --validate \
  ../tests/fixtures/gr/brill_lindquist_constraints.tn

./tools/driver/Tensorium_cc --dump-ast --dump-backend-expr \
  ../tests/fixtures/gr/brill_lindquist_constraints.tn
```

MLIR/LLVM generation is not enabled for constraint problems yet; the radial
solver executes directly from `ConstraintProblemIR`. MLIR lowering still fails
explicitly rather than silently discarding the problem.
