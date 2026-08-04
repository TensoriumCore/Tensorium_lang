# Initial-data constraint problems

Tensorium now distinguishes two forms of initial data:

- the existing analytic data (`metric4` or `alpha/beta/gamma`), evaluated
  directly;
- an **elliptic constraint problem**, described in the DSL and preserved in
  `ConstraintProblemIR` for spectral solver backends.

The executable backend now solves coupled scalar, rank-one, and rank-two radial
problems across multiple Chebyshev domains, including a compactified exterior
domain. Generic covariant tensor operators remain under development.

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
- scalar, rank-one, and rank-two unknowns, paired with residuals of identical
  variance;
- three spatial components for each rank-one unknown;
- nine row-major spatial components for each general rank-two unknown;
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

```tensorium
reconstruct ctt {
  conformal_factor = psi
  radial_vector = w
  mean_curvature = 3 * amplitude * r^(-1.5)
}
```

The reconstruction block identifies the solved conformal factor and radial
vector potential and supplies the prescribed mean-curvature expression. It
then creates physical spatial-metric and extrinsic-curvature profiles.

```sh
./tools/driver/Tensorium_cc \
  --solve-constraints --param amplitude=0.2 \
  ../tests/fixtures/gr/ctt_radial_vacuum_solve.tn
```

The profiles use the flat spherical orthonormal coframe
`(dr, r dtheta, r sin(theta) dphi)`. If `q = w' - w/r`, the stored diagonal
profiles are

```text
gamma_radial     = psi^4
gamma_tangential = psi^4
k_radial         = psi^(-2) (4/3) q + (1/3) psi^4 K
k_tangential     = psi^(-2) (-2/3) q + (1/3) psi^4 K.
```

They convert to the spherical coordinate basis as

```text
gamma_rr       = gamma_radial
gamma_thetatheta = r^2 gamma_tangential
gamma_phiphi     = r^2 sin(theta)^2 gamma_tangential

K_rr           = k_radial
K_thetatheta     = r^2 k_tangential
K_phiphi         = r^2 sin(theta)^2 k_tangential.
```

Export all collocation points to a CSV file with:

```sh
./tools/driver/Tensorium_cc \
  --export-constraint-csv ctt_initial_data.csv \
  --param amplitude=0.2 \
  ../tests/fixtures/gr/ctt_radial_vacuum_solve.tn
```

The CSV records the domain name, radius, solved `psi` and `w`, prescribed
mean curvature, and the four reconstructed physical profiles. Interface
points occur once for each adjacent domain, preserving the spectral domain
layout.

## Evolution-grid handoff

`interpolateRadialCttToGrid` transfers a converged reconstructed CTT solution
from its multidomain spectral grid to an arbitrary target grid. It selects the
source shell for every target radius and evaluates each physical profile with
Chebyshev-Lobatto barycentric interpolation. The same interpolation applies to
finite shells and to the compactified exterior coordinate.

The public API accepts `CttTargetGrid` and writes `CttEvolutionBuffers`.
Spatial metric, inverse spatial metric, and extrinsic curvature buffers use a
structure-of-arrays layout with nine component pointers each. Components are
row-major, so component `3*i + j` stores tensor entry `(i,j)` for all target
points. This is the same component-major convention used by generated tensor
kernels. Mean curvature is an optional scalar output buffer.

For `CttTargetCoordinates::Spherical`, the three input coordinates are
`(r, theta, phi)`. The handoff writes the spherical coordinate-basis tensors
shown above, including the radial scale and angular scale factors. For
`CttTargetCoordinates::Cartesian`, the input is `(x, y, z)`. With
`n_i = x_i/r`, a radial/tangential profile pair is lifted as

```text
gamma_ij = gamma_tangential delta_ij
         + (gamma_radial - gamma_tangential) n_i n_j

K_ij = k_tangential delta_ij
     + (k_radial - k_tangential) n_i n_j.
```

All target coordinates must be finite. Radii must also be positive and covered
by the solved domains.
The handoff deliberately does not construct lapse or shift: those are gauge
variables and are not determined by the CTT constraint equations.

## Cartesian BSSN initialization

`initializeBssnFromRadialCtt` converts the interpolated physical tensors into
the Cartesian BSSN variables used by evolution kernels. For
`gamma = det(gamma_ij)`, the conversion is

```text
chi             = gamma^(-1/3)
gamma_tilde_ij  = chi gamma_ij
gamma_tilde^ij  = gamma^ij / chi
A_tilde_ij      = chi (K_ij - gamma_ij K/3).
```

The resulting conformal metric has unit determinant and `A_tilde_ij` is
trace-free with respect to its inverse. `CttBssnBuffers` uses a
structure-of-arrays, component-major layout suitable for transfer into an
external evolution code. The conversion currently requires a Cartesian target
grid; spherical BSSN needs a reference-metric formulation and is not
approximated by this API.

`BssnGaugeSeed` supplies the initial lapse and shift when their optional output
buffers are present. These are explicit gauge choices rather than results of
the elliptic CTT solve.

A single DSL module may also contain both the constraint `initial_data` block
and an `evolution` block. Constraint equations execute through the spectral
solver, while the evolution block can still lower to MLIR and LLVM for
verification. For example, from `build`:

```sh
./tools/driver/Tensorium_cc \
  --solve-constraints --param amplitude=0.2 \
  --emit-llvm /tmp/ctt_bssn_handoff.ll \
  ../tests/fixtures/gr/ctt_bssn_handoff.tn
```

This emits `tensorium_rhs_grid_affine` after solving the same module's CTT
problem. The generated analytic initialization entry point is not the
constraint solver: a host must call `solveRadialConstraintProblem` followed by
`initializeBssnFromRadialCtt`.

Production time evolution is intentionally outside the constraint backend.
An external solver should copy or bind the resulting BSSN buffers into its own
grid storage and apply its own boundary conditions, gauge evolution,
time-integration scheme, and mesh-refinement policy. The generated Tensorium
RHS path is an optional verification path, not the required production
evolution engine.

External C and C++ solvers can use the versioned, exception-safe handoff ABI
described in [Constraint handoff C ABI](constraint_handoff_c_abi.md). It owns
the DSL-to-solution lifetime and writes physical CTT or Cartesian BSSN data
directly into caller-owned structure-of-arrays buffers.

This is a genuine coupled vacuum Einstein constraint solve on a bounded radial
interval under spherical symmetry and a conformally flat ansatz. It is not yet
a complete asymptotically flat data set or generic CTT/XCTS: the conformal
metric is fixed and `w` is a radial vector amplitude rather than three
independent angular fields. The reconstructed radial tensors can now be
interpolated into full spherical or Cartesian evolution buffers, but the
constraint solve itself remains spherically symmetric.

## Tensor component layout

Tensor equations preserve their free indices through semantic analysis and IR
lowering. The radial solver instantiates each index with spatial components
`0`, `1`, and `2`. Rank-one unknowns therefore have three components. General
rank-two unknowns have nine components in row-major order,
`component = 3*i + j`. This applies to `cov_tensor2`, `con_tensor2`, and
`mixed_tensor(up=1,down=1)`.

Scalar boundary or seed expressions are broadcast to every component, which
makes homogeneous tensor data concise:

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

Run the rank-two regression problem with:

```sh
./tools/driver/Tensorium_cc --solve-constraints \
  ../tests/fixtures/gr/rank_two_radial_solve.tn
```

It solves covariant, contravariant, and mixed rank-two equations together. The
covariant and contravariant residuals also access the transposed component,
which exercises cross-component entries in the automatic Newton Jacobian.

Solution buffers are component-major: all radial points for component `0`,
followed by all points for component `1`, and so on. The CLI reports both the
point count per component and the total number of stored values.

At this stage `laplacian(W[i])` and `laplacian(A[i,j])` apply the scalar radial
Laplacian to each component independently. They are not yet covariant tensor
Laplacians in a spherical basis. Rank-two storage is general and does not yet
compress a declared symmetric tensor from nine to six components. Connection
terms, contractions, and explicit tensor symmetry are the next geometric
lowering steps.

## Pipeline status

```text
constrained initial_data DSL
  -> typed AST
  -> unknown/residual/boundary/seed analysis
  -> ConstraintProblemIR
  -> multidomain radial maps and Chebyshev-Lobatto collocation [implemented]
  -> compactified exterior and C0/C1 matching [implemented]
  -> coupled scalar/rank-one/rank-two layout and Jacobian [implemented]
  -> radial vacuum CTT Hamiltonian-momentum system [implemented subset]
  -> reconstruct and export physical gamma_ij and K_ij profiles [implemented]
  -> interpolate profiles into spherical or Cartesian evolution buffers [implemented]
  -> initialize Cartesian BSSN buffers for external evolution [implemented]
  -> damped Newton and dense linear solve [implemented subset]
  -> [next] generic covariant contractions and tensor symmetry metadata
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

The radial solver executes constraint equations directly from
`ConstraintProblemIR`; it does not lower those elliptic equations to MLIR.
Constraint-only modules therefore still fail MLIR lowering explicitly rather
than silently discarding the problem. A combined constraint-and-evolution
module can lower its evolution RHS after the host-side constraint solve.
