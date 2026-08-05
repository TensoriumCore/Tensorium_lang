# Initial-data constraint problems

Tensorium now distinguishes two forms of initial data:

- the existing analytic data (`metric4` or `alpha/beta/gamma`), evaluated
  directly;
- an **elliptic constraint problem**, described in the DSL and preserved in
  `ConstraintProblemIR` for spectral solver backends.

The executable backend now solves coupled scalar, rank-one, and rank-two radial
problems across multiple Chebyshev domains, including a compactified exterior
domain. It also supports regular scalar problems on a spherical ball containing
the origin. Generic covariant tensor operators remain under development.

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

Finite shell domains provide their two physical bounds:

```tensorium
domain exterior {
  coordinates = spherical
  topology = shell
  resolution = [49]
  basis = chebyshev
  bounds = [1, 20]
}
```

A spherical ball contains the origin and provides only its outer radius:

```tensorium
domain interior {
  coordinates = spherical
  topology = ball
  resolution = [17]
  basis = chebyshev
  bounds = [2]
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

- an optional spherical ball as the first domain, followed by zero or more
  spherical shells in declaration order;
- an optional compactified final domain;
- scalar, rank-one, and rank-two unknowns, paired with residuals of identical
  variance;
- three spatial components for each rank-one unknown;
- nine row-major spatial components for each general rank-two unknown, or six
  components for an explicitly symmetric covariant/contravariant unknown;
- Chebyshev-Lobatto collocation;
- the radial spherical Laplacian
  `d2/dr2 + (2/r) d/dr`;
- `radial_derivative(f) = df/dr`;
- the flat conformal vector Laplacian for a radial vector amplitude;
- algebraic nonlinearities, powers, pointwise tensor products, and Einstein
  contractions over one or two repeated spatial indices;
- an optional fixed spherical-orthonormal background geometry, including
  composable covariant derivatives, scalar gradients, divergences, traces,
  and scalar/rank-one/rank-two rough Laplacians;
- curved-background longitudinal CTT, momentum-divergence, and sourced
  Hamiltonian residuals assembled directly from those tensor primitives;
- radial derivatives;
- global `inner` and `outer` Dirichlet conditions for shell-only layouts;
- an origin radial-derivative condition and an outer Dirichlet condition for
  layouts beginning with a ball;
- automatic `C0` and `C1` matching at every declared interface;
- forward-mode automatic differentiation of the complete coupled residual,
  including off-diagonal Jacobian blocks;
- damped Newton iteration and a pivoted dense linear solve.

### Regular scalar ball domains

For a layout whose first domain has `topology = ball`, the left boundary is
named `origin`. Its assignment is a radial-derivative condition, not a value
condition:

```tensorium
boundary origin {
  u = 0  # du/dr = 0 at r = 0
}
```

The backend evaluates the regular spherical-scalar limit
`laplacian(u)(0) = 3 * d2u/dr2(0)` instead of dividing by `r`. The current ball
implementation is intentionally limited to scalar unknowns on a flat
background. Rank-one/rank-two regularity, a nontrivial background geometry,
and expressions that contain their own explicit singular factors such as
`1/r` are rejected or remain outside this subset.

Run the manufactured regular Poisson problem from `build`:

```sh
./tools/driver/Tensorium_cc \
  --solve-constraints --param source=6 \
  ../tests/fixtures/gr/regular_ball_poisson_solve.tn
```

It solves `laplacian(u) = source` on `0 <= r <= 2`, with `u'(0) = 0` and
`u(2) = 2*source/3`. The exact solution is
`u(r) = source*r^2/6`; the regression checks every collocation value, including
the origin.

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

## Published exact-solution benchmarks

The test suite distinguishes published physical solutions from manufactured
operator tests. A fixture is labelled as paper-backed only when the equations,
symmetry assumptions, boundary data, and diagnostic quantities all fit the
current executable backend without an unmentioned reduction.

### Brill-Lindquist one-centre data

The multidomain fixture implements the one-centre sector of
[Brill and Lindquist, *Interaction Energy in Geometrostatics*](https://doi.org/10.1103/PhysRev.131.471).
It is the time-symmetric, conformally flat Schwarzschild slice

```text
psi(r) = 1 + M/(2 r),
Delta psi = 0.
```

The strict unit regression uses `M = 2`, so the inner shell boundary `r = 1`
is the Einstein-Rosen throat `r = M/2`. It checks every finite collocation
value against the exact conformal factor, recovers `M` from
`2 r (psi - 1)`, and verifies the throat areal radius
`R = psi^2 r = 2 M`. This is a genuine published solution, but only the
SO(3)-invariant one-hole member of the Brill-Lindquist family.

### Reissner-Nordstrom Einstein-Maxwell data

The charged nonlinear fixture follows the isotropic Reissner-Nordstrom data
and conformal Einstein-Maxwell equations reviewed by
[Bozzola and Paschalidis, *Initial data for general relativistic simulations of multiple electrically charged black holes with linear and angular momenta*](https://doi.org/10.1103/PhysRevD.99.104044),
especially Eqs. (21), (22), and (43a). For a time-symmetric slice with
conformal electric field `Ebar^r = Q/r^2`, the coupled solved system is

```text
Delta psi + (Ebar^r)^2 psi^(-3)/4 = 0,
d(Ebar^r)/dr + 2 Ebar^r/r = 0,
psi(r) = sqrt((1 + M/(2 r))^2 - Q^2/(4 r^2)).
```

Run the normalized `M = 1`, `Q = 0.6` benchmark from `build`:

```sh
./tools/driver/Tensorium_cc \
  --solve-constraints --param charge=0.6 \
  ../tests/fixtures/gr/reissner_nordstrom_einstein_maxwell_solve.tn
```

The solve joins `[0.4,2]` to compactified infinity. Newton iteration determines
both `psi` and the scalar SO(3) radial amplitude `electric` from independently
perturbed seeds. The unit regression checks `psi`, `electric = Q/r^2`, the
physical field `E^r = psi^(-6) electric`, and the Maxwell Gauss constraint at
all 58 collocation points. It also verifies that the inner isotropic horizon
`r_H = sqrt(M^2-Q^2)/2 = 0.4` has areal radius
`R_+ = M + sqrt(M^2-Q^2) = 1.8`.

The reconstruction declares the conformal electric amplitude explicitly:

```tensorium
reconstruct ctt {
  conformal_factor = psi
  conformal_electric_radial = electric
  mean_curvature = 0
}
```

No radial CTT vector potential is required for time-symmetric data; omitting
`radial_vector` reconstructs zero extrinsic curvature. The physical electric
vector can be interpolated to spherical or Cartesian external grids through
`interpolateRadialElectromagneticToGrid` or the versioned C handoff ABI. The
currently supported electrostatic reconstruction writes a zero magnetic
field.

### Published cases that are not yet faithful executable tests

| Published case | Missing capability |
| --- | --- |
| Multi-hole Brill-Lindquist or Misner data | Executable multidimensional or bispherical domains, multiple punctures or throats, and their matching conditions. |
| Boosted or spinning Bowen-York black holes | Angular dependence beyond SO(3), executable 3D vector/tensor fields, and puncture or excision treatment. The original data are explicitly non-spherical. |
| Generic charged black-hole binaries | Three-dimensional puncture domains, non-radial electric and magnetic fields, and the associated multidimensional Maxwell constraints. |
| TOV or rotating neutron-star data | An executable ball domain with regularity at `r = 0`, matter variables, an equation of state, surface matching, and coupled matter constraints. |
| Kerr/XCTS quasi-equilibrium data | Axisymmetric or 3D bases, coupled XCTS lapse and shift equations, and apparent-horizon/excision boundary conditions. |
| `f(R)`, DHOST, or other modified-gravity initial data | A theory-level declaration of the extra gravitational fields and their constraints, theory-specific source terms, physical boundary conditions, and reference solutions compatible with the chosen symmetry. Rank-two storage alone is not sufficient. |

Until those capabilities exist, promoting a manufactured radial equation or a
symmetry-reduced zero-momentum limit under one of these paper names would not
constitute a validation of the published physical case.

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

The reconstruction block identifies the solved conformal factor and optional
radial vector potential and supplies the prescribed mean-curvature expression.
When `radial_vector` is omitted, its longitudinal contribution is zero. The
block then creates physical spatial-metric and extrinsic-curvature profiles.

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

`interpolateRadialElectromagneticToGrid` uses the same target-grid contract.
It converts the solved conformal radial amplitude into the physical
contravariant field and writes three-component structure-of-arrays buffers:

```text
E^r = psi^(-6) Ebar^r                         (spherical)
E^i = psi^(-6) Ebar^r x^i/r                   (Cartesian)
B^i = 0                                       (electrostatic subset).
```

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

The legacy reconstruction fixture is a genuine coupled vacuum Einstein
constraint solve on a bounded radial interval under spherical symmetry and a
conformally flat ansatz. The curved-background fixture additionally solves a
three-component `W`, its symmetric longitudinal tensor, and a sourced
Hamiltonian residual, but remains an SO(3)-invariant radial reduction. It is
not yet a complete asymptotically flat data set or generic CTT/XCTS system.
Only the legacy flat reconstruction is currently interpolated into spherical
or Cartesian evolution buffers.

## Tensor component layout

Tensor equations preserve their free indices through semantic analysis and IR
lowering. The radial solver instantiates each index with spatial components
`0`, `1`, and `2`. Rank-one unknowns therefore have three components. General
rank-two unknowns have nine components in row-major order,
`component = 3*i + j`. This applies to `cov_tensor2`, `con_tensor2`, and
`mixed_tensor(up=1,down=1)`.

Purely covariant or contravariant rank-two unknowns may declare compact
symmetric storage:

```tensorium
unknown symmetric cov_tensor2 A[i,j]
unknown symmetric con_tensor2 B[i,j]
```

Their six stored components use the order `(00, 01, 02, 11, 12, 22)`.
`A[i,j]` and `A[j,i]` resolve to the same component, including inside the
automatically differentiated residual. A symmetric declaration is a contract;
Tensorium does not attempt to prove that an arbitrary residual preserves the
symmetry. Mixed tensors cannot use this modifier because exchanging an upper
and a lower index is not a tensor symmetry without an explicit metric.

Algebraic Einstein contractions are executable in radial residuals. For
example:

```tensorium
equation scalar hamiltonian =
    laplacian(psi) + contract(A[i,j] * B[i,j]) - 18
```

Here `A` is covariant and `B` is contravariant. The solver evaluates the full
ordered sum `sum_i sum_j A_ij B^ij`. Compact symmetric storage does not change
that mathematical sum: diagonal pairs occur once and off-diagonal pairs occur
twice. The same evaluator propagates automatic derivatives through both tensor
factors, so these terms populate the cross-field blocks of the Newton
Jacobian. The current executable limit is two simultaneously summed spatial
indices.

## Spherical-orthonormal background geometry

A radial constraint problem may declare one fixed background geometry:

```tensorium
geometry spherical_orthonormal {
  metric = gamma
  inverse_metric = gammaU
  radial_scale = 2
  tangential_scale = 1
}
```

The two positive scalar profiles define

```text
ds^2 = A(r)^2 dr^2 + (r B(r))^2 dOmega^2,
A = radial_scale,
B = tangential_scale.
```

Profiles may contain `r`, scalar parameters, arithmetic, powers, and the
executable scalar functions `sin`, `cos`, `sqrt`, and `exp`. They are fixed
during Newton iteration and cannot depend on an unknown or another field. Both
profiles must be finite and strictly positive at every collocation point.

Tensor components use the physical orthonormal frame
`(r_hat, theta_hat, phi_hat)`. Consequently, the declared `gamma[i,j]` and
`gammaU[i,j]` symbols both have numerical components `delta_ij`; they retain
opposite variance in the type system. The scale profiles enter frame
derivatives and the connection through

```text
e_r = (1/A) d/dr,
H = (1/A) (1/r + B'/B).
```

This first geometry backend represents SO(3)-invariant radial reductions. It
does not represent arbitrary angular dependence or general spherical-harmonic
modes; tangential components are interpreted in a transported orthonormal
frame, and the manufactured contractions below are valid in that reduction.

Within this geometry, `nabla_i` and `nabla^i` execute the connection corrections
for scalar and tensor expressions whose arguments have rank at most two. The
result may be differentiated again, so Hessians and derivatives of tensor
products or contractions remain covariant. Scalar `gradient`, `divergence`,
`trace`, and contractions compose with those operations and participate in
automatic differentiation. The scalar Laplacian is

```text
Delta f = 1/(A B^2 r^2) d/dr [(B^2 r^2/A) df/dr].
```

For a rank-one or rank-two tensor, `laplacian(T)` is the connection-aware rough
Laplacian

```text
D^a D_a T = sum_a [D_(e_a) D_(e_a) T - D_(D_(e_a) e_a) T].
```

The evaluator differentiates both the tensor-slot connection terms and the
covariant slot introduced by the first derivative. It therefore includes
radial derivatives of the connection and is not a component-wise scalar
shortcut.

Run the manufactured geometry regression with:

```sh
./tools/driver/Tensorium_cc --solve-constraints \
  ../tests/fixtures/gr/covariant_geometry_radial_solve.tn
```

For `A=2` and `B=1`, it verifies `Delta(r^2)=3/2`,
`T_ij=r delta_ij`, `trace(T)=3r`, and
`nabla^i T_ij=(1/2,0,0)`.

Run the second-order covariant regression with:

```sh
./tools/driver/Tensorium_cc --solve-constraints \
  ../tests/fixtures/gr/covariant_tensor_laplacian_radial_solve.tn
```

Writing `n_i = 2 nabla_i(r)`, its exact fields are
`V_i = r^2 n_i`, `T_ij = r^2 n_i n_j`, and
`Q_ij = D_i D_j(r^2) = delta_ij/2`. It verifies
`D^a D_a V_i = 2 nabla_i(r)` and
`D^a D_a T_ij = delta_ij/2`.

### Curved-background CTT residuals

The same primitives now express the conformal longitudinal operator without a
backend-specific shortcut:

```tensorium
equation con_tensor2 longitudinal[i,j] =
    L[i,j]
    - nabla^i(W[j])
    - nabla^j(W[i])
    + (2 / 3) * gammaU[i,j] * divergence(W[k])

equation vector momentum[j] =
    divergence(L[i,j]) - (2 / 3) * psi^6 * nabla^j(K)
```

These equations implement

```text
(L W)^ij = D^i W^j + D^j W^i - (2/3) gamma^ij D_k W^k,
D_i (L W)^ij - (2/3) psi^6 D^j K = 0.
```

The Hamiltonian residual can use an explicitly lowered covariant copy
`A_ij = gamma_ik gamma_jl L^kl` and the two-index contraction `L^ij A_ij`:

```text
Delta psi - (Rtilde/8) psi
  + (A_ij A^ij/8) psi^(-7)
  - (K^2/12) psi^5 + source = 0.
```

Run the coupled manufactured system with:

```sh
./tools/driver/Tensorium_cc --solve-constraints \
  ../tests/fixtures/gr/ctt_curved_background_solve.tn
```

It uses `A=2`, `B=1`, `Rtilde=3/(2 r^2)`, `psi=1`, and a polynomial radial
vector potential that vanishes with its longitudinal tensor at both shell
boundaries. A prescribed polynomial `K` makes the momentum equation exact;
the Hamiltonian includes a manufactured matter source. The solve recovers the
scalar, all three vector components, both compact symmetric tensors, and the
cross-field Newton derivatives simultaneously.

This curved residual path is not yet connected to `reconstruct ctt`, whose
current public export contract still accepts the legacy scalar radial vector
amplitude on a flat conformal background.

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

Run the nonlinear contraction regression with:

```sh
./tools/driver/Tensorium_cc --solve-constraints \
  ../tests/fixtures/gr/tensor_contraction_radial_solve.tn
```

Its manufactured solution is `psi = 1`, `A_ij = 1`, `B^ij = 2`, and
`C^i_j = 3`. It checks both the double contraction `A_ij B^ij = 18` and the
one-index contraction `B^ik A_kj = 6` in three spatial dimensions. The system
couples compact symmetric and general mixed rank-two fields and converges in
two Newton updates.

Solution buffers are component-major: all radial points for component `0`,
followed by all points for component `1`, and so on. The CLI reports both the
point count per component and the total number of stored values.

Without a `geometry` block, `laplacian(W[i])` and `laplacian(A[i,j])` retain the
legacy component-wise scalar radial behavior. With spherical-orthonormal
geometry, scalar, vector, and rank-two Laplacians include the frame and
connection terms described above.

## Generated multidimensional spectral constraints

The `constraints` kernel path is the current compiled multidimensional
counterpart to the host-side radial `initial_data` solver. A spectral
constraint remains an ordinary DSL equation: coordinates, parameters, local
scalar expressions, unknown derivatives, and nonlinear source terms are
lowered to generated point and grid residual kernels. Domain geometry, unknown
representation, Newton strategy, and final evolution-grid handoff remain
runtime policies.

`tests/fixtures/elliptic/spectral_two_puncture_hamiltonian_3d.tn` is the first
physical binary-black-hole example on this path. The DSL itself computes the
two puncture radii, arbitrary Bowen-York momentum and spin tensors, their
contraction, and the Lichnerowicz Hamiltonian residual. The runtime composes
that generated equation with the compact two-centre coordinate map and the
generic boundary-factor unknown map `U=(A-1)v`.

Generic tensor-product interpolation evaluates the solved collocation field at
fixed logical probes and compactified boundaries. The current regression uses
this to compare three resolutions and to extract the asymptotic regular field
for the ADM energy. Coordinate-specific global-charge formulas remain runtime
diagnostics rather than DSL/compiler intrinsics.

This separation is intentional. TwoPunctures is a validation target, not a
compiler mode: another DSL residual can reuse the same maps and solver, while
the same generated residual can be paired with a different domain or external
solver adapter.

## Pipeline status

```text
constrained initial_data DSL
  -> typed AST
  -> unknown/residual/boundary/seed analysis
  -> ConstraintProblemIR
  -> multidomain radial maps and Chebyshev-Lobatto collocation [implemented]
  -> regular scalar ball domain and origin Laplacian limit [implemented subset]
  -> compactified exterior and C0/C1 matching [implemented]
  -> coupled scalar/rank-one/rank-two layout and Jacobian [implemented]
  -> compact six-component symmetric rank-two storage [implemented]
  -> algebraic Einstein tensor contractions [implemented]
  -> fixed spherical-orthonormal geometry and covariant first derivatives [implemented]
  -> composable covariant derivatives and rank-one/rank-two rough Laplacians [implemented]
  -> radial vacuum CTT Hamiltonian-momentum system [implemented subset]
  -> curved-background longitudinal CTT and sourced Hamiltonian residuals [implemented radial subset]
  -> coupled radial Einstein-Maxwell Hamiltonian and Gauss constraints [implemented electrostatic subset]
  -> reconstruct and export physical gamma_ij and K_ij profiles [implemented]
  -> reconstruct and export physical electric and magnetic buffers [implemented electrostatic subset]
  -> interpolate profiles into spherical or Cartesian evolution buffers [implemented]
  -> initialize Cartesian BSSN buffers for external evolution [implemented]
  -> damped Newton and dense linear solve [implemented subset]
  -> [next] general matter-source declarations and regular tensor fields at the origin
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
