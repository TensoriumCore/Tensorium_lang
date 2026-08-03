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
- one scalar unknown and one scalar residual;
- Chebyshev-Lobatto collocation;
- the radial spherical Laplacian
  `d2/dr2 + (2/r) d/dr`;
- algebraic nonlinearities, powers, and radial derivatives;
- global `inner` and `outer` Dirichlet conditions;
- automatic `C0` and `C1` matching at every declared interface;
- forward-mode automatic differentiation of the discrete residual;
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

## Pipeline status

```text
constrained initial_data DSL
  -> typed AST
  -> unknown/residual/boundary/seed analysis
  -> ConstraintProblemIR
  -> multidomain radial maps and Chebyshev-Lobatto collocation [implemented]
  -> compactified exterior and C0/C1 matching [implemented]
  -> discrete residual and automatic Jacobian [implemented subset]
  -> damped Newton and dense linear solve [implemented subset]
  -> [next] coupled scalar/vector/tensor systems
  -> [next] export gamma_ij, K_ij, alpha, and beta^i
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
