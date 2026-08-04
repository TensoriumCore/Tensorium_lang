# Constraint handoff C ABI

Tensorium exposes a versioned C ABI for transferring solved initial data to an
external evolution code. The API performs the complete host-side sequence:

```text
Tensorium DSL source or file
  -> parse, semantic analysis, IR validation
  -> radial spectral CTT solve
  -> opaque converged solution
  -> physical CTT tensors, electromagnetic vectors, or Cartesian BSSN buffers
     owned by the caller
```

The public declarations are in
`include/tensorium/Solver/ConstraintHandoff.h`. Link against the
`tensoriumConstraintHandoff` CMake target. The header is valid C11 and C++;
the implementation catches C++ exceptions before returning through the ABI.

## Versioning and structure compatibility

ABI v1 is identified by `TENSORIUM_CONSTRAINT_HANDOFF_ABI_VERSION` and by the
`_v1` suffix on ABI types and functions. A caller should compare the compile-
time constant with `tensorium_constraint_handoff_abi_version()` before using
the interface.

Every public input or output structure begins with `struct_size`. Zero-
initialize the structure, then set that field to `sizeof(structure)`. The v1
implementation accepts a structure at least as large as the v1 definition,
which permits fields to be appended in a compatible future revision. A
smaller structure returns `TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH`.

All functions except destruction return `tensorium_constraint_status_v1`.
When an error buffer is supplied, its capacity includes the terminating null
byte. Tensorium truncates diagnostics to fit the buffer. A null error buffer
is valid only when its capacity is zero.

## Lifetime and ownership

`tensorium_solve_radial_constraints_source_v1` and
`tensorium_solve_radial_constraints_file_v1` return an opaque solution handle
only after a converged solve. The caller owns that handle and must release it
with `tensorium_constraint_solution_destroy_v1`. Passing null to the destroy
function is valid.

Tensorium does not retain pointers to parameter names, source text, target
coordinates, or output buffers after a call returns. Interpolation and BSSN
initialization read the solution handle and write directly into caller-owned
arrays. The caller must not destroy a handle while another call is using it.

## Buffer layout

Tensor fields use a structure-of-arrays layout. Each of the nine pointers
addresses `point_count` doubles, and pointer `3*i + j` stores tensor component
`(i,j)` at all grid points. Tensorium does not allocate target-grid arrays.

Physical CTT interpolation accepts either:

- spherical coordinates `(r, theta, phi)`, producing coordinate-basis tensors;
- Cartesian coordinates `(x, y, z)`, producing Cartesian tensors.

Cartesian BSSN initialization writes `chi`, `gamma_tilde_ij`,
`gamma_tilde^ij`, `A_tilde_ij`, and `K`. Lapse and shift buffers are optional
because gauge data is not fixed by the constraints. If any shift component is
provided, all three must be provided. BSSN initialization currently requires
a Cartesian target grid.

Einstein-Maxwell interpolation writes physical contravariant electric and
magnetic vectors through `tensorium_electromagnetic_buffers_v1`. Each of the
three component pointers addresses `point_count` doubles. The electrostatic
radial backend currently reconstructs `E^i = psi^(-6) Ebar^i` and writes a
zero magnetic field.

## Minimal C integration

```c
#include "tensorium/Solver/ConstraintHandoff.h"

char error[1024];
tensorium_constraint_parameter_v1 amplitude = {0};
amplitude.struct_size = sizeof(amplitude);
amplitude.name = "amplitude";
amplitude.value = 0.2;

tensorium_constraint_solution_v1 *solution = NULL;
tensorium_constraint_status_v1 status =
    tensorium_solve_radial_constraints_file_v1(
        "initial_data.tn", &amplitude, 1, &solution,
        error, sizeof(error));
if (status != TENSORIUM_CONSTRAINT_STATUS_OK) {
  /* Report error and abort initialization. */
}

tensorium_ctt_target_grid_v1 grid = {0};
grid.struct_size = sizeof(grid);
grid.coordinates = TENSORIUM_CTT_COORDINATES_CARTESIAN;
grid.point_count = point_count;
grid.coordinate_components[0] = x;
grid.coordinate_components[1] = y;
grid.coordinate_components[2] = z;

tensorium_ctt_bssn_buffers_v1 bssn = {0};
bssn.struct_size = sizeof(bssn);
bssn.point_count = point_count;
bssn.chi = chi;
bssn.mean_curvature = K;
for (int c = 0; c < 9; ++c) {
  bssn.conformal_metric[c] = gamma_tilde[c];
  bssn.inverse_conformal_metric[c] = gamma_tilde_inverse[c];
  bssn.trace_free_extrinsic_curvature[c] = A_tilde[c];
}

tensorium_bssn_gauge_seed_v1 gauge = {0};
gauge.struct_size = sizeof(gauge);
gauge.lapse = 1.0;
status = tensorium_initialize_bssn_from_radial_ctt_v1(
    solution, &grid, &bssn, &gauge, error, sizeof(error));

/* When reconstruct ctt declares conformal_electric_radial: */
tensorium_electromagnetic_buffers_v1 electromagnetic = {0};
electromagnetic.struct_size = sizeof(electromagnetic);
electromagnetic.point_count = point_count;
for (int c = 0; c < 3; ++c) {
  electromagnetic.electric_field[c] = electric[c];
  electromagnetic.magnetic_field[c] = magnetic[c];
}
status = tensorium_interpolate_radial_electromagnetic_v1(
    solution, &grid, &electromagnetic, error, sizeof(error));

tensorium_constraint_solution_destroy_v1(solution);
```

The external evolution code remains responsible for its own storage import,
ghost zones, boundary conditions, mesh refinement, gauge evolution, and time
integration. The ABI supplies initial data; it is not a time-stepping API.

## Current solver scope

ABI v1 exposes the current radial backend: multidomain Chebyshev-Lobatto
collocation, Newton iteration with a dense direct linear solve, spherical
symmetry, a flat conformal metric, and the reduced radial vector potential used
by the CTT fixture. It also exports the coupled radial electrostatic
Einstein-Maxwell solution. It does not yet provide generic three-dimensional
CTT/XCTS, non-radial electromagnetic fields, puncture or excision data,
distributed arrays, or parallel interpolation.
It also does not currently emit the BSSN conformal connection functions;
external BSSN codes must derive any additional formulation-specific state from
the transferred fields.

`Tensorium_constraint_handoff_c_probe` is compiled as C and exercises the
file-to-solution-to-BSSN path, coupled Einstein-Maxwell solve, Cartesian
electric-field transfer, numerical tensor invariants, gauge transfer, and ABI
mismatch handling.
