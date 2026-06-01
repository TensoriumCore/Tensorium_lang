# Tensorium Generated Kernel ABI (v1)

This document freezes the ABI contract used by generated functions:

- `tensorium_init`
- `tensorium_rhs`
- `tensorium_entry`
- `tensorium_init_point`
- `tensorium_init_grid_scf`
- `tensorium_init_grid_affine`
- `tensorium_rhs_grid_scf`
- `tensorium_rhs_grid_affine`
- `tensorium_rhs_grid_parallel`
- `tensorium_residual_grid_scf`
- `tensorium_residual_grid_affine`
- `tensorium_residual_grid_parallel`
- `tensorium_spectral_residual_<target>`
- `tensorium_spectral_residual_grid_<target>`

Source of truth for ABI constants:
`include/tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h`.

Source of truth for the host-callable kernel descriptor:
`include/tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h`.

Source of truth for the runtime spectral residual callback ABI:
`include/tensorium_mlir/Runtime/SpectralResidualKernel.h`.

Architectural context: `docs/language_mlir_abi_architecture.md` describes how
this ABI fits between Tensorium MLIR/LLVM kernels, generated host glue,
standalone execution, and optional AMReX integration.

## Versioning

- ABI version attribute: `tensorium.abi.version`
- Current value: `1`
- Memory layout attribute: `tensorium.abi.memory_layout = "soa_component_major"`
- Memref ABI attribute: `tensorium.abi.memref_abi = "strided_memref_rank1_f64"`

Every generated function above carries these attrs, plus a stable
`tensorium.abi.kind`.

## Metadata attrs per function

Generated functions expose argument-order metadata:

- `tensorium.abi.param_names`: runtime scalar parameter order (`f64` values)
- `tensorium.abi.coord_names`: coordinate buffer order (`x,y,z` or `r,theta,phi`)
- `tensorium.abi.field_names`: field buffer order (matching function args)
- `tensorium.abi.output_names`: written output field names
- `tensorium.abi.read_arg_indices`: absolute argument indices read by the kernel
- `tensorium.abi.write_arg_indices`: absolute argument indices (in function
  signature) written by the kernel
- `tensorium.abi.stencil_radius`: required interior ghost/radius width for RHS
  and residual grid kernels, derived from lowered stencil reads
- `tensorium.abi.residual_kernel`: internal marker on source residual modules
  that asks grid lowering to expose residual aliases

## C/C++ low-level memref contract

After LLVM lowering, each `memref<?xf64>` or `memref<Nxf64>` argument is lowered
to the 5-value descriptor:

1. `double *allocated`
2. `double *aligned`
3. `int64_t offset`
4. `int64_t size`
5. `int64_t stride`

The helper host-side shape is provided as:

`tensorium_mlir::abi::StridedMemRef1DF64`

in `include/tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h`.

## Generated C Host Header

The driver can emit a C header that mirrors the lowered ABI and adds thin
buffer wrappers:

```bash
Tensorium_cc --emit-host-header tensorium_generated_host.h <file.tn>
```

The header contains:

- raw `extern void tensorium_*` prototypes with expanded memref descriptors,
- `tensorium_memref1d_f64` for callers that need descriptor-level access,
- convenience wrappers such as `tensorium_call_init_grid_affine(...)` and
  `tensorium_call_rhs_grid_affine(...)`.

Convenience wrappers accept plain `double *` buffers and compute descriptor
sizes from ABI metadata and field tensor ranks. For example, a rank-2 spatial
field uses `9 * n_points` in 3D, and a rank-3 spatial field uses
`27 * n_points`.

The internal host module descriptor also exposes simulation metadata
(`dimension`, coordinate system, resolution, spatial scheme/order), field
descriptors (`name`, variance, rank, component count), and per-kernel
read/write/stencil metadata. This descriptor is the intended source for future
C++ runtime and AMReX wrappers.

## Runtime buffer contract

`HostModuleABI` now materializes the buffer-level contract that a runtime should
consume directly:

- each `HostFieldABI` records field name, variance (`up/down`), rank, and
  component count per grid point;
- each `HostKernelABI` records raw scalar/memref arguments and a `buffers`
  table;
- each `HostBufferABI` records logical buffer name, C-safe name, absolute
  function argument index, role (`Coordinate`, `Field`, `Output`), access
  (`Read`, `Write`, `ReadWrite`, `None`), variance/rank, and component count;
- `requiredBufferScalars(buffer, nPoints)` returns the exact scalar allocation
  size required by SoA component-major layout;
- `validateHostModuleABI(abi)` checks the descriptor before a runtime trusts it.
- `tensorium_mlir::runtime::HostFieldStorage` builds a deduplicated storage
  plan from the ABI and a uniform grid shape. It allocates one contiguous scalar
  arena for all logical buffers, then exposes stable per-kernel binding plans and
  rank-1 memref descriptors into that arena.
- `tensorium_mlir::runtime::GeneratedHostStorage` provides the same uniform-grid
  storage plan from generated C descriptor tables, so standalone runners can
  consume `tensorium_host_kernels` / `tensorium_host_buffers` without linking
  the compiler-side MLIR ABI builder.
- Generated host headers also emit `tensorium_host_kernel_adapters`, a uniform
  invocation table for grid kernels. `GeneratedHostStorage::invoke(...)` uses
  those adapters to bind runtime-owned buffers by descriptor order instead of
  requiring callers to spell every lowered field argument manually.
- `constraints` DSL blocks lower through the same scalarization path as RHS
  kernels but additionally expose `tensorium_residual_grid_affine` /
  `tensorium_residual_grid_scf` / `tensorium_residual_grid_parallel`
  host-callable symbols. These kernels compute residual buffers `F(u)`; the
  solver runtime is responsible for choosing the update method that drives
  those residuals toward zero.
- `tensorium_rhs_grid_parallel` and `tensorium_residual_grid_parallel` have the
  same low-level argument layout as the other field-grid kernels. Their MLIR
  body uses `scf.parallel`; LLVM emission lowers it through OpenMP runtime
  calls.
- `GeneratedHostStorage` also exposes a descriptor-level Euler helper for
  standalone runtime experiments: `eulerUpdatePairsFromDerivativePrefix()`
  discovers writable derivative fields named `dX` and maps them to state field
  `X`, then `applyEulerUpdate(...)` performs `X += dt * dX` over the runtime
  arena. This is intentionally minimal and should be replaced by the target
  runtime's integrator once AMReX owns the storage.
- Spectral initial-data experiments use `tensorium_spectral_residual_point` and
  `tensorium_spectral_residual_kernel_fn` as the pointwise residual callback
  ABI. The runtime constructs the derivative bundle on the selected spectral
  grid, applies an optional coordinate map, then calls the generated callback to
  compute one scalar `F(u)=0` value per collocation point. This keeps
  NRPy/Kadath/TwoPunctures-style formulations above the generic spectral grid
  and solver machinery.
- For `constraints` modules with `spatial { scheme = spectral order = 0 }`, the
  compiler emits `tensorium_spectral_residual_<target>` point kernels for scalar
  residuals whose RHS depends on one scalar unknown, its supplied spectral
  derivatives, and optional scalar auxiliary fields. Generated host headers
  expose these through `tensorium_spectral_residual_kernels`.
- `SpectralResidualProblem` is the runtime-side assembly surface for these point
  kernels. It binds the grid, generated callback, scalar params, optional
  auxiliary fields, optional coordinate map, and optional generated grid kernel.
  `assembleSpectralResidual(...)` computes the global collocation vector `F(u)`
  plus L2/max norms. When `SpectralResidualProblem::gridKernel` is set from
  `tensorium_spectral_residual_grid_kernels`, assembly uses the generated
  MLIR/LLVM global kernel; otherwise it falls back to the pointwise callback
  loop. `evaluateSpectralJacobianVectorProduct(...)` provides a
  finite-difference JVP hook for future Newton/Krylov elliptic solvers.
- `solveSpectralNewton(...)` is the first scalar spectral elliptic solve path.
  It uses finite-difference JVPs to assemble a dense Jacobian, solves the dense
  Newton system with pivoting, and performs a damped residual-decreasing line
  search. In `SpectralLinearSolveKind::Auto`, small problems use the dense
  Jacobian path and grids above `denseJacobianMaxUnknowns` use matrix-free GMRES
  over the same JVP interface. The current GMRES path is unpreconditioned and is
  intended as the scalable solve interface; production physical grids still need
  preconditioning and richer boundary/domain handling.
- The compiler also emits `tensorium_spectral_residual_grid_<target>` MLIR/LLVM
  kernels. These consume the runtime-computed spectral derivative buffers,
  auxiliary field buffers, coordinate buffers, scalar params, and one residual
  output buffer, then call the point kernel inside an MLIR `scf.for` loop. This
  moves global `F(u)` evaluation into generated code while keeping spectral
  differentiation in the runtime grid layer for now.
  Generated host headers expose uniform descriptors through
  `tensorium_spectral_residual_grid_kernels`.

Generated host headers also expose the same runtime contract in C-compatible
tables:

- `TENSORIUM_HOST_KERNEL_COUNT` / `tensorium_host_kernels`;
- `TENSORIUM_HOST_BUFFER_COUNT` / `tensorium_host_buffers`;
- `TENSORIUM_HOST_KERNEL_ADAPTER_COUNT` /
  `tensorium_host_kernel_adapters`;
- `tensorium_host_buffer_desc::component_count`, `role`, `access`, and
  `arg_index` are enough for a lightweight runtime to deduplicate buffers and
  bind generated wrappers without reconstructing tensor sizes by hand.

Runtime code should use this contract instead of reconstructing argument order
from names. For AMReX this means:

- allocate one `MultiFab` component group per logical field or map component
  ranges according to `componentCount`;
- allocate at least `stencilRadius` ghost cells for RHS kernels;
- bind read/write buffers according to `HostBufferABI::access`;
- reject descriptors with non-empty `validateHostModuleABI` diagnostics.

Generated grid loops should not perform per-point heap allocation. Scratch
buffers used by fallback init-grid lowering are hoisted outside the generated
loop and reused for every point. RHS old-state snapshots, when required by
read/write overlap, are full-grid temporaries allocated once before the loop and
released after it.

The development probe can print this contract for a fixture:

```bash
build/tools/runtime/Tensorium_abi_probe \
  tests/fixtures/gr/schwarzschild_bssn_constraints_analytic_3d.tn
```

## Signature contracts

### `tensorium_init_point`

MLIR-level:
- `(params..., coords..., alpha: memref<1xf64>, gamma: memref<9xf64>, gammaU: memref<9xf64>) -> ()`

LLVM-level:
- scalar `f64` params/coords first,
- then 3 memref descriptors (`alpha`, `gamma`, `gammaU`), each expanded to 5 C
  arguments.

### `tensorium_init_grid_{scf,affine}`

MLIR-level:
- `(params..., coord_buffers..., alpha, gamma, gammaU) -> ()`
- coord/output buffers are `memref<?xf64>`.

LLVM-level:
- scalar params first,
- then one descriptor per coord/output buffer.

### `tensorium_rhs_grid_{scf,affine}`

MLIR-level:
- `(nx:index, ny:index, nz:index, dx:f64, dy:f64, dz:f64, params..., fields...) -> ()`
- each field is `memref<?xf64>`.

LLVM-level:
- prefix: `i64,i64,i64,double,double,double`,
- then scalar params (`double`),
- then one 5-argument memref descriptor per field buffer.

### `tensorium_residual_grid_{scf,affine}`

Same low-level signature shape as `tensorium_rhs_grid_{scf,affine}`:
- prefix: `i64,i64,i64,double,double,double`,
- scalar params,
- one 5-argument memref descriptor per participating field buffer.

The semantic difference is the ABI kind:
- `tensorium.abi.kind = "residual_grid_scf"` or
  `"residual_grid_affine"`.

Residual grid kernels write the declared `residual` targets from a
`constraints` block. Host wrappers and descriptor tables expose those outputs in
the same `tensorium.abi.output_names` and access metadata used by RHS kernels.

### `tensorium_{rhs,residual}_grid_parallel`

Same low-level signature shape as `tensorium_rhs_grid_affine`:
- prefix: `i64,i64,i64,double,double,double`,
- scalar params,
- one 5-argument memref descriptor per participating field buffer.

The semantic difference is the ABI kind:
- `tensorium.abi.kind = "rhs_grid_parallel"` or
  `"residual_grid_parallel"`.

The Tensorium MLIR body is a three-dimensional `scf.parallel` over the interior
stencil domain. The LLVM lowering pipeline converts this path to OpenMP runtime
calls, so executables that link the generated object need an OpenMP runtime.

### `tensorium_spectral_residual_<target>`

MLIR-level:
- `(value, d1, d2, d3, d11, d12, d13, d22, d23, d33, aux..., x1, x2, x3, params...) -> f64`

Host callback-level:
- generated headers define a `tensorium_spectral_residual_kernel_desc` entry;
- the callback receives `tensorium_spectral_residual_point`, scalar params, and
  optional user data;
- the first `tensorium.abi.field_names` entry is the differentiated unknown;
  subsequent scalar fields are passed as `point.aux_values[]` in field-name
  order;
- `point.physical[]` is populated by the runtime coordinate map before the
  generated residual is called.

The initial compiler path supports scalar single-unknown residuals with scalar
auxiliary point fields. Multi-unknown systems are intentionally left to the next
ABI extension.

### `tensorium_spectral_residual_grid_<target>`

MLIR-level:
- `(n_points:index, params..., value, d1, d2, d3, d11, d12, d13, d22, d23, d33, aux..., x1, x2, x3, residual_out) -> ()`
- derivative, auxiliary, coordinate, and output buffers are `memref<?xf64>`.

Generated host wrapper:
- `tensorium_call_spectral_residual_grid_<target>(n_points, params..., value, d1, ..., d33, aux..., x1, x2, x3, residual_out)`
- `tensorium_spectral_residual_grid_kernels[]` exposes the same generated
  kernel through a uniform callback consumed by `SpectralResidualProblem`.

This kernel does not compute spectral derivatives itself. The runtime supplies
the derivative buffers from its selected spectral basis and coordinate mapping,
then the generated MLIR loop evaluates the residual at every collocation point.

## Memory layout contract (SoA, component-major)

Tensor field components are flattened as:

- `flat = component * nPoints + pointLinearIndex`

where:
- `component` is row-major over tensor indices,
- `pointLinearIndex` is row-major over `(x,y,z)` grid index.

Examples:
- covariant/convariant 2-tensor uses 9 components.
- rank-3 tensor uses 27 components.

## RHS read/write semantics

`tensorium_rhs_grid_*` snapshots all field buffers before stencil reads, then
writes `dt_assign` targets to original field buffers. This guarantees read
consistency within one RHS sweep and avoids write-after-read hazards.
