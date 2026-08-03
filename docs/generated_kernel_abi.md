# Tensorium Generated Kernel ABI (v2)

This document freezes the ABI contract used by generated functions:

- `tensorium_init`
- `tensorium_rhs`
- `tensorium_entry`
- `tensorium_init_point`
- `tensorium_init_grid_scf`
- `tensorium_init_grid_affine`
- `tensorium_rhs_grid_scf`
- `tensorium_rhs_grid_affine`

Source of truth in code: `include/tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h`.

## Versioning

- ABI version attribute: `tensorium.abi.version`
- Current value: `2`
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
- `tensorium.abi.write_arg_indices`: absolute argument indices (in function
  signature) written by the kernel
- `tensorium.abi.halo_width`: number of grid points excluded on every side of
  each axis by an RHS kernel

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
- `(nx:index, ny:index, nz:index, dx:f64, dy:f64, dz:f64, params..., inputs..., outputs...) -> ()`
- every input and output is a dynamically strided rank-one `f64` memref,
- `tensorium.abi.field_names` names the input memrefs,
- `tensorium.abi.output_names` names the output memrefs,
- `tensorium.abi.write_arg_indices` contains the absolute function argument
  index of each output memref.

LLVM-level:
- prefix: `i64,i64,i64,double,double,double`,
- then scalar params (`double`),
- then one 5-argument memref descriptor per input field,
- then one 5-argument memref descriptor per output field.

This output suffix is the incompatible change from ABI v1. Host code must not
pass only the input descriptors or expect input state to be overwritten.

## Memory layout contract (SoA, component-major)

Tensor field components are flattened as:

- `flat = component * nPoints + pointLinearIndex`

where:
- `component` is row-major over tensor indices,
- `pointLinearIndex` is row-major over `(x,y,z)` grid index.

The memref descriptor then maps a logical flattened index to storage as:

- `address = aligned + offset + flat * stride`

ABI v2 generated grid functions honor both `offset` and `stride`. The
`allocated` pointer is retained for the standard memref descriptor contract;
loads and stores use `aligned`.

Examples:
- covariant/convariant 2-tensor uses 9 components.
- rank-3 tensor uses 27 components.

## RHS read/write semantics

`tensorium_rhs_grid_*` treats every field listed by `field_names` as read-only
state and writes each `dt_assign` result to the separate output memref with the
same name in `output_names`. Inputs and outputs must not alias. The generated
kernel performs no heap allocation and does not make whole-grid snapshots.

For a field of tensor rank `r` in three spatial dimensions, the caller must
provide at least `3^r * nx * ny * nz` logical elements. Scalar fields need
`nx * ny * nz`. ABI v2 checks every descriptor's logical `size` and the minimum
grid extent before entering the loop nest. If any descriptor is too short or
an axis cannot contain both halos, the void kernel returns without writing any
output. The caller remains responsible for ensuring that the pointer and
physical allocation really cover the declared offset, size, and stride.

The kernel writes the half-open region
`[halo,nx-halo) x [halo,ny-halo) x [halo,nz-halo)`. The uniform halo is exposed
as `tensorium.abi.halo_width` and is computed from explicit reference offsets
and nested centered derivatives in the RHS expression graph. Second-order
centered derivatives use radius one and fourth-order centered derivatives use
radius two per nesting level. Output values in the excluded boundary region
are left untouched. Algebraic RHS kernels have a zero halo and therefore write
every grid point, including boundaries.

## Constraint-backed initial data

A module whose `initial_data` block is a constraint problem does not emit
`tensorium_init`, `tensorium_init_point`, or `tensorium_init_grid_*`. Those
symbols are reserved for analytic initial-data expressions. The host must run
the constraint solver, convert the result to evolution buffers, and then call
the generated RHS kernel.

## Native ABI regression test

`tools/dev/test_rhs_abi_v2_ll.sh` assembles and verifies generated LLVM IR,
then compiles and runs it at both `-O0` and `-O2`. Its C runner checks nonzero
offsets, non-unit strides, separate outputs, unchanged input state, untouched
padding, undersized-descriptor rejection, and zero-halo boundary writes.
