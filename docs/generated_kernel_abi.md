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

Source of truth in code: `include/tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h`.

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
- `tensorium.abi.write_arg_indices`: absolute argument indices (in function
  signature) written by the kernel

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
- `(nx:index, ny:index, nz:index, dx:f64, dy:f64, dz:f64, params..., fields...) -> ()`
- each field is `memref<?xf64>`.

LLVM-level:
- prefix: `i64,i64,i64,double,double,double`,
- then scalar params (`double`),
- then one 5-argument memref descriptor per field buffer.

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

