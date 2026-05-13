# Tensorium Language, MLIR, ABI, and Runtime Architecture

## Purpose
Tensorium is a research language for numerical relativity and differential
geometry. Its source programs should stay close to the mathematical
formulation: tensor fields, indices, geometric operators, constraints,
initial-data systems, and evolution equations.

The compiler must preserve that structure long enough to optimize it
aggressively before lowering to executable kernels. The generated host-side C
or C++ is ABI glue only; it is not the primary representation of Tensorium
semantics and it is not where mathematical optimization should happen.

## Pipeline Contract
The intended compilation path is:

```text
.tn source
  -> AST and semantic analysis
  -> backend tensor IR
  -> Tensorium MLIR dialect
  -> Tensorium and MLIR optimization/lowering passes
  -> LLVM IR / native object
  -> generated ABI glue and host/runtime integration
```

Each layer has a different responsibility:

- Front-end: parse source syntax, resolve symbols, enforce tensor and Einstein
  rules, and keep diagnostics close to source math.
- Backend tensor IR: represent canonical tensor expressions independently from
  source syntax.
- Tensorium MLIR: preserve high-level tensor/geometric structure for
  optimization passes.
- Lowered MLIR/LLVM: produce efficient kernels with stable, linkable symbols.
- Generated host glue: expose those kernels through safe C/C++ wrappers that
  hide large raw ABI signatures.
- Runtime integration: allocate and schedule grids through standalone C/C++ or
  optional HPC runtimes such as AMReX.

## Non-Goals
- Do not parse or embed C as part of the Tensorium language.
- Do not make generated C the main compute backend.
- Do not require AMReX for normal Tensorium execution.
- Do not expose massive lowered memref signatures as the user-facing API.
- Do not introduce dynamic callbacks inside cell or tile loops.

## Generated ABI Glue
Generated host code is allowed and expected when it removes ABI risk. Its role
is to connect MLIR/LLVM kernels to ordinary C/C++ callers without making users
write long, error-prone parameter lists by hand.

Examples of valid generated glue:

- raw external prototypes for lowered kernel symbols,
- compact wrapper structs for rank-1 strided memref descriptors,
- `tensorium_call_*` convenience wrappers over plain `double *` buffers,
- generated print/debug helpers derived from `.tn` `print(...)` requests,
- future C++ wrappers over runtime-owned field views,
- future AMReX wrappers over `MultiFab` / `Array4` views.

The generated glue must be derived from a single ABI description of the module:
kernel names, scalar parameters, coordinate buffers, field buffers, output
fields, tensor ranks, component layout, and requested debug prints.

## Standalone First, AMReX Optional
Tensorium must remain useful without AMReX. The same `.tn` program should be
able to target:

- standalone C runners for smoke tests and small numerical experiments,
- a Tensorium C++ runtime for local research workflows,
- AMReX wrappers for large-scale MPI/GPU execution.

AMReX is therefore a host/runtime target, not a front-end language dependency.
The source language and high-level MLIR should not contain AMReX-specific
concepts unless they describe a mathematical or grid contract that also has a
standalone meaning.

## AMReX Direction
The AMReX integration should sit on top of the same kernel ABI used by the
standalone path. A future wrapper can map AMReX views to Tensorium buffers and
schedule kernels over boxes or tiles:

```cpp
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
  tensorium_rhs_point(i, j, k, /* generated field views */);
});
```

The important constraint is that the inner compute path remains statically
visible, inlineable where possible, and device-friendly. Runtime-dispatched C
callbacks inside the cell loop should be avoided.

## External C/C++ Functions
Extern declarations in `.tn` are linkable symbols, not source imports. The
front-end owns type checking; MLIR owns symbol declaration and call lowering.

For a source declaration such as:

```tn
extern scalar eos_pressure(scalar rho, scalar eps)
```

the executable lowering should produce an MLIR declaration and call equivalent
to:

```mlir
func.func private @eos_pressure(f64, f64) -> f64
%p = func.call @eos_pressure(%rho, %eps) : (f64, f64) -> f64
```

The C/C++ implementation is compiled and linked as a normal object file. The
compiler driver may provide convenience options for link orchestration, but the
language semantics remain MLIR symbol semantics.

The first supported extern class should be scalar pure functions. Tensor,
buffer, grid, or runtime-aware externs should wait until the kernel ABI
descriptor is explicit enough to avoid accidental performance and portability
constraints.

## ABI Descriptor Direction
Host header generation should evolve from ad hoc string emission toward an
internal descriptor, for example:

```text
KernelABI
  name
  kind: init, rhs, point, grid
  dimensions
  scalar parameters
  coordinate buffers
  field buffers
  output fields
  tensor ranks and layout
  print requests
```

The current descriptor entry point is
`tensorium_mlir::buildHostModuleABI(...)`, declared in
`include/tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h`.

That descriptor should be the source for:

- raw C prototypes,
- C convenience wrappers,
- debug/print helpers,
- standalone runtime adapters,
- future C++ and AMReX wrappers.

This keeps the ABI stable while allowing multiple host targets to share the
same compiled kernels.

## Near-Term Work Order
1. Keep documenting the language/MLIR/ABI/runtime boundary in this file and
   `docs/generated_kernel_abi.md`.
2. Extract a reusable ABI descriptor from the current host-header generation.
3. Regenerate the existing C wrappers and print helpers from that descriptor.
4. Implement scalar extern lowering as MLIR symbol declarations and
   `func.call`.
5. Add a link smoke test with a small C helper object.
6. Design point or tile kernel ABI variants before adding AMReX wrappers.

The guiding rule is: users write mathematical Tensorium; Tensorium emits
optimized MLIR/LLVM kernels; generated C/C++ only makes the ABI safe and
ergonomic.
