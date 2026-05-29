#pragma once

#include <cstdint>

namespace tensorium_mlir::abi {

// Bump this value only when changing generated symbol signatures or
// argument-order contracts in a backward-incompatible way.
inline constexpr std::int64_t kGeneratedKernelABIVersion = 1;

// Module and function attributes used to expose the generated ABI contract.
inline constexpr const char kAttrABIVersion[] = "tensorium.abi.version";
inline constexpr const char kAttrABIKind[] = "tensorium.abi.kind";
inline constexpr const char kAttrParamNames[] = "tensorium.abi.param_names";
inline constexpr const char kAttrCoordNames[] = "tensorium.abi.coord_names";
inline constexpr const char kAttrFieldNames[] = "tensorium.abi.field_names";
inline constexpr const char kAttrOutputNames[] = "tensorium.abi.output_names";
inline constexpr const char kAttrReadArgIndices[] =
    "tensorium.abi.read_arg_indices";
inline constexpr const char kAttrWriteArgIndices[] =
    "tensorium.abi.write_arg_indices";
inline constexpr const char kAttrStencilRadius[] =
    "tensorium.abi.stencil_radius";
inline constexpr const char kAttrResidualKernel[] =
    "tensorium.abi.residual_kernel";
inline constexpr const char kAttrMemoryLayout[] = "tensorium.abi.memory_layout";
inline constexpr const char kAttrMemrefABI[] = "tensorium.abi.memref_abi";

// Stable metadata values for ABI v1.
inline constexpr const char kMemLayoutSoAComponentMajor[] =
    "soa_component_major";
inline constexpr const char kMemrefABI1DStridedF64[] =
    "strided_memref_rank1_f64";

// Stable generated symbol names.
inline constexpr const char kSymbolInit[] = "tensorium_init";
inline constexpr const char kSymbolRhs[] = "tensorium_rhs";
inline constexpr const char kSymbolEntry[] = "tensorium_entry";
inline constexpr const char kSymbolInitPoint[] = "tensorium_init_point";
inline constexpr const char kSymbolInitGridScf[] = "tensorium_init_grid_scf";
inline constexpr const char kSymbolInitGridAffine[] =
    "tensorium_init_grid_affine";
inline constexpr const char kSymbolRhsGridScf[] = "tensorium_rhs_grid_scf";
inline constexpr const char kSymbolRhsGridAffine[] = "tensorium_rhs_grid_affine";
inline constexpr const char kSymbolRhsGridParallel[] =
    "tensorium_rhs_grid_parallel";
inline constexpr const char kSymbolResidualGridScf[] =
    "tensorium_residual_grid_scf";
inline constexpr const char kSymbolResidualGridAffine[] =
    "tensorium_residual_grid_affine";
inline constexpr const char kSymbolResidualGridParallel[] =
    "tensorium_residual_grid_parallel";

// Stable function kind tags.
inline constexpr const char kKindInitSource[] = "init_source";
inline constexpr const char kKindRhsSource[] = "rhs_source";
inline constexpr const char kKindEntrySource[] = "entry_source";
inline constexpr const char kKindInitPoint[] = "init_point";
inline constexpr const char kKindInitGridScf[] = "init_grid_scf";
inline constexpr const char kKindInitGridAffine[] = "init_grid_affine";
inline constexpr const char kKindRhsGridScf[] = "rhs_grid_scf";
inline constexpr const char kKindRhsGridAffine[] = "rhs_grid_affine";
inline constexpr const char kKindRhsGridParallel[] = "rhs_grid_parallel";
inline constexpr const char kKindResidualGridScf[] = "residual_grid_scf";
inline constexpr const char kKindResidualGridAffine[] = "residual_grid_affine";
inline constexpr const char kKindResidualGridParallel[] =
    "residual_grid_parallel";

// C/C++ host-side view of the rank-1 memref descriptor shape used after LLVM
// lowering: (allocatedPtr, alignedPtr, offset, size, stride).
struct StridedMemRef1DF64 {
  double *allocated = nullptr;
  double *aligned = nullptr;
  std::int64_t offset = 0;
  std::int64_t size = 0;
  std::int64_t stride = 1;
};

inline StridedMemRef1DF64 makeContiguousMemRef(double *data,
                                               std::int64_t size) {
  return {data, data, 0, size, 1};
}

} // namespace tensorium_mlir::abi
