#pragma once

#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"
#include "tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tensorium_mlir::runtime {

struct HostGridShape {
  std::int64_t nx = 1;
  std::int64_t ny = 1;
  std::int64_t nz = 1;

  std::int64_t nPoints() const;
};

struct HostStorageBuffer {
  std::string key;
  std::string name;
  std::string cName;
  HostBufferRole role = HostBufferRole::Field;
  HostArgAccess access = HostArgAccess::None;
  int up = 0;
  int down = 0;
  int rank = 0;
  std::int64_t componentCount = 1;
  std::int64_t scalarCount = 0;
  std::int64_t scalarOffset = 0;
};

struct HostKernelBufferBinding {
  std::int64_t argIndex = -1;
  std::size_t storageIndex = 0;
  HostArgAccess access = HostArgAccess::None;
};

struct HostKernelBindingPlan {
  std::string symbolName;
  std::string wrapperName;
  std::string kind;
  std::vector<HostKernelBufferBinding> buffers;
};

class HostFieldStorage {
public:
  HostFieldStorage(const HostModuleABI &abi, HostGridShape shape);

  HostGridShape shape() const { return shape_; }
  std::int64_t nPoints() const { return shape_.nPoints(); }
  std::int64_t totalScalars() const {
    return static_cast<std::int64_t>(arena_.size());
  }
  std::size_t dataAllocationCount() const { return arena_.empty() ? 0u : 1u; }

  std::span<double> scalars() {
    return std::span<double>(arena_.data(), arena_.size());
  }
  std::span<const double> scalars() const {
    return std::span<const double>(arena_.data(), arena_.size());
  }

  const std::vector<HostStorageBuffer> &buffers() const { return buffers_; }
  const std::vector<HostKernelBindingPlan> &kernelPlans() const {
    return kernelPlans_;
  }

  HostStorageBuffer *findBuffer(std::string_view key);
  const HostStorageBuffer *findBuffer(std::string_view key) const;
  const HostKernelBindingPlan *findKernelPlan(std::string_view symbolName) const;

  double *data(std::size_t storageIndex);
  const double *data(std::size_t storageIndex) const;
  abi::StridedMemRef1DF64 memref(std::size_t storageIndex);
  abi::StridedMemRef1DF64 memref(const HostKernelBufferBinding &binding);

  static std::string storageKey(const HostBufferABI &buffer);

private:
  HostGridShape shape_;
  std::vector<double> arena_;
  std::vector<HostStorageBuffer> buffers_;
  std::unordered_map<std::string, std::size_t> indexByKey_;
  std::vector<HostKernelBindingPlan> kernelPlans_;

  std::size_t addOrMergeBuffer(const HostBufferABI &buffer);
  void finalizeArena();
  void buildKernelPlans(const HostModuleABI &abi);
};

} // namespace tensorium_mlir::runtime
