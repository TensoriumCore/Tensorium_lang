#pragma once

#include "tensorium_mlir/Runtime/GeneratedHostDescriptors.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorium_mlir::runtime {

struct GeneratedHostGridShape {
  std::int64_t nx = 1;
  std::int64_t ny = 1;
  std::int64_t nz = 1;

  std::int64_t nPoints() const {
    if (nx <= 0 || ny <= 0 || nz <= 0)
      return 0;
    if (nx > std::numeric_limits<std::int64_t>::max() / ny)
      return 0;
    const std::int64_t nxy = nx * ny;
    if (nxy > std::numeric_limits<std::int64_t>::max() / nz)
      return 0;
    return nxy * nz;
  }
};

struct GeneratedHostStorageBuffer {
  std::string key;
  std::string name;
  std::string cName;
  std::int64_t role = TENSORIUM_HOST_BUFFER_ROLE_FIELD;
  std::int64_t access = TENSORIUM_HOST_ARG_ACCESS_NONE;
  std::int64_t up = 0;
  std::int64_t down = 0;
  std::int64_t rank = 0;
  std::int64_t componentCount = 1;
  std::int64_t scalarCount = 0;
  std::int64_t scalarOffset = 0;
};

struct GeneratedHostKernelBufferBinding {
  std::int64_t argIndex = -1;
  std::size_t storageIndex = 0;
  std::int64_t access = TENSORIUM_HOST_ARG_ACCESS_NONE;
};

struct GeneratedHostKernelBindingPlan {
  std::string symbolName;
  std::string wrapperName;
  std::string kind;
  std::int64_t stencilRadius = 0;
  std::vector<GeneratedHostKernelBufferBinding> buffers;
};

struct GeneratedHostGridSpacing {
  double dx = 1.0;
  double dy = 1.0;
  double dz = 1.0;
};

struct GeneratedHostEulerUpdate {
  std::size_t stateStorageIndex = 0;
  std::size_t derivativeStorageIndex = 0;
  std::string stateKey;
  std::string derivativeKey;
  std::int64_t scalarCount = 0;
};

class GeneratedHostStorage {
public:
  GeneratedHostStorage(std::span<const tensorium_host_kernel_desc> kernels,
                       std::span<const tensorium_host_buffer_desc> buffers,
                       GeneratedHostGridShape shape)
      : shape_(shape) {
    require(shape_.nPoints() > 0, "grid shape must have positive dimensions");
    for (const auto &buffer : buffers)
      addOrMergeBuffer(buffer);
    finalizeArena();
    buildKernelPlans(kernels, buffers);
  }

  GeneratedHostGridShape shape() const { return shape_; }
  std::int64_t nPoints() const { return shape_.nPoints(); }
  std::int64_t totalScalars() const {
    return static_cast<std::int64_t>(arena_.size());
  }
  std::size_t dataAllocationCount() const { return arena_.empty() ? 0u : 1u; }
  std::size_t bufferCount() const { return buffers_.size(); }

  std::span<double> scalars() {
    return std::span<double>(arena_.data(), arena_.size());
  }
  std::span<const double> scalars() const {
    return std::span<const double>(arena_.data(), arena_.size());
  }

  const std::vector<GeneratedHostStorageBuffer> &buffers() const {
    return buffers_;
  }
  const std::vector<GeneratedHostKernelBindingPlan> &kernelPlans() const {
    return kernelPlans_;
  }

  GeneratedHostStorageBuffer *findBuffer(std::string_view key) {
    auto found = indexByKey_.find(std::string(key));
    if (found == indexByKey_.end())
      return nullptr;
    return &buffers_[found->second];
  }

  const GeneratedHostStorageBuffer *findBuffer(std::string_view key) const {
    auto found = indexByKey_.find(std::string(key));
    if (found == indexByKey_.end())
      return nullptr;
    return &buffers_[found->second];
  }

  const GeneratedHostKernelBindingPlan *
  findKernelPlan(std::string_view symbolName) const {
    for (const auto &plan : kernelPlans_) {
      if (plan.symbolName == symbolName)
        return &plan;
    }
    return nullptr;
  }

  double *data(std::size_t storageIndex) {
    if (storageIndex >= buffers_.size())
      throw std::out_of_range("generated host storage: index out of range");
    return arena_.data() + buffers_[storageIndex].scalarOffset;
  }

  const double *data(std::size_t storageIndex) const {
    if (storageIndex >= buffers_.size())
      throw std::out_of_range("generated host storage: index out of range");
    return arena_.data() + buffers_[storageIndex].scalarOffset;
  }

  double *data(std::string_view key) {
    const auto *buffer = findBuffer(key);
    if (!buffer)
      throw std::out_of_range("generated host storage: missing buffer " +
                              std::string(key));
    return arena_.data() + buffer->scalarOffset;
  }

  const double *data(std::string_view key) const {
    const auto *buffer = findBuffer(key);
    if (!buffer)
      throw std::out_of_range("generated host storage: missing buffer " +
                              std::string(key));
    return arena_.data() + buffer->scalarOffset;
  }

  abi::StridedMemRef1DF64 memref(std::size_t storageIndex) {
    if (storageIndex >= buffers_.size())
      throw std::out_of_range("generated host storage: index out of range");
    const auto &buffer = buffers_[storageIndex];
    return abi::makeContiguousMemRef(data(storageIndex), buffer.scalarCount);
  }

  abi::StridedMemRef1DF64
  memref(const GeneratedHostKernelBufferBinding &binding) {
    return memref(binding.storageIndex);
  }

  void invoke(const tensorium_host_kernel_adapter_desc &adapter,
              std::span<const double> params,
              GeneratedHostGridSpacing spacing) {
    require(adapter.symbol_name && adapter.symbol_name[0] != '\0',
            "adapter symbol name is empty");
    require(adapter.invoke != nullptr,
            "kernel has no generated runtime adapter: " +
                std::string(adapter.symbol_name));

    const auto *plan = findKernelPlan(adapter.symbol_name);
    require(plan != nullptr,
            "missing binding plan for " + std::string(adapter.symbol_name));

    std::vector<tensorium_memref1d_f64> refs;
    refs.reserve(plan->buffers.size());
    for (const auto &binding : plan->buffers) {
      auto ref = memref(binding);
      refs.push_back({ref.allocated, ref.aligned, ref.offset, ref.size,
                      ref.stride});
    }

    const tensorium_host_grid_desc grid{
        shape_.nx, shape_.ny, shape_.nz, spacing.dx, spacing.dy, spacing.dz,
        shape_.nPoints()};
    const int status =
        adapter.invoke(params.data(), static_cast<std::int64_t>(params.size()),
                       refs.data(), static_cast<std::int64_t>(refs.size()),
                       &grid);
    require(status == 0, "kernel adapter failed for " +
                             std::string(adapter.symbol_name) +
                             " with status " + std::to_string(status));
  }

  void invoke(std::span<const tensorium_host_kernel_adapter_desc> adapters,
              std::string_view symbolName, std::span<const double> params,
              GeneratedHostGridSpacing spacing) {
    for (const auto &adapter : adapters) {
      if (adapter.symbol_name && symbolName == adapter.symbol_name) {
        invoke(adapter, params, spacing);
        return;
      }
    }
    throw std::out_of_range("generated host storage: missing adapter " +
                            std::string(symbolName));
  }

  std::vector<GeneratedHostEulerUpdate>
  eulerUpdatePairsFromDerivativePrefix(char prefix = 'd') const {
    std::vector<GeneratedHostEulerUpdate> updates;
    for (std::size_t derivativeIndex = 0; derivativeIndex < buffers_.size();
         ++derivativeIndex) {
      const auto &derivative = buffers_[derivativeIndex];
      if (derivative.role != TENSORIUM_HOST_BUFFER_ROLE_FIELD ||
          !accessWrites(derivative.access) ||
          derivative.name.size() <= 1 || derivative.name.front() != prefix)
        continue;

      const std::string stateName = derivative.name.substr(1);
      const std::string stateKey = std::string("field:") + stateName;
      auto found = indexByKey_.find(stateKey);
      if (found == indexByKey_.end())
        continue;

      const auto &state = buffers_[found->second];
      require(state.componentCount == derivative.componentCount &&
                  state.scalarCount == derivative.scalarCount &&
                  state.rank == derivative.rank && state.up == derivative.up &&
                  state.down == derivative.down,
              "Euler update metadata mismatch for " + derivative.key +
                  " -> " + state.key);

      GeneratedHostEulerUpdate update;
      update.stateStorageIndex = found->second;
      update.derivativeStorageIndex = derivativeIndex;
      update.stateKey = state.key;
      update.derivativeKey = derivative.key;
      update.scalarCount = state.scalarCount;
      updates.push_back(std::move(update));
    }
    return updates;
  }

  void applyEulerUpdate(std::span<const GeneratedHostEulerUpdate> updates,
                        double dt) {
    for (const auto &update : updates) {
      require(update.stateStorageIndex < buffers_.size() &&
                  update.derivativeStorageIndex < buffers_.size(),
              "Euler update storage index out of range");
      const auto &state = buffers_[update.stateStorageIndex];
      const auto &derivative = buffers_[update.derivativeStorageIndex];
      require(update.scalarCount == state.scalarCount &&
                  update.scalarCount == derivative.scalarCount,
              "Euler update scalar count mismatch for " + update.derivativeKey +
                  " -> " + update.stateKey);

      double *stateData = data(update.stateStorageIndex);
      const double *derivativeData = data(update.derivativeStorageIndex);
#pragma omp parallel for schedule(static)
      for (std::int64_t i = 0; i < update.scalarCount; ++i)
        stateData[i] += dt * derivativeData[i];
    }
  }

  static std::string storageKey(const tensorium_host_buffer_desc &buffer) {
    const char *prefix =
        buffer.role == TENSORIUM_HOST_BUFFER_ROLE_COORDINATE ? "coord:"
                                                             : "field:";
    return std::string(prefix) + safeCString(buffer.name);
  }

private:
  GeneratedHostGridShape shape_;
  std::vector<double> arena_;
  std::vector<GeneratedHostStorageBuffer> buffers_;
  std::unordered_map<std::string, std::size_t> indexByKey_;
  std::vector<GeneratedHostKernelBindingPlan> kernelPlans_;

  static const char *safeCString(const char *value) { return value ? value : ""; }

  static bool isValidRole(std::int64_t role) {
    return role == TENSORIUM_HOST_BUFFER_ROLE_COORDINATE ||
           role == TENSORIUM_HOST_BUFFER_ROLE_FIELD ||
           role == TENSORIUM_HOST_BUFFER_ROLE_OUTPUT;
  }

  static bool isValidAccess(std::int64_t access) {
    return access == TENSORIUM_HOST_ARG_ACCESS_NONE ||
           access == TENSORIUM_HOST_ARG_ACCESS_READ ||
           access == TENSORIUM_HOST_ARG_ACCESS_WRITE ||
           access == TENSORIUM_HOST_ARG_ACCESS_READWRITE;
  }

  static bool accessWrites(std::int64_t access) {
    return access == TENSORIUM_HOST_ARG_ACCESS_WRITE ||
           access == TENSORIUM_HOST_ARG_ACCESS_READWRITE;
  }

  static std::int64_t storageRole(std::int64_t role) {
    return role == TENSORIUM_HOST_BUFFER_ROLE_COORDINATE
               ? TENSORIUM_HOST_BUFFER_ROLE_COORDINATE
               : TENSORIUM_HOST_BUFFER_ROLE_FIELD;
  }

  static std::int64_t combineAccess(std::int64_t lhs, std::int64_t rhs) {
    const bool reads = lhs == TENSORIUM_HOST_ARG_ACCESS_READ ||
                       lhs == TENSORIUM_HOST_ARG_ACCESS_READWRITE ||
                       rhs == TENSORIUM_HOST_ARG_ACCESS_READ ||
                       rhs == TENSORIUM_HOST_ARG_ACCESS_READWRITE;
    const bool writes = lhs == TENSORIUM_HOST_ARG_ACCESS_WRITE ||
                        lhs == TENSORIUM_HOST_ARG_ACCESS_READWRITE ||
                        rhs == TENSORIUM_HOST_ARG_ACCESS_WRITE ||
                        rhs == TENSORIUM_HOST_ARG_ACCESS_READWRITE;
    if (reads && writes)
      return TENSORIUM_HOST_ARG_ACCESS_READWRITE;
    if (reads)
      return TENSORIUM_HOST_ARG_ACCESS_READ;
    if (writes)
      return TENSORIUM_HOST_ARG_ACCESS_WRITE;
    return TENSORIUM_HOST_ARG_ACCESS_NONE;
  }

  static void require(bool cond, const std::string &message) {
    if (!cond)
      throw std::invalid_argument("generated host storage: " + message);
  }

  static std::int64_t requiredScalars(const tensorium_host_buffer_desc &buffer,
                                      std::int64_t nPoints) {
    require(buffer.component_count > 0,
            "component_count must be positive for " +
                std::string(safeCString(buffer.name)));
    if (buffer.component_count >
        std::numeric_limits<std::int64_t>::max() / nPoints)
      throw std::overflow_error("generated host storage: buffer is too large");
    return buffer.component_count * nPoints;
  }

  std::size_t addOrMergeBuffer(const tensorium_host_buffer_desc &buffer) {
    require(buffer.name && buffer.name[0] != '\0', "buffer name is empty");
    require(isValidRole(buffer.role), "invalid role for " +
                                          std::string(safeCString(buffer.name)));
    require(isValidAccess(buffer.access),
            "invalid access for " + std::string(safeCString(buffer.name)));
    require(buffer.arg_index >= 0,
            "negative arg_index for " + std::string(safeCString(buffer.name)));

    const std::string key = storageKey(buffer);
    auto found = indexByKey_.find(key);
    if (found != indexByKey_.end()) {
      GeneratedHostStorageBuffer &existing = buffers_[found->second];
      require(existing.componentCount == buffer.component_count,
              "component count mismatch for " + key);
      require(existing.rank == buffer.rank && existing.up == buffer.up &&
                  existing.down == buffer.down,
              "rank/variance mismatch for " + key);
      existing.access = combineAccess(existing.access, buffer.access);
      return found->second;
    }

    GeneratedHostStorageBuffer storage;
    storage.key = key;
    storage.name = safeCString(buffer.name);
    storage.cName = safeCString(buffer.c_name);
    storage.role = storageRole(buffer.role);
    storage.access = buffer.access;
    storage.up = buffer.up;
    storage.down = buffer.down;
    storage.rank = buffer.rank;
    storage.componentCount = buffer.component_count;
    storage.scalarCount = requiredScalars(buffer, shape_.nPoints());
    require(storage.scalarCount > 0, "empty allocation for " + key);

    const std::size_t index = buffers_.size();
    indexByKey_.emplace(storage.key, index);
    buffers_.push_back(std::move(storage));
    return index;
  }

  void finalizeArena() {
    std::int64_t offset = 0;
    for (auto &buffer : buffers_) {
      buffer.scalarOffset = offset;
      if (buffer.scalarCount >
          std::numeric_limits<std::int64_t>::max() - offset)
        throw std::overflow_error(
            "generated host storage: scalar arena is too large");
      offset += buffer.scalarCount;
    }
    arena_.assign(static_cast<std::size_t>(offset), 0.0);
  }

  void buildKernelPlans(std::span<const tensorium_host_kernel_desc> kernels,
                        std::span<const tensorium_host_buffer_desc> buffers) {
    kernelPlans_.reserve(kernels.size());
    for (std::size_t kernelIndex = 0; kernelIndex < kernels.size();
         ++kernelIndex) {
      const auto &kernel = kernels[kernelIndex];
      require(kernel.symbol_name && kernel.symbol_name[0] != '\0',
              "kernel symbol name is empty");
      require(kernel.buffer_begin >= 0 && kernel.buffer_count >= 0,
              "kernel buffer range is negative");
      require(kernel.buffer_begin <=
                  static_cast<std::int64_t>(buffers.size()) &&
              kernel.buffer_count <=
                  static_cast<std::int64_t>(buffers.size()) -
                      kernel.buffer_begin,
              "kernel buffer range is out of bounds");

      GeneratedHostKernelBindingPlan plan;
      plan.symbolName = safeCString(kernel.symbol_name);
      plan.wrapperName = safeCString(kernel.wrapper_name);
      plan.kind = safeCString(kernel.kind);
      plan.stencilRadius = kernel.stencil_radius;
      plan.buffers.reserve(static_cast<std::size_t>(kernel.buffer_count));

      for (std::int64_t i = 0; i < kernel.buffer_count; ++i) {
        const auto &buffer =
            buffers[static_cast<std::size_t>(kernel.buffer_begin + i)];
        require(buffer.kernel_symbol &&
                    std::string_view(buffer.kernel_symbol) == plan.symbolName,
                "kernel symbol mismatch in buffer descriptor");
        require(buffer.kernel_index == static_cast<std::int64_t>(kernelIndex),
                "kernel index mismatch in buffer descriptor");
        const std::string key = storageKey(buffer);
        auto found = indexByKey_.find(key);
        require(found != indexByKey_.end(),
                "missing storage buffer for " + plan.symbolName + "." +
                    std::string(safeCString(buffer.name)));

        GeneratedHostKernelBufferBinding binding;
        binding.argIndex = buffer.arg_index;
        binding.storageIndex = found->second;
        binding.access = buffer.access;
        plan.buffers.push_back(binding);
      }

      kernelPlans_.push_back(std::move(plan));
    }
  }
};

} // namespace tensorium_mlir::runtime
