#include "tensorium_mlir/Runtime/HostBuffers.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tensorium_mlir::runtime {
namespace {

HostArgAccess combineAccess(HostArgAccess lhs, HostArgAccess rhs) {
  const bool reads = lhs == HostArgAccess::Read ||
                     lhs == HostArgAccess::ReadWrite ||
                     rhs == HostArgAccess::Read ||
                     rhs == HostArgAccess::ReadWrite;
  const bool writes = lhs == HostArgAccess::Write ||
                      lhs == HostArgAccess::ReadWrite ||
                      rhs == HostArgAccess::Write ||
                      rhs == HostArgAccess::ReadWrite;
  if (reads && writes)
    return HostArgAccess::ReadWrite;
  if (reads)
    return HostArgAccess::Read;
  if (writes)
    return HostArgAccess::Write;
  return HostArgAccess::None;
}

HostBufferRole storageRole(HostBufferRole role) {
  return role == HostBufferRole::Coordinate ? HostBufferRole::Coordinate
                                            : HostBufferRole::Field;
}

void require(bool cond, const std::string &message) {
  if (!cond)
    throw std::invalid_argument("host field storage: " + message);
}

std::string validationMessage(const std::vector<std::string> &errors) {
  std::ostringstream os;
  os << "invalid host ABI descriptor";
  for (const auto &error : errors)
    os << "\n  - " << error;
  return os.str();
}

} // namespace

std::int64_t HostGridShape::nPoints() const {
  if (nx <= 0 || ny <= 0 || nz <= 0)
    return 0;
  if (nx > std::numeric_limits<std::int64_t>::max() / ny)
    return 0;
  const std::int64_t nxy = nx * ny;
  if (nxy > std::numeric_limits<std::int64_t>::max() / nz)
    return 0;
  return nxy * nz;
}

HostFieldStorage::HostFieldStorage(const HostModuleABI &abi,
                                   HostGridShape shape)
    : shape_(shape) {
  const auto abiErrors = validateHostModuleABI(abi);
  if (!abiErrors.empty())
    throw std::invalid_argument(validationMessage(abiErrors));

  require(shape_.nPoints() > 0, "grid shape must have positive dimensions");

  for (const auto &kernel : abi.kernels) {
    for (const auto &buffer : kernel.buffers)
      addOrMergeBuffer(buffer);
  }

  finalizeArena();
  buildKernelPlans(abi);
}

std::string HostFieldStorage::storageKey(const HostBufferABI &buffer) {
  const char *prefix =
      buffer.role == HostBufferRole::Coordinate ? "coord:" : "field:";
  return std::string(prefix) + buffer.name;
}

std::size_t HostFieldStorage::addOrMergeBuffer(const HostBufferABI &buffer) {
  const std::string key = storageKey(buffer);
  auto found = indexByKey_.find(key);
  if (found != indexByKey_.end()) {
    HostStorageBuffer &existing = buffers_[found->second];
    require(existing.componentCount == buffer.componentCount,
            "component count mismatch for " + key);
    require(existing.rank == buffer.rank && existing.up == buffer.up &&
                existing.down == buffer.down,
            "rank/variance mismatch for " + key);
    existing.access = combineAccess(existing.access, buffer.access);
    return found->second;
  }

  HostStorageBuffer storage;
  storage.key = key;
  storage.name = buffer.name;
  storage.cName = buffer.cName;
  storage.role = storageRole(buffer.role);
  storage.access = buffer.access;
  storage.up = buffer.up;
  storage.down = buffer.down;
  storage.rank = buffer.rank;
  storage.componentCount = buffer.componentCount;
  storage.scalarCount = requiredBufferScalars(buffer, shape_.nPoints());
  require(storage.scalarCount > 0, "empty allocation for " + key);

  const std::size_t index = buffers_.size();
  indexByKey_.emplace(storage.key, index);
  buffers_.push_back(std::move(storage));
  return index;
}

void HostFieldStorage::finalizeArena() {
  std::int64_t offset = 0;
  for (auto &buffer : buffers_) {
    buffer.scalarOffset = offset;
    if (buffer.scalarCount >
        std::numeric_limits<std::int64_t>::max() - offset)
      throw std::overflow_error("host field storage: scalar arena is too large");
    offset += buffer.scalarCount;
  }
  arena_.assign(static_cast<std::size_t>(offset), 0.0);
}

void HostFieldStorage::buildKernelPlans(const HostModuleABI &abi) {
  kernelPlans_.reserve(abi.kernels.size());
  for (const auto &kernel : abi.kernels) {
    HostKernelBindingPlan plan;
    plan.symbolName = kernel.symbolName;
    plan.wrapperName = kernel.wrapperName;
    plan.kind = kernel.kind;
    plan.buffers.reserve(kernel.buffers.size());

    for (const auto &buffer : kernel.buffers) {
      const std::string key = storageKey(buffer);
      auto found = indexByKey_.find(key);
      require(found != indexByKey_.end(),
              "missing storage buffer for kernel binding " +
                  kernel.symbolName + "." + buffer.name);
      HostKernelBufferBinding binding;
      binding.argIndex = buffer.argIndex;
      binding.storageIndex = found->second;
      binding.access = buffer.access;
      plan.buffers.push_back(binding);
    }

    kernelPlans_.push_back(std::move(plan));
  }
}

HostStorageBuffer *HostFieldStorage::findBuffer(std::string_view key) {
  auto found = indexByKey_.find(std::string(key));
  if (found == indexByKey_.end())
    return nullptr;
  return &buffers_[found->second];
}

const HostStorageBuffer *
HostFieldStorage::findBuffer(std::string_view key) const {
  auto found = indexByKey_.find(std::string(key));
  if (found == indexByKey_.end())
    return nullptr;
  return &buffers_[found->second];
}

const HostKernelBindingPlan *
HostFieldStorage::findKernelPlan(std::string_view symbolName) const {
  for (const auto &plan : kernelPlans_) {
    if (plan.symbolName == symbolName)
      return &plan;
  }
  return nullptr;
}

double *HostFieldStorage::data(std::size_t storageIndex) {
  if (storageIndex >= buffers_.size())
    throw std::out_of_range("host field storage: storage index out of range");
  return arena_.data() + buffers_[storageIndex].scalarOffset;
}

const double *HostFieldStorage::data(std::size_t storageIndex) const {
  if (storageIndex >= buffers_.size())
    throw std::out_of_range("host field storage: storage index out of range");
  return arena_.data() + buffers_[storageIndex].scalarOffset;
}

abi::StridedMemRef1DF64 HostFieldStorage::memref(std::size_t storageIndex) {
  if (storageIndex >= buffers_.size())
    throw std::out_of_range("host field storage: storage index out of range");
  const auto &buffer = buffers_[storageIndex];
  return abi::makeContiguousMemRef(data(storageIndex), buffer.scalarCount);
}

abi::StridedMemRef1DF64
HostFieldStorage::memref(const HostKernelBufferBinding &binding) {
  return memref(binding.storageIndex);
}

} // namespace tensorium_mlir::runtime
