#pragma once

#include "tensorium/Backend/DomainIR.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace tensorium_mlir {

enum class HostArgKind { F64, Index, Memref1DF64 };

struct HostArgABI {
  HostArgKind kind = HostArgKind::F64;
  std::string cName;
};

struct HostKernelABI {
  std::string symbolName;
  std::string wrapperName;
  std::string kind;
  std::vector<HostArgABI> rawArgs;
  std::vector<std::string> params;
  std::vector<std::string> coords;
  std::vector<std::string> fields;
  std::vector<std::string> outputs;
};

struct HostPrintABI {
  std::string label;
  std::string fieldName;
  int rank = 0;
};

struct HostModuleABI {
  int dimension = 3;
  std::unordered_map<std::string, std::int64_t> componentCounts;
  std::vector<HostKernelABI> kernels;
  std::vector<HostPrintABI> prints;
  std::vector<std::string> printFields;
};

std::string makeHostCIdentifier(std::string_view input,
                                std::string_view fallback);

HostModuleABI buildHostModuleABI(const tensorium::backend::ModuleIR &module,
                                 mlir::ModuleOp moduleOp);

} // namespace tensorium_mlir
