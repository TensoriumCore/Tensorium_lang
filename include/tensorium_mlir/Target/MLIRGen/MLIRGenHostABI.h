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

enum class HostReturnKind { Void, F64 };

enum class HostBufferRole { Coordinate, Field, Output };

enum class HostArgAccess { None, Read, Write, ReadWrite };

struct HostArgABI {
  HostArgKind kind = HostArgKind::F64;
  std::string cName;
};

struct HostFieldABI {
  std::string name;
  int up = 0;
  int down = 0;
  int rank = 0;
  std::int64_t componentCount = 1;
};

struct HostBufferABI {
  std::string name;
  std::string cName;
  std::int64_t argIndex = -1;
  HostBufferRole role = HostBufferRole::Field;
  HostArgAccess access = HostArgAccess::None;
  int up = 0;
  int down = 0;
  int rank = 0;
  std::int64_t componentCount = 1;
};

struct HostKernelABI {
  std::string symbolName;
  std::string wrapperName;
  std::string kind;
  HostReturnKind returnKind = HostReturnKind::Void;
  std::vector<HostArgABI> rawArgs;
  std::vector<HostBufferABI> buffers;
  std::vector<std::string> params;
  std::vector<std::string> coords;
  std::vector<std::string> fields;
  std::vector<std::string> outputs;
  std::vector<std::int64_t> readArgIndices;
  std::vector<std::int64_t> writeArgIndices;
  std::int64_t stencilRadius = 0;
};

struct HostPrintABI {
  std::string label;
  std::string fieldName;
  int rank = 0;
};

struct HostSpectralBoundaryConditionABI {
  std::string face;
  std::string kind;
  double valueCoefficient = 1.0;
  double normalDerivativeCoefficient = 0.0;
  double targetValue = 0.0;
  std::string derivativeKind = "normal";
  std::string valueCoefficientCoordinate;
  std::string normalDerivativeCoefficientCoordinate;
  std::string targetValueCoordinate;
};

struct HostSpectralResidualSystemEquationABI {
  std::string residualName;
  std::string unknownName;
  std::int64_t unknownIndex = -1;
  std::string pointKernelSymbol;
  std::string gridKernelSymbol;
  std::vector<std::string> params;
  std::vector<std::string> auxiliaryNames;
  std::vector<std::int64_t> auxiliaryUnknownIndices;
  std::vector<HostSpectralBoundaryConditionABI> boundaryConditions;
};

struct HostSpectralResidualSystemABI {
  std::string name;
  std::vector<std::string> unknownNames;
  std::vector<HostSpectralResidualSystemEquationABI> equations;
};

struct HostModuleABI {
  int dimension = 3;
  std::string coordSystem;
  std::vector<int> resolution;
  int spatialOrder = 0;
  std::string spatialScheme;
  std::string derivativeScheme;
  std::unordered_map<std::string, std::int64_t> componentCounts;
  std::vector<HostFieldABI> fields;
  std::vector<HostKernelABI> kernels;
  std::vector<HostSpectralResidualSystemABI> spectralResidualSystems;
  std::vector<HostPrintABI> prints;
  std::vector<std::string> printFields;
};

std::string makeHostCIdentifier(std::string_view input,
                                std::string_view fallback);

std::int64_t requiredBufferScalars(const HostBufferABI &buffer,
                                   std::int64_t nPoints);

std::vector<std::string> validateHostModuleABI(const HostModuleABI &abi);

HostModuleABI buildHostModuleABI(const tensorium::backend::ModuleIR &module,
                                 mlir::ModuleOp moduleOp);

} // namespace tensorium_mlir
