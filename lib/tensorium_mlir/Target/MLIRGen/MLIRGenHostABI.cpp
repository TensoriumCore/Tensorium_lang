#include "tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

#include <cctype>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace tensorium_mlir {
namespace {

std::vector<std::string> getStringArrayAttr(mlir::Operation *op,
                                            llvm::StringRef name) {
  std::vector<std::string> out;
  auto arr = op->getAttrOfType<mlir::ArrayAttr>(name);
  if (!arr)
    return out;
  out.reserve(arr.size());
  for (mlir::Attribute attr : arr) {
    if (auto str = llvm::dyn_cast<mlir::StringAttr>(attr))
      out.push_back(str.getValue().str());
  }
  return out;
}

std::string getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  if (auto str = op->getAttrOfType<mlir::StringAttr>(name))
    return str.getValue().str();
  return {};
}

std::int64_t getI64Attr(mlir::Operation *op, llvm::StringRef name,
                        std::int64_t fallback = 0) {
  if (auto value = op->getAttrOfType<mlir::IntegerAttr>(name))
    return value.getInt();
  return fallback;
}

std::vector<std::int64_t> getI64ArrayAttr(mlir::Operation *op,
                                          llvm::StringRef name) {
  std::vector<std::int64_t> out;
  auto arr = op->getAttrOfType<mlir::ArrayAttr>(name);
  if (!arr)
    return out;
  out.reserve(arr.size());
  for (mlir::Attribute attr : arr) {
    if (auto value = llvm::dyn_cast<mlir::IntegerAttr>(attr))
      out.push_back(value.getInt());
  }
  return out;
}

bool isSupportedHostScalarType(mlir::Type type) {
  return type.isF64() || type.isIndex();
}

bool isSupportedHostMemrefType(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.getElementType().isF64();
}

bool isSupportedHostType(mlir::Type type) {
  return isSupportedHostScalarType(type) || isSupportedHostMemrefType(type);
}

bool hasSupportedHostResults(mlir::func::FuncOp fn) {
  if (fn.getNumResults() == 0)
    return true;
  return fn.getNumResults() == 1 && fn.getResultTypes()[0].isF64();
}

bool hasOnlySupportedHostTypes(mlir::func::FuncOp fn) {
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    if (!isSupportedHostType(fn.getArgument(i).getType()))
      return false;
  }
  return true;
}

bool isHostCallableKind(llvm::StringRef kind) {
  return kind == tensorium_mlir::abi::kKindInitPoint ||
         kind == tensorium_mlir::abi::kKindInitGridScf ||
         kind == tensorium_mlir::abi::kKindInitGridAffine ||
         kind == tensorium_mlir::abi::kKindRhsGridScf ||
         kind == tensorium_mlir::abi::kKindRhsGridAffine ||
         kind == tensorium_mlir::abi::kKindRhsGridParallel ||
         kind == tensorium_mlir::abi::kKindResidualGridScf ||
         kind == tensorium_mlir::abi::kKindResidualGridAffine ||
         kind == tensorium_mlir::abi::kKindResidualGridParallel ||
         kind == tensorium_mlir::abi::kKindSpectralResidualPoint ||
         kind == tensorium_mlir::abi::kKindSpectralResidualJvpPoint ||
         kind == tensorium_mlir::abi::kKindSpectralResidualGrid;
}

bool isFieldGridKind(llvm::StringRef kind) {
  return kind == tensorium_mlir::abi::kKindRhsGridScf ||
         kind == tensorium_mlir::abi::kKindRhsGridAffine ||
         kind == tensorium_mlir::abi::kKindRhsGridParallel ||
         kind == tensorium_mlir::abi::kKindResidualGridScf ||
         kind == tensorium_mlir::abi::kKindResidualGridAffine ||
         kind == tensorium_mlir::abi::kKindResidualGridParallel;
}

std::vector<std::string> logicalArgNames(mlir::func::FuncOp fn) {
  const std::string kind =
      getStringAttr(fn.getOperation(), tensorium_mlir::abi::kAttrABIKind);
  std::vector<std::string> names;
  auto append = [&](const std::vector<std::string> &items) {
    names.insert(names.end(), items.begin(), items.end());
  };

  const auto params = getStringArrayAttr(fn.getOperation(),
                                         tensorium_mlir::abi::kAttrParamNames);
  const auto coords = getStringArrayAttr(fn.getOperation(),
                                         tensorium_mlir::abi::kAttrCoordNames);
  const auto fields = getStringArrayAttr(fn.getOperation(),
                                         tensorium_mlir::abi::kAttrFieldNames);
  const auto outputs = getStringArrayAttr(
      fn.getOperation(), tensorium_mlir::abi::kAttrOutputNames);

  if (kind == tensorium_mlir::abi::kKindInitPoint) {
    append(params);
    append(coords);
    append(outputs);
  } else if (kind == tensorium_mlir::abi::kKindInitGridScf ||
             kind == tensorium_mlir::abi::kKindInitGridAffine) {
    append(params);
    append(coords);
    append(outputs);
  } else if (kind == tensorium_mlir::abi::kKindRhsGridScf ||
             kind == tensorium_mlir::abi::kKindRhsGridAffine ||
             kind == tensorium_mlir::abi::kKindRhsGridParallel ||
             kind == tensorium_mlir::abi::kKindResidualGridScf ||
             kind == tensorium_mlir::abi::kKindResidualGridAffine ||
             kind == tensorium_mlir::abi::kKindResidualGridParallel) {
    names = {"nx", "ny", "nz", "dx", "dy", "dz"};
    append(params);
    append(fields);
  } else if (kind == tensorium_mlir::abi::kKindSpectralResidualPoint) {
    const char *derivatives[] = {"value", "d1",  "d2",  "d3",  "d11",
                                 "d12",   "d13", "d22", "d23", "d33"};
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (const char *derivative : derivatives) {
        names.push_back(field == 0 ? derivative
                                   : fields[field] + "_" + derivative);
      }
    }
    append(coords);
    append(params);
  } else if (kind == tensorium_mlir::abi::kKindSpectralResidualJvpPoint) {
    const char *derivatives[] = {"value", "d1",  "d2",  "d3",  "d11",
                                 "d12",   "d13", "d22", "d23", "d33"};
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (const char *derivative : derivatives) {
        names.push_back(field == 0 ? derivative
                                   : fields[field] + "_" + derivative);
      }
    }
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (const char *derivative : derivatives) {
        names.push_back(field == 0
                            ? std::string("direction_") + derivative
                            : "direction_" + fields[field] + "_" + derivative);
      }
    }
    append(coords);
    append(params);
  } else if (kind == tensorium_mlir::abi::kKindSpectralResidualGrid) {
    names = {"n_points"};
    append(params);
    const char *derivatives[] = {"value", "d1",  "d2",  "d3",  "d11",
                                 "d12",   "d13", "d22", "d23", "d33"};
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (const char *derivative : derivatives) {
        names.push_back(field == 0 ? derivative
                                   : fields[field] + "_" + derivative);
      }
    }
    append(coords);
    append(outputs);
  }

  while (names.size() < fn.getNumArguments())
    names.push_back("arg" + std::to_string(names.size()));
  if (names.size() > fn.getNumArguments())
    names.resize(fn.getNumArguments());
  for (std::string &name : names)
    name = makeHostCIdentifier(name, "arg");
  return names;
}

std::unordered_map<std::string, std::int64_t>
fieldComponentCounts(const tensorium::backend::ModuleIR &module) {
  int dim = 3;
  if (module.simulation && module.simulation->dimension > 0)
    dim = module.simulation->dimension;

  std::unordered_map<std::string, std::int64_t> out;
  for (const auto &field : module.fields) {
    const int rank = field.tensorType.up + field.tensorType.down;
    std::int64_t components = 1;
    for (int i = 0; i < rank; ++i)
      components *= dim;
    out[field.name] = components;
  }
  return out;
}

std::vector<HostFieldABI>
hostFields(const tensorium::backend::ModuleIR &module,
           const std::unordered_map<std::string, std::int64_t> &components) {
  std::vector<HostFieldABI> out;
  out.reserve(module.fields.size());
  for (const auto &field : module.fields) {
    HostFieldABI abi;
    abi.name = field.name;
    abi.up = field.tensorType.up;
    abi.down = field.tensorType.down;
    abi.rank = field.tensorType.rank();
    auto it = components.find(field.name);
    abi.componentCount = it == components.end() ? 1 : it->second;
    out.push_back(std::move(abi));
  }
  return out;
}

std::string coordSystemName(tensorium::backend::CoordSystem coords) {
  using tensorium::backend::CoordSystem;
  switch (coords) {
  case CoordSystem::Cartesian:
    return "cartesian";
  case CoordSystem::Spherical:
    return "spherical";
  case CoordSystem::Cylindrical:
    return "cylindrical";
  }
  return "cartesian";
}

std::string spatialSchemeName(tensorium::backend::SpatialScheme scheme) {
  using tensorium::backend::SpatialScheme;
  switch (scheme) {
  case SpatialScheme::FD:
    return "fd";
  case SpatialScheme::Spectral:
    return "spectral";
  }
  return "fd";
}

std::string derivativeSchemeName(tensorium::backend::DerivativeScheme scheme) {
  using tensorium::backend::DerivativeScheme;
  switch (scheme) {
  case DerivativeScheme::Centered:
    return "centered";
  case DerivativeScheme::Upwind:
    return "upwind";
  }
  return "centered";
}

std::string hostWrapperName(mlir::func::FuncOp fn) {
  llvm::StringRef symbol = fn.getSymName();
  symbol.consume_front("tensorium_");
  return "tensorium_call_" + symbol.str();
}

HostArgKind hostArgKindFor(mlir::Type type) {
  if (type.isIndex())
    return HostArgKind::Index;
  if (isSupportedHostMemrefType(type))
    return HostArgKind::Memref1DF64;
  return HostArgKind::F64;
}

const HostFieldABI *findHostField(const std::vector<HostFieldABI> &fields,
                                  llvm::StringRef name) {
  for (const auto &field : fields) {
    if (field.name == name)
      return &field;
  }
  return nullptr;
}

HostArgAccess combineAccess(bool reads, bool writes) {
  if (reads && writes)
    return HostArgAccess::ReadWrite;
  if (reads)
    return HostArgAccess::Read;
  if (writes)
    return HostArgAccess::Write;
  return HostArgAccess::None;
}

HostBufferABI makeHostBufferABI(const HostModuleABI &abi,
                                llvm::StringRef logicalName,
                                std::int64_t argIndex, HostBufferRole role,
                                HostArgAccess access) {
  HostBufferABI buffer;
  buffer.name = logicalName.str();
  buffer.cName = makeHostCIdentifier(logicalName.str(), "buffer");
  buffer.argIndex = argIndex;
  buffer.role = role;
  buffer.access = access;

  if (role != HostBufferRole::Coordinate) {
    if (const auto *field = findHostField(abi.fields, logicalName)) {
      buffer.up = field->up;
      buffer.down = field->down;
      buffer.rank = field->rank;
      buffer.componentCount = field->componentCount;
    }
  }
  return buffer;
}

std::vector<HostBufferABI> hostKernelBuffers(const HostModuleABI &abi,
                                             const HostKernelABI &kernel) {
  std::vector<HostBufferABI> buffers;
  std::unordered_set<std::int64_t> reads(kernel.readArgIndices.begin(),
                                         kernel.readArgIndices.end());
  std::unordered_set<std::int64_t> writes(kernel.writeArgIndices.begin(),
                                          kernel.writeArgIndices.end());
  auto accessFor = [&](std::int64_t argIndex, HostArgAccess fallback) {
    const HostArgAccess access =
        combineAccess(reads.count(argIndex) != 0, writes.count(argIndex) != 0);
    return access == HostArgAccess::None ? fallback : access;
  };

  if (kernel.kind == tensorium_mlir::abi::kKindInitPoint) {
    const std::int64_t outputBase =
        static_cast<std::int64_t>(kernel.params.size() + kernel.coords.size());
    for (std::size_t i = 0; i < kernel.outputs.size(); ++i) {
      const std::int64_t argIndex = outputBase + static_cast<std::int64_t>(i);
      buffers.push_back(makeHostBufferABI(
          abi, kernel.outputs[i], argIndex, HostBufferRole::Output,
          accessFor(argIndex, HostArgAccess::Write)));
    }
    return buffers;
  }

  if (kernel.kind == tensorium_mlir::abi::kKindInitGridScf ||
      kernel.kind == tensorium_mlir::abi::kKindInitGridAffine) {
    const std::int64_t coordBase =
        static_cast<std::int64_t>(kernel.params.size());
    for (std::size_t i = 0; i < kernel.coords.size(); ++i) {
      const std::int64_t argIndex = coordBase + static_cast<std::int64_t>(i);
      buffers.push_back(makeHostBufferABI(
          abi, kernel.coords[i], argIndex, HostBufferRole::Coordinate,
          accessFor(argIndex, HostArgAccess::Read)));
    }

    const std::int64_t outputBase =
        coordBase + static_cast<std::int64_t>(kernel.coords.size());
    for (std::size_t i = 0; i < kernel.outputs.size(); ++i) {
      const std::int64_t argIndex = outputBase + static_cast<std::int64_t>(i);
      buffers.push_back(makeHostBufferABI(
          abi, kernel.outputs[i], argIndex, HostBufferRole::Output,
          accessFor(argIndex, HostArgAccess::Write)));
    }
    return buffers;
  }

  if (isFieldGridKind(kernel.kind)) {
    const std::int64_t fieldBase =
        6 + static_cast<std::int64_t>(kernel.params.size());
    for (std::size_t i = 0; i < kernel.fields.size(); ++i) {
      const std::int64_t argIndex = fieldBase + static_cast<std::int64_t>(i);
      buffers.push_back(makeHostBufferABI(
          abi, kernel.fields[i], argIndex, HostBufferRole::Field,
          accessFor(argIndex, HostArgAccess::None)));
    }
  }

  return buffers;
}

bool appendHostKernelABI(HostModuleABI &abi, mlir::func::FuncOp fn) {
  HostKernelABI kernel;
  kernel.symbolName = fn.getSymName().str();
  kernel.wrapperName = hostWrapperName(fn);
  kernel.kind =
      getStringAttr(fn.getOperation(), tensorium_mlir::abi::kAttrABIKind);
  if (!isHostCallableKind(kernel.kind) || !hasOnlySupportedHostTypes(fn) ||
      !hasSupportedHostResults(fn))
    return false;
  if (fn.getNumResults() == 1)
    kernel.returnKind = HostReturnKind::F64;

  kernel.params = getStringArrayAttr(fn.getOperation(),
                                     tensorium_mlir::abi::kAttrParamNames);
  kernel.coords = getStringArrayAttr(fn.getOperation(),
                                     tensorium_mlir::abi::kAttrCoordNames);
  kernel.fields = getStringArrayAttr(fn.getOperation(),
                                     tensorium_mlir::abi::kAttrFieldNames);
  kernel.outputs = getStringArrayAttr(fn.getOperation(),
                                      tensorium_mlir::abi::kAttrOutputNames);
  kernel.readArgIndices = getI64ArrayAttr(
      fn.getOperation(), tensorium_mlir::abi::kAttrReadArgIndices);
  kernel.writeArgIndices = getI64ArrayAttr(
      fn.getOperation(), tensorium_mlir::abi::kAttrWriteArgIndices);
  kernel.stencilRadius =
      getI64Attr(fn.getOperation(), tensorium_mlir::abi::kAttrStencilRadius);

  const auto names = logicalArgNames(fn);
  kernel.rawArgs.reserve(fn.getNumArguments());
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    HostArgABI arg;
    arg.kind = hostArgKindFor(fn.getArgument(i).getType());
    arg.cName = makeHostCIdentifier(names[i], "arg");
    kernel.rawArgs.push_back(std::move(arg));
  }
  kernel.buffers = hostKernelBuffers(abi, kernel);

  abi.kernels.push_back(std::move(kernel));
  return true;
}

const HostKernelABI *
findSpectralResidualKernel(const std::vector<HostKernelABI> &kernels,
                           llvm::StringRef kind, llvm::StringRef residualName) {
  for (const auto &kernel : kernels) {
    if (kernel.kind == kind && kernel.outputs.size() == 1 &&
        kernel.outputs[0] == residualName)
      return &kernel;
  }
  return nullptr;
}

std::int64_t appendOrFindUnknown(std::vector<std::string> &unknowns,
                                 llvm::StringRef name) {
  for (std::size_t i = 0; i < unknowns.size(); ++i) {
    if (unknowns[i] == name)
      return static_cast<std::int64_t>(i);
  }
  unknowns.push_back(name.str());
  return static_cast<std::int64_t>(unknowns.size() - 1);
}

std::int64_t findUnknownIndex(const std::vector<std::string> &unknowns,
                              llvm::StringRef name) {
  for (std::size_t i = 0; i < unknowns.size(); ++i) {
    if (unknowns[i] == name)
      return static_cast<std::int64_t>(i);
  }
  return -1;
}

std::vector<HostSpectralResidualSystemABI>
hostSpectralResidualSystems(const tensorium::backend::ModuleIR &module,
                            const std::vector<HostKernelABI> &kernels) {
  std::vector<HostSpectralResidualSystemABI> systems;
  if (!module.hasResidualConstraints)
    return systems;

  for (const auto &evo : module.evolutions) {
    HostSpectralResidualSystemABI system;
    system.name = evo.name;
    bool complete = true;
    for (const auto &eq : evo.equations) {
      const HostKernelABI *pointKernel = findSpectralResidualKernel(
          kernels, tensorium_mlir::abi::kKindSpectralResidualPoint,
          eq.fieldName);
      if (!pointKernel || pointKernel->fields.empty()) {
        complete = false;
        break;
      }
      const HostKernelABI *gridKernel = findSpectralResidualKernel(
          kernels, tensorium_mlir::abi::kKindSpectralResidualGrid,
          eq.fieldName);

      HostSpectralResidualSystemEquationABI equation;
      equation.residualName = eq.fieldName;
      equation.unknownName = pointKernel->fields.front();
      equation.unknownIndex =
          appendOrFindUnknown(system.unknownNames, equation.unknownName);
      equation.pointKernelSymbol = pointKernel->symbolName;
      if (gridKernel)
        equation.gridKernelSymbol = gridKernel->symbolName;
      equation.params = pointKernel->params;
      if (pointKernel->fields.size() > 1) {
        equation.auxiliaryNames.assign(pointKernel->fields.begin() + 1,
                                       pointKernel->fields.end());
      }
      system.equations.push_back(std::move(equation));
    }
    if (!complete || system.equations.empty())
      continue;

    for (auto &equation : system.equations) {
      equation.auxiliaryUnknownIndices.reserve(equation.auxiliaryNames.size());
      for (const auto &auxiliary : equation.auxiliaryNames)
        equation.auxiliaryUnknownIndices.push_back(
            findUnknownIndex(system.unknownNames, auxiliary));
    }
    systems.push_back(std::move(system));
  }
  return systems;
}

} // namespace

std::string makeHostCIdentifier(std::string_view input,
                                std::string_view fallback) {
  std::string out;
  for (char ch : input) {
    unsigned char u = static_cast<unsigned char>(ch);
    if (std::isalnum(u) || ch == '_')
      out.push_back(ch);
    else
      out.push_back('_');
  }
  if (out.empty())
    out = std::string(fallback);
  if (std::isdigit(static_cast<unsigned char>(out.front())))
    out.insert(out.begin(), '_');
  return out;
}

std::int64_t requiredBufferScalars(const HostBufferABI &buffer,
                                   std::int64_t nPoints) {
  if (nPoints <= 0 || buffer.componentCount <= 0)
    return 0;
  return buffer.componentCount * nPoints;
}

std::vector<std::string> validateHostModuleABI(const HostModuleABI &abi) {
  std::vector<std::string> errors;
  auto add = [&](std::string message) { errors.push_back(std::move(message)); };

  if (abi.dimension <= 0)
    add("dimension must be positive");
  if (abi.coordSystem.empty())
    add("coordinate system must be set");
  if (abi.spatialOrder < 0)
    add("spatial order must not be negative");

  std::unordered_set<std::string> fieldNames;
  for (const auto &field : abi.fields) {
    if (field.name.empty())
      add("field name must not be empty");
    if (!fieldNames.insert(field.name).second)
      add("duplicate field descriptor: " + field.name);
    if (field.rank != field.up + field.down)
      add("field rank/variance mismatch: " + field.name);
    if (field.componentCount <= 0)
      add("field component count must be positive: " + field.name);
    auto it = abi.componentCounts.find(field.name);
    if (it == abi.componentCounts.end())
      add("missing component count entry: " + field.name);
    else if (it->second != field.componentCount)
      add("component count map mismatch: " + field.name);
  }

  for (const auto &kernel : abi.kernels) {
    if (kernel.symbolName.empty())
      add("kernel symbol name must not be empty");
    if (kernel.wrapperName.empty())
      add("kernel wrapper name must not be empty: " + kernel.symbolName);
    if (kernel.kind.empty())
      add("kernel kind must not be empty: " + kernel.symbolName);
    if (kernel.stencilRadius < 0)
      add("kernel stencil radius must not be negative: " + kernel.symbolName);

    auto validateArgIndex = [&](std::int64_t idx, const char *attrName) {
      if (idx < 0 || static_cast<std::size_t>(idx) >= kernel.rawArgs.size()) {
        std::ostringstream oss;
        oss << "kernel " << kernel.symbolName << " has out-of-range "
            << attrName << " index " << idx;
        add(oss.str());
        return;
      }
      if (kernel.rawArgs[static_cast<std::size_t>(idx)].kind !=
          HostArgKind::Memref1DF64) {
        std::ostringstream oss;
        oss << "kernel " << kernel.symbolName << " has non-buffer " << attrName
            << " index " << idx;
        add(oss.str());
      }
    };
    for (std::int64_t idx : kernel.readArgIndices)
      validateArgIndex(idx, "read_arg_indices");
    for (std::int64_t idx : kernel.writeArgIndices)
      validateArgIndex(idx, "write_arg_indices");

    for (const auto &buffer : kernel.buffers) {
      if (buffer.name.empty())
        add("kernel buffer name must not be empty: " + kernel.symbolName);
      if (buffer.argIndex < 0 ||
          static_cast<std::size_t>(buffer.argIndex) >= kernel.rawArgs.size()) {
        std::ostringstream oss;
        oss << "kernel " << kernel.symbolName
            << " has buffer with out-of-range arg index " << buffer.argIndex;
        add(oss.str());
        continue;
      }
      if (kernel.rawArgs[static_cast<std::size_t>(buffer.argIndex)].kind !=
          HostArgKind::Memref1DF64)
        add("kernel buffer does not map to a memref arg: " + kernel.symbolName +
            "." + buffer.name);
      if (buffer.componentCount <= 0)
        add("kernel buffer component count must be positive: " +
            kernel.symbolName + "." + buffer.name);
      if (buffer.role != HostBufferRole::Coordinate &&
          !fieldNames.count(buffer.name))
        add("kernel buffer references unknown field: " + kernel.symbolName +
            "." + buffer.name);
    }

    const bool isFieldGrid = isFieldGridKind(kernel.kind);
    if (isFieldGrid) {
      if (kernel.rawArgs.size() < 6) {
        add("field grid kernel must expose grid prefix args: " +
            kernel.symbolName);
      } else if (kernel.rawArgs[0].kind != HostArgKind::Index ||
                 kernel.rawArgs[1].kind != HostArgKind::Index ||
                 kernel.rawArgs[2].kind != HostArgKind::Index ||
                 kernel.rawArgs[3].kind != HostArgKind::F64 ||
                 kernel.rawArgs[4].kind != HostArgKind::F64 ||
                 kernel.rawArgs[5].kind != HostArgKind::F64) {
        add("field grid kernel prefix arg kinds mismatch: " +
            kernel.symbolName);
      }
      for (const auto &field : kernel.fields) {
        if (!fieldNames.count(field))
          add("field grid kernel references unknown field: " +
              kernel.symbolName + "." + field);
      }
    }
  }

  return errors;
}

HostModuleABI buildHostModuleABI(const tensorium::backend::ModuleIR &module,
                                 mlir::ModuleOp moduleOp) {
  HostModuleABI abi;
  if (module.simulation && module.simulation->dimension > 0) {
    abi.dimension = module.simulation->dimension;
    abi.coordSystem = coordSystemName(module.simulation->coords);
    abi.resolution = module.simulation->resolution;
    abi.spatialOrder = module.simulation->spatial.order;
    abi.spatialScheme = spatialSchemeName(module.simulation->spatial.scheme);
    abi.derivativeScheme =
        derivativeSchemeName(module.simulation->spatial.derivative);
  }
  abi.componentCounts = fieldComponentCounts(module);
  abi.fields = hostFields(module, abi.componentCounts);

  moduleOp.walk([&](mlir::func::FuncOp fn) { appendHostKernelABI(abi, fn); });
  abi.spectralResidualSystems =
      hostSpectralResidualSystems(module, abi.kernels);
  if (module.spectralInitialData) {
    const auto &source = *module.spectralInitialData;
    HostSpectralInitialDataABI initialData;
    initialData.name = source.name;
    initialData.system = source.system;
    initialData.coordinateMap = source.coordinateMap;
    initialData.resolution.assign(source.resolution.begin(),
                                  source.resolution.end());
    initialData.basis = source.basis;
    initialData.coordinateParameters = source.coordinateParameters;
    initialData.unknownMap = source.unknownMap;
    initialData.unknownMapParameters = source.unknownMapParameters;
    initialData.fieldProjector = source.fieldProjector;
    initialData.reconstruction = source.reconstruction;
    for (const auto &binding : source.parameters) {
      initialData.parameterNames.push_back(binding.name);
      initialData.parameterValues.push_back(binding.value);
    }
    initialData.solve = source.solve;
    abi.spectralInitialData = std::move(initialData);
  }

  for (const auto &print : module.prints) {
    HostPrintABI out;
    out.label = print.label;
    out.fieldName = print.fieldName;
    out.rank = print.tensorType.rank();
    abi.prints.push_back(std::move(out));

    bool seen = false;
    for (const auto &field : abi.printFields) {
      if (field == print.fieldName) {
        seen = true;
        break;
      }
    }
    if (!seen)
      abi.printFields.push_back(print.fieldName);
  }

  return abi;
}

} // namespace tensorium_mlir
