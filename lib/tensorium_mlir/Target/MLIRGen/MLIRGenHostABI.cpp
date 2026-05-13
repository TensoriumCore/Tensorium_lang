#include "tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

#include <cctype>

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
         kind == tensorium_mlir::abi::kKindRhsGridAffine;
}

std::vector<std::string> logicalArgNames(mlir::func::FuncOp fn) {
  const std::string kind =
      getStringAttr(fn.getOperation(), tensorium_mlir::abi::kAttrABIKind);
  std::vector<std::string> names;
  auto append = [&](const std::vector<std::string> &items) {
    names.insert(names.end(), items.begin(), items.end());
  };

  const auto params =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrParamNames);
  const auto coords =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrCoordNames);
  const auto fields =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrFieldNames);
  const auto outputs =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrOutputNames);

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
             kind == tensorium_mlir::abi::kKindRhsGridAffine) {
    names = {"nx", "ny", "nz", "dx", "dy", "dz"};
    append(params);
    append(fields);
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

bool appendHostKernelABI(HostModuleABI &abi, mlir::func::FuncOp fn) {
  HostKernelABI kernel;
  kernel.symbolName = fn.getSymName().str();
  kernel.wrapperName = hostWrapperName(fn);
  kernel.kind =
      getStringAttr(fn.getOperation(), tensorium_mlir::abi::kAttrABIKind);
  if (!isHostCallableKind(kernel.kind) || !hasOnlySupportedHostTypes(fn))
    return false;

  kernel.params =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrParamNames);
  kernel.coords =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrCoordNames);
  kernel.fields =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrFieldNames);
  kernel.outputs =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrOutputNames);

  const auto names = logicalArgNames(fn);
  kernel.rawArgs.reserve(fn.getNumArguments());
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    HostArgABI arg;
    arg.kind = hostArgKindFor(fn.getArgument(i).getType());
    arg.cName = makeHostCIdentifier(names[i], "arg");
    kernel.rawArgs.push_back(std::move(arg));
  }

  abi.kernels.push_back(std::move(kernel));
  return true;
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

HostModuleABI buildHostModuleABI(const tensorium::backend::ModuleIR &module,
                                 mlir::ModuleOp moduleOp) {
  HostModuleABI abi;
  if (module.simulation && module.simulation->dimension > 0)
    abi.dimension = module.simulation->dimension;
  abi.componentCounts = fieldComponentCounts(module);

  moduleOp.walk([&](mlir::func::FuncOp fn) { appendHostKernelABI(abi, fn); });

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
