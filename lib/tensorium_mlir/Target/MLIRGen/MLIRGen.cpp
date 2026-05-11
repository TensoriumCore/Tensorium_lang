#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"
#include "MLIRGenExpr.h"
#include "MLIRGenInitialData.h"
#include "MLIRGenPipeline.h"
#include "MLIRGenShared.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace tensorium_mlir {
namespace {

llvm::StringRef coordSystemToAttr(tensorium::backend::CoordSystem coords) {
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

std::string cIdentifier(llvm::StringRef input, llvm::StringRef fallback) {
  std::string out;
  for (char ch : input) {
    unsigned char u = static_cast<unsigned char>(ch);
    if (std::isalnum(u) || ch == '_')
      out.push_back(ch);
    else
      out.push_back('_');
  }
  if (out.empty())
    out = fallback.str();
  if (std::isdigit(static_cast<unsigned char>(out.front())))
    out.insert(out.begin(), '_');
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
    name = cIdentifier(name, "arg");
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

std::int64_t componentCountFor(
    llvm::StringRef name,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  auto it = componentCounts.find(name.str());
  if (it == componentCounts.end() || it->second < 1)
    return 1;
  return it->second;
}

std::string sizeExprFor(
    llvm::StringRef name, llvm::StringRef nExpr,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  const std::int64_t components = componentCountFor(name, componentCounts);
  if (components == 1)
    return nExpr.str();
  return std::to_string(components) + " * " + nExpr.str();
}

std::string hostWrapperName(mlir::func::FuncOp fn) {
  llvm::StringRef symbol = fn.getSymName();
  symbol.consume_front("tensorium_");
  return "tensorium_call_" + symbol.str();
}

void appendComma(std::ostringstream &os, bool &first) {
  if (!first)
    os << ", ";
  first = false;
}

void emitRawFormal(std::ostringstream &os, mlir::Type type,
                   llvm::StringRef baseName, bool &first) {
  const std::string base = cIdentifier(baseName, "arg");
  if (type.isF64()) {
    appendComma(os, first);
    os << "double " << base;
    return;
  }
  if (type.isIndex()) {
    appendComma(os, first);
    os << "int64_t " << base;
    return;
  }
  if (isSupportedHostMemrefType(type)) {
    appendComma(os, first);
    os << "double *" << base << "_alloc";
    appendComma(os, first);
    os << "double *" << base << "_aligned";
    appendComma(os, first);
    os << "int64_t " << base << "_offset";
    appendComma(os, first);
    os << "int64_t " << base << "_size";
    appendComma(os, first);
    os << "int64_t " << base << "_stride";
  }
}

void emitRawPrototype(std::ostringstream &os, mlir::func::FuncOp fn) {
  auto names = logicalArgNames(fn);
  os << "extern void " << fn.getSymName().str() << "(";
  bool first = true;
  for (unsigned i = 0; i < fn.getNumArguments(); ++i)
    emitRawFormal(os, fn.getArgument(i).getType(), names[i], first);
  if (first)
    os << "void";
  os << ");\n";
}

bool hasOnlySupportedHostTypes(mlir::func::FuncOp fn) {
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    if (!isSupportedHostType(fn.getArgument(i).getType()))
      return false;
  }
  return true;
}

void emitScalarFormal(std::ostringstream &os, llvm::StringRef type,
                      llvm::StringRef name, bool &first) {
  appendComma(os, first);
  os << type.str() << " " << cIdentifier(name, "arg");
}

void emitBufferFormal(std::ostringstream &os, llvm::StringRef name,
                      bool &first) {
  appendComma(os, first);
  os << "double *" << cIdentifier(name, "buffer");
}

void emitDescriptorCallArgs(std::ostringstream &os, llvm::StringRef name,
                            llvm::StringRef sizeExpr, bool &first) {
  const std::string cName = cIdentifier(name, "buffer");
  appendComma(os, first);
  os << cName;
  appendComma(os, first);
  os << cName;
  appendComma(os, first);
  os << "0";
  appendComma(os, first);
  os << sizeExpr.str();
  appendComma(os, first);
  os << "1";
}

void emitInitPointWrapper(
    std::ostringstream &os, mlir::func::FuncOp fn,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  const auto params =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrParamNames);
  const auto coords =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrCoordNames);
  const auto outputs =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrOutputNames);
  if (outputs.empty())
    return;

  os << "static inline void " << hostWrapperName(fn) << "(";
  bool first = true;
  for (const auto &param : params)
    emitScalarFormal(os, "double", param, first);
  for (const auto &coord : coords)
    emitScalarFormal(os, "double", coord, first);
  for (const auto &output : outputs)
    emitBufferFormal(os, output, first);
  os << ") {\n  " << fn.getSymName().str() << "(";

  first = true;
  for (const auto &param : params) {
    appendComma(os, first);
    os << cIdentifier(param, "param");
  }
  for (const auto &coord : coords) {
    appendComma(os, first);
    os << cIdentifier(coord, "coord");
  }
  for (const auto &output : outputs) {
    emitDescriptorCallArgs(os, output,
                           std::to_string(componentCountFor(output,
                                                            componentCounts)),
                           first);
  }
  os << ");\n}\n";
}

void emitInitGridWrapper(
    std::ostringstream &os, mlir::func::FuncOp fn,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  const auto params =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrParamNames);
  const auto coords =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrCoordNames);
  const auto outputs =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrOutputNames);
  if (outputs.empty())
    return;

  os << "static inline void " << hostWrapperName(fn) << "(";
  bool first = true;
  for (const auto &param : params)
    emitScalarFormal(os, "double", param, first);
  for (const auto &coord : coords)
    emitBufferFormal(os, coord, first);
  for (const auto &output : outputs)
    emitBufferFormal(os, output, first);
  emitScalarFormal(os, "int64_t", "n_points", first);
  os << ") {\n  " << fn.getSymName().str() << "(";

  first = true;
  for (const auto &param : params) {
    appendComma(os, first);
    os << cIdentifier(param, "param");
  }
  for (const auto &coord : coords)
    emitDescriptorCallArgs(os, coord, "n_points", first);
  for (const auto &output : outputs)
    emitDescriptorCallArgs(os, output,
                           sizeExprFor(output, "n_points", componentCounts),
                           first);
  os << ");\n}\n";
}

void emitRhsGridWrapper(
    std::ostringstream &os, mlir::func::FuncOp fn,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  const auto params =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrParamNames);
  const auto fields =
      getStringArrayAttr(fn.getOperation(), tensorium_mlir::abi::kAttrFieldNames);
  if (fields.empty())
    return;

  os << "static inline void " << hostWrapperName(fn) << "(";
  bool first = true;
  emitScalarFormal(os, "int64_t", "nx", first);
  emitScalarFormal(os, "int64_t", "ny", first);
  emitScalarFormal(os, "int64_t", "nz", first);
  emitScalarFormal(os, "double", "dx", first);
  emitScalarFormal(os, "double", "dy", first);
  emitScalarFormal(os, "double", "dz", first);
  for (const auto &param : params)
    emitScalarFormal(os, "double", param, first);
  for (const auto &field : fields)
    emitBufferFormal(os, field, first);
  os << ") {\n"
     << "  const int64_t n_points = nx * ny * nz;\n"
     << "  " << fn.getSymName().str() << "(";

  first = true;
  for (llvm::StringRef name : {"nx", "ny", "nz", "dx", "dy", "dz"}) {
    appendComma(os, first);
    os << name.str();
  }
  for (const auto &param : params) {
    appendComma(os, first);
    os << cIdentifier(param, "param");
  }
  for (const auto &field : fields)
    emitDescriptorCallArgs(os, field,
                           sizeExprFor(field, "n_points", componentCounts),
                           first);
  os << ");\n}\n";
}

void emitConvenienceWrapper(
    std::ostringstream &os, mlir::func::FuncOp fn,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  const std::string kind =
      getStringAttr(fn.getOperation(), tensorium_mlir::abi::kAttrABIKind);
  if (kind == tensorium_mlir::abi::kKindInitPoint) {
    emitInitPointWrapper(os, fn, componentCounts);
  } else if (kind == tensorium_mlir::abi::kKindInitGridScf ||
             kind == tensorium_mlir::abi::kKindInitGridAffine) {
    emitInitGridWrapper(os, fn, componentCounts);
  } else if (kind == tensorium_mlir::abi::kKindRhsGridScf ||
             kind == tensorium_mlir::abi::kKindRhsGridAffine) {
    emitRhsGridWrapper(os, fn, componentCounts);
  }
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
buildMLIRModule(const tensorium::backend::ModuleIR &module,
                mlir::MLIRContext &ctx, const MLIRGenOptions &opts,
                bool *pipelineSuccess) {
  if (opts.mlirDisableThreading)
    ctx.disableMultithreading();
  ctx.printOpOnDiagnostic(opts.mlirPrintOpOnDiagnostic);

  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<tensorium::mlir::TensoriumDialect>();

  mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();
  auto moduleOp = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc));

  const auto fields = extractFields(module);
  llvm::SmallVector<mlir::Type, 8> allArgTypes;
  allArgTypes.reserve(fields.size());
  for (const auto &fd : fields) {
    allArgTypes.push_back(
        tensorium::mlir::FieldType::get(&ctx, b.getF64Type(), fd.up, fd.down));
  }

  std::vector<unsigned> initArgIndices = collectInitArgIndices(module, fields);
  std::vector<unsigned> rhsArgIndices = collectRhsArgIndices(module, fields);
  std::vector<unsigned> allArgIndices(fields.size());
  std::iota(allArgIndices.begin(), allArgIndices.end(), 0u);

  auto buildTypeFromIndices = [&](const std::vector<unsigned> &indices) {
    llvm::SmallVector<mlir::Type, 8> types;
    types.reserve(indices.size());
    for (unsigned idx : indices)
      types.push_back(allArgTypes[idx]);
    return b.getFunctionType(types, {});
  };

  auto initFunc = mlir::func::FuncOp::create(
      loc, tensorium_mlir::abi::kSymbolInit, buildTypeFromIndices(initArgIndices));
  auto rhsFunc = mlir::func::FuncOp::create(
      loc, tensorium_mlir::abi::kSymbolRhs, buildTypeFromIndices(rhsArgIndices));
  auto entryFunc = mlir::func::FuncOp::create(
      loc, tensorium_mlir::abi::kSymbolEntry, b.getFunctionType(allArgTypes, {}));

  auto mapFieldArgs = [&](mlir::Block *block, const std::vector<unsigned> &indices) {
    llvm::DenseMap<llvm::StringRef, mlir::Value> fieldArg;
    for (unsigned i = 0; i < indices.size(); ++i)
      fieldArg[fields[indices[i]].name] = block->getArgument(i);
    return fieldArg;
  };

  auto makeFieldNamesAttr = [&](const std::vector<unsigned> &indices) {
    llvm::SmallVector<mlir::Attribute> attrs;
    attrs.reserve(indices.size());
    for (unsigned idx : indices)
      attrs.push_back(b.getStringAttr(fields[idx].name));
    return b.getArrayAttr(attrs);
  };

  auto setCommonABIAttrs = [&](mlir::func::FuncOp fn, llvm::StringRef kind) {
    fn->setAttr(tensorium_mlir::abi::kAttrABIVersion,
                b.getI64IntegerAttr(
                    tensorium_mlir::abi::kGeneratedKernelABIVersion));
    fn->setAttr(tensorium_mlir::abi::kAttrABIKind, b.getStringAttr(kind));
    fn->setAttr(tensorium_mlir::abi::kAttrMemoryLayout,
                b.getStringAttr(
                    tensorium_mlir::abi::kMemLayoutSoAComponentMajor));
    fn->setAttr(tensorium_mlir::abi::kAttrMemrefABI,
                b.getStringAttr(tensorium_mlir::abi::kMemrefABI1DStridedF64));
  };

  setCommonABIAttrs(initFunc, tensorium_mlir::abi::kKindInitSource);
  setCommonABIAttrs(rhsFunc, tensorium_mlir::abi::kKindRhsSource);
  setCommonABIAttrs(entryFunc, tensorium_mlir::abi::kKindEntrySource);
  initFunc->setAttr(tensorium_mlir::abi::kAttrFieldNames,
                    makeFieldNamesAttr(initArgIndices));
  rhsFunc->setAttr(tensorium_mlir::abi::kAttrFieldNames,
                   makeFieldNamesAttr(rhsArgIndices));
  entryFunc->setAttr(tensorium_mlir::abi::kAttrFieldNames,
                     makeFieldNamesAttr(allArgIndices));

  auto *initBlock = initFunc.addEntryBlock();
  b.setInsertionPointToEnd(initBlock);
  auto initFieldArg = mapFieldArgs(initBlock, initArgIndices);
  emitInitialDataOps(b, loc, module, initFieldArg);
  b.create<mlir::func::ReturnOp>(loc);

  auto *rhsBlock = rhsFunc.addEntryBlock();
  b.setInsertionPointToEnd(rhsBlock);
  auto rhsFieldArg = mapFieldArgs(rhsBlock, rhsArgIndices);
  emitEvolutionOps(b, loc, module, rhsFieldArg);
  b.create<mlir::func::ReturnOp>(loc);

  auto *entryBlock = entryFunc.addEntryBlock();
  b.setInsertionPointToEnd(entryBlock);
  llvm::SmallVector<mlir::Value, 8> initCallArgs;
  initCallArgs.reserve(initArgIndices.size());
  for (unsigned idx : initArgIndices)
    initCallArgs.push_back(entryBlock->getArgument(idx));

  llvm::SmallVector<mlir::Value, 8> rhsCallArgs;
  rhsCallArgs.reserve(rhsArgIndices.size());
  for (unsigned idx : rhsArgIndices)
    rhsCallArgs.push_back(entryBlock->getArgument(idx));

  b.create<mlir::func::CallOp>(loc, tensorium_mlir::abi::kSymbolInit, mlir::TypeRange{},
                             initCallArgs);
  b.create<mlir::func::CallOp>(loc, tensorium_mlir::abi::kSymbolRhs, mlir::TypeRange{},
                             rhsCallArgs);
  b.create<mlir::func::ReturnOp>(loc);

  moduleOp->getOperation()->setAttr(
      tensorium_mlir::abi::kAttrABIVersion,
      b.getI64IntegerAttr(tensorium_mlir::abi::kGeneratedKernelABIVersion));
  moduleOp->getOperation()->setAttr(
      tensorium_mlir::abi::kAttrMemoryLayout,
      b.getStringAttr(tensorium_mlir::abi::kMemLayoutSoAComponentMajor));
  moduleOp->getOperation()->setAttr(
      tensorium_mlir::abi::kAttrMemrefABI,
      b.getStringAttr(tensorium_mlir::abi::kMemrefABI1DStridedF64));

  if (module.simulation) {
    moduleOp->getOperation()->setAttr(
        "tensorium.sim.dim", b.getI64IntegerAttr(module.simulation->dimension));
    moduleOp->getOperation()->setAttr(
        "tensorium.sim.coords",
        b.getStringAttr(coordSystemToAttr(module.simulation->coords)));
  }
  moduleOp->push_back(initFunc);
  moduleOp->push_back(rhsFunc);
  moduleOp->push_back(entryFunc);

  MLIRGenOptions pipelineOpts = opts;
  if (module.simulation) {
    pipelineOpts.order = module.simulation->spatial.order;
    if (!module.simulation->resolution.empty() &&
        module.simulation->resolution[0] > 0) {
      pipelineOpts.dx =
          1.0 / static_cast<double>(module.simulation->resolution[0]);
    }
  }

  mlir::PassManager pm(&ctx);
  if (pipelineOpts.mlirPrintIRAfterFailure) {
    pm.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/true);
  }
  addEinsteinPipelineSafe(pm, pipelineOpts);
  addPostMLIRNormalizationPipeline(pm, pipelineOpts);

  const bool ok = mlir::succeeded(pm.run(*moduleOp));
  if (!ok)
    llvm::errs() << "Pipeline failed\n";
  if (pipelineSuccess)
    *pipelineSuccess = ok;
  return moduleOp;
}

bool emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts, std::string *mlirText) {
  mlir::MLIRContext ctx;
  bool pipelineOk = true;
  auto moduleOp = buildMLIRModule(module, ctx, opts, &pipelineOk);
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  moduleOp->print(stream);
  stream.flush();

  if (mlirText) {
    *mlirText = std::move(buffer);
  } else {
    llvm::outs() << buffer;
  }
  return pipelineOk;
}

bool emitLLVMIR(const tensorium::backend::ModuleIR &module,
                const MLIRGenOptions &opts, std::string *llvmIRText) {
  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  ctx.appendDialectRegistry(registry);
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  bool pipelineOk = true;
  auto moduleOp = buildMLIRModule(module, ctx, opts, &pipelineOk);
  if (!pipelineOk)
    return false;

  if (!lowerModuleToLLVM(*moduleOp, ctx, opts)) {
    llvm::errs() << "LLVM lowering pipeline failed\n";
    return false;
  }

  llvm::LLVMContext llvmCtx;
  auto llvmModule = mlir::translateModuleToLLVMIR(*moduleOp, llvmCtx);
  if (!llvmModule) {
    llvm::errs() << "LLVM IR translation failed\n";
    return false;
  }

  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  llvmModule->print(stream, nullptr);
  stream.flush();

  if (llvmIRText) {
    *llvmIRText = std::move(buffer);
  } else {
    llvm::outs() << buffer;
  }
  return true;
}

bool emitHostHeader(const tensorium::backend::ModuleIR &module,
                    const MLIRGenOptions &opts, std::string *headerText) {
  mlir::MLIRContext ctx;
  bool pipelineOk = true;
  auto moduleOp = buildMLIRModule(module, ctx, opts, &pipelineOk);
  if (!pipelineOk)
    return false;

  std::vector<mlir::func::FuncOp> hostFns;
  moduleOp->walk([&](mlir::func::FuncOp fn) {
    const std::string kind =
        getStringAttr(fn.getOperation(), tensorium_mlir::abi::kAttrABIKind);
    if (!isHostCallableKind(kind))
      return;
    if (!hasOnlySupportedHostTypes(fn))
      return;
    hostFns.push_back(fn);
  });

  const auto componentCounts = fieldComponentCounts(module);
  std::ostringstream os;
  os << "#ifndef TENSORIUM_GENERATED_HOST_H\n"
     << "#define TENSORIUM_GENERATED_HOST_H\n\n"
     << "#include <stdint.h>\n\n"
     << "typedef struct tensorium_memref1d_f64 {\n"
     << "  double *allocated;\n"
     << "  double *aligned;\n"
     << "  int64_t offset;\n"
     << "  int64_t size;\n"
     << "  int64_t stride;\n"
     << "} tensorium_memref1d_f64;\n\n"
     << "static inline tensorium_memref1d_f64 "
        "tensorium_make_memref1d_f64(double *data, int64_t size) {\n"
     << "  tensorium_memref1d_f64 ref = {data, data, 0, size, 1};\n"
     << "  return ref;\n"
     << "}\n\n"
     << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";

  for (auto fn : hostFns)
    emitRawPrototype(os, fn);

  os << "\n#ifdef __cplusplus\n}\n#endif\n\n";

  for (auto fn : hostFns) {
    emitConvenienceWrapper(os, fn, componentCounts);
    os << "\n";
  }

  os << "#endif /* TENSORIUM_GENERATED_HOST_H */\n";

  if (headerText) {
    *headerText = os.str();
  } else {
    llvm::outs() << os.str();
  }
  return true;
}

} // namespace tensorium_mlir
