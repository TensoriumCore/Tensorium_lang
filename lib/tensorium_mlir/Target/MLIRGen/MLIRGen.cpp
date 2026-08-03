#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "MLIRGenExpr.h"
#include "MLIRGenInitialData.h"
#include "MLIRGenPipeline.h"
#include "MLIRGenShared.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include <stdexcept>

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

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
buildMLIRModule(const tensorium::backend::ModuleIR &module,
                mlir::MLIRContext &ctx, const MLIRGenOptions &opts,
                bool *pipelineSuccess) {
  if (module.constraintProblem && module.evolutions.empty()) {
    throw std::runtime_error(
        "constraint problem MLIR lowering is not implemented; use "
        "--validate or --dump-backend-expr");
  }
  if (opts.mlirDisableThreading)
    ctx.disableMultithreading();
  ctx.printOpOnDiagnostic(opts.mlirPrintOpOnDiagnostic);

  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<tensorium::mlir::TensoriumDialect>();

  mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();
  auto moduleOp =
      mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc));

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

  auto initFunc =
      mlir::func::FuncOp::create(loc, tensorium_mlir::abi::kSymbolInit,
                                 buildTypeFromIndices(initArgIndices));
  auto rhsFunc =
      mlir::func::FuncOp::create(loc, tensorium_mlir::abi::kSymbolRhs,
                                 buildTypeFromIndices(rhsArgIndices));
  auto entryFunc =
      mlir::func::FuncOp::create(loc, tensorium_mlir::abi::kSymbolEntry,
                                 b.getFunctionType(allArgTypes, {}));

  auto mapFieldArgs = [&](mlir::Block *block,
                          const std::vector<unsigned> &indices) {
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
    fn->setAttr(
        tensorium_mlir::abi::kAttrABIVersion,
        b.getI64IntegerAttr(tensorium_mlir::abi::kGeneratedKernelABIVersion));
    fn->setAttr(tensorium_mlir::abi::kAttrABIKind, b.getStringAttr(kind));
    fn->setAttr(
        tensorium_mlir::abi::kAttrMemoryLayout,
        b.getStringAttr(tensorium_mlir::abi::kMemLayoutSoAComponentMajor));
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

  b.create<mlir::func::CallOp>(loc, tensorium_mlir::abi::kSymbolInit,
                               mlir::TypeRange{}, initCallArgs);
  b.create<mlir::func::CallOp>(loc, tensorium_mlir::abi::kSymbolRhs,
                               mlir::TypeRange{}, rhsCallArgs);
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
    pm.enableIRPrinting([](mlir::Pass *, mlir::Operation *) { return false; },
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

} // namespace tensorium_mlir
