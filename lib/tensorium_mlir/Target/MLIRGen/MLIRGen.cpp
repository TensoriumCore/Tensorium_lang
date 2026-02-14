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
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

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

  auto buildTypeFromIndices = [&](const std::vector<unsigned> &indices) {
    llvm::SmallVector<mlir::Type, 8> types;
    types.reserve(indices.size());
    for (unsigned idx : indices)
      types.push_back(allArgTypes[idx]);
    return b.getFunctionType(types, {});
  };

  auto initFunc = mlir::func::FuncOp::create(
      loc, "tensorium_init", buildTypeFromIndices(initArgIndices));
  auto rhsFunc = mlir::func::FuncOp::create(
      loc, "tensorium_rhs", buildTypeFromIndices(rhsArgIndices));
  auto entryFunc = mlir::func::FuncOp::create(
      loc, "tensorium_entry", b.getFunctionType(allArgTypes, {}));

  auto mapFieldArgs = [&](mlir::Block *block, const std::vector<unsigned> &indices) {
    llvm::DenseMap<llvm::StringRef, mlir::Value> fieldArg;
    for (unsigned i = 0; i < indices.size(); ++i)
      fieldArg[fields[indices[i]].name] = block->getArgument(i);
    return fieldArg;
  };

  auto *initBlock = initFunc.addEntryBlock();
  b.setInsertionPointToEnd(initBlock);
  auto initFieldArg = mapFieldArgs(initBlock, initArgIndices);
  emitInitialDataOps(b, loc, module, initFieldArg);
  mlir::func::ReturnOp::create(b, loc);

  auto *rhsBlock = rhsFunc.addEntryBlock();
  b.setInsertionPointToEnd(rhsBlock);
  auto rhsFieldArg = mapFieldArgs(rhsBlock, rhsArgIndices);
  emitEvolutionOps(b, loc, module, rhsFieldArg);
  mlir::func::ReturnOp::create(b, loc);

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

  mlir::func::CallOp::create(b, loc, "tensorium_init", mlir::TypeRange{},
                             initCallArgs);
  mlir::func::CallOp::create(b, loc, "tensorium_rhs", mlir::TypeRange{},
                             rhsCallArgs);
  mlir::func::ReturnOp::create(b, loc);

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
              const MLIRGenOptions &opts) {
  mlir::MLIRContext ctx;
  bool pipelineOk = true;
  auto moduleOp = buildMLIRModule(module, ctx, opts, &pipelineOk);
  moduleOp->print(llvm::outs());
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
