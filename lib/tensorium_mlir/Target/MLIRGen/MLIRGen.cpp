#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

namespace tensorium_mlir {

void emitMLIR(const tensorium::backend::ModuleIR & /*module*/,
              const MLIRGenOptions &opts) {
  std::cerr << "[MLIR] emitMLIR called\n";

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();

  ctx.getOrLoadDialect<tensorium::mlir::TensoriumDialect>();
  mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();

  mlir::ModuleOp moduleOp = mlir::ModuleOp::create(loc);

  auto funcTy = b.getFunctionType({}, {});
  auto f = b.create<mlir::func::FuncOp>(loc, "tensorium_entry", funcTy);
  auto *entry = f.addEntryBlock();
  b.setInsertionPointToEnd(entry);
  b.create<mlir::func::ReturnOp>(loc);
  moduleOp.push_back(f);

  mlir::PassManager pm(&ctx);
  auto fieldTy = tensorium::mlir::FieldType::get(&ctx, b.getF64Type(), 0);

  llvm::errs() << fieldTy << "\n";

  if (opts.enableAnalysisPass) {
    pm.addPass(tensorium::mlir::createTensoriumAnalysisPass());
  }

  if (opts.enableNoOpPass) {
    pm.addPass(tensorium::mlir::createTensoriumNoOpPass());
  }

  if (mlir::failed(pm.run(moduleOp))) {
    std::cerr << "[MLIR] pass pipeline failed\n";
    return;
  }

  moduleOp.print(llvm::outs());
  llvm::outs() << "\n";
}

} // namespace tensorium_mlir
