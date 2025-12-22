#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

namespace tensorium_mlir {

void emitMLIR(const tensorium::backend::ModuleIR & /*module*/) {
  std::cerr << "[MLIR] emitMLIR called\n";

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();
  mlir::ModuleOp module = mlir::ModuleOp::create(loc);

  auto funcTy = b.getFunctionType({}, {});
  auto f = b.create<mlir::func::FuncOp>(loc, "tensorium_entry", funcTy);

  auto *entry = f.addEntryBlock();
  b.setInsertionPointToEnd(entry);
  b.create<mlir::func::ReturnOp>(loc);

  module.push_back(f);
  module.print(llvm::outs());
  llvm::outs() << "\n";
}

} // namespace tensorium_mlir
