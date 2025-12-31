#include "tensorium_mlir/Dialect/Tensorium/Transform/EinsteinValidityPass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tensorium::mlir {
namespace {

struct TensoriumEinsteinValidityPass final
    : public PassWrapper<TensoriumEinsteinValidityPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensoriumEinsteinValidityPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensorium::mlir::TensoriumDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    bool ok = true;

    m.walk([&](tensorium::mlir::DtAssignOp op) {
      auto v = op->getAttrOfType<BoolAttr>("tin.idx.valid");
      if (!v || !v.getValue()) {
        op->emitError("invalid Einstein indices on dt_assign");
        ok = false;
      }
    });

    if (!ok)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumEinsteinValidityPass() {
  return std::make_unique<TensoriumEinsteinValidityPass>();
}

} // namespace tensorium::mlir
