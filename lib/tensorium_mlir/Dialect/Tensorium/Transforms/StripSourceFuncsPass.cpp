#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tensorium::mlir {
namespace {

struct StripSourceFuncsPass
    : public PassWrapper<StripSourceFuncsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripSourceFuncsPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

    const bool hasInitReplacement =
        module.lookupSymbol<func::FuncOp>("tensorium_init_point") ||
        module.lookupSymbol<func::FuncOp>("tensorium_init_grid_scf") ||
        module.lookupSymbol<func::FuncOp>("tensorium_init_grid_affine");
    const bool hasRhsReplacement =
        module.lookupSymbol<func::FuncOp>("tensorium_rhs_grid_scf") ||
        module.lookupSymbol<func::FuncOp>("tensorium_rhs_grid_affine") ||
        module.lookupSymbol<func::FuncOp>("tensorium_residual_grid_scf") ||
        module.lookupSymbol<func::FuncOp>("tensorium_residual_grid_affine");

    auto eraseIfPresent = [&](const char *name) {
      if (auto fn = module.lookupSymbol<func::FuncOp>(name))
        fn.erase();
    };

    if (hasInitReplacement)
      eraseIfPresent("tensorium_init");
    if (hasRhsReplacement)
      eraseIfPresent("tensorium_rhs");
    // tensorium_entry calls source init/rhs; once either source function is
    // replaced and erased, keeping entry can leave dangling call targets.
    if (hasInitReplacement || hasRhsReplacement)
      eraseIfPresent("tensorium_entry");
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumStripSourceFuncsPass() {
  return std::make_unique<StripSourceFuncsPass>();
}

} // namespace tensorium::mlir
