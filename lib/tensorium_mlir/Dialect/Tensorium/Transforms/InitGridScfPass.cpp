#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tensorium::mlir {
namespace {

struct InitGridScfPass
    : public PassWrapper<InitGridScfPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitGridScfPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto initPoint = module.lookupSymbol<func::FuncOp>("tensorium_init_point");
    if (!initPoint)
      return;

    if (module.lookupSymbol<func::FuncOp>("tensorium_init_grid_scf"))
      return;

    OpBuilder b(&getContext());
    Location loc = initPoint.getLoc();
    Type f64 = b.getF64Type();

    auto dynMemTy = MemRefType::get({ShapedType::kDynamic}, f64);
    auto gridTy = b.getFunctionType(
        {f64, dynMemTy, dynMemTy, dynMemTy, dynMemTy, dynMemTy, dynMemTy}, {});

    auto gridFn = func::FuncOp::create(loc, "tensorium_init_grid_scf", gridTy);
    Block *entry = gridFn.addEntryBlock();
    b.setInsertionPointToEnd(entry);

    Value mArg = entry->getArgument(0);
    Value rArg = entry->getArgument(1);
    Value thetaArg = entry->getArgument(2);
    Value phiArg = entry->getArgument(3);
    Value alphaArg = entry->getArgument(4);
    Value gammaArg = entry->getArgument(5);
    Value gammaUArg = entry->getArgument(6);

    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value n = memref::DimOp::create(b, loc, rArg, c0);

    auto forOp = scf::ForOp::create(b, loc, c0, n, c1);
    b.setInsertionPointToStart(forOp.getBody());
    Value i = forOp.getInductionVar();

    auto mem1Ty = MemRefType::get({1}, f64);
    auto mem9Ty = MemRefType::get({9}, f64);
    Value tmpAlpha = memref::AllocOp::create(b, loc, mem1Ty);
    Value tmpGamma = memref::AllocOp::create(b, loc, mem9Ty);
    Value tmpGammaU = memref::AllocOp::create(b, loc, mem9Ty);

    Value rVal = memref::LoadOp::create(b, loc, rArg, ValueRange{i});
    Value thetaVal = memref::LoadOp::create(b, loc, thetaArg, ValueRange{i});
    Value phiVal = memref::LoadOp::create(b, loc, phiArg, ValueRange{i});

    func::CallOp::create(b, loc, initPoint.getSymName(), TypeRange{},
                         ValueRange{mArg, rVal, thetaVal, phiVal, tmpAlpha,
                                    tmpGamma, tmpGammaU});

    Value a0 = memref::LoadOp::create(b, loc, tmpAlpha, ValueRange{c0});
    memref::StoreOp::create(b, loc, a0, alphaArg, ValueRange{i});

    for (int64_t comp = 0; comp < 9; ++comp) {
      Value cComp = arith::ConstantIndexOp::create(b, loc, comp);
      Value base = arith::MulIOp::create(b, loc, cComp, n);
      Value flat = arith::AddIOp::create(b, loc, base, i);

      Value g = memref::LoadOp::create(b, loc, tmpGamma, ValueRange{cComp});
      memref::StoreOp::create(b, loc, g, gammaArg, ValueRange{flat});

      Value gU = memref::LoadOp::create(b, loc, tmpGammaU, ValueRange{cComp});
      memref::StoreOp::create(b, loc, gU, gammaUArg, ValueRange{flat});
    }

    memref::DeallocOp::create(b, loc, tmpAlpha);
    memref::DeallocOp::create(b, loc, tmpGamma);
    memref::DeallocOp::create(b, loc, tmpGammaU);

    b.setInsertionPointAfter(forOp);
    func::ReturnOp::create(b, loc);

    module.push_back(gridFn);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumInitGridScfPass() {
  return std::make_unique<InitGridScfPass>();
}

} // namespace tensorium::mlir
