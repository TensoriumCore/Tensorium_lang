#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tensorium::mlir {
namespace {

struct InitGridAffinePass
    : public PassWrapper<InitGridAffinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitGridAffinePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto initPoint = module.lookupSymbol<func::FuncOp>("tensorium_init_point");
    if (!initPoint)
      return;

    if (module.lookupSymbol<func::FuncOp>("tensorium_init_grid_affine"))
      return;

    OpBuilder b(&getContext());
    Location loc = initPoint.getLoc();
    Type f64 = b.getF64Type();

    auto dynMemTy = MemRefType::get({ShapedType::kDynamic}, f64);
    auto gridTy = b.getFunctionType(
        {f64, dynMemTy, dynMemTy, dynMemTy, dynMemTy, dynMemTy, dynMemTy}, {});

    auto gridFn = func::FuncOp::create(loc, "tensorium_init_grid_affine", gridTy);
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
    Value n = memref::DimOp::create(b, loc, rArg, c0);

    AffineMap lbMap = AffineMap::getConstantMap(0, &getContext());
    AffineExpr s0 = b.getAffineSymbolExpr(0);
    AffineMap ubMap = AffineMap::get(0, 1, s0);

    auto loop = affine::AffineForOp::create(
        b, loc, ValueRange{}, lbMap, ValueRange{n}, ubMap, 1);

    OpBuilder ib = OpBuilder::atBlockTerminator(loop.getBody());
    Value i = loop.getInductionVar();

    auto mem1Ty = MemRefType::get({1}, f64);
    auto mem9Ty = MemRefType::get({9}, f64);
    Value tmpAlpha = memref::AllocOp::create(ib, loc, mem1Ty);
    Value tmpGamma = memref::AllocOp::create(ib, loc, mem9Ty);
    Value tmpGammaU = memref::AllocOp::create(ib, loc, mem9Ty);

    Value rVal = memref::LoadOp::create(ib, loc, rArg, ValueRange{i});
    Value thetaVal = memref::LoadOp::create(ib, loc, thetaArg, ValueRange{i});
    Value phiVal = memref::LoadOp::create(ib, loc, phiArg, ValueRange{i});

    func::CallOp::create(ib, loc, initPoint.getSymName(), TypeRange{},
                         ValueRange{mArg, rVal, thetaVal, phiVal, tmpAlpha,
                                    tmpGamma, tmpGammaU});

    Value a0 = memref::LoadOp::create(ib, loc, tmpAlpha, ValueRange{c0});
    memref::StoreOp::create(ib, loc, a0, alphaArg, ValueRange{i});

    for (int64_t comp = 0; comp < 9; ++comp) {
      Value cComp = arith::ConstantIndexOp::create(ib, loc, comp);
      Value base = arith::MulIOp::create(ib, loc, cComp, n);
      Value flat = arith::AddIOp::create(ib, loc, base, i);

      Value g = memref::LoadOp::create(ib, loc, tmpGamma, ValueRange{cComp});
      memref::StoreOp::create(ib, loc, g, gammaArg, ValueRange{flat});

      Value gU = memref::LoadOp::create(ib, loc, tmpGammaU, ValueRange{cComp});
      memref::StoreOp::create(ib, loc, gU, gammaUArg, ValueRange{flat});
    }

    memref::DeallocOp::create(ib, loc, tmpAlpha);
    memref::DeallocOp::create(ib, loc, tmpGamma);
    memref::DeallocOp::create(ib, loc, tmpGammaU);

    b.setInsertionPointAfter(loop);
    func::ReturnOp::create(b, loc);

    module.push_back(gridFn);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumInitGridAffinePass() {
  return std::make_unique<InitGridAffinePass>();
}

} // namespace tensorium::mlir
