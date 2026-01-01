#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

using namespace mlir;
using namespace tensorium::mlir;

namespace {

struct StencilPoint {
  int offset;
  double weight;
};

static std::vector<StencilPoint> getKO4Stencil() {
  return {{-3, 1.0 / 64.0},  {-2, -6.0 / 64.0}, {-1, 15.0 / 64.0},
          {0, -20.0 / 64.0}, {1, 15.0 / 64.0},  {2, -6.0 / 64.0},
          {3, 1.0 / 64.0}};
}

static SmallVector<int64_t> makeOffsets(unsigned dim, int dir, int delta) {
  SmallVector<int64_t> off(dim, 0);
  if (dir >= 0 && dir < (int)dim)
    off[dir] = delta;
  return off;
}

struct AddDissipation : public OpRewritePattern<DtAssignOp> {
  double strength;
  double dx;

  AddDissipation(MLIRContext *ctx, double strength, double dx)
      : OpRewritePattern<DtAssignOp>(ctx), strength(strength), dx(dx) {}

  LogicalResult matchAndRewrite(DtAssignOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("dissipation_added"))
      return failure();

    Location loc = op.getLoc();

    Value currentRHS = op.getOperand(1);
    Value field = op.getField();

    ArrayAttr tensorIndices = op.getIndices();

    unsigned spatialDim = 3;
    auto stencil = getKO4Stencil();
    double factor = strength * (1.0 / dx);

    Value dissipationTotal = rewriter.create<ConstOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.0));

    for (unsigned dir = 0; dir < spatialDim; ++dir) {
      Value dirSum = rewriter.create<ConstOp>(loc, rewriter.getF64Type(),
                                              rewriter.getF64FloatAttr(0.0));

      for (const auto &pt : stencil) {
        auto offAttr =
            rewriter.getI64ArrayAttr(makeOffsets(spatialDim, dir, pt.offset));

        Value val = rewriter.create<RefOp>(loc, rewriter.getF64Type(), field,
                                           rewriter.getStringAttr("field"),
                                           tensorIndices, offAttr);

        Value w = rewriter.create<ConstOp>(loc, rewriter.getF64Type(),
                                           rewriter.getF64FloatAttr(pt.weight));
        Value term = rewriter.create<MulOp>(loc, val, w);
        dirSum = rewriter.create<AddOp>(loc, dirSum, term);
      }
      dissipationTotal = rewriter.create<AddOp>(loc, dissipationTotal, dirSum);
    }

    Value sigmaVal = rewriter.create<ConstOp>(loc, rewriter.getF64Type(),
                                              rewriter.getF64FloatAttr(factor));
    Value dissipationScaled =
        rewriter.create<MulOp>(loc, dissipationTotal, sigmaVal);
    Value newRHS = rewriter.create<AddOp>(loc, currentRHS, dissipationScaled);

    rewriter.modifyOpInPlace(op, [&] {
      op.setOperand(1, newRHS);
      op->setAttr("dissipation_added", rewriter.getUnitAttr());
    });

    return success();
  }
};

struct DissipationPass
    : public PassWrapper<DissipationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DissipationPass)

  double strength;
  double dx;

  DissipationPass() : strength(0.1), dx(1.0) {}
  DissipationPass(double strength, double dx) : strength(strength), dx(dx) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TensoriumDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddDissipation>(&getContext(), strength, dx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      // signalPassFailure();
    }
  }
};

} // namespace

namespace tensorium::mlir {
std::unique_ptr<Pass> createTensoriumDissipationPass(double strength,
                                                     double dx) {
  return std::make_unique<DissipationPass>(strength, dx);
}
} // namespace tensorium::mlir
