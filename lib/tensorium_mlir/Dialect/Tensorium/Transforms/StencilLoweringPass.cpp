#include "tensorium_mlir/Dialect/Tensorium/Transform/StencilLoweringPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

using namespace mlir;
using namespace tensorium::mlir;

namespace {

struct StencilPoint {
  int offset;
  double weight;
};

static std::vector<StencilPoint> getCoefficients(int order) {
  if (order == 2) {
    return {{-1, -1.0 / 2.0}, {1, 1.0 / 2.0}};
  } else if (order == 4) {
    return {
        {-2, 1.0 / 12.0}, {-1, -2.0 / 3.0}, {1, 2.0 / 3.0}, {2, -1.0 / 12.0}};
  }
  return {{-1, -0.5}, {1, 0.5}};
}

static SmallVector<int64_t> makeOffsets(unsigned spatialDim, int dim,
                                        int delta) {
  SmallVector<int64_t> off(spatialDim, 0);
  if (dim >= 0 && dim < (int)spatialDim)
    off[dim] = delta;
  return off;
}

struct LowerDerivToStencil : public OpRewritePattern<tensorium::mlir::DerivOp> {
  double dx;
  int order;

  LowerDerivToStencil(MLIRContext *ctx, double dx, int order)
      : OpRewritePattern<tensorium::mlir::DerivOp>(ctx), dx(dx), order(order) {}

  LogicalResult matchAndRewrite(tensorium::mlir::DerivOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getIn();
    auto refOp = input.getDefiningOp<tensorium::mlir::RefOp>();
    if (!refOp)
      return failure();

    auto derivIdxAttr = op->getAttrOfType<StringAttr>("index");
    if (!derivIdxAttr)
      return failure();
    StringRef idxName = derivIdxAttr.getValue();

    int dim = -1;
    if (idxName == "i" || idxName == "x")
      dim = 0;
    else if (idxName == "j" || idxName == "y")
      dim = 1;
    else if (idxName == "k" || idxName == "z")
      dim = 2;
    if (dim == -1)
      return failure();

    auto stencil = getCoefficients(order);
    unsigned spatialDim = 3;

    Location loc = op.getLoc();
    Value sum = rewriter.create<ConstOp>(loc, rewriter.getF64Type(),
                                         rewriter.getF64FloatAttr(0.0));
    for (const auto &pt : stencil) {
      auto offAttr =
          rewriter.getI64ArrayAttr(makeOffsets(spatialDim, dim, pt.offset));
      Value val = rewriter.create<RefOp>(loc, op.getType(), refOp.getSource(),
                                         refOp.getKindAttr(),
                                         refOp.getIndicesAttr(), offAttr);

      Value weight = rewriter.create<ConstOp>(
          loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(pt.weight));
      Value term = rewriter.create<MulOp>(loc, val, weight);

      sum = rewriter.create<AddOp>(loc, sum, term);
    }

    double invDx = (dx > 1e-12) ? (1.0 / dx) : 1.0;
    Value invDxVal = rewriter.create<ConstOp>(loc, rewriter.getF64Type(),
                                              rewriter.getF64FloatAttr(invDx));
    Value res = rewriter.create<MulOp>(loc, sum, invDxVal);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct StencilLoweringPass
    : public PassWrapper<StencilLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StencilLoweringPass)

  double dx;
  int order;

  StencilLoweringPass() : dx(1.0), order(2) {}
  StencilLoweringPass(double dx, int order) : dx(dx), order(order) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TensoriumDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerDerivToStencil>(&getContext(), dx, order);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace tensorium::mlir {
std::unique_ptr<Pass> createTensoriumStencilLoweringPass(double dx, int order) {
  return std::make_unique<StencilLoweringPass>(dx, order);
}
} // namespace tensorium::mlir
