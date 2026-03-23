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

static FieldType getScalarFieldType(MLIRContext *ctx) {
  return FieldType::get(ctx, Float64Type::get(ctx), 0, 0);
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

    auto stencil = getCoefficients(order);
    auto mod = op->getParentOfType<ModuleOp>();
    auto dimAttr = mod->getAttrOfType<IntegerAttr>("tensorium.sim.dim");
    if (!dimAttr)
      return failure();
    unsigned spatialDim = (unsigned)dimAttr.getInt();
    if (spatialDim != 3)
      return failure();

    Location loc = op.getLoc();
    auto inputTy = llvm::dyn_cast<FieldType>(input.getType());
    auto resultType = op.getType();
    auto resultTy = llvm::dyn_cast<FieldType>(resultType);
    if (!inputTy || !resultTy)
      return failure();
    if (resultTy.getRank() != inputTy.getRank() + 1)
      return failure();

    auto scalarTy = getScalarFieldType(rewriter.getContext());
    auto covectorTy = FieldType::get(rewriter.getContext(),
                                     Float64Type::get(rewriter.getContext()),
                                     /*up=*/0, /*down=*/1);
    Value zero =
        rewriter.create<ConstOp>(loc, scalarTy, rewriter.getF64FloatAttr(0.0))
            .getResult();
    Value one =
        rewriter.create<ConstOp>(loc, scalarTy, rewriter.getF64FloatAttr(1.0))
            .getResult();

    double invDx = (dx > 1e-12) ? (1.0 / dx) : 1.0;
    Value invDxVal =
        rewriter.create<ConstOp>(loc, scalarTy, rewriter.getF64FloatAttr(invDx))
            .getResult();

    Value derivOut;
    bool firstAxis = true;
    for (unsigned dim = 0; dim < spatialDim; ++dim) {
      Value sum;
      bool firstTerm = true;
      for (const auto &pt : stencil) {
        auto offAttr =
            rewriter.getI64ArrayAttr(makeOffsets(spatialDim, dim, pt.offset));
        Value val =
            rewriter.create<RefOp>(loc, input.getType(), refOp.getSource(),
                          refOp.getKindAttr(), refOp.getIndicesAttr(), offAttr)
                .getResult();

        Value weight = rewriter.create<ConstOp>(loc, scalarTy,
                                       rewriter.getF64FloatAttr(pt.weight))
                           .getResult();
        Value term =
            rewriter.create<MulOp>(loc, input.getType(), val, weight).getResult();

        if (firstTerm) {
          sum = term;
          firstTerm = false;
          continue;
        }
        sum = rewriter.create<AddOp>(loc, input.getType(), sum, term).getResult();
      }

      Value axisDeriv =
          rewriter.create<MulOp>(loc, input.getType(), sum, invDxVal).getResult();

      SmallVector<Value, 3> basisComps = {zero, zero, zero};
      basisComps[dim] = one;
      Value basis =
          rewriter.create<BuildCovectorOp>(loc, covectorTy, basisComps).getOut();

      Value lifted =
          rewriter.create<MulOp>(loc, resultType, axisDeriv, basis).getResult();

      if (firstAxis) {
        derivOut = lifted;
        firstAxis = false;
        continue;
      }
      derivOut =
          rewriter.create<AddOp>(loc, resultType, derivOut, lifted).getResult();
    }

    rewriter.replaceOp(op, derivOut);
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
