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

static FieldType getScalarFieldType(MLIRContext *ctx) {
  return FieldType::get(ctx, Float64Type::get(ctx), 0, 0);
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

    bool hasSpatialTerms = false;
    std::vector<Value> worklist;
    worklist.push_back(op.getOperand(1));
    llvm::SmallPtrSet<Operation *, 16> visited;

    while (!worklist.empty()) {
      Value val = worklist.back();
      worklist.pop_back();

      Operation *defOp = val.getDefiningOp();
      if (!defOp)
        continue;

      if (!visited.insert(defOp).second)
        continue;

      if (auto ref = dyn_cast<RefOp>(defOp)) {
        if (ArrayAttr offsets = ref.getOffsetsAttr()) {
          for (auto attr : offsets) {
            if (cast<IntegerAttr>(attr).getInt() != 0) {
              hasSpatialTerms = true;
              worklist.clear();
              break;
            }
          }
        }
      }
      if (hasSpatialTerms)
        break;

      for (Value operand : defOp->getOperands()) {
        worklist.push_back(operand);
      }
    }

    if (!hasSpatialTerms)
      return failure();

    Location loc = op.getLoc();
    Value currentRHS = op.getOperand(1);
    Value field = op.getField();
    ArrayAttr tensorIndices = op.getIndices();

    auto mod = op->getParentOfType<ModuleOp>();

    auto dimAttr = mod->getAttrOfType<IntegerAttr>("tensorium.sim.dim");
    if (!dimAttr) {
      dimAttr = mod->getAttrOfType<IntegerAttr>("tensorium.sim.dimension");
    }
    if (!dimAttr)
      return failure();

    unsigned spatialDim = (unsigned)dimAttr.getInt();
    auto stencil = getKO4Stencil();
    double invDx = (dx > 1e-12) ? (1.0 / dx) : 1.0;
    double factor = strength * invDx * invDx * invDx * invDx * invDx;

    auto fieldTy = currentRHS.getType();
    auto scalarTy = getScalarFieldType(rewriter.getContext());
    Value dissipationTotal = rewriter.create<ConstOp>(
        loc, fieldTy, rewriter.getF64FloatAttr(0.0));

    for (unsigned dir = 0; dir < spatialDim; ++dir) {
      Value dirSum = rewriter.create<ConstOp>(loc, fieldTy,
                                              rewriter.getF64FloatAttr(0.0));

      for (const auto &pt : stencil) {
        auto offAttr =
            rewriter.getI64ArrayAttr(makeOffsets(spatialDim, dir, pt.offset));

        Value val = rewriter.create<RefOp>(loc, field.getType(), field,
                                           rewriter.getStringAttr("field"),
                                           tensorIndices, offAttr);

        Value w = rewriter.create<ConstOp>(loc, scalarTy,
                                           rewriter.getF64FloatAttr(pt.weight));
        Value term = rewriter.create<MulOp>(loc, fieldTy, val, w);
        dirSum = rewriter.create<AddOp>(loc, fieldTy, dirSum, term);
      }
      dissipationTotal =
          rewriter.create<AddOp>(loc, fieldTy, dissipationTotal, dirSum);
    }

    Value sigmaVal = rewriter.create<ConstOp>(loc, scalarTy,
                                              rewriter.getF64FloatAttr(factor));
    Value dissipationScaled =
        rewriter.create<MulOp>(loc, fieldTy, dissipationTotal, sigmaVal);
    Value newRHS = rewriter.create<AddOp>(loc, fieldTy, currentRHS,
                                          dissipationScaled);

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
