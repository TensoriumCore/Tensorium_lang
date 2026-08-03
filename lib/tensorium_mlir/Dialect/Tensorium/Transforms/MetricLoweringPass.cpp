#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

#include <array>
#include <llvm/ADT/SmallPtrSet.h>
#include <vector>

using namespace mlir;

namespace tensorium::mlir {
namespace {

static Value addScalar(OpBuilder &b, Location loc, FieldType scalarTy,
                       Value lhs, Value rhs) {
  return b.create<AddOp>(loc, scalarTy, lhs, rhs).getRes();
}

static Value subScalar(OpBuilder &b, Location loc, FieldType scalarTy,
                       Value lhs, Value rhs) {
  return b.create<SubOp>(loc, scalarTy, lhs, rhs).getRes();
}

static Value mulScalar(OpBuilder &b, Location loc, FieldType scalarTy,
                       Value lhs, Value rhs) {
  return b.create<MulOp>(loc, scalarTy, lhs, rhs).getRes();
}

static Value divScalar(OpBuilder &b, Location loc, FieldType scalarTy,
                       Value lhs, Value rhs) {
  return b.create<DivOp>(loc, scalarTy, lhs, rhs).getRes();
}

struct MetricLoweringPass
    : public PassWrapper<MetricLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MetricLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TensoriumDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(&getContext());
    auto scalarTy = FieldType::get(&getContext(), b.getF64Type(), 0, 0);

    std::vector<Operation *> toErase;
    llvm::SmallPtrSet<Operation *, 8> queuedForErase;
    auto queueErase = [&](Operation *op) {
      if (!op)
        return;
      if (queuedForErase.insert(op).second)
        toErase.push_back(op);
    };

    module.walk([&](Decompose3P1FromMetricOp decomp) {
      auto metric = decomp.getMetric4().template getDefiningOp<Metric4Op>();
      if (!metric)
        return;
      if (metric.getComponents().size() != 16)
        return;

      b.setInsertionPoint(decomp);
      Location loc = decomp.getLoc();
      auto comps = metric.getComponents();

      // Spatial metric gamma_ij as 3x3 block of g_{mu,nu}.
      std::array<Value, 9> gamma = {comps[5],  comps[6],  comps[7],
                                    comps[9],  comps[10], comps[11],
                                    comps[13], comps[14], comps[15]};

      // beta_i = g_{0i}.
      std::array<Value, 3> beta = {comps[1], comps[2], comps[3]};

      Value gammaVal =
          b.create<BuildCovTensor2Op>(
               loc, llvm::cast<FieldType>(decomp.getGamma().getType()),
               ValueRange(gamma))
              .getOut();
      Value betaVal =
          b.create<BuildCovectorOp>(
               loc, llvm::cast<FieldType>(decomp.getBeta().getType()),
               ValueRange(beta))
              .getOut();

      // Inverse gammaU = inverse(gamma) via adjugate/determinant.
      Value a = gamma[0], bb = gamma[1], c = gamma[2];
      Value d = gamma[3], e = gamma[4], f = gamma[5];
      Value g = gamma[6], h = gamma[7], i = gamma[8];

      Value ei = mulScalar(b, loc, scalarTy, e, i);
      Value fh = mulScalar(b, loc, scalarTy, f, h);
      Value di = mulScalar(b, loc, scalarTy, d, i);
      Value fg = mulScalar(b, loc, scalarTy, f, g);
      Value dh = mulScalar(b, loc, scalarTy, d, h);
      Value eg = mulScalar(b, loc, scalarTy, e, g);
      Value bi = mulScalar(b, loc, scalarTy, bb, i);
      Value ch = mulScalar(b, loc, scalarTy, c, h);
      Value ai = mulScalar(b, loc, scalarTy, a, i);
      Value cg = mulScalar(b, loc, scalarTy, c, g);
      Value ah = mulScalar(b, loc, scalarTy, a, h);
      Value bg = mulScalar(b, loc, scalarTy, bb, g);
      Value bf = mulScalar(b, loc, scalarTy, bb, f);
      Value ce = mulScalar(b, loc, scalarTy, c, e);
      Value af = mulScalar(b, loc, scalarTy, a, f);
      Value ae = mulScalar(b, loc, scalarTy, a, e);
      Value cd = mulScalar(b, loc, scalarTy, c, d);
      Value bd = mulScalar(b, loc, scalarTy, bb, d);

      Value c00 = subScalar(b, loc, scalarTy, ei, fh);
      Value c01 = subScalar(b, loc, scalarTy, fg, di);
      Value c02 = subScalar(b, loc, scalarTy, dh, eg);
      Value c10 = subScalar(b, loc, scalarTy, ch, bi);
      Value c11 = subScalar(b, loc, scalarTy, ai, cg);
      Value c12 = subScalar(b, loc, scalarTy, bg, ah);
      Value c20 = subScalar(b, loc, scalarTy, bf, ce);
      Value c21 = subScalar(b, loc, scalarTy, cd, af);
      Value c22 = subScalar(b, loc, scalarTy, ae, bd);

      Value aC00 = mulScalar(b, loc, scalarTy, a, c00);
      Value bC01 = mulScalar(b, loc, scalarTy, bb, c01);
      Value cC02 = mulScalar(b, loc, scalarTy, c, c02);
      Value det = addScalar(b, loc, scalarTy,
                            addScalar(b, loc, scalarTy, aC00, bC01), cC02);

      std::array<Value, 9> gammaU = {divScalar(b, loc, scalarTy, c00, det),
                                     divScalar(b, loc, scalarTy, c10, det),
                                     divScalar(b, loc, scalarTy, c20, det),
                                     divScalar(b, loc, scalarTy, c01, det),
                                     divScalar(b, loc, scalarTy, c11, det),
                                     divScalar(b, loc, scalarTy, c21, det),
                                     divScalar(b, loc, scalarTy, c02, det),
                                     divScalar(b, loc, scalarTy, c12, det),
                                     divScalar(b, loc, scalarTy, c22, det)};

      Value gammaUVal =
          b.create<BuildConTensor2Op>(
               loc, llvm::cast<FieldType>(decomp.getGammaU().getType()),
               ValueRange(gammaU))
              .getOut();

      // alpha = sqrt(beta_i * beta^i - g00), beta^i = gammaU^{ij} beta_j.
      Value betaUp0 =
          addScalar(b, loc, scalarTy,
                    addScalar(b, loc, scalarTy,
                              mulScalar(b, loc, scalarTy, gammaU[0], beta[0]),
                              mulScalar(b, loc, scalarTy, gammaU[1], beta[1])),
                    mulScalar(b, loc, scalarTy, gammaU[2], beta[2]));
      Value betaUp1 =
          addScalar(b, loc, scalarTy,
                    addScalar(b, loc, scalarTy,
                              mulScalar(b, loc, scalarTy, gammaU[3], beta[0]),
                              mulScalar(b, loc, scalarTy, gammaU[4], beta[1])),
                    mulScalar(b, loc, scalarTy, gammaU[5], beta[2]));
      Value betaUp2 =
          addScalar(b, loc, scalarTy,
                    addScalar(b, loc, scalarTy,
                              mulScalar(b, loc, scalarTy, gammaU[6], beta[0]),
                              mulScalar(b, loc, scalarTy, gammaU[7], beta[1])),
                    mulScalar(b, loc, scalarTy, gammaU[8], beta[2]));

      Value betaDot =
          addScalar(b, loc, scalarTy,
                    addScalar(b, loc, scalarTy,
                              mulScalar(b, loc, scalarTy, beta[0], betaUp0),
                              mulScalar(b, loc, scalarTy, beta[1], betaUp1)),
                    mulScalar(b, loc, scalarTy, beta[2], betaUp2));

      Value alphaSq = subScalar(b, loc, scalarTy, betaDot, comps[0]);
      Value alphaVal =
          b.create<SqrtOp>(
               loc, llvm::cast<FieldType>(decomp.getAlpha().getType()), alphaSq)
              .getOut();

      decomp.getAlpha().replaceAllUsesWith(alphaVal);
      decomp.getBeta().replaceAllUsesWith(betaVal);
      decomp.getGamma().replaceAllUsesWith(gammaVal);
      decomp.getGammaU().replaceAllUsesWith(gammaUVal);

      queueErase(decomp.getOperation());
      queueErase(metric.getOperation());
    });

    module.walk([&](Init3P1Op init3p1) {
      init3p1.getAlpha().replaceAllUsesWith(init3p1.getAlphaIn());
      init3p1.getBeta().replaceAllUsesWith(init3p1.getBetaIn());
      init3p1.getGamma().replaceAllUsesWith(init3p1.getGammaIn());
      init3p1.getGammaU().replaceAllUsesWith(init3p1.getGammaUIn());
      queueErase(init3p1.getOperation());
    });

    bool erasedAny = false;
    do {
      erasedAny = false;
      for (Operation *&op : toErase) {
        if (!op)
          continue;
        if (op->getBlock() == nullptr) {
          op = nullptr;
          continue;
        }
        if (!op->use_empty())
          continue;
        op->erase();
        op = nullptr;
        erasedAny = true;
      }
    } while (erasedAny);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumMetricLoweringPass() {
  return std::make_unique<MetricLoweringPass>();
}

} // namespace tensorium::mlir
