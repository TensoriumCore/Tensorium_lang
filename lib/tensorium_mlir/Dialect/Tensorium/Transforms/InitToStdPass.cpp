#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include <array>

using namespace mlir;

namespace tensorium::mlir {
namespace {

struct InitToStdPass
    : public PassWrapper<InitToStdPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitToStdPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TensoriumDialect, func::FuncDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto initFn = module.lookupSymbol<func::FuncOp>("tensorium_init");
    if (!initFn)
      return;

    if (module.lookupSymbol<func::FuncOp>("tensorium_init_point"))
      return;

    Block &srcBlock = initFn.getBody().front();

    OpBuilder b(&getContext());
    Location loc = initFn.getLoc();
    Type f64 = b.getF64Type();
    Type mem1 = MemRefType::get({1}, f64);
    Type mem9 = MemRefType::get({9}, f64);

    auto loweredTy =
        b.getFunctionType({f64, f64, f64, f64, mem1, mem9, mem9}, {});
    auto lowered =
        func::FuncOp::create(loc, "tensorium_init_point", loweredTy);
    Block *dstBlock = lowered.addEntryBlock();
    b.setInsertionPointToEnd(dstBlock);

    Value mArg = dstBlock->getArgument(0);
    Value rArg = dstBlock->getArgument(1);
    Value thetaArg = dstBlock->getArgument(2);
    Value phiArg = dstBlock->getArgument(3);
    Value alphaOut = dstBlock->getArgument(4);
    Value gammaOut = dstBlock->getArgument(5);
    Value gammaUOut = dstBlock->getArgument(6);

    DenseMap<Value, Value> scalarVals;
    DenseMap<Value, std::array<Value, 3>> covectorVals;
    DenseMap<Value, std::array<Value, 9>> covTensor2Vals;
    DenseMap<Value, std::array<Value, 9>> conTensor2Vals;

    auto requireScalar = [&](Operation *user, Value v) -> FailureOr<Value> {
      auto it = scalarVals.find(v);
      if (it == scalarVals.end()) {
        user->emitError("init-to-std: expected scalar SSA value");
        return failure();
      }
      return it->second;
    };

    auto storeScalarAt = [&](Value memref, int64_t idx, Value scalar) {
      Value index = arith::ConstantIndexOp::create(b, loc, idx);
      memref::StoreOp::create(b, loc, scalar, memref, ValueRange{index});
    };

    for (Operation &op : srcBlock.without_terminator()) {
      if (auto c = dyn_cast<ConstOp>(&op)) {
        Value v = arith::ConstantFloatOp::create(
            b, loc, llvm::cast<FloatType>(f64), c.getValue());
        scalarVals[c.getResult()] = v;
        continue;
      }

      if (auto p = dyn_cast<ParamOp>(&op)) {
        if (p.getName() == "M") {
          scalarVals[p.getResult()] = mArg;
          continue;
        }
        op.emitError("init-to-std: unsupported runtime parameter")
            << " '" << p.getName() << "'";
        signalPassFailure();
        return;
      }

      if (auto c = dyn_cast<CoordOp>(&op)) {
        if (c.getName() == "r")
          scalarVals[c.getResult()] = rArg;
        else if (c.getName() == "theta")
          scalarVals[c.getResult()] = thetaArg;
        else if (c.getName() == "phi")
          scalarVals[c.getResult()] = phiArg;
        else {
          op.emitError("init-to-std: unsupported coordinate symbol")
              << " '" << c.getName() << "'";
          signalPassFailure();
          return;
        }
        continue;
      }

      if (auto a = dyn_cast<AddOp>(&op)) {
        auto lhs = requireScalar(&op, a.getLhs());
        auto rhs = requireScalar(&op, a.getRhs());
        if (failed(lhs) || failed(rhs)) {
          signalPassFailure();
          return;
        }
        scalarVals[a.getRes()] = arith::AddFOp::create(b, loc, *lhs, *rhs);
        continue;
      }

      if (auto s = dyn_cast<SubOp>(&op)) {
        auto lhs = requireScalar(&op, s.getLhs());
        auto rhs = requireScalar(&op, s.getRhs());
        if (failed(lhs) || failed(rhs)) {
          signalPassFailure();
          return;
        }
        scalarVals[s.getRes()] = arith::SubFOp::create(b, loc, *lhs, *rhs);
        continue;
      }

      if (auto m = dyn_cast<MulOp>(&op)) {
        auto lhs = requireScalar(&op, m.getLhs());
        auto rhs = requireScalar(&op, m.getRhs());
        if (failed(lhs) || failed(rhs)) {
          signalPassFailure();
          return;
        }
        scalarVals[m.getRes()] = arith::MulFOp::create(b, loc, *lhs, *rhs);
        continue;
      }

      if (auto d = dyn_cast<DivOp>(&op)) {
        auto lhs = requireScalar(&op, d.getLhs());
        auto rhs = requireScalar(&op, d.getRhs());
        if (failed(lhs) || failed(rhs)) {
          signalPassFailure();
          return;
        }
        scalarVals[d.getRes()] = arith::DivFOp::create(b, loc, *lhs, *rhs);
        continue;
      }

      if (auto sin = dyn_cast<SinOp>(&op)) {
        auto in = requireScalar(&op, sin.getIn());
        if (failed(in)) {
          signalPassFailure();
          return;
        }
        scalarVals[sin.getOut()] = math::SinOp::create(b, loc, *in);
        continue;
      }

      if (auto sq = dyn_cast<SqrtOp>(&op)) {
        auto in = requireScalar(&op, sq.getIn());
        if (failed(in)) {
          signalPassFailure();
          return;
        }
        scalarVals[sq.getOut()] = math::SqrtOp::create(b, loc, *in);
        continue;
      }

      if (auto cov = dyn_cast<BuildCovectorOp>(&op)) {
        if (cov.getComponents().size() != 3) {
          op.emitError("init-to-std: build_covector expects 3 components");
          signalPassFailure();
          return;
        }
        std::array<Value, 3> vec{};
        for (unsigned i = 0; i < 3; ++i) {
          auto scalar = requireScalar(&op, cov.getComponents()[i]);
          if (failed(scalar)) {
            signalPassFailure();
            return;
          }
          vec[i] = *scalar;
        }
        covectorVals[cov.getOut()] = vec;
        continue;
      }

      if (auto cov2 = dyn_cast<BuildCovTensor2Op>(&op)) {
        if (cov2.getComponents().size() != 9) {
          op.emitError("init-to-std: build_cov_tensor2 expects 9 components");
          signalPassFailure();
          return;
        }
        std::array<Value, 9> mat{};
        for (unsigned i = 0; i < 9; ++i) {
          auto scalar = requireScalar(&op, cov2.getComponents()[i]);
          if (failed(scalar)) {
            signalPassFailure();
            return;
          }
          mat[i] = *scalar;
        }
        covTensor2Vals[cov2.getOut()] = mat;
        continue;
      }

      if (auto con2 = dyn_cast<BuildConTensor2Op>(&op)) {
        if (con2.getComponents().size() != 9) {
          op.emitError("init-to-std: build_con_tensor2 expects 9 components");
          signalPassFailure();
          return;
        }
        std::array<Value, 9> mat{};
        for (unsigned i = 0; i < 9; ++i) {
          auto scalar = requireScalar(&op, con2.getComponents()[i]);
          if (failed(scalar)) {
            signalPassFailure();
            return;
          }
          mat[i] = *scalar;
        }
        conTensor2Vals[con2.getOut()] = mat;
        continue;
      }

      if (auto assign = dyn_cast<AssignOp>(&op)) {
        auto fieldArg = dyn_cast<BlockArgument>(assign.getField());
        if (!fieldArg || fieldArg.getOwner() != &srcBlock) {
          op.emitError("init-to-std: assign target must be init function argument");
          signalPassFailure();
          return;
        }

        switch (fieldArg.getArgNumber()) {
        case 0: {
          auto rhs = requireScalar(&op, assign.getRhs());
          if (failed(rhs)) {
            signalPassFailure();
            return;
          }
          storeScalarAt(alphaOut, 0, *rhs);
          break;
        }
        case 1: {
          auto it = covTensor2Vals.find(assign.getRhs());
          if (it == covTensor2Vals.end()) {
            op.emitError("init-to-std: gamma assign expects build_cov_tensor2 RHS");
            signalPassFailure();
            return;
          }
          for (unsigned i = 0; i < 9; ++i)
            storeScalarAt(gammaOut, i, it->second[i]);
          break;
        }
        case 2: {
          auto it = conTensor2Vals.find(assign.getRhs());
          if (it == conTensor2Vals.end()) {
            op.emitError("init-to-std: gammaU assign expects build_con_tensor2 RHS");
            signalPassFailure();
            return;
          }
          for (unsigned i = 0; i < 9; ++i)
            storeScalarAt(gammaUOut, i, it->second[i]);
          break;
        }
        default:
          op.emitError("init-to-std: unsupported init field argument index");
          signalPassFailure();
          return;
        }
        continue;
      }

      if (isa<Metric4Op, Decompose3P1FromMetricOp, Init3P1Op>(&op)) {
        op.emitError("init-to-std: run metric lowering before init-to-std");
      } else {
        op.emitError("init-to-std: unsupported op in tensorium_init");
      }
      signalPassFailure();
      return;
    }

    func::ReturnOp::create(b, loc);
    module.push_back(lowered);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumInitToStdPass() {
  return std::make_unique<InitToStdPass>();
}

} // namespace tensorium::mlir
