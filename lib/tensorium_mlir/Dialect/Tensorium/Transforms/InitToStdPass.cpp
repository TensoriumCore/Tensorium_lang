#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <llvm/ADT/StringSet.h>

using namespace mlir;

namespace tensorium::mlir {
namespace {

enum class CoordFamily { Unknown, Cartesian, Spherical, Cylindrical };

static std::vector<std::string> defaultCoordsForFamily(CoordFamily family,
                                                       int dim) {
  const int clampedDim = std::max(1, std::min(3, dim));
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(clampedDim));
  if (family == CoordFamily::Cartesian) {
    static const char *k[] = {"x", "y", "z"};
    for (int i = 0; i < clampedDim; ++i)
      out.push_back(k[i]);
    return out;
  }
  if (family == CoordFamily::Cylindrical) {
    static const char *k[] = {"rho", "phi", "z"};
    for (int i = 0; i < clampedDim; ++i)
      out.push_back(k[i]);
    return out;
  }
  if (family == CoordFamily::Spherical) {
    static const char *k[] = {"r", "theta", "phi"};
    for (int i = 0; i < clampedDim; ++i)
      out.push_back(k[i]);
    return out;
  }
  return out;
}

static CoordFamily parseCoordFamilyAttr(StringAttr attr) {
  if (!attr)
    return CoordFamily::Unknown;
  const StringRef v = attr.getValue();
  if (v == "cartesian")
    return CoordFamily::Cartesian;
  if (v == "spherical")
    return CoordFamily::Spherical;
  if (v == "cylindrical")
    return CoordFamily::Cylindrical;
  return CoordFamily::Unknown;
}

struct InitToStdPass
    : public PassWrapper<InitToStdPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitToStdPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TensoriumDialect, func::FuncDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto initFn =
        module.lookupSymbol<func::FuncOp>(tensorium_mlir::abi::kSymbolInit);
    if (!initFn)
      return;

    if (module.lookupSymbol<func::FuncOp>(tensorium_mlir::abi::kSymbolInitPoint))
      return;

    Block &srcBlock = initFn.getBody().front();

    OpBuilder b(&getContext());
    Location loc = initFn.getLoc();
    Type f64 = b.getF64Type();
    Type mem1 = MemRefType::get({1}, f64);
    Type mem9 = MemRefType::get({9}, f64);

    llvm::StringSet<> seenParams;
    llvm::StringSet<> seenCoords;
    std::vector<std::string> paramNames;
    std::vector<std::string> coordNames;
    CoordFamily family = parseCoordFamilyAttr(
        module->getAttrOfType<StringAttr>("tensorium.sim.coords"));
    for (Operation &op : srcBlock.without_terminator()) {
      if (auto p = dyn_cast<ParamOp>(&op)) {
        if (seenParams.insert(p.getName()).second)
          paramNames.push_back(p.getName().str());
      } else if (auto c = dyn_cast<CoordOp>(&op)) {
        const std::string name = c.getName().str();
        if (seenCoords.insert(c.getName()).second)
          coordNames.push_back(name);
        if (family == CoordFamily::Unknown) {
          if (name == "x" || name == "y" || name == "z")
            family = CoordFamily::Cartesian;
          else if (name == "rho")
            family = CoordFamily::Cylindrical;
          else if (name == "r" || name == "theta" || name == "phi")
            family = CoordFamily::Spherical;
        }
      }
    }
    if (auto it = std::find(paramNames.begin(), paramNames.end(), "M");
        it != paramNames.end() && it != paramNames.begin()) {
      std::rotate(paramNames.begin(), it, it + 1);
    }
    if (family != CoordFamily::Unknown) {
      int dim = 3;
      if (auto dimAttr = module->getAttrOfType<IntegerAttr>("tensorium.sim.dim"))
        dim = static_cast<int>(dimAttr.getInt());
      coordNames = defaultCoordsForFamily(family, dim);
    }

    SmallVector<Type> loweredArgTypes;
    loweredArgTypes.reserve(paramNames.size() + coordNames.size() + 3);
    for (std::size_t i = 0; i < paramNames.size(); ++i)
      loweredArgTypes.push_back(f64);
    for (std::size_t i = 0; i < coordNames.size(); ++i)
      loweredArgTypes.push_back(f64);
    loweredArgTypes.push_back(mem1);
    loweredArgTypes.push_back(mem9);
    loweredArgTypes.push_back(mem9);

    auto loweredTy = b.getFunctionType(loweredArgTypes, {});
    auto lowered = func::FuncOp::create(loc, tensorium_mlir::abi::kSymbolInitPoint,
                                        loweredTy);
    auto makeStrArrayAttr = [&](const std::vector<std::string> &names) {
      SmallVector<StringRef> refs;
      refs.reserve(names.size());
      for (const auto &name : names)
        refs.push_back(name);
      return b.getStrArrayAttr(refs);
    };
    auto makeI64ArrayAttr = [&](const std::vector<int64_t> &values) {
      SmallVector<Attribute> attrs;
      attrs.reserve(values.size());
      for (int64_t value : values)
        attrs.push_back(b.getI64IntegerAttr(value));
      return b.getArrayAttr(attrs);
    };
    auto setCommonABIAttrs = [&](func::FuncOp fn, StringRef kind) {
      fn->setAttr(tensorium_mlir::abi::kAttrABIVersion,
                  b.getI64IntegerAttr(
                      tensorium_mlir::abi::kGeneratedKernelABIVersion));
      fn->setAttr(tensorium_mlir::abi::kAttrABIKind, b.getStringAttr(kind));
      fn->setAttr(tensorium_mlir::abi::kAttrMemoryLayout,
                  b.getStringAttr(
                      tensorium_mlir::abi::kMemLayoutSoAComponentMajor));
      fn->setAttr(tensorium_mlir::abi::kAttrMemrefABI,
                  b.getStringAttr(
                      tensorium_mlir::abi::kMemrefABI1DStridedF64));
    };

    setCommonABIAttrs(lowered, tensorium_mlir::abi::kKindInitPoint);
    lowered->setAttr("tensorium.init.param_names",
                     makeStrArrayAttr(paramNames));
    lowered->setAttr("tensorium.init.coord_names",
                     makeStrArrayAttr(coordNames));
    lowered->setAttr(tensorium_mlir::abi::kAttrParamNames,
                     makeStrArrayAttr(paramNames));
    lowered->setAttr(tensorium_mlir::abi::kAttrCoordNames,
                     makeStrArrayAttr(coordNames));
    lowered->setAttr(tensorium_mlir::abi::kAttrOutputNames,
                     makeStrArrayAttr({"alpha", "gamma", "gammaU"}));
    const int64_t firstOutputArg =
        static_cast<int64_t>(paramNames.size() + coordNames.size());
    lowered->setAttr(tensorium_mlir::abi::kAttrWriteArgIndices,
                     makeI64ArrayAttr(
                         {firstOutputArg, firstOutputArg + 1, firstOutputArg + 2}));
    Block *dstBlock = lowered.addEntryBlock();
    b.setInsertionPointToEnd(dstBlock);

    unsigned argIdx = 0;
    llvm::StringMap<Value> paramArgs;
    llvm::StringMap<Value> coordArgs;
    for (const auto &name : paramNames)
      paramArgs[name] = dstBlock->getArgument(argIdx++);
    for (const auto &name : coordNames)
      coordArgs[name] = dstBlock->getArgument(argIdx++);
    Value alphaOut = dstBlock->getArgument(argIdx++);
    Value gammaOut = dstBlock->getArgument(argIdx++);
    Value gammaUOut = dstBlock->getArgument(argIdx++);

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
      Value index = b.create<arith::ConstantIndexOp>(loc, idx);
      b.create<memref::StoreOp>(loc, scalar, memref, ValueRange{index});
    };

    for (Operation &op : srcBlock.without_terminator()) {
      if (auto c = dyn_cast<ConstOp>(&op)) {
        Value v = b.create<arith::ConstantFloatOp>(
            loc, c.getValue(), llvm::cast<FloatType>(f64));
        scalarVals[c.getResult()] = v;
        continue;
      }

      if (auto p = dyn_cast<ParamOp>(&op)) {
        auto it = paramArgs.find(p.getName());
        if (it == paramArgs.end()) {
          op.emitError("init-to-std: missing runtime parameter")
              << " '" << p.getName() << "'";
          signalPassFailure();
          return;
        }
        scalarVals[p.getResult()] = it->second;
        continue;
      }

      if (auto c = dyn_cast<CoordOp>(&op)) {
        auto it = coordArgs.find(c.getName());
        if (it == coordArgs.end()) {
          op.emitError("init-to-std: missing coordinate value")
              << " '" << c.getName() << "'";
          signalPassFailure();
          return;
        }
        scalarVals[c.getResult()] = it->second;
        continue;
      }

      if (auto a = dyn_cast<AddOp>(&op)) {
        auto lhs = requireScalar(&op, a.getLhs());
        auto rhs = requireScalar(&op, a.getRhs());
        if (failed(lhs) || failed(rhs)) {
          signalPassFailure();
          return;
        }
        scalarVals[a.getRes()] = b.create<arith::AddFOp>(loc, *lhs, *rhs);
        continue;
      }

      if (auto s = dyn_cast<SubOp>(&op)) {
        auto lhs = requireScalar(&op, s.getLhs());
        auto rhs = requireScalar(&op, s.getRhs());
        if (failed(lhs) || failed(rhs)) {
          signalPassFailure();
          return;
        }
        scalarVals[s.getRes()] = b.create<arith::SubFOp>(loc, *lhs, *rhs);
        continue;
      }

      if (auto m = dyn_cast<MulOp>(&op)) {
        auto lhs = requireScalar(&op, m.getLhs());
        auto rhs = requireScalar(&op, m.getRhs());
        if (failed(lhs) || failed(rhs)) {
          signalPassFailure();
          return;
        }
        scalarVals[m.getRes()] = b.create<arith::MulFOp>(loc, *lhs, *rhs);
        continue;
      }

      if (auto d = dyn_cast<DivOp>(&op)) {
        auto lhs = requireScalar(&op, d.getLhs());
        auto rhs = requireScalar(&op, d.getRhs());
        if (failed(lhs) || failed(rhs)) {
          signalPassFailure();
          return;
        }
        scalarVals[d.getRes()] = b.create<arith::DivFOp>(loc, *lhs, *rhs);
        continue;
      }

      if (auto sin = dyn_cast<SinOp>(&op)) {
        auto in = requireScalar(&op, sin.getIn());
        if (failed(in)) {
          signalPassFailure();
          return;
        }
        scalarVals[sin.getOut()] = b.create<math::SinOp>(loc, *in);
        continue;
      }

      if (auto sq = dyn_cast<SqrtOp>(&op)) {
        auto in = requireScalar(&op, sq.getIn());
        if (failed(in)) {
          signalPassFailure();
          return;
        }
        scalarVals[sq.getOut()] = b.create<math::SqrtOp>(loc, *in);
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

    b.create<func::ReturnOp>(loc);
    module.push_back(lowered);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumInitToStdPass() {
  return std::make_unique<InitToStdPass>();
}

} // namespace tensorium::mlir
