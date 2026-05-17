#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;

namespace tensorium::mlir {
namespace {

static std::vector<std::string> parseStringArrayAttr(ArrayAttr arr) {
  std::vector<std::string> out;
  if (!arr)
    return out;
  out.reserve(arr.size());
  for (Attribute attr : arr) {
    if (auto s = dyn_cast<StringAttr>(attr))
      out.push_back(s.getValue().str());
  }
  return out;
}

struct ConstantInitPointStores {
  std::array<std::optional<double>, 1> alpha;
  std::array<std::optional<double>, 9> gamma;
  std::array<std::optional<double>, 9> gammaU;

  bool complete() const {
    for (const auto &value : alpha)
      if (!value)
        return false;
    for (const auto &value : gamma)
      if (!value)
        return false;
    for (const auto &value : gammaU)
      if (!value)
        return false;
    return true;
  }
};

static std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto indexOp = value.getDefiningOp<arith::ConstantIndexOp>())
    return indexOp.value();
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (!value.getType().isIndex())
      return std::nullopt;
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

static std::optional<double> getConstantF64Value(Value value) {
  if (auto floatOp = value.getDefiningOp<arith::ConstantFloatOp>())
    return floatOp.value().convertToDouble();
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue()))
      return floatAttr.getValue().convertToDouble();
  }
  return std::nullopt;
}

static std::optional<ConstantInitPointStores>
collectConstantInitPointStores(func::FuncOp initPoint,
                               unsigned firstOutputArg) {
  ConstantInitPointStores out;
  for (Operation &op : initPoint.getBody().front().without_terminator()) {
    if (isa<arith::ConstantFloatOp, arith::ConstantIndexOp,
            arith::ConstantOp>(&op))
      continue;

    auto store = dyn_cast<memref::StoreOp>(&op);
    if (!store)
      return std::nullopt;

    auto outputArg = dyn_cast<BlockArgument>(store.getMemref());
    if (!outputArg || outputArg.getOwner() != &initPoint.getBody().front())
      return std::nullopt;
    const unsigned argNumber = outputArg.getArgNumber();
    if (argNumber < firstOutputArg || argNumber > firstOutputArg + 2)
      return std::nullopt;

    if (store.getIndices().size() != 1)
      return std::nullopt;
    auto component = getConstantIndexValue(store.getIndices().front());
    auto value = getConstantF64Value(store.getValue());
    if (!component || !value)
      return std::nullopt;

    if (argNumber == firstOutputArg) {
      if (*component != 0)
        return std::nullopt;
      out.alpha[0] = *value;
      continue;
    }
    if (*component < 0 || *component >= 9)
      return std::nullopt;
    if (argNumber == firstOutputArg + 1)
      out.gamma[static_cast<std::size_t>(*component)] = *value;
    else
      out.gammaU[static_cast<std::size_t>(*component)] = *value;
  }

  if (!out.complete())
    return std::nullopt;
  return out;
}

static Value soaFlatIndex(OpBuilder &b, Location loc, Value n, Value point,
                          int64_t component) {
  if (component == 0)
    return point;
  Value cComp = b.create<arith::ConstantIndexOp>(loc, component);
  Value base = b.create<arith::MulIOp>(loc, cComp, n);
  return b.create<arith::AddIOp>(loc, base, point);
}

static bool canInlineInitPointBody(func::FuncOp initPoint,
                                   unsigned firstOutputArg) {
  Block &body = initPoint.getBody().front();
  for (Operation &op : body.without_terminator()) {
    if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      auto outputArg = dyn_cast<BlockArgument>(store.getMemref());
      if (!outputArg || outputArg.getOwner() != &body)
        return false;

      const unsigned outputArgNumber = outputArg.getArgNumber();
      if (outputArgNumber < firstOutputArg ||
          outputArgNumber > firstOutputArg + 2)
        return false;
      if (store.getIndices().size() != 1)
        return false;

      auto component = getConstantIndexValue(store.getIndices().front());
      if (!component)
        return false;
      if (outputArgNumber == firstOutputArg) {
        if (*component != 0)
          return false;
        continue;
      }
      if (*component < 0 || *component >= 9)
        return false;
      continue;
    }

    if (isa<memref::AllocOp, memref::DeallocOp, func::CallOp>(&op))
      return false;
    if (op.getName().getDialectNamespace() == "tensorium")
      return false;
    if (op.getNumRegions() != 0)
      return false;
    for (Value operand : op.getOperands()) {
      auto operandArg = dyn_cast<BlockArgument>(operand);
      if (operandArg && operandArg.getOwner() == &body &&
          operandArg.getArgNumber() >= firstOutputArg)
        return false;
    }
  }
  return true;
}

static bool tryInlineInitPointBody(OpBuilder &ib, Location loc,
                                   func::FuncOp initPoint,
                                   unsigned firstOutputArg,
                                   ArrayRef<Value> paramArgs,
                                   ArrayRef<Value> coordMemrefs, Value point,
                                   Value n, Value alphaArg, Value gammaArg,
                                   Value gammaUArg) {
  if (!canInlineInitPointBody(initPoint, firstOutputArg))
    return false;

  IRMapping mapper;
  Block &body = initPoint.getBody().front();
  unsigned argIdx = 0;
  for (Value param : paramArgs)
    mapper.map(body.getArgument(argIdx++), param);
  for (Value coordMemref : coordMemrefs) {
    Value coord = ib.create<memref::LoadOp>(loc, coordMemref, ValueRange{point});
    mapper.map(body.getArgument(argIdx++), coord);
  }

  for (Operation &op : body.without_terminator()) {
    if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      auto outputArg = dyn_cast<BlockArgument>(store.getMemref());
      if (!outputArg || outputArg.getOwner() != &body)
        return false;

      const unsigned outputArgNumber = outputArg.getArgNumber();
      if (outputArgNumber < firstOutputArg ||
          outputArgNumber > firstOutputArg + 2)
        return false;
      if (store.getIndices().size() != 1)
        return false;

      auto component = getConstantIndexValue(store.getIndices().front());
      if (!component)
        return false;

      Value stored = mapper.lookupOrDefault(store.getValue());
      if (outputArgNumber == firstOutputArg) {
        if (*component != 0)
          return false;
        ib.create<memref::StoreOp>(loc, stored, alphaArg, ValueRange{point});
        continue;
      }

      if (*component < 0 || *component >= 9)
        return false;
      Value flat = soaFlatIndex(ib, loc, n, point, *component);
      Value dst = outputArgNumber == firstOutputArg + 1 ? gammaArg : gammaUArg;
      ib.create<memref::StoreOp>(loc, stored, dst, ValueRange{flat});
      continue;
    }

    if (isa<memref::AllocOp, memref::DeallocOp, func::CallOp>(&op))
      return false;
    if (op.getName().getDialectNamespace() == "tensorium")
      return false;

    ib.clone(op, mapper);
  }
  return true;
}

struct InitGridAffinePass
    : public PassWrapper<InitGridAffinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitGridAffinePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    arith::ArithDialect, math::MathDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto initPoint =
        module.lookupSymbol<func::FuncOp>(tensorium_mlir::abi::kSymbolInitPoint);
    if (!initPoint)
      return;

    if (module.lookupSymbol<func::FuncOp>(tensorium_mlir::abi::kSymbolInitGridAffine))
      return;

    OpBuilder b(&getContext());
    Location loc = initPoint.getLoc();
    Type f64 = b.getF64Type();

    auto dynMemTy = MemRefType::get({ShapedType::kDynamic}, f64);

    std::vector<std::string> paramNames =
        parseStringArrayAttr(initPoint->getAttrOfType<ArrayAttr>(
            "tensorium.init.param_names"));
    std::vector<std::string> coordNames =
        parseStringArrayAttr(initPoint->getAttrOfType<ArrayAttr>(
            "tensorium.init.coord_names"));

    const unsigned expectedInitArgs = static_cast<unsigned>(
        paramNames.size() + coordNames.size() + 3u);
    if (initPoint.getNumArguments() != expectedInitArgs) {
      initPoint.emitError(
          "init-grid-affine: tensorium_init_point signature does not match "
          "param/coord metadata");
      signalPassFailure();
      return;
    }

    SmallVector<Type> gridArgTypes;
    gridArgTypes.reserve(paramNames.size() + coordNames.size() + 3);
    for (std::size_t i = 0; i < paramNames.size(); ++i)
      gridArgTypes.push_back(f64);
    for (std::size_t i = 0; i < coordNames.size(); ++i)
      gridArgTypes.push_back(dynMemTy);
    gridArgTypes.push_back(dynMemTy); // alpha
    gridArgTypes.push_back(dynMemTy); // gamma
    gridArgTypes.push_back(dynMemTy); // gammaU

    auto gridTy = b.getFunctionType(gridArgTypes, {});

    auto gridFn =
        func::FuncOp::create(loc, tensorium_mlir::abi::kSymbolInitGridAffine, gridTy);
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
    setCommonABIAttrs(gridFn, tensorium_mlir::abi::kKindInitGridAffine);
    gridFn->setAttr(tensorium_mlir::abi::kAttrParamNames,
                    makeStrArrayAttr(paramNames));
    gridFn->setAttr(tensorium_mlir::abi::kAttrCoordNames,
                    makeStrArrayAttr(coordNames));
    gridFn->setAttr(tensorium_mlir::abi::kAttrOutputNames,
                    makeStrArrayAttr({"alpha", "gamma", "gammaU"}));
    const int64_t firstOutputArg =
        static_cast<int64_t>(paramNames.size() + coordNames.size());
    gridFn->setAttr(tensorium_mlir::abi::kAttrWriteArgIndices,
                    makeI64ArrayAttr(
                        {firstOutputArg, firstOutputArg + 1, firstOutputArg + 2}));
    Block *entry = gridFn.addEntryBlock();
    b.setInsertionPointToEnd(entry);

    unsigned gridArgIdx = 0;
    SmallVector<Value> paramArgs;
    SmallVector<Value> coordMemrefs;
    paramArgs.reserve(paramNames.size());
    coordMemrefs.reserve(coordNames.size());
    for (std::size_t i = 0; i < paramNames.size(); ++i)
      paramArgs.push_back(entry->getArgument(gridArgIdx++));
    for (std::size_t i = 0; i < coordNames.size(); ++i)
      coordMemrefs.push_back(entry->getArgument(gridArgIdx++));
    Value alphaArg = entry->getArgument(gridArgIdx++);
    Value gammaArg = entry->getArgument(gridArgIdx++);
    Value gammaUArg = entry->getArgument(gridArgIdx++);

    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value n = coordMemrefs.empty() ? b.create<memref::DimOp>(loc, alphaArg, c0)
                                   : b.create<memref::DimOp>(loc,
                                                           coordMemrefs.front(), c0);
    const unsigned initPointFirstOutputArg = gridArgIdx - 3;
    auto constantStores =
        collectConstantInitPointStores(initPoint, initPointFirstOutputArg);
    const bool canInlineInitPoint =
        !constantStores &&
        canInlineInitPointBody(initPoint, initPointFirstOutputArg);
    const bool needsScratchBuffers = !constantStores && !canInlineInitPoint;

    Value tmpAlpha;
    Value tmpGamma;
    Value tmpGammaU;
    if (needsScratchBuffers) {
      auto mem1Ty = MemRefType::get({1}, f64);
      auto mem9Ty = MemRefType::get({9}, f64);
      tmpAlpha = b.create<memref::AllocOp>(loc, mem1Ty);
      tmpGamma = b.create<memref::AllocOp>(loc, mem9Ty);
      tmpGammaU = b.create<memref::AllocOp>(loc, mem9Ty);
    }

    AffineMap lbMap = AffineMap::getConstantMap(0, &getContext());
    AffineExpr s0 = b.getAffineSymbolExpr(0);
    AffineMap ubMap = AffineMap::get(0, 1, s0);

    auto loop = b.create<affine::AffineForOp>(loc, ValueRange{}, lbMap,
                                              ValueRange{n}, ubMap, 1);

    OpBuilder ib = OpBuilder::atBlockTerminator(loop.getBody());
    Value i = loop.getInductionVar();

    bool emittedDirect = false;
    if (constantStores) {
      auto f64Attr = [&](double value) {
        return ib.getF64FloatAttr(value);
      };
      auto f64Const = [&](double value) {
        return ib.create<arith::ConstantOp>(loc, f64, f64Attr(value));
      };
      ib.create<memref::StoreOp>(loc, f64Const(*constantStores->alpha[0]),
                                 alphaArg, ValueRange{i});

      for (int64_t comp = 0; comp < 9; ++comp) {
        Value cComp = ib.create<arith::ConstantIndexOp>(loc, comp);
        Value base = ib.create<arith::MulIOp>(loc, cComp, n);
        Value flat = ib.create<arith::AddIOp>(loc, base, i);

        ib.create<memref::StoreOp>(
            loc,
            f64Const(*constantStores->gamma[static_cast<std::size_t>(comp)]),
            gammaArg, ValueRange{flat});
        ib.create<memref::StoreOp>(
            loc,
            f64Const(*constantStores->gammaU[static_cast<std::size_t>(comp)]),
            gammaUArg, ValueRange{flat});
      }
      emittedDirect = true;
    } else if (canInlineInitPoint) {
      emittedDirect =
          tryInlineInitPointBody(ib, loc, initPoint, initPointFirstOutputArg,
                                 paramArgs, coordMemrefs, i, n, alphaArg,
                                 gammaArg, gammaUArg);
    }

    if (!emittedDirect && !needsScratchBuffers) {
      initPoint.emitError("init-grid-affine: failed to inline init_point body");
      signalPassFailure();
      return;
    }

    if (!emittedDirect) {
      SmallVector<Value> callArgs;
      callArgs.reserve(paramArgs.size() + coordMemrefs.size() + 3);
      callArgs.append(paramArgs.begin(), paramArgs.end());
      for (Value coordMemref : coordMemrefs)
        callArgs.push_back(
            ib.create<memref::LoadOp>(loc, coordMemref, ValueRange{i}));
      callArgs.push_back(tmpAlpha);
      callArgs.push_back(tmpGamma);
      callArgs.push_back(tmpGammaU);

      ib.create<func::CallOp>(loc, initPoint.getSymName(), TypeRange{},
                              callArgs);

      Value a0 = ib.create<memref::LoadOp>(loc, tmpAlpha, ValueRange{c0});
      ib.create<memref::StoreOp>(loc, a0, alphaArg, ValueRange{i});

      for (int64_t comp = 0; comp < 9; ++comp) {
        Value cComp = ib.create<arith::ConstantIndexOp>(loc, comp);
        Value base = ib.create<arith::MulIOp>(loc, cComp, n);
        Value flat = ib.create<arith::AddIOp>(loc, base, i);

        Value g = ib.create<memref::LoadOp>(loc, tmpGamma, ValueRange{cComp});
        ib.create<memref::StoreOp>(loc, g, gammaArg, ValueRange{flat});

        Value gU = ib.create<memref::LoadOp>(loc, tmpGammaU, ValueRange{cComp});
        ib.create<memref::StoreOp>(loc, gU, gammaUArg, ValueRange{flat});
      }
    }

    b.setInsertionPointAfter(loop);
    if (needsScratchBuffers) {
      b.create<memref::DeallocOp>(loc, tmpAlpha);
      b.create<memref::DeallocOp>(loc, tmpGamma);
      b.create<memref::DeallocOp>(loc, tmpGammaU);
    }
    b.create<func::ReturnOp>(loc);

    module.push_back(gridFn);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumInitGridAffinePass() {
  return std::make_unique<InitGridAffinePass>();
}

} // namespace tensorium::mlir
