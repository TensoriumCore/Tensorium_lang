#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

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

struct InitGridScfPass
    : public PassWrapper<InitGridScfPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitGridScfPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto initPoint =
        module.lookupSymbol<func::FuncOp>(tensorium_mlir::abi::kSymbolInitPoint);
    if (!initPoint)
      return;

    if (module.lookupSymbol<func::FuncOp>(tensorium_mlir::abi::kSymbolInitGridScf))
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
      initPoint.emitError("init-grid-scf: tensorium_init_point signature does not "
                          "match param/coord metadata");
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

    auto gridFn = func::FuncOp::create(loc, tensorium_mlir::abi::kSymbolInitGridScf,
                                       gridTy);
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
    setCommonABIAttrs(gridFn, tensorium_mlir::abi::kKindInitGridScf);
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
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value n = coordMemrefs.empty() ? b.create<memref::DimOp>(loc, alphaArg, c0)
                                   : b.create<memref::DimOp>(loc,
                                                           coordMemrefs.front(), c0);

    auto mem1Ty = MemRefType::get({1}, f64);
    auto mem9Ty = MemRefType::get({9}, f64);
    Value tmpAlpha = b.create<memref::AllocOp>(loc, mem1Ty);
    Value tmpGamma = b.create<memref::AllocOp>(loc, mem9Ty);
    Value tmpGammaU = b.create<memref::AllocOp>(loc, mem9Ty);

    auto forOp = b.create<scf::ForOp>(loc, c0, n, c1);
    b.setInsertionPointToStart(forOp.getBody());
    Value i = forOp.getInductionVar();

    SmallVector<Value> callArgs;
    callArgs.reserve(paramArgs.size() + coordMemrefs.size() + 3);
    callArgs.append(paramArgs.begin(), paramArgs.end());
    for (Value coordMemref : coordMemrefs)
      callArgs.push_back(b.create<memref::LoadOp>(loc, coordMemref, ValueRange{i}));
    callArgs.push_back(tmpAlpha);
    callArgs.push_back(tmpGamma);
    callArgs.push_back(tmpGammaU);

    b.create<func::CallOp>(loc, initPoint.getSymName(), TypeRange{}, callArgs);

    Value a0 = b.create<memref::LoadOp>(loc, tmpAlpha, ValueRange{c0});
    b.create<memref::StoreOp>(loc, a0, alphaArg, ValueRange{i});

    for (int64_t comp = 0; comp < 9; ++comp) {
      Value cComp = b.create<arith::ConstantIndexOp>(loc, comp);
      Value base = b.create<arith::MulIOp>(loc, cComp, n);
      Value flat = b.create<arith::AddIOp>(loc, base, i);

      Value g = b.create<memref::LoadOp>(loc, tmpGamma, ValueRange{cComp});
      b.create<memref::StoreOp>(loc, g, gammaArg, ValueRange{flat});

      Value gU = b.create<memref::LoadOp>(loc, tmpGammaU, ValueRange{cComp});
      b.create<memref::StoreOp>(loc, gU, gammaUArg, ValueRange{flat});
    }

    b.setInsertionPointAfter(forOp);
    b.create<memref::DeallocOp>(loc, tmpAlpha);
    b.create<memref::DeallocOp>(loc, tmpGamma);
    b.create<memref::DeallocOp>(loc, tmpGammaU);
    b.create<func::ReturnOp>(loc);

    module.push_back(gridFn);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumInitGridScfPass() {
  return std::make_unique<InitGridScfPass>();
}

} // namespace tensorium::mlir
