#include "tensorium_mlir/Dialect/Tensorium/Transform/EinsteinLoweringPass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
static ArrayAttr getIndicesAttr(::mlir::Operation *op) {
  return op->getAttrOfType<ArrayAttr>("indices");
}

static bool
collectTensorRefsAndScalars(mlir::Value v,
                            llvm::SmallVector<mlir::Operation *, 8> &tensorRefs,
                            llvm::SmallVector<mlir::Value, 8> &scalars) {

  if (!v)
    return false;

  if (auto ref = v.getDefiningOp<tensorium::mlir::RefOp>()) {
    auto idx = getIndicesAttr(ref.getOperation());
    if (idx) {
      tensorRefs.push_back(ref.getOperation());
    } else {
      scalars.push_back(ref.getResult());
    }
    return true;
  }

  if (auto cst = v.getDefiningOp<tensorium::mlir::ConstOp>()) {
    scalars.push_back(cst.getResult());
    return true;
  }

  if (auto ctr = v.getDefiningOp<tensorium::mlir::ContractOp>()) {
    return collectTensorRefsAndScalars(ctr.getIn(), tensorRefs, scalars);
  }

  if (auto mul = v.getDefiningOp<tensorium::mlir::MulOp>()) {
    return collectTensorRefsAndScalars(mul.getLhs(), tensorRefs, scalars) &&
           collectTensorRefsAndScalars(mul.getRhs(), tensorRefs, scalars);
  }

  return false;
}
namespace tensorium::mlir {
namespace {

static llvm::SmallVector<std::string, 4> arrayAttrToStrings(ArrayAttr a) {
  llvm::SmallVector<std::string, 4> out;
  if (!a)
    return out;
  out.reserve(a.size());
  for (auto attr : a) {
    auto s = dyn_cast<StringAttr>(attr);
    if (!s)
      return {};
    out.push_back(s.getValue().str());
  }
  return out;
}

static bool collectIndexedRefOperands(Value v,
                                      llvm::SmallVector<Operation *, 8> &refs) {
  if (!v)
    return false;

  if (auto ref = v.getDefiningOp<tensorium::mlir::RefOp>()) {
    auto idx = getIndicesAttr(ref.getOperation());
    if (!idx)
      return false;
    refs.push_back(ref.getOperation());
    return true;
  }

  if (auto mul = v.getDefiningOp<tensorium::mlir::MulOp>()) {
    return collectIndexedRefOperands(mul.getLhs(), refs) &&
           collectIndexedRefOperands(mul.getRhs(), refs);
  }

  return false;
}

static std::string buildSpec(const llvm::SmallVector<Operation *, 8> &refs,
                             ArrayAttr lhsIndices) {
  auto outIdx = arrayAttrToStrings(lhsIndices);

  std::string spec;
  for (size_t r = 0; r < refs.size(); ++r) {
    auto idxA = getIndicesAttr(refs[r]);
    auto idx = arrayAttrToStrings(idxA);

    if (r != 0)
      spec += ",";

    for (auto &s : idx)
      spec += s;
  }

  spec += "->";
  for (auto &s : outIdx)
    spec += s;

  return spec;
}

struct LowerContractToEinsum final
    : OpRewritePattern<tensorium::mlir::DtAssignOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensorium::mlir::DtAssignOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value destField = op.getField();
    Value rhs = op.getRhs();

    auto lhsIdxAttr = cast<ArrayAttr>(op->getAttr("indices"));

    llvm::SmallVector<Operation *, 8> tensorRefs;
    llvm::SmallVector<Value, 8> scalars;

    auto process = [&](Value expr) -> LogicalResult {
      tensorRefs.clear();
      scalars.clear();

      if (!collectTensorRefsAndScalars(expr, tensorRefs, scalars))
        return failure();

      if (tensorRefs.size() == 0)
        return failure();

      llvm::SmallVector<llvm::StringRef, 8> lhsIdx;
      for (auto a : lhsIdxAttr)
        lhsIdx.push_back(cast<StringAttr>(a).getValue());

      llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 4>, 8> ins;
      for (auto *r : tensorRefs) {
        llvm::SmallVector<llvm::StringRef, 4> v;
        for (auto a : getIndicesAttr(r))
          v.push_back(cast<StringAttr>(a).getValue());
        ins.push_back(v);
      }

      llvm::DenseMap<llvm::StringRef, unsigned> counts;
      for (auto &v : ins)
        for (auto s : v)
          counts[s]++;

      llvm::SmallVector<llvm::StringRef, 4> rhsFree;
      for (auto &it : counts)
        if (it.second == 1)
          rhsFree.push_back(it.first);

      if (rhsFree.size() != lhsIdx.size())
        return failure();

      for (auto s : rhsFree)
        if (!llvm::is_contained(lhsIdx, s))
          return failure();
      std::string spec;
      for (unsigned i = 0; i < ins.size(); ++i) {
        if (i)
          spec += ",";
        for (auto s : ins[i])
          spec += s.str();
      }
      spec += "->";
      for (auto s : lhsIdx)
        spec += s.str();

      llvm::SmallVector<Value, 8> inputs;
      for (auto *o : tensorRefs)
        inputs.push_back(o->getResult(0));

      auto specAttr =
          rewriter.getNamedAttr("spec", rewriter.getStringAttr(spec));

      auto eins = rewriter.create<tensorium::mlir::EinsumOp>(
          loc, rewriter.getF64Type(), inputs,
          llvm::ArrayRef<NamedAttribute>{specAttr});

      Value out = eins.getResult();
      for (Value s : scalars)
        out = rewriter.create<tensorium::mlir::MulOp>(loc, out, s).getResult();

      rewriter.create<tensorium::mlir::DtAssignOp>(loc, destField, out,
                                                   lhsIdxAttr);

      return success();
    };
    if (failed(process(rhs)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerContractOp final : OpRewritePattern<tensorium::mlir::ContractOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensorium::mlir::ContractOp op,
                                PatternRewriter &rewriter) const override {

    llvm::SmallVector<Operation *, 8> tensorRefs;
    llvm::SmallVector<Value, 8> scalars;

    if (!collectTensorRefsAndScalars(op.getIn(), tensorRefs, scalars))
      return failure();

    if (tensorRefs.size() < 2)
      return failure();

    std::string spec;
    for (unsigned i = 0; i < tensorRefs.size(); ++i) {
      if (i)
        spec += ",";
      auto idxAttr = getIndicesAttr(tensorRefs[i]);
      for (auto a : idxAttr)
        spec += cast<StringAttr>(a).getValue().str();
    }
    spec += "->";

    llvm::SmallVector<Value, 4> inputs;
    for (auto *o : tensorRefs)
      inputs.push_back(o->getResult(0));

    auto eins = rewriter.create<tensorium::mlir::EinsumOp>(
        op.getLoc(), rewriter.getF64Type(), inputs,
        rewriter.getNamedAttr("spec", rewriter.getStringAttr(spec)));

    Value out = eins.getResult();
    for (Value s : scalars)
      out = rewriter.create<tensorium::mlir::MulOp>(op.getLoc(), out, s);

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct TensoriumEinsteinLoweringPass final
    : PassWrapper<TensoriumEinsteinLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensoriumEinsteinLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensorium::mlir::TensoriumDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerContractToEinsum>(&getContext()); // DtAssignOp
    patterns.add<LowerContractOp>(&getContext());       // ContractOp

    GreedyRewriteConfig cfg;
    cfg.useTopDownTraversal = true;

    if (failed(applyPatternsGreedily(m, std::move(patterns), cfg)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumEinsteinLoweringPass() {
  return std::make_unique<TensoriumEinsteinLoweringPass>();
}

} // namespace tensorium::mlir
