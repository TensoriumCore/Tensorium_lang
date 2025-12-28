#include "tensorium_mlir/Dialect/Tensorium/Transform/EinsteinLoweringPass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace tensorium::mlir {
namespace {

static ArrayAttr getIndicesAttr(Operation *op) {
  if (!op)
    return {};
  return op->getAttrOfType<ArrayAttr>("indices");
}

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

static void collectRefOperands(Value v,
                               llvm::SmallVector<Operation *, 8> &refs) {
  if (!v)
    return;

  if (auto ref = v.getDefiningOp<tensorium::mlir::RefOp>()) {
    refs.push_back(ref.getOperation());
    return;
  }

  if (auto mul = v.getDefiningOp<tensorium::mlir::MulOp>()) {
    collectRefOperands(mul.getLhs(), refs);
    collectRefOperands(mul.getRhs(), refs);
  }
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
    auto ctr = op.getRhs().getDefiningOp<tensorium::mlir::ContractOp>();
    if (!ctr)
      return failure();

    llvm::SmallVector<Operation *, 8> refs;
    collectRefOperands(ctr.getIn(), refs);
    if (refs.empty())
      return failure();

    auto lhsIdx = op->getAttrOfType<ArrayAttr>("indices");
    if (!lhsIdx)
      lhsIdx = rewriter.getArrayAttr({});

    std::string spec = buildSpec(refs, lhsIdx);

    llvm::SmallVector<Value, 8> ins;
    ins.reserve(refs.size());
    for (auto *r : refs) {
      if (r->getNumResults() != 1)
        return failure();
      ins.push_back(r->getResult(0));
    }

    OperationState st(op.getLoc(), "tensorium.einsum");
    st.addOperands(ins);
    st.addTypes(rewriter.getF64Type());
    st.addAttribute("spec", rewriter.getStringAttr(spec));
    Operation *einsOp = rewriter.create(st);
    Value eins = einsOp->getResult(0);

    Value lhsField = op->getOperand(0);

    rewriter.create<tensorium::mlir::DtAssignOp>(op.getLoc(), lhsField, eins,
                                                 lhsIdx);

    rewriter.eraseOp(op);

    if (ctr->use_empty())
      rewriter.eraseOp(ctr);
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
    patterns.add<LowerContractToEinsum>(&getContext());

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
