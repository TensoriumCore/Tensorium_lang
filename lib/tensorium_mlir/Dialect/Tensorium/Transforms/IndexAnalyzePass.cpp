#include "tensorium_mlir/Dialect/Tensorium/Transform/IndexAnalyzePass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "tensorium_mlir/Semantic/Einstein.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace tensorium::mlir {
namespace {

static bool isStringArray(ArrayAttr a) {
  if (!a)
    return false;
  for (Attribute x : a)
    if (!isa<StringAttr>(x))
      return false;
  return true;
}

static ::llvm::SmallVector<::llvm::StringRef, 8> toRefs(ArrayAttr a) {
  ::llvm::SmallVector<::llvm::StringRef, 8> out;
  if (!a)
    return out;
  out.reserve(a.size());
  for (Attribute x : a) {
    auto s = dyn_cast<StringAttr>(x);
    if (!s)
      return {};
    out.push_back(s.getValue());
  }
  return out;
}

static ArrayAttr fromRefs(::mlir::OpBuilder &b,
                          ::mlir::ArrayRef<::llvm::StringRef> xs) {
  ::llvm::SmallVector<Attribute, 8> attrs;
  attrs.reserve(xs.size());
  for (auto s : xs)
    attrs.push_back(b.getStringAttr(s));
  return b.getArrayAttr(attrs);
}

static ArrayAttr arrayOfArraysFromVec(
    ::mlir::OpBuilder &b,
    ::mlir::ArrayRef<::llvm::SmallVector<::llvm::StringRef, 8>> vv) {
  ::llvm::SmallVector<Attribute, 8> out;
  out.reserve(vv.size());
  for (auto &v : vv)
    out.push_back(fromRefs(b, v));
  return b.getArrayAttr(out);
}

static DictionaryAttr
makeCounts(::mlir::OpBuilder &b,
           const ::llvm::MapVector<::llvm::StringRef, int64_t> &counts) {
  ::llvm::SmallVector<NamedAttribute, 16> kv;
  kv.reserve(counts.size());
  for (auto &it : counts)
    kv.push_back(b.getNamedAttr(it.first, b.getI64IntegerAttr(it.second)));
  return DictionaryAttr::get(b.getContext(), kv);
}


static ::llvm::SmallVector<::llvm::StringRef, 8>
readLhsIndices(tensorium::mlir::DtAssignOp op) {
  auto lhsIdx = op->getAttrOfType<ArrayAttr>("indices");
  if (!lhsIdx || !isStringArray(lhsIdx))
    return {};
  return toRefs(lhsIdx);
}


static DictionaryAttr makeRolesDictFromSemantic(
    ::mlir::OpBuilder &b,
    const tensorium::semantic::EinsteinAnalysisResult &sem) {
  ::llvm::SmallVector<NamedAttribute, 16> kv;
  kv.reserve(sem.all.size());
  for (auto idx : sem.all) {
    auto it = sem.roles.find(idx);
    auto role = (it == sem.roles.end())
                    ? llvm::StringRef("invalid")
                    : tensorium::semantic::roleToString(it->second);
    kv.push_back(b.getNamedAttr(idx, b.getStringAttr(role)));
  }
  return DictionaryAttr::get(b.getContext(), kv);
}

static llvm::SmallVector<llvm::StringRef, 32> collectIndices(Value v) {
  if (!v)
    return {};

  if (auto ref = v.getDefiningOp<tensorium::mlir::RefOp>()) {
    auto idx = ref->getAttrOfType<ArrayAttr>("indices");
    if (!idx)
      return {};
    llvm::SmallVector<llvm::StringRef, 8> out;
    for (auto a : idx)
      out.push_back(cast<StringAttr>(a).getValue());
    return out;
  }

  auto *def = v.getDefiningOp();
  if (!def)
    return {};

  auto name = def->getName().getStringRef();

  if (name == "tensorium.einsum") {
    llvm::SmallVector<llvm::StringRef, 32> out;
    for (auto in : def->getOperands()) {
      auto sub = collectIndices(in);
      out.append(sub.begin(), sub.end());
    }
    return out;
  }

  if (name == "tensorium.mul") { 
    auto a = collectIndices(def->getOperand(0));
    auto b = collectIndices(def->getOperand(1));
    a.append(b.begin(), b.end());
    return a;
  }

  if (name == "tensorium.add" ||
      name == "tensorium.sub") {
    return collectIndices(def->getOperand(0));
  }

  if (name == "tensorium.div") {
    return collectIndices(def->getOperand(0));
  }

  if (name == "tensorium.contract") {
    return collectIndices(def->getOperand(0));
  }

  if (name == "tensorium.deriv") {
    auto base = collectIndices(def->getOperand(0));
    auto idx = def->getAttrOfType<StringAttr>("index");
    if (idx)
      base.push_back(idx.getValue());
    return base;
  }

  return {};
}

struct TensoriumIndexAnalyzePass final
    : public ::mlir::PassWrapper<TensoriumIndexAnalyzePass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensoriumIndexAnalyzePass)

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<tensorium::mlir::TensoriumDialect>();
  }

  void runOnOperation() override {
    ::mlir::ModuleOp m = this->getOperation();
    llvm::errs() << "[IndexAnalyze] ran\n";
    ::mlir::OpBuilder b(&this->getContext());

    m.walk([&](tensorium::mlir::DtAssignOp op) {
      b.setInsertionPoint(op);

      auto out = readLhsIndices(op);
      auto rhs = collectIndices(op->getOperand(1));

      tensorium::semantic::EinsteinAnalyzeOptions opt;
      opt.allowSummed = false;
      opt.allowDangling = false;

      auto sem = tensorium::semantic::analyzeEinstein(out, rhs, opt);

      ::llvm::SmallVector<::llvm::SmallVector<::llvm::StringRef, 8>, 4> ins =
          sem.ins;

      auto outAttr = fromRefs(b, sem.out);
      auto allAttr = fromRefs(b, sem.all);
      auto countsAttr = makeCounts(b, sem.counts);
      auto rolesAttr = makeRolesDictFromSemantic(b, sem);

      op->setAttr("tin.idx.ins", arrayOfArraysFromVec(b, ins));
      op->setAttr("tin.idx.out", outAttr);
      op->setAttr("tin.idx.all", allAttr);
      op->setAttr("tin.idx.counts", countsAttr);
      op->setAttr("tin.idx.roles", rolesAttr);
      op->setAttr("tin.idx.valid", b.getBoolAttr(sem.valid));

      if (!sem.valid) {
        op.emitError("invalid Einstein indices in dt_assign");
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumIndexAnalyzePass() {
  return std::unique_ptr<::mlir::Pass>(new TensoriumIndexAnalyzePass());
}

} // namespace tensorium::mlir
