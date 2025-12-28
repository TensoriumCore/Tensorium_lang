#include "tensorium_mlir/Dialect/Tensorium/Transform/IndexAnalyzePass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tensorium::mlir {
namespace {

static ArrayAttr getArrayAttr(::mlir::Operation *op, ::llvm::StringRef name) {
  if (!op)
    return {};
  return op->getAttrOfType<ArrayAttr>(name);
}

static ArrayAttr getIndicesAttr(::mlir::Operation *op) {
  if (!op)
    return {};
  return op->getAttrOfType<ArrayAttr>("indices");
}

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

static ::llvm::SmallVector<::llvm::SmallVector<::llvm::StringRef, 8>, 8>
readTinIdxInsOrReconstruct(tensorium::mlir::EinsumOp op) {
  auto insAttr = getArrayAttr(op.getOperation(), "tin.idx.ins");
  if (insAttr) {
    ::llvm::SmallVector<::llvm::SmallVector<::llvm::StringRef, 8>, 8> ins;
    ins.reserve(insAttr.size());
    for (Attribute a : insAttr) {
      auto aa = dyn_cast<ArrayAttr>(a);
      if (!aa || !isStringArray(aa))
        return {};
      ins.push_back(toRefs(aa));
    }
    return ins;
  }

  ::llvm::SmallVector<::llvm::SmallVector<::llvm::StringRef, 8>, 8> ins;
  ins.reserve(op.getNumOperands());
  for (Value v : op.getOperands()) {
    ::mlir::Operation *def = v.getDefiningOp();
    if (!def) {
      ins.push_back({});
      continue;
    }
    auto idx = getIndicesAttr(def);
    if (!idx || !isStringArray(idx)) {
      ins.push_back({});
      continue;
    }
    ins.push_back(toRefs(idx));
  }
  return ins;
}

static ::llvm::SmallVector<::llvm::StringRef, 8>
readTinIdxOutOrFallback(tensorium::mlir::EinsumOp op) {
  auto outAttr = getArrayAttr(op.getOperation(), "tin.idx.out");
  if (outAttr && isStringArray(outAttr))
    return toRefs(outAttr);

  ::mlir::Operation *user = nullptr;
  for (::mlir::Operation *u : op->getUsers()) {
    user = u;
    break;
  }
  if (!user)
    return {};

  auto lhsIdx = user->getAttrOfType<ArrayAttr>("indices");
  if (!lhsIdx || !isStringArray(lhsIdx))
    return {};
  return toRefs(lhsIdx);
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

static DictionaryAttr
makeRoles(::mlir::OpBuilder &b, ::mlir::ArrayRef<::llvm::StringRef> all,
          ::mlir::ArrayRef<::llvm::StringRef> out,
          const ::llvm::MapVector<::llvm::StringRef, int64_t> &counts,
          bool &valid) {
  ::llvm::SmallDenseSet<::llvm::StringRef, 16> outSet;
  for (auto x : out)
    outSet.insert(x);

  ::llvm::SmallVector<NamedAttribute, 16> kv;
  kv.reserve(all.size());

  valid = true;

  for (auto idx : all) {
    auto it = counts.find(idx);
    int64_t c = (it == counts.end()) ? 0 : it->second;

    ::llvm::StringRef role = "invalid";

    if (outSet.contains(idx)) {
      role = "free";
      if (c != 1)
        valid = false;
    } else {
      if (c == 2)
        role = "contracted";
      else if (c > 2)
        role = "summed";
      else
        role = "invalid";
      if (c < 2)
        valid = false;
    }

    kv.push_back(b.getNamedAttr(idx, b.getStringAttr(role)));
  }

  return DictionaryAttr::get(b.getContext(), kv);
}

static ::llvm::SmallVector<::llvm::StringRef, 16> computeAllSorted(
    ::mlir::ArrayRef<::llvm::SmallVector<::llvm::StringRef, 8>> ins,
    ::mlir::ArrayRef<::llvm::StringRef> out) {
  ::llvm::SmallVector<::llvm::StringRef, 16> all;
  ::llvm::SmallDenseSet<::llvm::StringRef, 32> seen;

  for (auto x : out)
    if (seen.insert(x).second)
      all.push_back(x);

  for (auto &vec : ins)
    for (auto x : vec)
      if (seen.insert(x).second)
        all.push_back(x);

  return all;
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
    ::mlir::OpBuilder b(&this->getContext());

    m.walk([&](tensorium::mlir::EinsumOp op) {
      b.setInsertionPoint(op);

      auto ins = readTinIdxInsOrReconstruct(op);
      auto out = readTinIdxOutOrFallback(op);

      if (ins.empty() || out.empty()) {
        op.emitError("einsum index analysis failed: missing tin.idx.ins/out "
                     "and cannot reconstruct");
        op->setAttr("tin.idx.valid", b.getBoolAttr(false));
        return;
      }

      ::llvm::MapVector<::llvm::StringRef, int64_t> counts;
      for (auto &vec : ins)
        for (auto idx : vec)
          counts[idx] += 1;

      auto all = computeAllSorted(ins, out);

      bool valid = true;
      auto roles = makeRoles(b, all, out, counts, valid);

      op->setAttr("tin.idx.ins", arrayOfArraysFromVec(b, ins));
      op->setAttr("tin.idx.out", fromRefs(b, out));
      op->setAttr("tin.idx.all", fromRefs(b, all));
      op->setAttr("tin.idx.counts", makeCounts(b, counts));
      op->setAttr("tin.idx.roles", roles);
      op->setAttr("tin.idx.valid", b.getBoolAttr(valid));

      if (!valid) {
        auto specAttr = op->getAttrOfType<StringAttr>("spec");
        auto d = op.emitError("invalid einsum indices");
        if (specAttr)
          d.attachNote(op.getLoc()) << "spec: " << specAttr.getValue();
      }
    });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumIndexAnalyzePass() {
  return std::unique_ptr<::mlir::Pass>(new TensoriumIndexAnalyzePass());
}

} // namespace tensorium::mlir
