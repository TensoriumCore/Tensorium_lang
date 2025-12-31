#include "tensorium_mlir/Dialect/Tensorium/Transform/IndexAnalyzePass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

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

static ::llvm::SmallVector<::llvm::StringRef, 8>
readLhsIndices(tensorium::mlir::DtAssignOp op) {
  auto lhsIdx = op->getAttrOfType<ArrayAttr>("indices");
  if (!lhsIdx || !isStringArray(lhsIdx))
    return {};
  return toRefs(lhsIdx);
}

static void appendAll(::llvm::SmallVector<::llvm::StringRef, 32> &dst,
                      ::llvm::ArrayRef<::llvm::StringRef> src) {
  dst.append(src.begin(), src.end());
}

static ::llvm::SmallVector<::llvm::StringRef, 32>
contractOnce(::llvm::ArrayRef<::llvm::StringRef> in) {
  ::llvm::MapVector<::llvm::StringRef, int64_t> counts;
  for (auto s : in)
    counts[s] += 1;

  ::llvm::SmallVector<::llvm::StringRef, 32> out;
  out.reserve(in.size());
  for (auto s : in)
    if (counts[s] == 1)
      out.push_back(s);

  return out;
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

  if (name == "tensorium.mul" || name == "tensorium.add" ||
      name == "tensorium.sub" || name == "tensorium.div") {
    auto a = collectIndices(def->getOperand(0));
    auto b = collectIndices(def->getOperand(1));
    a.append(b.begin(), b.end());
    return a;
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

      // LHS indices
      auto out = readLhsIndices(op);

      // RHS indices (flat list)
      auto rhs = collectIndices(op->getOperand(1));

      // Count RHS occurrences
      ::llvm::MapVector<::llvm::StringRef, int64_t> counts;
      for (auto idx : rhs)
        counts[idx] += 1;

      // Inputs list (single RHS for now)
      ::llvm::SmallVector<::llvm::SmallVector<::llvm::StringRef, 8>, 1> ins;
      ins.emplace_back(rhs.begin(), rhs.end());

      // All indices (stable order)
      auto all = computeAllSorted(ins, out);

      // Determine roles + validity
      bool valid = true;
      auto roles = makeRoles(b, all, out, counts, valid);

      // Write attrs
      op->setAttr("tin.idx.ins", arrayOfArraysFromVec(b, ins));
      op->setAttr("tin.idx.out", fromRefs(b, out));
      op->setAttr("tin.idx.all", fromRefs(b, all));
      op->setAttr("tin.idx.counts", makeCounts(b, counts));
      op->setAttr("tin.idx.roles", roles);
      op->setAttr("tin.idx.valid", b.getBoolAttr(valid));

      if (!valid) {
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
