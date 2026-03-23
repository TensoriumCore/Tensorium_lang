#include "tensorium_mlir/Dialect/Tensorium/Transform/EinsteinCanonicalizePass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace mlir;

namespace tensorium::mlir {
namespace {

[[maybe_unused]] static bool isStringArray(ArrayAttr a) {
  if (!a) return false;
  for (Attribute x : a)
    if (!isa<StringAttr>(x))
      return false;
  return true;
}

static llvm::SmallVector<StringRef, 8> toRefs(ArrayAttr a) {
  llvm::SmallVector<StringRef, 8> out;
  if (!a) return out;
  for (Attribute x : a)
    out.push_back(cast<StringAttr>(x).getValue());
  return out;
}

static ArrayAttr fromRefs(OpBuilder &b, ArrayRef<StringRef> xs) {
  llvm::SmallVector<Attribute, 8> attrs;
  for (auto s : xs)
    attrs.push_back(b.getStringAttr(s));
  return b.getArrayAttr(attrs);
}

static ArrayAttr fromRefs2D(OpBuilder &b,
                            ArrayRef<llvm::SmallVector<StringRef, 8>> vv) {
  llvm::SmallVector<Attribute, 8> out;
  for (auto &v : vv)
    out.push_back(fromRefs(b, v));
  return b.getArrayAttr(out);
}

static DictionaryAttr
makeCounts(OpBuilder &b, const llvm::MapVector<StringRef, int64_t> &counts) {
  llvm::SmallVector<NamedAttribute, 16> kv;
  for (auto &it : counts)
    kv.push_back(b.getNamedAttr(it.first, b.getI64IntegerAttr(it.second)));
  return DictionaryAttr::get(b.getContext(), kv);
}

static DictionaryAttr
makeRolesStrict(OpBuilder &b, ArrayRef<StringRef> all, ArrayRef<StringRef> out,
                const llvm::MapVector<StringRef, int64_t> &counts,
                bool &valid) {
  llvm::SmallDenseSet<StringRef, 16> outSet(out.begin(), out.end());
  llvm::SmallVector<NamedAttribute, 16> kv;
  valid = true;

  for (auto idx : all) {
    int64_t c = counts.lookup(idx);
    StringRef role = "invalid";

    if (outSet.contains(idx)) {
      role = "free";
      if (c != 1) valid = false;
    } else {
      role = (c == 2) ? "contracted" : "invalid";
      if (c != 2) valid = false;
    }

    kv.push_back(b.getNamedAttr(idx, b.getStringAttr(role)));
  }

  return DictionaryAttr::get(b.getContext(), kv);
}

static std::string joinNoSep(ArrayRef<StringRef> xs) {
  std::string s;
  for (auto x : xs) s += x.str();
  return s;
}

static std::string joinCommaNoSep(ArrayRef<std::string> xs) {
  std::string s;
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i) s += ",";
    s += xs[i];
  }
  return s;
}

static llvm::SmallVector<StringRef, 16>
computeAllCanonical(ArrayRef<llvm::SmallVector<StringRef, 8>> ins,
                    ArrayRef<StringRef> outSorted) {
  llvm::SmallDenseSet<StringRef, 32> seen(outSorted.begin(), outSorted.end());
  llvm::SmallVector<StringRef, 16> rest;

  for (auto &v : ins)
    for (auto x : v)
      if (seen.insert(x).second)
        rest.push_back(x);

  llvm::sort(rest);
  llvm::SmallVector<StringRef, 16> all(outSorted.begin(), outSorted.end());
  all.append(rest.begin(), rest.end());
  return all;
}

static bool parseSpecToIdx(MLIRContext *ctx, StringRef spec,
                           llvm::SmallVector<llvm::SmallVector<StringRef, 8>, 8> &ins,
                           llvm::SmallVector<StringRef, 8> &out) {
  auto parts = spec.split("->");
  if (parts.second.empty()) return false;

  llvm::SmallVector<StringRef, 8> lhs;
  parts.first.split(lhs, ',', -1, false);

  ins.clear();
  out.clear();

  for (auto t : lhs) {
    llvm::SmallVector<StringRef, 8> v;
    for (char c : t.trim())
      v.push_back(StringAttr::get(ctx, StringRef(&c, 1)).getValue());
    ins.push_back(v);
  }

  for (char c : parts.second.trim())
    out.push_back(StringAttr::get(ctx, StringRef(&c, 1)).getValue());

  return true;
}

struct TensoriumEinsteinCanonicalizePass final
    : public PassWrapper<TensoriumEinsteinCanonicalizePass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TensoriumEinsteinCanonicalizePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensorium::mlir::TensoriumDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(&getContext());

    m.walk([&](tensorium::mlir::EinsumOp op) {
      b.setInsertionPoint(op);

      llvm::SmallVector<llvm::SmallVector<StringRef, 8>, 8> ins;
      llvm::SmallVector<StringRef, 8> out;

      auto insAttr = op->getAttrOfType<ArrayAttr>("tin.idx.ins");
      auto outAttr = op->getAttrOfType<ArrayAttr>("tin.idx.out");

      if (insAttr && outAttr) {
        for (auto a : insAttr)
          ins.push_back(toRefs(cast<ArrayAttr>(a)));
        out = toRefs(outAttr);
      } else {
        auto specAttr = op->getAttrOfType<StringAttr>("spec");
        if (!specAttr || !parseSpecToIdx(&getContext(), specAttr.getValue(), ins, out))
          return;
      }

      if (ins.size() != op.getNumOperands()) return;

      struct Item { std::string key; unsigned pos; };
      std::vector<Item> order;
      for (unsigned i = 0; i < ins.size(); ++i)
        order.push_back({joinNoSep(ins[i]), i});

      std::stable_sort(order.begin(), order.end(),
        [](auto &a, auto &b){ return a.key < b.key; });

      llvm::SmallVector<Value, 8> newOps;
      llvm::SmallVector<llvm::SmallVector<StringRef, 8>, 8> newIns;
      std::vector<std::string> specIn;

      for (auto &it : order) {
        newOps.push_back(op.getOperand(it.pos));
        newIns.push_back(ins[it.pos]);
        specIn.push_back(joinNoSep(ins[it.pos]));
      }

      std::string spec = joinCommaNoSep(specIn) + "->" + joinNoSep(out);

      llvm::MapVector<StringRef, int64_t> counts;
      for (auto &v : newIns)
        for (auto x : v)
          counts[x]++;

      auto all = computeAllCanonical(newIns, out);
      bool valid = true;
      auto roles = makeRolesStrict(b, all, out, counts, valid);

      auto newOp = b.create<tensorium::mlir::EinsumOp>(
          op.getLoc(), op.getResult().getType(), newOps);

      newOp->setAttr("spec", b.getStringAttr(spec));
      newOp->setAttr("tin.idx.ins", fromRefs2D(b, newIns));
      newOp->setAttr("tin.idx.out", fromRefs(b, out));
      newOp->setAttr("tin.idx.all", fromRefs(b, all));
      newOp->setAttr("tin.idx.counts", makeCounts(b, counts));
      newOp->setAttr("tin.idx.roles", roles);
      newOp->setAttr("tin.idx.valid", b.getBoolAttr(valid));

      op.replaceAllUsesWith(newOp.getResult());
      op.erase();
    });
  }
};

} 

std::unique_ptr<::mlir::Pass> createTensoriumEinsteinCanonicalizePass() {
  return std::make_unique<TensoriumEinsteinCanonicalizePass>();
}

} 
