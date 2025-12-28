#include "tensorium_mlir/Dialect/Tensorium/Transform/IndexRoleAnalysisPass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tensorium::mlir {
namespace {

static bool isAllStringAttrs(ArrayAttr a) {
  if (!a)
    return true;
  for (Attribute x : a)
    if (!isa<StringAttr>(x))
      return false;
  return true;
}

static llvm::SmallVector<StringRef, 8> toRefs(ArrayAttr a) {
  llvm::SmallVector<StringRef, 8> out;
  if (!a)
    return out;
  out.reserve(a.size());
  for (Attribute x : a)
    out.push_back(cast<StringAttr>(x).getValue());
  return out;
}

static bool contains(const llvm::SmallVector<StringRef, 8> &v, StringRef x) {
  for (auto s : v)
    if (s == x)
      return true;
  return false;
}

static void bumpCounts(const llvm::SmallVector<StringRef, 8> &idx,
                       llvm::DenseMap<StringRef, int64_t> &counts) {
  for (StringRef s : idx)
    counts[s] += 1;
}

static DictionaryAttr
makeCountsDict(MLIRContext &ctx,
               const llvm::DenseMap<StringRef, int64_t> &counts) {
  SmallVector<NamedAttribute, 16> attrs;
  attrs.reserve(counts.size());
  for (auto &kv : counts) {
    attrs.push_back(NamedAttribute(
        StringAttr::get(&ctx, kv.first),
        IntegerAttr::get(IntegerType::get(&ctx, 64), kv.second)));
  }
  return DictionaryAttr::get(&ctx, attrs);
}

static DictionaryAttr
makeRolesDict(MLIRContext &ctx,
              const llvm::DenseMap<StringRef, StringRef> &roles) {
  SmallVector<NamedAttribute, 16> attrs;
  attrs.reserve(roles.size());
  for (auto &kv : roles) {
    attrs.push_back(NamedAttribute(StringAttr::get(&ctx, kv.first),
                                   StringAttr::get(&ctx, kv.second)));
  }
  return DictionaryAttr::get(&ctx, attrs);
}

static ArrayAttr
makeAllIdxArray(MLIRContext &ctx,
                const llvm::DenseMap<StringRef, int64_t> &counts,
                const llvm::SmallVector<StringRef, 8> &outIdx) {
  llvm::DenseMap<StringRef, bool> seen;
  SmallVector<Attribute, 16> out;

  for (auto &kv : counts) {
    if (!seen[kv.first]) {
      seen[kv.first] = true;
      out.push_back(StringAttr::get(&ctx, kv.first));
    }
  }
  for (auto s : outIdx) {
    if (!seen[s]) {
      seen[s] = true;
      out.push_back(StringAttr::get(&ctx, s));
    }
  }

  return ArrayAttr::get(&ctx, out);
}

struct TensoriumIndexRoleAnalysisPass
    : public PassWrapper<TensoriumIndexRoleAnalysisPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensoriumIndexRoleAnalysisPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensorium::mlir::TensoriumDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext &ctx = getContext();

    m.walk([&](tensorium::mlir::DtAssignOp dt) {
      auto eins = dt.getRhs().getDefiningOp<tensorium::mlir::EinsumOp>();
      if (!eins)
        return;

      auto outIdxAttr = dt->getAttrOfType<ArrayAttr>("indices");
      if (!outIdxAttr)
        outIdxAttr = ArrayAttr::get(&ctx, {});
      if (!isAllStringAttrs(outIdxAttr)) {
        eins->emitError("dt_assign.indices must be ArrayAttr<StringAttr>");
        signalPassFailure();
        return;
      }
      auto outIdx = toRefs(outIdxAttr);

      llvm::SmallVector<Attribute, 8> insLists;
      llvm::DenseMap<StringRef, int64_t> counts;

      for (Value v : eins->getOperands()) {
        Operation *def = v.getDefiningOp();
        ArrayAttr inIdxAttr =
            def ? def->getAttrOfType<ArrayAttr>("indices") : ArrayAttr();
        if (!inIdxAttr)
          inIdxAttr = ArrayAttr::get(&ctx, {});
        if (!isAllStringAttrs(inIdxAttr)) {
          eins->emitError("einsum input indices must be ArrayAttr<StringAttr>");
          signalPassFailure();
          return;
        }
        insLists.push_back(inIdxAttr);
        bumpCounts(toRefs(inIdxAttr), counts);
      }

      llvm::DenseMap<StringRef, StringRef> roles;
      bool valid = true;

      auto all = makeAllIdxArray(ctx, counts, outIdx);
      for (Attribute a : all) {
        StringRef name = cast<StringAttr>(a).getValue();
        int64_t c = counts.lookup(name);
        bool inOut = contains(outIdx, name);

        if (inOut) {
          if (c == 1) {
            roles[name] = "free";
          } else {
            roles[name] = "invalid";
            valid = false;
          }
        } else {
          if (c >= 2) {
            roles[name] = "contracted";
          } else if (c == 1) {
            roles[name] = "dangling";
            valid = false;
          } else {
            roles[name] = "invalid";
            valid = false;
          }
        }
      }

      eins->setAttr("tin.idx.ins", ArrayAttr::get(&ctx, insLists));
      eins->setAttr("tin.idx.out", outIdxAttr);
      eins->setAttr("tin.idx.all", all);
      eins->setAttr("tin.idx.counts", makeCountsDict(ctx, counts));
      eins->setAttr("tin.idx.roles", makeRolesDict(ctx, roles));
      eins->setAttr("tin.idx.valid", BoolAttr::get(&ctx, valid));
    });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumIndexRoleAnalysisPass() {
  return std::make_unique<TensoriumIndexRoleAnalysisPass>();
}

} // namespace tensorium::mlir
