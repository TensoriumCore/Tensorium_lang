
#include "tensorium_mlir/Dialect/Tensorium/Transform/EinsteinAnalyzeEinsumPass.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Semantic/Einstein.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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

static llvm::SmallVector<llvm::StringRef, 8> toRefs(ArrayAttr a) {
  llvm::SmallVector<llvm::StringRef, 8> out;
  if (!a)
    return out;
  out.reserve(a.size());
  for (Attribute x : a)
    out.push_back(cast<StringAttr>(x).getValue());
  return out;
}

static ArrayAttr fromRefs(OpBuilder &b, ArrayRef<llvm::StringRef> xs) {
  llvm::SmallVector<Attribute, 8> attrs;
  attrs.reserve(xs.size());
  for (auto s : xs)
    attrs.push_back(b.getStringAttr(s));
  return b.getArrayAttr(attrs);
}

static ArrayAttr fromRefs2D(
    OpBuilder &b,
    const llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 8>, 4> &vv) {
  llvm::SmallVector<Attribute, 8> out;
  out.reserve(vv.size());
  for (auto &v : vv)
    out.push_back(fromRefs(b, v));
  return b.getArrayAttr(out);
}

static DictionaryAttr
makeCountsDict(OpBuilder &b,
               const llvm::MapVector<llvm::StringRef, int64_t> &counts) {
  llvm::SmallVector<NamedAttribute, 16> kv;
  kv.reserve(counts.size());
  for (auto &it : counts)
    kv.push_back(b.getNamedAttr(it.first, b.getI64IntegerAttr(it.second)));
  return DictionaryAttr::get(b.getContext(), kv);
}

static DictionaryAttr makeRolesDict(
    OpBuilder &b, ArrayRef<llvm::StringRef> all,
    const llvm::DenseMap<llvm::StringRef, tensorium::semantic::IndexRoleKind>
        &roles) {
  llvm::SmallVector<NamedAttribute, 16> kv;
  kv.reserve(all.size());
  for (auto idx : all) {
    auto it = roles.find(idx);
    llvm::StringRef r = (it == roles.end())
                            ? llvm::StringRef("invalid")
                            : tensorium::semantic::roleToString(it->second);
    kv.push_back(b.getNamedAttr(idx, b.getStringAttr(r)));
  }
  return DictionaryAttr::get(b.getContext(), kv);
}

static bool
parseSpecToIdx(MLIRContext *ctx, StringRef spec,
               llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 8>, 4> &ins,
               llvm::SmallVector<llvm::StringRef, 8> &out) {
  auto parts = spec.split("->");
  if (parts.first.empty())
    return false;

  SmallVector<StringRef, 8> lhsTensors;
  parts.first.split(lhsTensors, ',', -1, false);

  ins.clear();
  out.clear();

  auto pushIndexChars = [&](StringRef s,
                            llvm::SmallVector<llvm::StringRef, 8> &dst) {
    s = s.trim();
    for (size_t i = 0; i < s.size(); ++i) {
      char ch = s[i];
      if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
        continue;

      StringRef one(&s.data()[i], 1);
      dst.push_back(StringAttr::get(ctx, one).getValue());
    }
  };

  for (auto t : lhsTensors) {
    llvm::SmallVector<llvm::StringRef, 8> v;
    pushIndexChars(t, v);
    ins.push_back(std::move(v));
  }

  StringRef rhs = parts.second.trim();
  pushIndexChars(rhs, out);

  return true;
}

struct TensoriumEinsteinAnalyzeEinsumPass final
    : public PassWrapper<TensoriumEinsteinAnalyzeEinsumPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TensoriumEinsteinAnalyzeEinsumPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensorium::mlir::TensoriumDialect>();
  }

  void runOnOperation() override {
    ::mlir::ModuleOp mod = getOperation();
    ::mlir::MLIRContext &ctx = getContext();
    ::mlir::OpBuilder b(&ctx);

    tensorium::semantic::EinsteinAnalyzeOptions opt;
    opt.allowSummed = false;
    opt.allowDangling = false;

    bool failed = false;

    mod.walk([&](tensorium::mlir::EinsumOp op) {
      b.setInsertionPoint(op);

      llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 8>, 4> ins;
      llvm::SmallVector<llvm::StringRef, 8> out;

      auto insAttr = op->getAttrOfType<::mlir::ArrayAttr>("tin.idx.ins");
      auto outAttr = op->getAttrOfType<::mlir::ArrayAttr>("tin.idx.out");

      if (insAttr) {
        for (::mlir::Attribute a : insAttr) {
          auto aa = ::mlir::dyn_cast<::mlir::ArrayAttr>(a);
          if (!aa || !isAllStringAttrs(aa)) {
            op.emitError(
                "tin.idx.ins must be ArrayAttr<ArrayAttr<StringAttr>>");
            failed = true;
            return;
          }
          ins.push_back(toRefs(aa));
        }
      }

      if (outAttr) {
        if (!isAllStringAttrs(outAttr)) {
          op.emitError("tin.idx.out must be ArrayAttr<StringAttr>");
          failed = true;
          return;
        }
        out = toRefs(outAttr);
      }

      if (ins.empty() || !outAttr) {
        auto specAttr = op->getAttrOfType<::mlir::StringAttr>("spec");
        if (!specAttr) {
          op.emitError("einsum missing 'spec' and missing tin.idx.ins/out");
          failed = true;
          return;
        }
        if (!parseSpecToIdx(&ctx, specAttr.getValue(), ins, out)) {
          op.emitError("failed to parse einsum spec");
          failed = true;
          return;
        }
      }

      if (ins.size() != op.getNumOperands()) {
        op.emitError("tin.idx.ins arity mismatch with einsum operands");
        failed = true;
        return;
      }

      llvm::SmallVector<llvm::StringRef, 32> rhsFlat;
      for (auto &v : ins)
        rhsFlat.append(v.begin(), v.end());

      auto sem = tensorium::semantic::analyzeEinstein(out, rhsFlat, opt);

      op->setAttr("tin.idx.ins", fromRefs2D(b, ins));
      op->setAttr("tin.idx.out", fromRefs(b, sem.out));
      op->setAttr("tin.idx.all", fromRefs(b, sem.all));
      op->setAttr("tin.idx.counts", makeCountsDict(b, sem.counts));
      op->setAttr("tin.idx.roles", makeRolesDict(b, sem.all, sem.roles));
      op->setAttr("tin.idx.valid", b.getBoolAttr(sem.valid));

      if (!sem.valid) {
        op.emitError("invalid Einstein indices on einsum");
        failed = true;
      }
    });

    if (failed)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumEinsteinAnalyzeEinsumPass() {
  return std::make_unique<TensoriumEinsteinAnalyzeEinsumPass>();
}

} // namespace tensorium::mlir
