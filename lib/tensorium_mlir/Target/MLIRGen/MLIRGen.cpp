#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorium_mlir {

namespace {

struct FieldDesc {
  std::string name;
  unsigned rank = 0;
  tensorium::mlir::Variance variance = tensorium::mlir::Variance::Scalar;
};

template <class M>
concept HasFieldsMember = requires(const M &m) { m.fields; };

template <class M>
concept HasGetFields = requires(const M &m) { m.getFields(); };

template <class F>
concept HasNameMember = requires(const F &f) { f.name; };

template <class F>
concept HasGetName = requires(const F &f) { f.getName(); };

template <class F>
concept HasRankMember = requires(const F &f) { f.rank; };

template <class F>
concept HasGetRank = requires(const F &f) { f.getRank(); };

template <class F>
concept HasUpMember = requires(const F &f) { f.up; };

template <class F>
concept HasDownMember = requires(const F &f) { f.down; };

template <class F>
concept HasGetUp = requires(const F &f) { f.getUp(); };

template <class F>
concept HasGetDown = requires(const F &f) { f.getDown(); };

template <class F>
concept HasIndicesMember = requires(const F &f) { f.indices; };

template <class F>
concept HasGetIndices = requires(const F &f) { f.getIndices(); };

template <class F>
concept HasVarianceMember = requires(const F &f) { f.variance; };

template <class F>
concept HasGetVariance = requires(const F &f) { f.getVariance(); };

static ::mlir::ArrayAttr
makeIndexArrayAttr(::mlir::OpBuilder &b, const std::vector<std::string> &idx) {
  llvm::SmallVector<::mlir::Attribute, 4> names;
  names.reserve(idx.size());
  for (auto &s : idx)
    names.push_back(b.getStringAttr(s));
  return b.getArrayAttr(names);
}

static bool startsWith(const std::string &s, const char *prefix) {
  size_t n = std::char_traits<char>::length(prefix);
  return s.size() >= n && s.compare(0, n, prefix) == 0;
}
template <class F> static unsigned getUpCount(const F &f) {
  if constexpr (HasUpMember<F>) {
    return static_cast<unsigned>(f.up);
  } else if constexpr (HasGetUp<F>) {
    return static_cast<unsigned>(f.getUp());
  } else {
    return 0u;
  }
}

template <class F> static unsigned getDownCount(const F &f) {
  if constexpr (HasDownMember<F>) {
    return static_cast<unsigned>(f.down);
  } else if constexpr (HasGetDown<F>) {
    return static_cast<unsigned>(f.getDown());
  } else {
    return 0u;
  }
}

template <class F> static unsigned getRankFromAny(const F &f) {
  if constexpr (HasRankMember<F>) {
    return static_cast<unsigned>(f.rank);
  } else if constexpr (HasGetRank<F>) {
    return static_cast<unsigned>(f.getRank());
  } else if constexpr (HasUpMember<F> || HasGetUp<F> || HasDownMember<F> ||
                       HasGetDown<F>) {
    return getUpCount(f) + getDownCount(f);
  } else if constexpr (HasIndicesMember<F>) {
    return static_cast<unsigned>(f.indices.size());
  } else if constexpr (HasGetIndices<F>) {
    return static_cast<unsigned>(f.getIndices().size());
  } else {
    return 0u;
  }
}

template <class F>
static tensorium::mlir::Variance getVarianceFromAny(const F &f) {
  if constexpr (HasVarianceMember<F>) {
    return f.variance;
  } else if constexpr (HasGetVariance<F>) {
    return f.getVariance();
  } else if constexpr (HasUpMember<F> || HasGetUp<F> || HasDownMember<F> ||
                       HasGetDown<F>) {
    unsigned up = getUpCount(f);
    unsigned down = getDownCount(f);
    if (up == 0 && down == 0)
      return tensorium::mlir::Variance::Scalar;
    if (up > 0 && down == 0)
      return tensorium::mlir::Variance::Contravariant;
    if (up == 0 && down > 0)
      return tensorium::mlir::Variance::Covariant;
    return tensorium::mlir::Variance::Mixed;
  } else {
    return tensorium::mlir::Variance::Scalar;
  }
}

template <class M> static auto getFieldRange(const M &module) {
  if constexpr (HasFieldsMember<M>) {
    return module.fields;
  } else if constexpr (HasGetFields<M>) {
    return module.getFields();
  } else {
    return std::vector<int>{};
  }
}

template <class F> static std::string getFieldName(const F &f) {
  if constexpr (HasNameMember<F>) {
    return std::string(f.name);
  } else if constexpr (HasGetName<F>) {
    return std::string(f.getName());
  } else {
    return "field";
  }
}

template <class M>
static std::vector<FieldDesc> extractFields(const M &module) {
  std::vector<FieldDesc> out;

  auto range = getFieldRange(module);
  using ElemT = std::remove_reference_t<decltype(*std::begin(range))>;

  if constexpr (std::is_same_v<ElemT, int>) {
    return out;
  } else {
    for (const auto &f : range) {
      FieldDesc d;
      d.name = getFieldName(f);
      d.rank = getRankFromAny(f);
      d.variance = getVarianceFromAny(f);
      out.push_back(std::move(d));
    }
    return out;
  }
}

static const char *varianceTag(tensorium::mlir::Variance v) {
  switch (v) {
  case tensorium::mlir::Variance::Scalar:
    return "scalar";
  case tensorium::mlir::Variance::Covariant:
    return "cov";
  case tensorium::mlir::Variance::Contravariant:
    return "con";
  case tensorium::mlir::Variance::Mixed:
    return "mixed";
  }
  return "scalar";
}

static ::mlir::Value
emitExpr(::mlir::OpBuilder &b, ::mlir::Location loc, ::mlir::Type f64,
         const tensorium::backend::ExprIR *e,
         const llvm::DenseMap<llvm::StringRef, ::mlir::Value> &fieldArg) {
  using namespace tensorium::backend;

  if (!e)
    return {};

  switch (e->kind) {
  case ExprIR::Kind::Number: {
    auto *n = static_cast<const NumberIR *>(e);
    auto c = b.create<tensorium::mlir::ConstOp>(loc, f64,
                                                b.getF64FloatAttr(n->value));
    return c.getResult();
  }

  case ExprIR::Kind::Var: {
    auto *v = static_cast<const VarIR *>(e);

    auto it = fieldArg.find(v->name);
    if (it == fieldArg.end()) {
      llvm::errs() << "[MLIRGen][error] unknown var: " << v->name << "\n";
      return {};
    }

    auto kindAttr = b.getStringAttr("field");
    auto r = b.create<tensorium::mlir::RefOp>(loc, f64, it->second, kindAttr);

    if (!v->tensorIndexNames.empty()) {
      llvm::SmallVector<::mlir::Attribute, 4> idxAttr;
      idxAttr.reserve(v->tensorIndexNames.size());
      for (auto &s : v->tensorIndexNames)
        idxAttr.push_back(b.getStringAttr(s));
      r->setAttr("indices", b.getArrayAttr(idxAttr));
    }

    return r.getResult();
  }

  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const BinaryIR *>(e);
    auto L = emitExpr(b, loc, f64, bin->lhs.get(), fieldArg);
    auto R = emitExpr(b, loc, f64, bin->rhs.get(), fieldArg);
    if (!L || !R)
      return {};

    if (bin->op == "+") {
      auto op = b.create<tensorium::mlir::AddOp>(loc, f64, L, R);
      return op.getResult();
    }
    if (bin->op == "*") {
      auto op = b.create<tensorium::mlir::MulOp>(loc, f64, L, R);
      return op.getResult();
    }

    if (bin->op == "-") {
      if (auto *Ln = dynamic_cast<const NumberIR *>(bin->lhs.get())) {
        if (Ln->value == 0.0) {
          if (auto *Rn = dynamic_cast<const NumberIR *>(bin->rhs.get())) {
            auto c = b.create<tensorium::mlir::ConstOp>(
                loc, f64, b.getF64FloatAttr(-Rn->value));
            return c.getResult();
          }
        }
      }
      auto op = b.create<tensorium::mlir::SubOp>(loc, f64, L, R);
      return op.getResult();
    }
    if (bin->op == "/") {
      llvm::errs() << "[MLIRGen][error] '/' not supported yet\n";
      return {};
    }

    llvm::errs() << "[MLIRGen][error] unknown binary op: " << bin->op << "\n";
    return {};
  }

  case ExprIR::Kind::Call: {
    auto *c = static_cast<const CallIR *>(e);

    if (startsWith(c->callee, "d_") && c->callee.size() == 3) {
      std::string idxName(1, c->callee[2]);

      ::mlir::Value arg0 =
          c->args.empty() ? ::mlir::Value()
                          : emitExpr(b, loc, f64, c->args[0].get(), fieldArg);
      if (!arg0)
        return {};

      auto deriv = b.create<tensorium::mlir::DerivOp>(loc, f64, arg0);
      deriv->setAttr("index", b.getStringAttr(idxName));
      return deriv.getResult();
    }

    if (c->callee == "contract") {
      if (c->args.size() != 1) {
        llvm::errs()
            << "[MLIRGen][error] contract() expects exactly one argument\n";
        return {};
      }

      ::mlir::Value arg0 = emitExpr(b, loc, f64, c->args[0].get(), fieldArg);
      if (!arg0)
        return {};

      auto ctr = b.create<tensorium::mlir::ContractOp>(loc, f64, arg0);
      return ctr.getResult();
    }

    llvm::errs() << "[MLIRGen][error] unknown call: " << c->callee << "\n";
    return {};
  }

    return {};
  }
}
} // namespace

void emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts) {
  std::cerr << "[MLIR] emitMLIR called\n";

  ::mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<::mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<tensorium::mlir::TensoriumDialect>();

  ::mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();

  ::mlir::ModuleOp moduleOp = ::mlir::ModuleOp::create(loc);

  llvm::SmallVector<::mlir::Type, 8> argTypes;

  const auto fields = extractFields(module);
  for (const auto &fd : fields) {
    auto ty = tensorium::mlir::FieldType::get(&ctx, b.getF64Type(), fd.rank,
                                              fd.variance);
    argTypes.push_back(ty);
    llvm::errs() << "[MLIRGen] field '" << fd.name << "' -> " << ty
                 << " (rank=" << fd.rank << ", var=" << varianceTag(fd.variance)
                 << ")\n";
  }

  auto funcTy = b.getFunctionType(argTypes, {});
  auto f = b.create<::mlir::func::FuncOp>(loc, "tensorium_entry", funcTy);

  auto *entry = f.addEntryBlock();
  b.setInsertionPointToEnd(entry);

  llvm::DenseMap<llvm::StringRef, ::mlir::Value> fieldArg;
  fieldArg.reserve(fields.size());

  for (unsigned i = 0; i < fields.size(); ++i) {
    fieldArg[fields[i].name] = entry->getArgument(i);
  }
  auto f64 = b.getF64Type();
  for (const auto &evo : module.evolutions) {
    for (const auto &eq : evo.equations) {

      auto it = fieldArg.find(eq.fieldName);
      if (it == fieldArg.end()) {
        llvm::errs() << "[MLIRGen][error] unknown lhs field: " << eq.fieldName
                     << "\n";
        continue;
      }

      auto rhsV = emitExpr(b, loc, f64, eq.rhs.get(), fieldArg);
      if (!rhsV) {
        llvm::errs() << "[MLIRGen][warn] skipping equation dt " << eq.fieldName
                     << " (rhs not lowered)\n";
        continue;
      }

      auto idxAttr = makeIndexArrayAttr(b, eq.indices);

      b.create<tensorium::mlir::DtAssignOp>(loc, it->second, rhsV, idxAttr);
    }
  }
  b.create<::mlir::func::ReturnOp>(loc);
  moduleOp.push_back(f);

  ::mlir::PassManager pm(&ctx);

  if (opts.enableAnalysisPass) {
    pm.addPass(tensorium::mlir::createTensoriumAnalysisPass());
  }

  if (opts.enableNoOpPass) {
    pm.addPass(tensorium::mlir::createTensoriumNoOpPass());
  }

  if (opts.enableAnalysisPass)
    pm.addPass(tensorium::mlir::createTensoriumAnalysisPass());

  if (opts.enableNoOpPass)
    pm.addPass(tensorium::mlir::createTensoriumNoOpPass());

  if (opts.enableEinsteinLoweringPass)
    pm.addPass(tensorium::mlir::createTensoriumEinsteinLoweringPass());

  if (opts.enableIndexRoleAnalysisPass)
    pm.addPass(tensorium::mlir::createTensoriumIndexRoleAnalysisPass());

  if (opts.enableEinsteinValidityPass)
    pm.addPass(tensorium::mlir::createTensoriumEinsteinValidityPass());

  pm.addPass(::mlir::createCanonicalizerPass());
  pm.addPass(::mlir::createCSEPass());

  if (::mlir::failed(pm.run(moduleOp))) {
    std::cerr << "[MLIR] pass pipeline failed\n";
    return;
  }

  moduleOp.print(llvm::outs());
  llvm::outs() << "\n";
}

} // namespace tensorium_mlir
