#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

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

template <class F>
static unsigned getUpCount(const F &f) {
  if constexpr (HasUpMember<F>) {
    return static_cast<unsigned>(f.up);
  } else if constexpr (HasGetUp<F>) {
    return static_cast<unsigned>(f.getUp());
  } else {
    return 0u;
  }
}

template <class F>
static unsigned getDownCount(const F &f) {
  if constexpr (HasDownMember<F>) {
    return static_cast<unsigned>(f.down);
  } else if constexpr (HasGetDown<F>) {
    return static_cast<unsigned>(f.getDown());
  } else {
    return 0u;
  }
}

template <class F>
static unsigned getRankFromAny(const F &f) {
  if constexpr (HasRankMember<F>) {
    return static_cast<unsigned>(f.rank);
  } else if constexpr (HasGetRank<F>) {
    return static_cast<unsigned>(f.getRank());
  } else if constexpr (HasUpMember<F> || HasGetUp<F> || HasDownMember<F> || HasGetDown<F>) {
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
  } else if constexpr (HasUpMember<F> || HasGetUp<F> || HasDownMember<F> || HasGetDown<F>) {
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

template <class M>
static auto getFieldRange(const M &module) {
  if constexpr (HasFieldsMember<M>) {
    return module.fields;
  } else if constexpr (HasGetFields<M>) {
    return module.getFields();
  } else {
    return std::vector<int>{};
  }
}

template <class F>
static std::string getFieldName(const F &f) {
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
    auto ty = tensorium::mlir::FieldType::get(&ctx,
                                              b.getF64Type(),
                                              fd.rank,
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
  b.create<::mlir::func::ReturnOp>(loc);
  moduleOp.push_back(f);

  ::mlir::PassManager pm(&ctx);

  if (opts.enableAnalysisPass) {
    pm.addPass(tensorium::mlir::createTensoriumAnalysisPass());
  }

  if (opts.enableNoOpPass) {
    pm.addPass(tensorium::mlir::createTensoriumNoOpPass());
  }

  if (::mlir::failed(pm.run(moduleOp))) {
    std::cerr << "[MLIR] pass pipeline failed\n";
    return;
  }

  moduleOp.print(llvm::outs());
  llvm::outs() << "\n";
}

} // namespace tensorium_mlir
