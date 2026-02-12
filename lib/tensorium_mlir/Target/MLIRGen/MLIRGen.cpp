#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include <algorithm>
#include <set>
#include <stdexcept>
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

namespace tensorium_mlir {

namespace {

struct FieldDesc {
  std::string name;
  unsigned up = 0;
  unsigned down = 0;
};

[[noreturn]] static void emitUnsupportedExprError(mlir::Location loc,
                                                 const std::string &detail);
static mlir::Value
emitExpr(mlir::OpBuilder &b, mlir::Location loc,
         const tensorium::backend::ExprIR *e,
         const llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg,
         llvm::StringMap<mlir::Value> *localTemps);

static mlir::ArrayAttr makeIndexArrayAttr(mlir::OpBuilder &b,
                                          const std::vector<std::string> &idx) {
  llvm::SmallVector<mlir::Attribute, 4> names;
  for (const auto &s : idx)
    names.push_back(b.getStringAttr(s));
  return b.getArrayAttr(names);
}

static tensorium::mlir::FieldType
asFieldType(mlir::OpBuilder &b, const tensorium::ir::TensorType &desc) {
  auto *ctx = b.getContext();
  auto elementType = b.getF64Type();
  unsigned up = desc.up < 0 ? 0u : static_cast<unsigned>(desc.up);
  unsigned down = desc.down < 0 ? 0u : static_cast<unsigned>(desc.down);
  return tensorium::mlir::FieldType::get(ctx, elementType, up, down);
}

static bool startsWith(const std::string &s, const char *prefix) {
  size_t n = std::char_traits<char>::length(prefix);
  return s.size() >= n && s.compare(0, n, prefix) == 0;
}

static std::vector<FieldDesc>
extractFields(const tensorium::backend::ModuleIR &module) {
  std::vector<FieldDesc> out;
  for (const auto &f : module.fields) {
    FieldDesc d;
    d.name = f.name;
    d.up = static_cast<unsigned>(std::max(0, f.tensorType.up));
    d.down = static_cast<unsigned>(std::max(0, f.tensorType.down));
    out.push_back(std::move(d));
  }
  return out;
}

static mlir::ArrayAttr makeStringArrayAttr(mlir::OpBuilder &b,
                                           const std::vector<std::string> &v) {
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  attrs.reserve(v.size());
  for (const auto &s : v)
    attrs.push_back(b.getStringAttr(s));
  return b.getArrayAttr(attrs);
}

static bool isCoordinateName(std::string_view name) {
  return name == "t" || name == "x" || name == "y" || name == "z" ||
         name == "r" || name == "rho" || name == "theta" || name == "phi";
}

static bool initExprStructuralEqual(const tensorium::backend::InitExprIR *lhs,
                                    const tensorium::backend::InitExprIR *rhs) {
  using namespace tensorium::backend;
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs || lhs->kind != rhs->kind)
    return false;

  switch (lhs->kind) {
  case InitExprIR::Kind::Number: {
    auto *l = static_cast<const InitNumberIR *>(lhs);
    auto *r = static_cast<const InitNumberIR *>(rhs);
    return l->value == r->value;
  }
  case InitExprIR::Kind::Symbol: {
    auto *l = static_cast<const InitSymbolIR *>(lhs);
    auto *r = static_cast<const InitSymbolIR *>(rhs);
    return l->name == r->name;
  }
  case InitExprIR::Kind::Binary: {
    auto *l = static_cast<const InitBinaryIR *>(lhs);
    auto *r = static_cast<const InitBinaryIR *>(rhs);
    return l->op == r->op &&
           initExprStructuralEqual(l->lhs.get(), r->lhs.get()) &&
           initExprStructuralEqual(l->rhs.get(), r->rhs.get());
  }
  case InitExprIR::Kind::Call: {
    auto *l = static_cast<const InitCallIR *>(lhs);
    auto *r = static_cast<const InitCallIR *>(rhs);
    if (l->callee != r->callee || l->args.size() != r->args.size())
      return false;
    for (size_t i = 0; i < l->args.size(); ++i) {
      if (!initExprStructuralEqual(l->args[i].get(), r->args[i].get()))
        return false;
    }
    return true;
  }
  }
  return false;
}

static bool metric4HasSymmetricComponents(
    const tensorium::backend::InitialDataIR &init) {
  if (!init.hasMetric4 || init.metric4.components.size() != 16)
    return false;

  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      const size_t a = static_cast<size_t>(i * 4 + j);
      const size_t b = static_cast<size_t>(j * 4 + i);
      if (!initExprStructuralEqual(init.metric4.components[a].get(),
                                   init.metric4.components[b].get())) {
        return false;
      }
    }
  }
  return true;
}

static bool initExprIsZero(const tensorium::backend::InitExprIR *expr) {
  using namespace tensorium::backend;
  if (!expr || expr->kind != InitExprIR::Kind::Number)
    return false;
  auto *num = static_cast<const InitNumberIR *>(expr);
  return num->value == 0.0;
}

static bool metric4HasZeroTimeSpaceTerms(
    const tensorium::backend::InitialDataIR &init) {
  if (!init.hasMetric4 || init.metric4.components.size() != 16)
    return false;

  // 4D index order is assumed as (t, x1, x2, x3): require g_ti == g_it == 0
  // for beta=0-only decomposition support in decompose3p1_from_metric.
  constexpr int kPairs[6][2] = {
      {0, 1}, {1, 0}, {0, 2}, {2, 0}, {0, 3}, {3, 0}};
  for (const auto &pair : kPairs) {
    const size_t flat = static_cast<size_t>(pair[0] * 4 + pair[1]);
    if (!initExprIsZero(init.metric4.components[flat].get()))
      return false;
  }
  return true;
}

static void collectExprFieldNames(const tensorium::backend::ExprIR *expr,
                                  llvm::StringSet<> &out) {
  using namespace tensorium::backend;
  if (!expr)
    return;
  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return;
  case ExprIR::Kind::Var: {
    auto *v = static_cast<const VarIR *>(expr);
    if (v->vkind == VarKind::Field)
      out.insert(v->name);
    return;
  }
  case ExprIR::Kind::Binary: {
    auto *b = static_cast<const BinaryIR *>(expr);
    collectExprFieldNames(b->lhs.get(), out);
    collectExprFieldNames(b->rhs.get(), out);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *c = static_cast<const CallIR *>(expr);
    for (const auto &arg : c->args)
      collectExprFieldNames(arg.get(), out);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *p = static_cast<const TensorProductIR *>(expr);
    collectExprFieldNames(p->lhs.get(), out);
    collectExprFieldNames(p->rhs.get(), out);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *c = static_cast<const ContractionIR *>(expr);
    collectExprFieldNames(c->in.get(), out);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *r = static_cast<const IndexRenameIR *>(expr);
    collectExprFieldNames(r->in.get(), out);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *p = static_cast<const IndexPermuteIR *>(expr);
    collectExprFieldNames(p->in.get(), out);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *t = static_cast<const TraceIR *>(expr);
    collectExprFieldNames(t->in.get(), out);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *d = static_cast<const PartialDerivativeIR *>(expr);
    collectExprFieldNames(d->in.get(), out);
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *g = static_cast<const GradientIR *>(expr);
    collectExprFieldNames(g->in.get(), out);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *d = static_cast<const CovariantDerivativeIR *>(expr);
    collectExprFieldNames(d->in.get(), out);
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *d = static_cast<const DivergenceIR *>(expr);
    collectExprFieldNames(d->in.get(), out);
    return;
  }
  }
}

static void collectInitExprFieldNames(
    const tensorium::backend::InitExprIR *expr,
    const llvm::StringSet<> &knownFieldNames, llvm::StringSet<> &out) {
  using namespace tensorium::backend;
  if (!expr)
    return;
  switch (expr->kind) {
  case InitExprIR::Kind::Number:
    return;
  case InitExprIR::Kind::Symbol: {
    auto *s = static_cast<const InitSymbolIR *>(expr);
    if (knownFieldNames.contains(s->name))
      out.insert(s->name);
    return;
  }
  case InitExprIR::Kind::Binary: {
    auto *b = static_cast<const InitBinaryIR *>(expr);
    collectInitExprFieldNames(b->lhs.get(), knownFieldNames, out);
    collectInitExprFieldNames(b->rhs.get(), knownFieldNames, out);
    return;
  }
  case InitExprIR::Kind::Call: {
    auto *c = static_cast<const InitCallIR *>(expr);
    for (const auto &arg : c->args)
      collectInitExprFieldNames(arg.get(), knownFieldNames, out);
    return;
  }
  }
}

static bool exprUsesFieldName(const tensorium::backend::ExprIR *expr,
                              llvm::StringRef fieldName) {
  using namespace tensorium::backend;
  if (!expr)
    return false;
  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return false;
  case ExprIR::Kind::Var: {
    auto *v = static_cast<const VarIR *>(expr);
    return v->vkind == VarKind::Field && v->name == fieldName;
  }
  case ExprIR::Kind::Binary: {
    auto *b = static_cast<const BinaryIR *>(expr);
    return exprUsesFieldName(b->lhs.get(), fieldName) ||
           exprUsesFieldName(b->rhs.get(), fieldName);
  }
  case ExprIR::Kind::Call: {
    auto *c = static_cast<const CallIR *>(expr);
    for (const auto &arg : c->args) {
      if (exprUsesFieldName(arg.get(), fieldName))
        return true;
    }
    return false;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *p = static_cast<const TensorProductIR *>(expr);
    return exprUsesFieldName(p->lhs.get(), fieldName) ||
           exprUsesFieldName(p->rhs.get(), fieldName);
  }
  case ExprIR::Kind::Contraction: {
    auto *c = static_cast<const ContractionIR *>(expr);
    return exprUsesFieldName(c->in.get(), fieldName);
  }
  case ExprIR::Kind::IndexRename: {
    auto *r = static_cast<const IndexRenameIR *>(expr);
    return exprUsesFieldName(r->in.get(), fieldName);
  }
  case ExprIR::Kind::IndexPermute: {
    auto *p = static_cast<const IndexPermuteIR *>(expr);
    return exprUsesFieldName(p->in.get(), fieldName);
  }
  case ExprIR::Kind::Trace: {
    auto *t = static_cast<const TraceIR *>(expr);
    return exprUsesFieldName(t->in.get(), fieldName);
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *d = static_cast<const PartialDerivativeIR *>(expr);
    return exprUsesFieldName(d->in.get(), fieldName);
  }
  case ExprIR::Kind::Gradient: {
    auto *g = static_cast<const GradientIR *>(expr);
    return exprUsesFieldName(g->in.get(), fieldName);
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *d = static_cast<const CovariantDerivativeIR *>(expr);
    return exprUsesFieldName(d->in.get(), fieldName);
  }
  case ExprIR::Kind::Divergence: {
    auto *d = static_cast<const DivergenceIR *>(expr);
    return exprUsesFieldName(d->in.get(), fieldName);
  }
  }
  return false;
}

static bool moduleUsesFieldName(const tensorium::backend::ModuleIR &module,
                                llvm::StringRef fieldName) {
  for (const auto &evo : module.evolutions) {
    for (const auto &tmp : evo.temporaries) {
      if (exprUsesFieldName(tmp.rhs.get(), fieldName))
        return true;
    }
    for (const auto &eq : evo.equations) {
      if (eq.fieldName == fieldName)
        continue;
      if (exprUsesFieldName(eq.rhs.get(), fieldName))
        return true;
    }
  }
  return false;
}

static std::vector<unsigned>
collectInitArgIndices(const tensorium::backend::ModuleIR &module,
                      const std::vector<FieldDesc> &fields) {
  llvm::StringMap<unsigned> indexByName;
  llvm::StringSet<> knownFieldNames;
  for (unsigned i = 0; i < fields.size(); ++i) {
    indexByName[fields[i].name] = i;
    knownFieldNames.insert(fields[i].name);
  }

  llvm::StringSet<> needed;
  auto markIfField = [&](const std::string &name) {
    if (indexByName.contains(name))
      needed.insert(name);
  };

  if (module.initialData) {
    const auto &init = *module.initialData;
    if (init.split3p1.enabled) {
      if (init.split3p1.hasAlpha && !init.split3p1.alphaField.empty())
        markIfField(init.split3p1.alphaField);
      if (init.split3p1.hasBeta && !init.split3p1.betaField.empty())
        markIfField(init.split3p1.betaField);
      if (init.split3p1.hasGamma && !init.split3p1.gammaField.empty())
        markIfField(init.split3p1.gammaField);
      if (init.split3p1.hasGammaU && !init.split3p1.gammaUField.empty())
        markIfField(init.split3p1.gammaUField);
    }

    if (init.hasMetric4) {
      for (const auto &comp : init.metric4.components)
        collectInitExprFieldNames(comp.get(), knownFieldNames, needed);
    } else if (init.hasDecomposed) {
      if (init.decomposed.alphaExpr)
        collectInitExprFieldNames(init.decomposed.alphaExpr.get(),
                                  knownFieldNames, needed);
      for (const auto &expr : init.decomposed.betaExpr)
        collectInitExprFieldNames(expr.get(), knownFieldNames, needed);
      for (const auto &expr : init.decomposed.gammaExpr)
        collectInitExprFieldNames(expr.get(), knownFieldNames, needed);
      for (const auto &expr : init.decomposed.gammaUExpr)
        collectInitExprFieldNames(expr.get(), knownFieldNames, needed);
    }
  }

  std::vector<unsigned> out;
  for (unsigned i = 0; i < fields.size(); ++i) {
    if (needed.contains(fields[i].name))
      out.push_back(i);
  }
  return out;
}

static std::vector<unsigned>
collectRhsArgIndices(const tensorium::backend::ModuleIR &module,
                     const std::vector<FieldDesc> &fields) {
  llvm::StringMap<unsigned> indexByName;
  for (unsigned i = 0; i < fields.size(); ++i)
    indexByName[fields[i].name] = i;

  llvm::StringSet<> needed;
  for (const auto &evo : module.evolutions) {
    for (const auto &tmp : evo.temporaries)
      collectExprFieldNames(tmp.rhs.get(), needed);
    for (const auto &eq : evo.equations) {
      needed.insert(eq.fieldName);
      collectExprFieldNames(eq.rhs.get(), needed);
    }
  }

  std::vector<unsigned> out;
  for (unsigned i = 0; i < fields.size(); ++i) {
    if (needed.contains(fields[i].name))
      out.push_back(i);
  }
  return out;
}

static mlir::Value emitInitExpr(
    mlir::OpBuilder &b, mlir::Location loc,
    const tensorium::backend::InitExprIR *expr,
    llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg,
    llvm::StringMap<mlir::Value> &paramValues,
    llvm::StringMap<mlir::Value> &coordValues) {
  using namespace tensorium::backend;
  if (!expr)
    emitUnsupportedExprError(loc, "null initial_data expression");

  auto *ctx = b.getContext();
  auto scalarTy = tensorium::mlir::FieldType::get(ctx, b.getF64Type(), 0, 0);

  switch (expr->kind) {
  case InitExprIR::Kind::Number: {
    auto *n = static_cast<const InitNumberIR *>(expr);
    return tensorium::mlir::ConstOp::create(b, loc, scalarTy,
                                            b.getF64FloatAttr(n->value))
        .getResult();
  }
  case InitExprIR::Kind::Symbol: {
    auto *s = static_cast<const InitSymbolIR *>(expr);
    if (auto it = fieldArg.find(s->name); it != fieldArg.end()) {
      return tensorium::mlir::RefOp::create(
                 b, loc, scalarTy, it->second, b.getStringAttr("field"),
                 mlir::ArrayAttr(), mlir::ArrayAttr())
          .getResult();
    }

    if (isCoordinateName(s->name)) {
      if (auto it = coordValues.find(s->name); it != coordValues.end())
        return it->second;
      auto coord = tensorium::mlir::CoordOp::create(b, loc, scalarTy,
                                                    b.getStringAttr(s->name));
      coordValues[s->name] = coord.getResult();
      return coord.getResult();
    }

    if (auto it = paramValues.find(s->name); it != paramValues.end())
      return it->second;
    auto param = tensorium::mlir::ParamOp::create(b, loc, scalarTy,
                                                  b.getStringAttr(s->name));
    paramValues[s->name] = param.getResult();
    return param.getResult();
  }
  case InitExprIR::Kind::Binary: {
    auto *bin = static_cast<const InitBinaryIR *>(expr);
    auto L = emitInitExpr(b, loc, bin->lhs.get(), fieldArg, paramValues,
                          coordValues);
    auto R = emitInitExpr(b, loc, bin->rhs.get(), fieldArg, paramValues,
                          coordValues);

    if (bin->op == '+')
      return tensorium::mlir::AddOp::create(b, loc, scalarTy, L, R).getResult();
    if (bin->op == '-')
      return tensorium::mlir::SubOp::create(b, loc, scalarTy, L, R).getResult();
    if (bin->op == '*')
      return tensorium::mlir::MulOp::create(b, loc, scalarTy, L, R).getResult();
    if (bin->op == '/')
      return tensorium::mlir::DivOp::create(b, loc, scalarTy, L, R).getResult();
    if (bin->op == '^') {
      auto *rhsNum = dynamic_cast<const InitNumberIR *>(bin->rhs.get());
      if (!rhsNum)
        emitUnsupportedExprError(
            loc, "initial_data exponentiation expects numeric literal exponent");
      int exp = static_cast<int>(rhsNum->value);
      if (rhsNum->value != static_cast<double>(exp) || exp < 0 || exp > 4)
        emitUnsupportedExprError(
            loc, "initial_data exponentiation supports integer exponents 0..4");
      if (exp == 0) {
        return tensorium::mlir::ConstOp::create(b, loc, scalarTy,
                                                b.getF64FloatAttr(1.0))
            .getResult();
      }
      mlir::Value acc = L;
      for (int i = 1; i < exp; ++i)
        acc = tensorium::mlir::MulOp::create(b, loc, scalarTy, acc, L)
                  .getResult();
      return acc;
    }
    emitUnsupportedExprError(
        loc, "unsupported initial_data binary operator");
  }
  case InitExprIR::Kind::Call: {
    auto *call = static_cast<const InitCallIR *>(expr);
    if (call->callee == "sin") {
      if (call->args.size() != 1)
        emitUnsupportedExprError(loc, "sin() expects 1 argument");
      auto arg = emitInitExpr(b, loc, call->args[0].get(), fieldArg, paramValues,
                              coordValues);
      return tensorium::mlir::SinOp::create(b, loc, scalarTy, arg).getResult();
    }
    if (call->callee == "sqrt") {
      if (call->args.size() != 1)
        emitUnsupportedExprError(loc, "sqrt() expects 1 argument");
      auto arg = emitInitExpr(b, loc, call->args[0].get(), fieldArg, paramValues,
                              coordValues);
      return tensorium::mlir::SqrtOp::create(b, loc, scalarTy, arg).getResult();
    }
    emitUnsupportedExprError(
        loc, "unsupported initial_data function '" + call->callee + "'");
  }
  }

  emitUnsupportedExprError(loc, "unsupported initial_data expression");
}

static void emitInitialDataOps(mlir::OpBuilder &b, mlir::Location loc,
                               const tensorium::backend::ModuleIR &module,
                               llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg) {
  if (!module.initialData)
    return;
  const auto &init = *module.initialData;

  if (!init.hasMetric4 && !init.hasDecomposed) {
    emitUnsupportedExprError(
        loc, "initial_data is present but no metric4 or alpha/beta/gamma data "
             "was provided");
  }

  llvm::StringMap<mlir::Value> paramValues;
  llvm::StringMap<mlir::Value> coordValues;

  auto *ctx = b.getContext();
  auto elemTy = b.getF64Type();
  auto scalarTy = tensorium::mlir::FieldType::get(ctx, elemTy, 0, 0);
  auto metricTy = tensorium::mlir::FieldType::get(ctx, elemTy, 0, 2);
  auto betaTy = tensorium::mlir::FieldType::get(ctx, elemTy, 0, 1);
  auto gammaTy = tensorium::mlir::FieldType::get(ctx, elemTy, 0, 2);
  auto gammaUTy = tensorium::mlir::FieldType::get(ctx, elemTy, 2, 0);

  tensorium::mlir::Init3P1Op init3p1;

  if (init.hasMetric4) {
    if (init.metric4.components.size() != 16) {
      emitUnsupportedExprError(
          loc, "metric4 initial_data must provide 16 component expressions");
    }
    if (!metric4HasSymmetricComponents(init)) {
      emitUnsupportedExprError(
          loc, "decompose3p1_from_metric requires symmetric metric components");
    }
    if (!metric4HasZeroTimeSpaceTerms(init)) {
      emitUnsupportedExprError(
          loc, "decompose3p1_from_metric requires g_ti = 0 (beta unsupported)");
    }

    llvm::SmallVector<mlir::Value, 16> metricComps;
    metricComps.reserve(16);
    for (const auto &comp : init.metric4.components) {
      metricComps.push_back(emitInitExpr(b, loc, comp.get(), fieldArg, paramValues,
                                         coordValues));
    }

    auto metric = tensorium::mlir::Metric4Op::create(
        b, loc, metricTy, metricComps, b.getStringAttr(init.metric4.name),
        b.getStringAttr(init.metric4.coordSystem),
        makeStringArrayAttr(b, init.metric4.indices),
        b.getBoolAttr(init.metric4.enforceSymmetry));

    auto decomp = tensorium::mlir::Decompose3P1FromMetricOp::create(
        b, loc, mlir::TypeRange{scalarTy, betaTy, gammaTy, gammaUTy},
        metric.getMetric());

    init3p1 = tensorium::mlir::Init3P1Op::create(
        b, loc, mlir::TypeRange{scalarTy, betaTy, gammaTy, gammaUTy},
        decomp.getAlpha(), decomp.getBeta(), decomp.getGamma(), decomp.getGammaU());
  } else if (init.hasDecomposed) {
    if (!init.decomposed.alphaExpr || init.decomposed.betaExpr.size() != 3 ||
        init.decomposed.gammaExpr.size() != 9 || init.decomposed.gammaUExpr.size() != 9) {
      emitUnsupportedExprError(
          loc, "decomposed initial_data requires alpha, beta[3], gamma[3x3], gammaU[3x3]");
    }

    auto alpha = emitInitExpr(b, loc, init.decomposed.alphaExpr.get(), fieldArg,
                              paramValues, coordValues);

    llvm::SmallVector<mlir::Value, 3> betaComponents;
    betaComponents.reserve(3);
    for (const auto &expr : init.decomposed.betaExpr) {
      betaComponents.push_back(emitInitExpr(b, loc, expr.get(), fieldArg,
                                            paramValues, coordValues));
    }
    auto beta = tensorium::mlir::BuildCovectorOp::create(b, loc, betaTy,
                                                          betaComponents)
                    .getResult();

    llvm::SmallVector<mlir::Value, 9> gammaComponents;
    gammaComponents.reserve(9);
    for (const auto &expr : init.decomposed.gammaExpr) {
      gammaComponents.push_back(emitInitExpr(b, loc, expr.get(), fieldArg,
                                             paramValues, coordValues));
    }
    auto gamma = tensorium::mlir::BuildCovTensor2Op::create(b, loc, gammaTy,
                                                             gammaComponents)
                     .getResult();

    llvm::SmallVector<mlir::Value, 9> gammaUComponents;
    gammaUComponents.reserve(9);
    for (const auto &expr : init.decomposed.gammaUExpr) {
      gammaUComponents.push_back(emitInitExpr(b, loc, expr.get(), fieldArg,
                                              paramValues, coordValues));
    }
    auto gammaU = tensorium::mlir::BuildConTensor2Op::create(b, loc, gammaUTy,
                                                              gammaUComponents)
                      .getResult();

    init3p1 = tensorium::mlir::Init3P1Op::create(
        b, loc, mlir::TypeRange{scalarTy, betaTy, gammaTy, gammaUTy},
        alpha, beta, gamma, gammaU);
  }

  if (init.split3p1.enabled) {
    auto bindToField = [&](llvm::StringRef name, mlir::Value rhs) {
      auto it = fieldArg.find(name);
      if (it == fieldArg.end()) {
        emitUnsupportedExprError(
            loc, "split_3p1 target field '" + name.str() +
                     "' is not available in entry function arguments");
      }
      llvm::SmallVector<mlir::Attribute, 0> noIndices;
      tensorium::mlir::AssignOp::create(b, loc, it->second, rhs,
                                        b.getArrayAttr(noIndices));
    };

    if (init.split3p1.hasAlpha && !init.split3p1.alphaField.empty())
      bindToField(init.split3p1.alphaField, init3p1.getAlpha());
    if (init.split3p1.hasBeta && !init.split3p1.betaField.empty())
      bindToField(init.split3p1.betaField, init3p1.getBeta());
    if (init.split3p1.hasGamma && !init.split3p1.gammaField.empty())
      bindToField(init.split3p1.gammaField, init3p1.getGamma());
    if (init.split3p1.hasGammaU && !init.split3p1.gammaUField.empty())
      bindToField(init.split3p1.gammaUField, init3p1.getGammaU());
  }

  if (moduleUsesFieldName(module, "gammaU")) {
    if (!(init.split3p1.enabled && init.split3p1.hasGammaU)) {
      emitUnsupportedExprError(
          loc, "field 'gammaU' is used in equations but split_3p1 does not bind gammaU");
    }
  }
}

static void emitEvolutionOps(
    mlir::OpBuilder &b, mlir::Location loc,
    const tensorium::backend::ModuleIR &module,
    const llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg) {
  for (const auto &evo : module.evolutions) {
    llvm::StringMap<mlir::Value> tempValues;

    for (const auto &tmp : evo.temporaries) {
      if (!tmp.indexOffsets.empty()) {
        emitUnsupportedExprError(
            loc, "non-scalar temporary '" + tmp.name +
                     "' is not supported in executable mode");
      }
      auto rhsV = emitExpr(b, loc, tmp.rhs.get(), fieldArg, &tempValues);
      tempValues[tmp.name] = rhsV;
    }

    for (const auto &eq : evo.equations) {
      auto it = fieldArg.find(eq.fieldName);
      if (it == fieldArg.end())
        continue;
      auto fieldTy = mlir::dyn_cast<tensorium::mlir::FieldType>(it->second.getType());
      if (!fieldTy)
        emitUnsupportedExprError(loc, "field argument lacks tensorium.field type");
      auto rhsV = emitExpr(b, loc, eq.rhs.get(), fieldArg, &tempValues);
      if (!rhsV)
        continue;
      auto rhsTy = mlir::dyn_cast<tensorium::mlir::FieldType>(rhsV.getType());
      if (!rhsTy)
        emitUnsupportedExprError(loc, "rhs expression did not produce tensorium.field type");
      if (rhsTy.getRank() == 0) {
        rhsV = tensorium::mlir::PromoteOp::create(b, loc, fieldTy, rhsV).getResult();
      } else if (fieldTy != rhsTy) {
        emitUnsupportedExprError(loc, "tensor assignment variance mismatch");
      }
      tensorium::mlir::DtAssignOp::create(b, loc, it->second, rhsV,
                                          makeIndexArrayAttr(b, eq.indices));
    }
  }
}

[[noreturn]] static void emitUnsupportedExprError(mlir::Location loc,
                                                 const std::string &detail) {
  mlir::emitError(loc) << "unsupported Tensorium expression in MLIR emission: "
                       << detail;
  throw std::runtime_error(detail);
}

[[noreturn]] static void emitExternLoweringError(mlir::Location loc,
                                                 const std::string &callee) {
  const std::string detail =
      "extern function '" + callee + "' lowering is not implemented yet";
  mlir::emitError(loc) << detail;
  throw std::runtime_error(detail);
}

static mlir::Value
emitExpr(mlir::OpBuilder &b, mlir::Location loc,
         const tensorium::backend::ExprIR *e,
         const llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg,
         llvm::StringMap<mlir::Value> *localTemps) {
  using namespace tensorium::backend;
  if (!e)
    emitUnsupportedExprError(loc, "null expression");

  auto desiredType = asFieldType(b, e->exprType);

  switch (e->kind) {
  case ExprIR::Kind::Number: {
    auto *n = static_cast<const NumberIR *>(e);
    return tensorium::mlir::ConstOp::create(b, loc, desiredType,
                                            b.getF64FloatAttr(n->value))
        .getResult();
  }
  case ExprIR::Kind::Var: {
    auto *v = static_cast<const VarIR *>(e);
    if (v->vkind == VarKind::Local) {
      if (!localTemps)
        emitUnsupportedExprError(loc, "temporary '" + v->name +
                                         "' is not supported in this context");
      auto itLocal = localTemps->find(v->name);
      if (itLocal == localTemps->end()) {
        emitUnsupportedExprError(
            loc, "temporary '" + v->name + "' referenced before definition");
      }
      return itLocal->second;
    }

    auto it = fieldArg.find(v->name);
    if (it == fieldArg.end())
      emitUnsupportedExprError(loc, "unknown field reference '" + v->name +
                                       "' in MLIR emission");

    mlir::ArrayAttr indicesAttr;
    if (!v->tensorIndexNames.empty()) {
      llvm::SmallVector<mlir::Attribute, 4> idxList;
      for (const auto &s : v->tensorIndexNames)
        idxList.push_back(b.getStringAttr(s));
      indicesAttr = b.getArrayAttr(idxList);
    }

    auto sourceType =
        mlir::dyn_cast<tensorium::mlir::FieldType>(it->second.getType());
    if (!sourceType)
      emitUnsupportedExprError(loc, "field argument '" + v->name +
                                       "' does not have tensorium.field type");

    auto r = tensorium::mlir::RefOp::create(
        b, loc, sourceType, it->second, b.getStringAttr("field"), indicesAttr,
        mlir::ArrayAttr());

    return r.getResult();
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const BinaryIR *>(e);
    auto L = emitExpr(b, loc, bin->lhs.get(), fieldArg, localTemps);
    auto R = emitExpr(b, loc, bin->rhs.get(), fieldArg, localTemps);

    if (bin->op == "+")
      return tensorium::mlir::AddOp::create(b, loc, desiredType, L, R)
          .getResult();
    if (bin->op == "*")
      return tensorium::mlir::MulOp::create(b, loc, desiredType, L, R)
          .getResult();
    if (bin->op == "-")
      return tensorium::mlir::SubOp::create(b, loc, desiredType, L, R)
          .getResult();
    if (bin->op == "/")
      return tensorium::mlir::DivOp::create(b, loc, desiredType, L, R)
          .getResult();

    emitUnsupportedExprError(loc,
                             "binary operator '" + bin->op +
                                 "' is not supported during MLIR emission");
  }
  case ExprIR::Kind::Call: {
    auto *c = static_cast<const CallIR *>(e);
    if (startsWith(c->callee, "d_") && c->callee.size() == 3) {
      if (c->args.empty())
        emitUnsupportedExprError(loc,
                                 "d_* expects exactly one argument in MLIR emission");
      auto arg0 = emitExpr(b, loc, c->args[0].get(), fieldArg, localTemps);
      auto deriv = tensorium::mlir::DerivOp::create(b, loc, desiredType, arg0);
      deriv->setAttr("index", b.getStringAttr(std::string(1, c->callee[2])));
      return deriv.getResult();
    }
    if (c->callee == "contract") {
      if (c->args.empty())
        emitUnsupportedExprError(loc,
                                 "contract() expects exactly one argument in MLIR emission");
      auto arg0 = emitExpr(b, loc, c->args[0].get(), fieldArg, localTemps);
      return tensorium::mlir::ContractOp::create(b, loc, desiredType, arg0)
          .getResult();
    }
    if (c->isExtern)
      emitExternLoweringError(loc, c->callee);

    emitUnsupportedExprError(loc, "call to '" + c->callee +
                                       "' is not supported during MLIR emission");
  }
  case ExprIR::Kind::TensorProduct: {
    auto *p = static_cast<const TensorProductIR *>(e);
    auto L = emitExpr(b, loc, p->lhs.get(), fieldArg, localTemps);
    auto R = emitExpr(b, loc, p->rhs.get(), fieldArg, localTemps);
    return tensorium::mlir::MulOp::create(b, loc, desiredType, L, R).getResult();
  }
  case ExprIR::Kind::Contraction: {
    auto *c = static_cast<const ContractionIR *>(e);
    auto in = emitExpr(b, loc, c->in.get(), fieldArg, localTemps);
    auto out = tensorium::mlir::ContractOp::create(b, loc, desiredType, in);
    if (!c->summedIndices.empty()) {
      out->setAttr("sum_indices", makeIndexArrayAttr(b, c->summedIndices));
    }
    return out.getResult();
  }
  case ExprIR::Kind::IndexRename: {
    auto *r = static_cast<const IndexRenameIR *>(e);
    return emitExpr(b, loc, r->in.get(), fieldArg, localTemps);
  }
  case ExprIR::Kind::IndexPermute: {
    auto *p = static_cast<const IndexPermuteIR *>(e);
    return emitExpr(b, loc, p->in.get(), fieldArg, localTemps);
  }
  case ExprIR::Kind::Trace: {
    auto *t = static_cast<const TraceIR *>(e);
    auto in = emitExpr(b, loc, t->in.get(), fieldArg, localTemps);
    auto out = tensorium::mlir::ContractOp::create(b, loc, desiredType, in);
    if (!t->tracedIndices.empty()) {
      out->setAttr("sum_indices", makeIndexArrayAttr(b, t->tracedIndices));
    }
    return out.getResult();
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *d = static_cast<const PartialDerivativeIR *>(e);
    auto in = emitExpr(b, loc, d->in.get(), fieldArg, localTemps);
    auto deriv = tensorium::mlir::DerivOp::create(b, loc, desiredType, in);
    deriv->setAttr("index", b.getStringAttr(d->coordIndex));
    return deriv.getResult();
  }
  case ExprIR::Kind::Gradient: {
    auto *g = static_cast<const GradientIR *>(e);
    (void)g;
    emitUnsupportedExprError(
        loc, "gradient lowering requires explicit coordinate index; use d_i(...)");
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *d = static_cast<const CovariantDerivativeIR *>(e);
    if (!d->hasConnectionTensor) {
      emitUnsupportedExprError(
          loc, "covariant derivative requires connection tensor Gamma");
    }
    auto in = emitExpr(b, loc, d->in.get(), fieldArg, localTemps);
    auto deriv = tensorium::mlir::DerivOp::create(b, loc, desiredType, in);
    deriv->setAttr("index", b.getStringAttr(d->derivIndex));
    deriv->setAttr("covariant", b.getBoolAttr(true));
    deriv->setAttr("contravariant", b.getBoolAttr(d->contravariant));
    return deriv.getResult();
  }
  case ExprIR::Kind::Divergence: {
    auto *d = static_cast<const DivergenceIR *>(e);
    auto in = emitExpr(b, loc, d->in.get(), fieldArg, localTemps);
    auto out = tensorium::mlir::ContractOp::create(b, loc, desiredType, in);
    if (!d->contractedIndex.empty()) {
      std::vector<std::string> idx = {d->contractedIndex};
      out->setAttr("sum_indices", makeIndexArrayAttr(b, idx));
    }
    return out.getResult();
  }
  }

  emitUnsupportedExprError(loc, "unknown expression kind");
}
} // namespace

static void addEinsteinPipelineSafe(::mlir::PassManager &pm,
                                    const MLIRGenOptions &opts) {

  if (opts.enableEinsteinLoweringPass) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinLoweringPass());
  }

  const bool needValidity = opts.enableEinsteinValidityPass;
  const bool needCanon = opts.enableEinsteinCanonicalizePass;
  const bool needAnalyze = opts.enableEinsteinAnalyzeEinsumPass || needValidity;
  const bool needIndex = opts.enableIndexAnalyzePass || needValidity;

  if (needIndex) {
    pm.addPass(tensorium::mlir::createTensoriumIndexAnalyzePass());
  }

  if (needAnalyze) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinAnalyzeEinsumPass());
  }

  if (needCanon) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinCanonicalizePass());
  }

  if (needValidity) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinValidityPass());
  }

  if (opts.enableStencilLoweringPass) {
    pm.addPass(tensorium::mlir::createTensoriumStencilLoweringPass(opts.dx,
                                                                   opts.order));
  }
  if (opts.enableDissipationPass) {
    pm.addPass(tensorium::mlir::createTensoriumDissipationPass(
        opts.dissipationStrength, opts.dx));
  }
}

static void addPostMLIRNormalizationPipeline(::mlir::PassManager &pm,
                                             const MLIRGenOptions &opts) {
  if (opts.enableMLIRInlinePass)
    pm.addPass(mlir::createInlinerPass());
  if (opts.enableMLIRCanonicalizePass)
    pm.addPass(mlir::createCanonicalizerPass());
  if (opts.enableMLIRCSEPass)
    pm.addPass(mlir::createCSEPass());
}

mlir::OwningOpRef<mlir::ModuleOp>
buildMLIRModule(const tensorium::backend::ModuleIR &module,
                mlir::MLIRContext &ctx, const MLIRGenOptions &opts) {
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<tensorium::mlir::TensoriumDialect>();

  mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();
  auto moduleOp = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc));

  const auto fields = extractFields(module);
  llvm::SmallVector<mlir::Type, 8> allArgTypes;
  allArgTypes.reserve(fields.size());
  for (const auto &fd : fields) {
    allArgTypes.push_back(
        tensorium::mlir::FieldType::get(&ctx, b.getF64Type(), fd.up, fd.down));
  }

  std::vector<unsigned> initArgIndices = collectInitArgIndices(module, fields);
  std::vector<unsigned> rhsArgIndices = collectRhsArgIndices(module, fields);

  auto buildTypeFromIndices = [&](const std::vector<unsigned> &indices) {
    llvm::SmallVector<mlir::Type, 8> types;
    types.reserve(indices.size());
    for (unsigned idx : indices)
      types.push_back(allArgTypes[idx]);
    return b.getFunctionType(types, {});
  };

  auto initFunc =
      mlir::func::FuncOp::create(loc, "tensorium_init", buildTypeFromIndices(initArgIndices));
  auto rhsFunc =
      mlir::func::FuncOp::create(loc, "tensorium_rhs", buildTypeFromIndices(rhsArgIndices));
  auto entryFunc = mlir::func::FuncOp::create(loc, "tensorium_entry",
                                              b.getFunctionType(allArgTypes, {}));

  auto mapFieldArgs = [&](mlir::Block *block, const std::vector<unsigned> &indices) {
    llvm::DenseMap<llvm::StringRef, mlir::Value> fieldArg;
    for (unsigned i = 0; i < indices.size(); ++i)
      fieldArg[fields[indices[i]].name] = block->getArgument(i);
    return fieldArg;
  };

  auto *initBlock = initFunc.addEntryBlock();
  b.setInsertionPointToEnd(initBlock);
  auto initFieldArg = mapFieldArgs(initBlock, initArgIndices);
  emitInitialDataOps(b, loc, module, initFieldArg);
  mlir::func::ReturnOp::create(b, loc);

  auto *rhsBlock = rhsFunc.addEntryBlock();
  b.setInsertionPointToEnd(rhsBlock);
  auto rhsFieldArg = mapFieldArgs(rhsBlock, rhsArgIndices);
  emitEvolutionOps(b, loc, module, rhsFieldArg);
  mlir::func::ReturnOp::create(b, loc);

  auto *entryBlock = entryFunc.addEntryBlock();
  b.setInsertionPointToEnd(entryBlock);
  llvm::SmallVector<mlir::Value, 8> initCallArgs;
  initCallArgs.reserve(initArgIndices.size());
  for (unsigned idx : initArgIndices)
    initCallArgs.push_back(entryBlock->getArgument(idx));

  llvm::SmallVector<mlir::Value, 8> rhsCallArgs;
  rhsCallArgs.reserve(rhsArgIndices.size());
  for (unsigned idx : rhsArgIndices)
    rhsCallArgs.push_back(entryBlock->getArgument(idx));

  mlir::func::CallOp::create(b, loc, "tensorium_init", mlir::TypeRange{},
                             initCallArgs);
  mlir::func::CallOp::create(b, loc, "tensorium_rhs", mlir::TypeRange{},
                             rhsCallArgs);
  mlir::func::ReturnOp::create(b, loc);
  if (module.simulation) {
    moduleOp->getOperation()->setAttr("tensorium.sim.dim",
                                      b.getI64IntegerAttr(module.simulation->dimension));
  }
  moduleOp->push_back(initFunc);
  moduleOp->push_back(rhsFunc);
  moduleOp->push_back(entryFunc);

  MLIRGenOptions pipelineOpts = opts;
  if (module.simulation) {
    pipelineOpts.order = module.simulation->spatial.order;
    if (!module.simulation->resolution.empty() &&
        module.simulation->resolution[0] > 0) {
      pipelineOpts.dx =
          1.0 / static_cast<double>(module.simulation->resolution[0]);
    }
  }

  mlir::PassManager pm(&ctx);
  addEinsteinPipelineSafe(pm, pipelineOpts);
  addPostMLIRNormalizationPipeline(pm, pipelineOpts);
  if (mlir::failed(pm.run(*moduleOp))) {
    llvm::errs() << "Pipeline failed\n";
  }
  return moduleOp;
}

void emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts) {
  mlir::MLIRContext ctx;
  auto moduleOp = buildMLIRModule(module, ctx, opts);
  moduleOp->print(llvm::outs());
}

} // namespace tensorium_mlir
