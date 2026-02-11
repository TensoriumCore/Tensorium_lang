#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include <algorithm>
#include <stdexcept>
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
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

static bool isZeroInitExpr(const tensorium::backend::InitExprIR *expr) {
  using tensorium::backend::InitExprIR;
  if (!expr)
    return false;
  if (expr->kind == InitExprIR::Kind::Number) {
    auto *n = static_cast<const tensorium::backend::InitNumberIR *>(expr);
    return n->value == 0.0;
  }
  return false;
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

  if (!init.hasMetric4)
    return;

  if (init.metric4.components.size() != 16) {
    emitUnsupportedExprError(
        loc, "metric4 initial_data must provide 16 component expressions");
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

  llvm::SmallVector<mlir::Value, 16> metricComps;
  metricComps.reserve(16);
  for (const auto &comp : init.metric4.components) {
    metricComps.push_back(emitInitExpr(b, loc, comp.get(), fieldArg, paramValues,
                                       coordValues));
  }

  [[maybe_unused]] auto metric = tensorium::mlir::Metric4Op::create(
      b, loc, metricTy, metricComps, b.getStringAttr(init.metric4.name),
      b.getStringAttr(init.metric4.coordSystem),
      makeStringArrayAttr(b, init.metric4.indices),
      b.getBoolAttr(init.metric4.enforceSymmetry));

  const bool betaZero = isZeroInitExpr(init.metric4.components[1].get()) &&
                        isZeroInitExpr(init.metric4.components[2].get()) &&
                        isZeroInitExpr(init.metric4.components[3].get()) &&
                        isZeroInitExpr(init.metric4.components[4].get()) &&
                        isZeroInitExpr(init.metric4.components[8].get()) &&
                        isZeroInitExpr(init.metric4.components[12].get());
  if (!betaZero) {
    emitUnsupportedExprError(
        loc, "metric4 -> init3p1 lowering with non-zero shift is not implemented yet");
  }

  auto one = tensorium::mlir::ConstOp::create(b, loc, scalarTy,
                                              b.getF64FloatAttr(1.0))
                 .getResult();
  auto zero = tensorium::mlir::ConstOp::create(b, loc, scalarTy,
                                               b.getF64FloatAttr(0.0))
                  .getResult();
  auto negOne = tensorium::mlir::ConstOp::create(b, loc, scalarTy,
                                                 b.getF64FloatAttr(-1.0))
                    .getResult();

  mlir::Value g00 = metricComps[0];
  mlir::Value g01 = metricComps[1];
  mlir::Value g02 = metricComps[2];
  mlir::Value g03 = metricComps[3];
  mlir::Value g11 = metricComps[5];
  mlir::Value g22 = metricComps[10];
  mlir::Value g33 = metricComps[15];

  auto minusG00 =
      tensorium::mlir::MulOp::create(b, loc, scalarTy, negOne, g00).getResult();
  auto alpha =
      tensorium::mlir::SqrtOp::create(b, loc, scalarTy, minusG00).getResult();

  llvm::SmallVector<mlir::Value, 3> betaComponents = {g01, g02, g03};
  auto beta = tensorium::mlir::BuildCovectorOp::create(b, loc, betaTy,
                                                        betaComponents)
                  .getResult();

  llvm::SmallVector<mlir::Value, 9> gammaComponents = {g11, zero, zero, zero,
                                                       g22, zero, zero, zero,
                                                       g33};
  auto gamma = tensorium::mlir::BuildCovTensor2Op::create(b, loc, gammaTy,
                                                           gammaComponents)
                   .getResult();

  auto inv11 = tensorium::mlir::DivOp::create(b, loc, scalarTy, one, g11).getResult();
  auto inv22 = tensorium::mlir::DivOp::create(b, loc, scalarTy, one, g22).getResult();
  auto inv33 = tensorium::mlir::DivOp::create(b, loc, scalarTy, one, g33).getResult();
  llvm::SmallVector<mlir::Value, 9> gammaUComponents = {
      inv11, zero, zero, zero, inv22, zero, zero, zero, inv33};
  auto gammaU = tensorium::mlir::BuildConTensor2Op::create(b, loc, gammaUTy,
                                                            gammaUComponents)
                    .getResult();

  auto init3p1 = tensorium::mlir::Init3P1Op::create(
      b, loc, mlir::TypeRange{scalarTy, betaTy, gammaTy, gammaUTy},
      alpha, beta, gamma, gammaU);

  if (init.split3p1.enabled) {
    if (init.split3p1.hasAlpha && !init.split3p1.alphaField.empty())
      fieldArg[init.split3p1.alphaField] = init3p1.getAlpha();
    if (init.split3p1.hasBeta && !init.split3p1.betaField.empty())
      fieldArg[init.split3p1.betaField] = init3p1.getBeta();
    if (init.split3p1.hasGamma && !init.split3p1.gammaField.empty())
      fieldArg[init.split3p1.gammaField] = init3p1.getGamma();
    if (init.split3p1.hasGammaU && !init.split3p1.gammaUField.empty())
      fieldArg[init.split3p1.gammaUField] = init3p1.getGammaU();
  }

  if (moduleUsesFieldName(module, "gammaU")) {
    if (!(init.split3p1.enabled && init.split3p1.hasGammaU)) {
      emitUnsupportedExprError(
          loc, "field 'gammaU' is used in equations but split_3p1 does not bind gammaU");
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

void emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<tensorium::mlir::TensoriumDialect>();

  mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();
  auto moduleOp = mlir::ModuleOp::create(loc);

  const auto fields = extractFields(module);
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (const auto &fd : fields) {
    argTypes.push_back(
        tensorium::mlir::FieldType::get(&ctx, b.getF64Type(), fd.up, fd.down));
  }

  auto funcTy = b.getFunctionType(argTypes, {});
  auto f = mlir::func::FuncOp::create(loc, "tensorium_entry", funcTy);
  auto *entry = f.addEntryBlock();
  b.setInsertionPointToEnd(entry);

  llvm::DenseMap<llvm::StringRef, mlir::Value> fieldArg;
  for (unsigned i = 0; i < fields.size(); ++i) {
    fieldArg[fields[i].name] = entry->getArgument(i);
  }

  emitInitialDataOps(b, loc, module, fieldArg);

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
        rhsV =
            tensorium::mlir::PromoteOp::create(b, loc, fieldTy, rhsV).getResult();
      } else if (fieldTy != rhsTy) {
        emitUnsupportedExprError(loc, "tensor assignment variance mismatch");
      }
      tensorium::mlir::DtAssignOp::create(b, loc, it->second, rhsV,
                                          makeIndexArrayAttr(b, eq.indices));
    }
  }
  if (module.simulation) {
    moduleOp->setAttr("tensorium.sim.dim",
                      b.getI64IntegerAttr(module.simulation->dimension));
  }
  mlir::func::ReturnOp::create(b, loc);
  moduleOp.push_back(f);

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

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(moduleOp))) {
    llvm::errs() << "Pipeline failed\n";
  }
  moduleOp.print(llvm::outs());
}

} // namespace tensorium_mlir
