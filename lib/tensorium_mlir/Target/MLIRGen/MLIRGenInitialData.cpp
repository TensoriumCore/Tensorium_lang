#include "MLIRGenInitialData.h"
#include "MLIRGenShared.h"
#include "mlir/IR/Builders.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/StringMap.h"

namespace tensorium_mlir {
namespace {

bool isCoordinateName(std::string_view name) {
  return name == "t" || name == "x" || name == "y" || name == "z" ||
         name == "r" || name == "rho" || name == "theta" || name == "phi";
}

bool initExprStructuralEqual(const tensorium::backend::InitExprIR *lhs,
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

bool metric4HasSymmetricComponents(
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

mlir::Value emitInitExpr(mlir::OpBuilder &b, mlir::Location loc,
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
    return b
        .create<tensorium::mlir::ConstOp>(loc, scalarTy,
                                          b.getF64FloatAttr(n->value))
        .getResult();
  }
  case InitExprIR::Kind::Symbol: {
    auto *s = static_cast<const InitSymbolIR *>(expr);
    if (auto it = fieldArg.find(s->name); it != fieldArg.end()) {
      return b
          .create<tensorium::mlir::RefOp>(loc, scalarTy, it->second,
                                          b.getStringAttr("field"),
                                          mlir::ArrayAttr(), mlir::ArrayAttr())
          .getResult();
    }

    if (isCoordinateName(s->name)) {
      if (auto it = coordValues.find(s->name); it != coordValues.end())
        return it->second;
      auto coord = b.create<tensorium::mlir::CoordOp>(loc, scalarTy,
                                                      b.getStringAttr(s->name));
      coordValues[s->name] = coord.getResult();
      return coord.getResult();
    }

    if (auto it = paramValues.find(s->name); it != paramValues.end())
      return it->second;
    auto param = b.create<tensorium::mlir::ParamOp>(loc, scalarTy,
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
      return b.create<tensorium::mlir::AddOp>(loc, scalarTy, L, R).getResult();
    if (bin->op == '-')
      return b.create<tensorium::mlir::SubOp>(loc, scalarTy, L, R).getResult();
    if (bin->op == '*')
      return b.create<tensorium::mlir::MulOp>(loc, scalarTy, L, R).getResult();
    if (bin->op == '/')
      return b.create<tensorium::mlir::DivOp>(loc, scalarTy, L, R).getResult();
    if (bin->op == '^') {
      auto *rhsNum = dynamic_cast<const InitNumberIR *>(bin->rhs.get());
      if (!rhsNum)
        emitUnsupportedExprError(
            loc,
            "initial_data exponentiation expects numeric literal exponent");
      int exp = static_cast<int>(rhsNum->value);
      if (rhsNum->value != static_cast<double>(exp) || exp < 0 || exp > 4)
        emitUnsupportedExprError(
            loc, "initial_data exponentiation supports integer exponents 0..4");
      if (exp == 0) {
        return b
            .create<tensorium::mlir::ConstOp>(loc, scalarTy,
                                              b.getF64FloatAttr(1.0))
            .getResult();
      }
      mlir::Value acc = L;
      for (int i = 1; i < exp; ++i)
        acc =
            b.create<tensorium::mlir::MulOp>(loc, scalarTy, acc, L).getResult();
      return acc;
    }
    emitUnsupportedExprError(loc, "unsupported initial_data binary operator");
  }
  case InitExprIR::Kind::Call: {
    auto *call = static_cast<const InitCallIR *>(expr);
    if (call->callee == "sin") {
      if (call->args.size() != 1)
        emitUnsupportedExprError(loc, "sin() expects 1 argument");
      auto arg = emitInitExpr(b, loc, call->args[0].get(), fieldArg,
                              paramValues, coordValues);
      return b.create<tensorium::mlir::SinOp>(loc, scalarTy, arg).getResult();
    }
    if (call->callee == "sqrt") {
      if (call->args.size() != 1)
        emitUnsupportedExprError(loc, "sqrt() expects 1 argument");
      auto arg = emitInitExpr(b, loc, call->args[0].get(), fieldArg,
                              paramValues, coordValues);
      return b.create<tensorium::mlir::SqrtOp>(loc, scalarTy, arg).getResult();
    }
    emitUnsupportedExprError(loc, "unsupported initial_data function '" +
                                      call->callee + "'");
  }
  }

  emitUnsupportedExprError(loc, "unsupported initial_data expression");
}

} // namespace

void emitInitialDataOps(
    mlir::OpBuilder &b, mlir::Location loc,
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

    llvm::SmallVector<mlir::Value, 16> metricComps;
    metricComps.reserve(16);
    for (const auto &comp : init.metric4.components) {
      metricComps.push_back(
          emitInitExpr(b, loc, comp.get(), fieldArg, paramValues, coordValues));
    }

    auto metric = b.create<tensorium::mlir::Metric4Op>(
        loc, metricTy, metricComps, b.getStringAttr(init.metric4.name),
        b.getStringAttr(init.metric4.coordSystem),
        makeStringArrayAttr(b, init.metric4.indices),
        b.getBoolAttr(init.metric4.enforceSymmetry));

    auto decomp = b.create<tensorium::mlir::Decompose3P1FromMetricOp>(
        loc, mlir::TypeRange{scalarTy, betaTy, gammaTy, gammaUTy},
        metric.getMetric());

    init3p1 = b.create<tensorium::mlir::Init3P1Op>(
        loc, mlir::TypeRange{scalarTy, betaTy, gammaTy, gammaUTy},
        decomp.getAlpha(), decomp.getBeta(), decomp.getGamma(),
        decomp.getGammaU());
  } else if (init.hasDecomposed) {
    if (!init.decomposed.alphaExpr || init.decomposed.betaExpr.size() != 3 ||
        init.decomposed.gammaExpr.size() != 9 ||
        init.decomposed.gammaUExpr.size() != 9) {
      emitUnsupportedExprError(
          loc, "decomposed initial_data requires alpha, beta[3], gamma[3x3], "
               "gammaU[3x3]");
    }

    auto alpha = emitInitExpr(b, loc, init.decomposed.alphaExpr.get(), fieldArg,
                              paramValues, coordValues);

    llvm::SmallVector<mlir::Value, 3> betaComponents;
    betaComponents.reserve(3);
    for (const auto &expr : init.decomposed.betaExpr) {
      betaComponents.push_back(
          emitInitExpr(b, loc, expr.get(), fieldArg, paramValues, coordValues));
    }
    auto beta =
        b.create<tensorium::mlir::BuildCovectorOp>(loc, betaTy, betaComponents)
            .getResult();

    llvm::SmallVector<mlir::Value, 9> gammaComponents;
    gammaComponents.reserve(9);
    for (const auto &expr : init.decomposed.gammaExpr) {
      gammaComponents.push_back(
          emitInitExpr(b, loc, expr.get(), fieldArg, paramValues, coordValues));
    }
    auto gamma = b.create<tensorium::mlir::BuildCovTensor2Op>(loc, gammaTy,
                                                              gammaComponents)
                     .getResult();

    llvm::SmallVector<mlir::Value, 9> gammaUComponents;
    gammaUComponents.reserve(9);
    for (const auto &expr : init.decomposed.gammaUExpr) {
      gammaUComponents.push_back(
          emitInitExpr(b, loc, expr.get(), fieldArg, paramValues, coordValues));
    }
    auto gammaU = b.create<tensorium::mlir::BuildConTensor2Op>(loc, gammaUTy,
                                                               gammaUComponents)
                      .getResult();

    init3p1 = b.create<tensorium::mlir::Init3P1Op>(
        loc, mlir::TypeRange{scalarTy, betaTy, gammaTy, gammaUTy}, alpha, beta,
        gamma, gammaU);
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
      b.create<tensorium::mlir::AssignOp>(loc, it->second, rhs,
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
          loc, "field 'gammaU' is used in equations but split_3p1 does not "
               "bind gammaU");
    }
  }
}

} // namespace tensorium_mlir
