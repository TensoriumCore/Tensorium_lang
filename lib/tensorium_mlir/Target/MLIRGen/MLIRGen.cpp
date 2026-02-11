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
#include <cctype>

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

static std::string trimExprString(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.pop_back();
  return s;
}

static bool isZeroExprString(const std::string &expr) {
  const std::string t = trimExprString(expr);
  return t == "0" || t == "0.0" || t == "(0)" || t == "-(0)";
}

static std::string deriveAlphaExpr(const std::string &g00Expr, bool betaZero) {
  if (!betaZero)
    return "sqrt(-1/g^00)";

  std::string t = trimExprString(g00Expr);
  if (t.rfind("-(", 0) == 0 && t.size() >= 3 && t.back() == ')') {
    std::string inner = t.substr(2, t.size() - 3);
    return "sqrt(" + inner + ")";
  }
  if (!t.empty() && t.front() == '-') {
    return "sqrt(" + t.substr(1) + ")";
  }
  return "sqrt(-1/g^00)";
}

static mlir::ArrayAttr makeStringArrayAttr(mlir::OpBuilder &b,
                                           const std::vector<std::string> &v) {
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  attrs.reserve(v.size());
  for (const auto &s : v)
    attrs.push_back(b.getStringAttr(s));
  return b.getArrayAttr(attrs);
}

static void emitInitialDataOps(mlir::OpBuilder &b, mlir::Location loc,
                               const tensorium::backend::ModuleIR &module) {
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

  auto *ctx = b.getContext();
  auto elemTy = b.getF64Type();
  auto metricTy = tensorium::mlir::FieldType::get(ctx, elemTy, 0, 2);
  auto alphaTy = tensorium::mlir::FieldType::get(ctx, elemTy, 0, 0);
  auto betaTy = tensorium::mlir::FieldType::get(ctx, elemTy, 0, 1);
  auto gammaTy = tensorium::mlir::FieldType::get(ctx, elemTy, 0, 2);
  auto gammaUTy = tensorium::mlir::FieldType::get(ctx, elemTy, 2, 0);

  auto metric = tensorium::mlir::Metric4Op::create(
      b, loc, metricTy, b.getStringAttr(init.metric4.name),
      b.getStringAttr(init.metric4.coordSystem),
      makeStringArrayAttr(b, init.metric4.indices),
      makeStringArrayAttr(b, init.metric4.components),
      b.getBoolAttr(init.metric4.enforceSymmetry));

  const bool betaZero =
      isZeroExprString(init.metric4.components[1]) &&
      isZeroExprString(init.metric4.components[2]) &&
      isZeroExprString(init.metric4.components[3]) &&
      isZeroExprString(init.metric4.components[4]) &&
      isZeroExprString(init.metric4.components[8]) &&
      isZeroExprString(init.metric4.components[12]);

  std::vector<std::string> gammaDiag = {init.metric4.components[5],
                                        init.metric4.components[10],
                                        init.metric4.components[15]};
  std::string alphaExpr = deriveAlphaExpr(init.metric4.components[0], betaZero);

  (void)tensorium::mlir::Split3P1Op::create(
      b, loc, mlir::TypeRange{alphaTy, betaTy, gammaTy, gammaUTy},
      metric.getResult(), b.getBoolAttr(betaZero),
      makeStringArrayAttr(b, gammaDiag), b.getStringAttr(alphaExpr),
      b.getBoolAttr(true));
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

  emitInitialDataOps(b, loc, module);

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
