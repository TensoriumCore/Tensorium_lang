#include "MLIRGenShared.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <stdexcept>

namespace tensorium_mlir {

mlir::ArrayAttr makeIndexArrayAttr(mlir::OpBuilder &b,
                                   const std::vector<std::string> &idx) {
  llvm::SmallVector<mlir::Attribute, 4> names;
  for (const auto &s : idx)
    names.push_back(b.getStringAttr(s));
  return b.getArrayAttr(names);
}

mlir::ArrayAttr makeStringArrayAttr(mlir::OpBuilder &b,
                                    const std::vector<std::string> &v) {
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  attrs.reserve(v.size());
  for (const auto &s : v)
    attrs.push_back(b.getStringAttr(s));
  return b.getArrayAttr(attrs);
}

tensorium::mlir::FieldType asFieldType(mlir::OpBuilder &b,
                                       const tensorium::ir::TensorType &desc) {
  auto *ctx = b.getContext();
  auto elementType = b.getF64Type();
  unsigned up = desc.up < 0 ? 0u : static_cast<unsigned>(desc.up);
  unsigned down = desc.down < 0 ? 0u : static_cast<unsigned>(desc.down);
  return tensorium::mlir::FieldType::get(ctx, elementType, up, down);
}

bool startsWith(const std::string &s, const char *prefix) {
  size_t n = std::char_traits<char>::length(prefix);
  return s.size() >= n && s.compare(0, n, prefix) == 0;
}

std::vector<FieldDesc> extractFields(const tensorium::backend::ModuleIR &module) {
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

void collectExprFieldNames(const tensorium::backend::ExprIR *expr,
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

void collectInitExprFieldNames(const tensorium::backend::InitExprIR *expr,
                               const llvm::StringSet<> &knownFieldNames,
                               llvm::StringSet<> &out) {
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

static bool exprUsesCovariantConnection(const tensorium::backend::ExprIR *expr) {
  using namespace tensorium::backend;
  if (!expr)
    return false;
  switch (expr->kind) {
  case ExprIR::Kind::Number:
  case ExprIR::Kind::Var:
    return false;
  case ExprIR::Kind::Binary: {
    auto *b = static_cast<const BinaryIR *>(expr);
    return exprUsesCovariantConnection(b->lhs.get()) ||
           exprUsesCovariantConnection(b->rhs.get());
  }
  case ExprIR::Kind::Call: {
    auto *c = static_cast<const CallIR *>(expr);
    for (const auto &arg : c->args) {
      if (exprUsesCovariantConnection(arg.get()))
        return true;
    }
    return false;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *p = static_cast<const TensorProductIR *>(expr);
    return exprUsesCovariantConnection(p->lhs.get()) ||
           exprUsesCovariantConnection(p->rhs.get());
  }
  case ExprIR::Kind::Contraction: {
    auto *c = static_cast<const ContractionIR *>(expr);
    return exprUsesCovariantConnection(c->in.get());
  }
  case ExprIR::Kind::IndexRename: {
    auto *r = static_cast<const IndexRenameIR *>(expr);
    return exprUsesCovariantConnection(r->in.get());
  }
  case ExprIR::Kind::IndexPermute: {
    auto *p = static_cast<const IndexPermuteIR *>(expr);
    return exprUsesCovariantConnection(p->in.get());
  }
  case ExprIR::Kind::Trace: {
    auto *t = static_cast<const TraceIR *>(expr);
    return exprUsesCovariantConnection(t->in.get());
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *d = static_cast<const PartialDerivativeIR *>(expr);
    return exprUsesCovariantConnection(d->in.get());
  }
  case ExprIR::Kind::Gradient: {
    auto *g = static_cast<const GradientIR *>(expr);
    return exprUsesCovariantConnection(g->in.get());
  }
  case ExprIR::Kind::CovariantDerivative:
    return true;
  case ExprIR::Kind::Divergence: {
    auto *d = static_cast<const DivergenceIR *>(expr);
    return exprUsesCovariantConnection(d->in.get());
  }
  }
  return false;
}

static const FieldDesc *
selectConnectionFieldForRhs(const std::vector<FieldDesc> &fields) {
  auto findNamed = [&](llvm::StringRef name) -> const FieldDesc * {
    for (const auto &field : fields) {
      if (field.name != name)
        continue;
      if ((field.up + field.down) == 3 && field.up == 1 && field.down == 2)
        return &field;
    }
    return nullptr;
  };

  if (const FieldDesc *preferred = findNamed("Christoffel"))
    return preferred;
  if (const FieldDesc *preferred = findNamed("Gamma"))
    return preferred;

  for (const auto &field : fields) {
    if ((field.up + field.down) == 3 && field.up == 1 && field.down == 2)
      return &field;
  }
  return nullptr;
}

bool moduleUsesFieldName(const tensorium::backend::ModuleIR &module,
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

std::vector<unsigned>
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

std::vector<unsigned>
collectRhsArgIndices(const tensorium::backend::ModuleIR &module,
                     const std::vector<FieldDesc> &fields) {
  llvm::StringSet<> needed;
  bool needsConnectionField = false;
  for (const auto &evo : module.evolutions) {
    for (const auto &tmp : evo.temporaries) {
      collectExprFieldNames(tmp.rhs.get(), needed);
      needsConnectionField |= exprUsesCovariantConnection(tmp.rhs.get());
    }
    for (const auto &eq : evo.equations) {
      needed.insert(eq.fieldName);
      collectExprFieldNames(eq.rhs.get(), needed);
      needsConnectionField |= exprUsesCovariantConnection(eq.rhs.get());
    }
  }

  if (needsConnectionField) {
    if (const FieldDesc *connection = selectConnectionFieldForRhs(fields)) {
      needed.insert(connection->name);
    }
  }

  std::vector<unsigned> out;
  for (unsigned i = 0; i < fields.size(); ++i) {
    if (needed.contains(fields[i].name))
      out.push_back(i);
  }
  return out;
}

[[noreturn]] void emitUnsupportedExprError(mlir::Location loc,
                                           const std::string &detail) {
  mlir::emitError(loc) << "unsupported Tensorium expression in MLIR emission: "
                       << detail;
  throw std::runtime_error(detail);
}

[[noreturn]] void emitExternLoweringError(mlir::Location loc,
                                          const std::string &callee) {
  const std::string detail =
      "extern function '" + callee + "' lowering is not implemented yet";
  mlir::emitError(loc) << detail;
  throw std::runtime_error(detail);
}

} // namespace tensorium_mlir
