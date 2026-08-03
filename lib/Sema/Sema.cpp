#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Sema/CallSupport.hpp"
#include "tensorium/Sema/tensor_type_checker.hpp"
#include <stdexcept>
#include <utility>

namespace tensorium {

static std::unique_ptr<IndexedCall> makeDeriv(const std::string &idx,
                                              std::unique_ptr<IndexedExpr> arg,
                                              TensorTypeChecker &checker) {
  auto call = std::make_unique<IndexedCall>();
  call->callee = "d_" + idx;
  call->args.push_back(std::move(arg));
  checker.infer(call.get());
  return call;
}

static std::unique_ptr<IndexedCall>
makeContract(std::unique_ptr<IndexedExpr> arg, TensorTypeChecker &checker) {
  auto call = std::make_unique<IndexedCall>();
  call->callee = "contract";
  call->args.push_back(std::move(arg));
  call->isExtern = false;
  checker.infer(call.get());
  return call;
}

static TensorType tensorTypeFromDesc(const TensorTypeDesc &desc) {
  return TensorType{desc.up, desc.down};
}

void SemanticAnalyzer::validateSpatialIndex(const std::string &idx) {
  if (!core::isSpatialIndexName(idx)) {
    throw std::runtime_error("Invalid tensor index '" + idx +
                             "'. Allowed: {i, j, k, l, m, n}.");
  }
}

int SemanticAnalyzer::resolveIndex(const std::string &name) {
  auto it = coordIndex.find(name);
  if (it == coordIndex.end())
    throw std::runtime_error("Unknown tensor index: " + name);
  return it->second;
}

// --- TRANSFORM EXPR MISE À JOUR ---

std::unique_ptr<IndexedExpr> SemanticAnalyzer::transformExpr(const Expr *e) {
  if (auto n = dynamic_cast<const NumberExpr *>(e))
    return std::make_unique<IndexedNumber>(n->value);

  if (auto v = dynamic_cast<const VarExpr *>(e)) {
    if (auto itLocal = locals.find(v->name); itLocal != locals.end()) {
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Local);
      iv->tensorKind = itLocal->second.kind;
      iv->up = itLocal->second.up;
      iv->down = itLocal->second.down;
      return iv;
    }

    if (params.count(v->name)) {
      auto iv =
          std::make_unique<IndexedVar>(v->name, IndexedVarKind::Parameter);
      iv->tensorKind = TensorKind::Scalar;
      return iv;
    }

    if (auto itf = fields.find(v->name); itf != fields.end()) {
      const FieldDecl *fd = itf->second;
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Field);
      iv->tensorKind = fd->kind;
      iv->up = fd->up;
      iv->down = fd->down;
      return iv;
    }

    if (auto itu = unknowns.find(v->name); itu != unknowns.end()) {
      const ConstraintUnknownDecl *decl = itu->second;
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Unknown);
      iv->tensorKind = decl->type.kind;
      iv->up = decl->type.up;
      iv->down = decl->type.down;
      return iv;
    }

    if (auto it = coordIndex.find(v->name); it != coordIndex.end()) {
      auto iv =
          std::make_unique<IndexedVar>(v->name, IndexedVarKind::Coordinate);
      iv->coordIndex = it->second;
      iv->tensorKind = TensorKind::Scalar;
      return iv;
    }

    throw std::runtime_error("Unknown identifier: " + v->name);
  }

  if (auto b = dynamic_cast<const BinaryExpr *>(e))
    return std::make_unique<IndexedBinary>(b->op, transformExpr(b->lhs.get()),
                                           transformExpr(b->rhs.get()));

  if (auto p = dynamic_cast<const ParenExpr *>(e))
    return transformExpr(p->inner.get());

  if (auto iv = dynamic_cast<const IndexedVarExpr *>(e)) {
    IndexedVarKind outKind = IndexedVarKind::Field;
    TensorKind tensorKind = TensorKind::Scalar;
    int up = 0;
    int down = 0;

    if (auto itLocal = locals.find(iv->base); itLocal != locals.end()) {
      outKind = IndexedVarKind::Local;
      tensorKind = itLocal->second.kind;
      up = itLocal->second.up;
      down = itLocal->second.down;
    } else if (auto it = fields.find(iv->base); it != fields.end()) {
      const FieldDecl *fd = it->second;
      tensorKind = fd->kind;
      up = fd->up;
      down = fd->down;
    } else if (auto it = unknowns.find(iv->base); it != unknowns.end()) {
      const ConstraintUnknownDecl *decl = it->second;
      outKind = IndexedVarKind::Unknown;
      tensorKind = decl->type.kind;
      up = decl->type.up;
      down = decl->type.down;
    } else {
      throw std::runtime_error("Unknown indexed tensor: " + iv->base);
    }

    size_t expected = static_cast<size_t>(up + down);

    if (iv->indices.size() != expected)
      throw std::runtime_error("Tensor '" + iv->base + "' expects " +
                               std::to_string(expected) + " indices, got " +
                               std::to_string(iv->indices.size()));

    auto out = std::make_unique<IndexedVar>(iv->base, outKind);
    out->tensorKind = tensorKind;
    out->up = up;
    out->down = down;

    size_t pos = 0;
    for (size_t i = 0; i < iv->indices.size(); ++i) {
      const auto &idx = iv->indices[i];
      if (!coordIndex.count(idx)) {
        validateSpatialIndex(idx);
        coordIndex[idx] = -2;
      }
      int off = resolveIndex(idx);
      out->tensorIndices.push_back(off);
      out->tensorIndexNames.push_back(idx);
      int idxOff = 0;
      if (i < iv->indexOffsets.size())
        idxOff = iv->indexOffsets[i];
      out->tensorIndexOffsets.push_back(idxOff);
      bool isUp = pos < static_cast<size_t>(up);
      out->tensorIndexIsUp.push_back(isUp);
      ++pos;
    }
    return out;
  }

  if (auto c = dynamic_cast<const CallExpr *>(e)) {
    if (c->callee == "trace") {
      if (c->args.size() != 1)
        throw std::runtime_error("trace() expects exactly 1 argument");
      TensorTypeChecker checker;
      auto arg = transformExpr(c->args[0].get());
      auto out = makeContract(std::move(arg), checker);
      checker.infer(out.get());
      return out;
    }

    if (c->callee == "laplacian") {
      if (c->args.size() != 1)
        throw std::runtime_error("laplacian() expects exactly 1 argument");
      TensorTypeChecker checker;
      auto arg = transformExpr(c->args[0].get());
      TensorType argT = checker.infer(arg.get());
      if (!argT.isScalar())
        throw std::runtime_error("laplacian() expects scalar argument");

      auto d1 = makeDeriv("i", std::move(arg), checker);
      auto d2 = makeDeriv("i", std::move(d1), checker);
      auto out = makeContract(std::move(d2), checker);
      checker.infer(out.get());
      return out;
    }

    const bool isCovariantNabla =
        c->callee.size() == 7 && c->callee.rfind("nabla_", 0) == 0;
    const bool isContravariantNabla =
        c->callee.size() == 7 && c->callee.rfind("nabla^", 0) == 0;
    if (isCovariantNabla || isContravariantNabla) {
      return transformNablaCall(*c, isContravariantNabla);
    }

    const ExternDecl *externDecl = nullptr;
    if (auto itExt = externSignatures.find(c->callee);
        itExt != externSignatures.end()) {
      externDecl = itExt->second;
      if (c->args.size() != externDecl->params.size()) {
        throw std::runtime_error(
            "extern function '" + c->callee + "' expects " +
            std::to_string(externDecl->params.size()) + " arguments, got " +
            std::to_string(c->args.size()));
      }
    }

    if (mode == CompilationMode::Executable &&
        !isExecutableBuiltin(c->callee) && !externDecl) {
      throw std::runtime_error(
          "executable mode requires implementation for function '" + c->callee +
          "'");
    }
    auto out = std::make_unique<IndexedCall>();
    out->callee = c->callee;
    out->isExtern = externDecl != nullptr;
    out->declaredArity = c->args.size();
    if (externDecl) {
      out->returnType = externDecl->returnType;
      out->paramTypes = externDecl->params;
    }
    TensorTypeChecker argChecker(hasConnectionTensor);
    for (size_t i = 0; i < c->args.size(); ++i) {
      auto transformed = transformExpr(c->args[i].get());
      if (externDecl) {
        TensorType actual = argChecker.infer(transformed.get());
        TensorType expected = tensorTypeFromDesc(externDecl->params[i]);
        if (!actual.sameVariance(expected)) {
          throw std::runtime_error("extern function '" + c->callee +
                                   "' argument " + std::to_string(i + 1) +
                                   " variance mismatch");
        }
      }
      out->args.push_back(std::move(transformed));
    }
    return out;
  }

  throw std::runtime_error("Unsupported expr in semantic analyzer");
}

} // namespace tensorium
