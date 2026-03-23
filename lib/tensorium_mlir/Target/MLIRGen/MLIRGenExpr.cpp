#include "MLIRGenExpr.h"
#include "MLIRGenShared.h"
#include "mlir/IR/Builders.h"
#include "tensorium/Core/IndexSet.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/StringMap.h"

#include <unordered_set>

namespace tensorium_mlir {
namespace {

static mlir::ArrayAttr makeIndicesAttr(mlir::OpBuilder &b,
                                       llvm::ArrayRef<std::string> names) {
  if (names.empty())
    return mlir::ArrayAttr();
  llvm::SmallVector<mlir::Attribute, 4> idxList;
  idxList.reserve(names.size());
  for (const auto &s : names)
    idxList.push_back(b.getStringAttr(s));
  return b.getArrayAttr(idxList);
}

static mlir::Value emitFieldRefFromSource(mlir::OpBuilder &b, mlir::Location loc,
                                          mlir::Value source,
                                          llvm::ArrayRef<std::string> names) {
  auto sourceType = mlir::dyn_cast<tensorium::mlir::FieldType>(source.getType());
  if (!sourceType)
    emitUnsupportedExprError(loc, "field source does not have tensorium.field type");

  auto ref = b.create<tensorium::mlir::RefOp>(
      loc, sourceType, source, b.getStringAttr("field"),
      makeIndicesAttr(b, names), mlir::ArrayAttr());
  return ref.getResult();
}

static mlir::Value emitFieldRefByName(
    mlir::OpBuilder &b, mlir::Location loc, llvm::StringRef fieldName,
    llvm::ArrayRef<std::string> names,
    const llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg) {
  auto it = fieldArg.find(fieldName);
  if (it == fieldArg.end())
    emitUnsupportedExprError(loc, "unknown field reference '" + fieldName.str() +
                                     "' in MLIR emission");
  return emitFieldRefFromSource(b, loc, it->second, names);
}

static mlir::Value
findConnectionFieldValue(const llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg) {
  auto isConnectionType = [](mlir::Type ty) {
    auto fieldTy = mlir::dyn_cast<tensorium::mlir::FieldType>(ty);
    return fieldTy && fieldTy.getRank() == 3 && fieldTy.getUp() == 1 &&
           fieldTy.getDown() == 2;
  };

  auto pickNamed = [&](llvm::StringRef name) -> mlir::Value {
    auto it = fieldArg.find(name);
    if (it == fieldArg.end())
      return mlir::Value();
    if (!isConnectionType(it->second.getType()))
      return mlir::Value();
    return it->second;
  };

  if (auto preferred = pickNamed("Christoffel"))
    return preferred;
  if (auto preferred = pickNamed("Gamma"))
    return preferred;

  for (const auto &entry : fieldArg) {
    if (isConnectionType(entry.second.getType()))
      return entry.second;
  }
  return mlir::Value();
}

static std::string pickDummyIndex(llvm::ArrayRef<std::string> used) {
  std::unordered_set<std::string> usedSet;
  usedSet.reserve(used.size());
  for (const auto &idx : used)
    usedSet.insert(idx);
  for (char c : tensorium::core::kTensorIndices) {
    std::string name(1, c);
    if (!usedSet.count(name))
      return name;
  }
  return "l";
}

mlir::Value emitExpr(mlir::OpBuilder &b, mlir::Location loc,
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
    return b.create<tensorium::mlir::ConstOp>(loc, desiredType,
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

    if (v->vkind == VarKind::Param) {
      if (desiredType.getRank() != 0) {
        emitUnsupportedExprError(
            loc, "parameter '" + v->name + "' must lower as scalar");
      }
      return b.create<tensorium::mlir::ParamOp>(loc, desiredType,
                                              b.getStringAttr(v->name))
          .getResult();
    }

    if (v->vkind == VarKind::Coord) {
      if (desiredType.getRank() != 0) {
        emitUnsupportedExprError(
            loc, "coordinate '" + v->name + "' must lower as scalar");
      }
      return b.create<tensorium::mlir::CoordOp>(loc, desiredType,
                                              b.getStringAttr(v->name))
          .getResult();
    }

    if (v->vkind != VarKind::Field) {
      emitUnsupportedExprError(loc,
                               "unsupported variable kind in MLIR emission");
    }

    return emitFieldRefByName(b, loc, v->name, v->tensorIndexNames, fieldArg);
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const BinaryIR *>(e);
    auto L = emitExpr(b, loc, bin->lhs.get(), fieldArg, localTemps);
    auto R = emitExpr(b, loc, bin->rhs.get(), fieldArg, localTemps);

    if (bin->op == "+")
      return b.create<tensorium::mlir::AddOp>(loc, desiredType, L, R)
          .getResult();
    if (bin->op == "*")
      return b.create<tensorium::mlir::MulOp>(loc, desiredType, L, R)
          .getResult();
    if (bin->op == "-")
      return b.create<tensorium::mlir::SubOp>(loc, desiredType, L, R)
          .getResult();
    if (bin->op == "/")
      return b.create<tensorium::mlir::DivOp>(loc, desiredType, L, R)
          .getResult();

    emitUnsupportedExprError(
        loc, "binary operator '" + bin->op +
                 "' is not supported during MLIR emission");
  }
  case ExprIR::Kind::Call: {
    auto *c = static_cast<const CallIR *>(e);
    if (startsWith(c->callee, "d_") && c->callee.size() == 3) {
      if (c->args.empty())
        emitUnsupportedExprError(
            loc, "d_* expects exactly one argument in MLIR emission");
      auto arg0 = emitExpr(b, loc, c->args[0].get(), fieldArg, localTemps);
      auto deriv = b.create<tensorium::mlir::DerivOp>(loc, desiredType, arg0);
      deriv->setAttr("index", b.getStringAttr(std::string(1, c->callee[2])));
      return deriv.getResult();
    }
    if (c->callee == "contract") {
      if (c->args.empty())
        emitUnsupportedExprError(
            loc, "contract() expects exactly one argument in MLIR emission");
      auto arg0 = emitExpr(b, loc, c->args[0].get(), fieldArg, localTemps);
      return b.create<tensorium::mlir::ContractOp>(loc, desiredType, arg0)
          .getResult();
    }
    if (c->callee == "laplacian") {
      if (c->args.size() != 1) {
        emitUnsupportedExprError(
            loc, "laplacian() expects exactly one argument in MLIR emission");
      }

      auto arg0 = emitExpr(b, loc, c->args[0].get(), fieldArg, localTemps);
      auto argTy = mlir::dyn_cast<tensorium::mlir::FieldType>(arg0.getType());
      if (!argTy || argTy.getRank() != 0) {
        emitUnsupportedExprError(
            loc, "laplacian() lowering expects scalar argument");
      }
      if (desiredType.getRank() != 0) {
        emitUnsupportedExprError(
            loc, "laplacian() lowering expects scalar result type");
      }

      tensorium::ir::TensorType gradDesc;
      gradDesc.up = 0;
      gradDesc.down = 1;
      auto gradTy = asFieldType(b, gradDesc);

      tensorium::ir::TensorType hessianDesc;
      hessianDesc.up = 0;
      hessianDesc.down = 2;
      auto hessianTy = asFieldType(b, hessianDesc);

      auto firstDeriv =
          b.create<tensorium::mlir::DerivOp>(loc, gradTy, arg0);
      firstDeriv->setAttr("index", b.getStringAttr("i"));

      auto secondDeriv =
          b.create<tensorium::mlir::DerivOp>(loc, hessianTy, firstDeriv.getResult());
      secondDeriv->setAttr("index", b.getStringAttr("i"));

      auto lap =
          b.create<tensorium::mlir::ContractOp>(loc, desiredType,
                                              secondDeriv.getResult());
      lap->setAttr("sum_indices", makeIndexArrayAttr(b, {"i"}));
      return lap.getResult();
    }
    if (c->isExtern)
      emitExternLoweringError(loc, c->callee);

    emitUnsupportedExprError(
        loc, "call to '" + c->callee +
                 "' is not supported during MLIR emission");
  }
  case ExprIR::Kind::TensorProduct: {
    auto *p = static_cast<const TensorProductIR *>(e);
    auto L = emitExpr(b, loc, p->lhs.get(), fieldArg, localTemps);
    auto R = emitExpr(b, loc, p->rhs.get(), fieldArg, localTemps);
    return b.create<tensorium::mlir::MulOp>(loc, desiredType, L, R).getResult();
  }
  case ExprIR::Kind::Contraction: {
    auto *c = static_cast<const ContractionIR *>(e);
    auto in = emitExpr(b, loc, c->in.get(), fieldArg, localTemps);
    auto out = b.create<tensorium::mlir::ContractOp>(loc, desiredType, in);
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
    auto out = b.create<tensorium::mlir::ContractOp>(loc, desiredType, in);
    if (!t->tracedIndices.empty()) {
      out->setAttr("sum_indices", makeIndexArrayAttr(b, t->tracedIndices));
    }
    return out.getResult();
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *d = static_cast<const PartialDerivativeIR *>(e);
    auto in = emitExpr(b, loc, d->in.get(), fieldArg, localTemps);
    auto deriv = b.create<tensorium::mlir::DerivOp>(loc, desiredType, in);
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

    if (d->contravariant) {
      emitUnsupportedExprError(
          loc,
          "contravariant covariant derivative (nabla^) lowering is not implemented");
    }

    auto *inVar = dynamic_cast<const VarIR *>(d->in.get());
    if (!inVar || inVar->vkind != VarKind::Field) {
      emitUnsupportedExprError(
          loc, "covariant derivative lowering requires a field reference input");
    }

    auto inValue =
        emitFieldRefByName(b, loc, inVar->name, inVar->tensorIndexNames, fieldArg);
    auto partial = b.create<tensorium::mlir::DerivOp>(loc, desiredType, inValue);
    partial->setAttr("index", b.getStringAttr(d->derivIndex));

    const int rank = inVar->exprType.rank();
    if (rank == 0) {
      // For scalars, ∇_k phi == ∂_k phi.
      return partial.getResult();
    }

    if (rank != 1 || inVar->tensorIndexNames.size() != 1) {
      emitUnsupportedExprError(
          loc,
          "covariant derivative lowering currently supports rank-1 field inputs");
    }

    const bool isVector = inVar->exprType.up == 1 && inVar->exprType.down == 0;
    const bool isCovector =
        inVar->exprType.up == 0 && inVar->exprType.down == 1;
    if (!isVector && !isCovector) {
      emitUnsupportedExprError(
          loc,
          "covariant derivative lowering supports vector/covector rank-1 fields");
    }

    if (inVar->tensorIndexNames[0].empty() || d->derivIndex.empty()) {
      emitUnsupportedExprError(loc, "covariant derivative requires explicit indices");
    }

    mlir::Value connection = findConnectionFieldValue(fieldArg);
    if (!connection) {
      emitUnsupportedExprError(
          loc, "covariant derivative requires a rank-3 mixed connection field "
               "(prefer 'Christoffel' or 'Gamma')");
    }

    const std::string tensorIndex = inVar->tensorIndexNames[0];
    const std::string dummy = pickDummyIndex({tensorIndex, d->derivIndex});

    std::vector<std::string> gammaIndices;
    if (isVector) {
      // +Gamma^i_{k m} V^m
      gammaIndices = {tensorIndex, d->derivIndex, dummy};
    } else {
      // -Gamma^m_{i k} V_m
      gammaIndices = {dummy, tensorIndex, d->derivIndex};
    }

    auto gammaRef = emitFieldRefFromSource(b, loc, connection, gammaIndices);
    auto shiftedTensorRef =
        emitFieldRefByName(b, loc, inVar->name, {dummy}, fieldArg);

    tensorium::ir::TensorType productTypeDesc;
    productTypeDesc.up = 1 + inVar->exprType.up;
    productTypeDesc.down = 2 + inVar->exprType.down;
    auto productType = asFieldType(b, productTypeDesc);

    auto product = b.create<tensorium::mlir::MulOp>(loc, productType, gammaRef,
                                                   shiftedTensorRef);
    auto correction =
        b.create<tensorium::mlir::ContractOp>(loc, desiredType, product.getRes());
    correction->setAttr("sum_indices", makeIndexArrayAttr(b, {dummy}));

    if (isVector) {
      return b.create<tensorium::mlir::AddOp>(loc, desiredType,
                                            partial.getResult(),
                                            correction.getOut())
          .getResult();
    }
    return b.create<tensorium::mlir::SubOp>(loc, desiredType,
                                          partial.getResult(),
                                          correction.getOut())
        .getResult();
  }
  case ExprIR::Kind::Divergence: {
    auto *d = static_cast<const DivergenceIR *>(e);
    auto in = emitExpr(b, loc, d->in.get(), fieldArg, localTemps);
    auto out = b.create<tensorium::mlir::ContractOp>(loc, desiredType, in);
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

void emitEvolutionOps(
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
      auto fieldTy =
          mlir::dyn_cast<tensorium::mlir::FieldType>(it->second.getType());
      if (!fieldTy)
        emitUnsupportedExprError(loc, "field argument lacks tensorium.field type");
      auto rhsV = emitExpr(b, loc, eq.rhs.get(), fieldArg, &tempValues);
      if (!rhsV)
        continue;
      auto rhsTy = mlir::dyn_cast<tensorium::mlir::FieldType>(rhsV.getType());
      if (!rhsTy)
        emitUnsupportedExprError(loc,
                                 "rhs expression did not produce tensorium.field type");
      if (rhsTy.getRank() == 0) {
        rhsV = b.create<tensorium::mlir::PromoteOp>(loc, fieldTy, rhsV)
                   .getResult();
      } else if (fieldTy != rhsTy) {
        emitUnsupportedExprError(loc, "tensor assignment variance mismatch");
      }
      b.create<tensorium::mlir::DtAssignOp>(loc, it->second, rhsV,
                                          makeIndexArrayAttr(b, eq.indices));
    }
  }
}

} // namespace tensorium_mlir
