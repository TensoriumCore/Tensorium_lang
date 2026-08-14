#include "MLIRGenSpectralResidual.h"

#include "MLIRGenShared.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"
#include "tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorium_mlir {
namespace {

using tensorium::backend::BinaryIR;
using tensorium::backend::CallIR;
using tensorium::backend::ContractionIR;
using tensorium::backend::EquationIR;
using tensorium::backend::EvolutionIR;
using tensorium::backend::ExprIR;
using tensorium::backend::FieldIR;
using tensorium::backend::IndexPermuteIR;
using tensorium::backend::IndexRenameIR;
using tensorium::backend::ModuleIR;
using tensorium::backend::NumberIR;
using tensorium::backend::PartialDerivativeIR;
using tensorium::backend::SpatialScheme;
using tensorium::backend::TempAssignIR;
using tensorium::backend::TensorProductIR;
using tensorium::backend::TraceIR;
using tensorium::backend::VarIR;
using tensorium::backend::VarKind;

struct SpectralPointArgs {
  mlir::Value value;
  mlir::Value d1;
  mlir::Value d2;
  mlir::Value d3;
  mlir::Value d11;
  mlir::Value d12;
  mlir::Value d13;
  mlir::Value d22;
  mlir::Value d23;
  mlir::Value d33;
  mlir::Value x1;
  mlir::Value x2;
  mlir::Value x3;
};

struct SpectralCandidate {
  std::string target;
  std::string unknown;
  std::vector<std::string> auxiliaryFields;
};

bool isScalarField(const ModuleIR &module, const std::string &name) {
  for (const FieldIR &field : module.fields) {
    if (field.name == name)
      return field.tensorType.rank() == 0;
  }
  return false;
}

std::vector<std::string> spectralCoordNames(const ModuleIR &module) {
  using tensorium::backend::CoordSystem;
  if (module.simulation) {
    switch (module.simulation->coords) {
    case CoordSystem::Cartesian:
      return {"x", "y", "z"};
    case CoordSystem::Spherical:
      return {"r", "theta", "phi"};
    case CoordSystem::Cylindrical:
      return {"rho", "phi", "z"};
    }
  }
  return {"x", "y", "z"};
}

void setCommonABIAttrs(mlir::OpBuilder &b, mlir::func::FuncOp fn,
                       llvm::StringRef kind) {
  fn->setAttr(
      tensorium_mlir::abi::kAttrABIVersion,
      b.getI64IntegerAttr(tensorium_mlir::abi::kGeneratedKernelABIVersion));
  fn->setAttr(tensorium_mlir::abi::kAttrABIKind, b.getStringAttr(kind));
  fn->setAttr(
      tensorium_mlir::abi::kAttrMemoryLayout,
      b.getStringAttr(tensorium_mlir::abi::kMemLayoutSoAComponentMajor));
  fn->setAttr(tensorium_mlir::abi::kAttrMemrefABI,
              b.getStringAttr(tensorium_mlir::abi::kMemrefABI1DStridedF64));
}

std::unordered_map<std::string, const ExprIR *>
tempDefsFor(const EvolutionIR &evo) {
  std::unordered_map<std::string, const ExprIR *> out;
  for (const TempAssignIR &tmp : evo.temporaries)
    out[tmp.name] = tmp.rhs.get();
  return out;
}

void collectFieldNames(
    const ExprIR *expr,
    const std::unordered_map<std::string, const ExprIR *> &temps,
    std::unordered_set<std::string> &out,
    std::unordered_set<std::string> &visiting) {
  if (!expr)
    return;
  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return;
  case ExprIR::Kind::Var: {
    const auto *var = static_cast<const VarIR *>(expr);
    if (var->vkind == VarKind::Field) {
      out.insert(var->name);
      return;
    }
    if (var->vkind == VarKind::Local && visiting.insert(var->name).second) {
      auto it = temps.find(var->name);
      if (it != temps.end())
        collectFieldNames(it->second, temps, out, visiting);
      visiting.erase(var->name);
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    const auto *bin = static_cast<const BinaryIR *>(expr);
    collectFieldNames(bin->lhs.get(), temps, out, visiting);
    collectFieldNames(bin->rhs.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::Call: {
    const auto *call = static_cast<const CallIR *>(expr);
    for (const auto &arg : call->args)
      collectFieldNames(arg.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    const auto *prod = static_cast<const TensorProductIR *>(expr);
    collectFieldNames(prod->lhs.get(), temps, out, visiting);
    collectFieldNames(prod->rhs.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    const auto *deriv = static_cast<const PartialDerivativeIR *>(expr);
    collectFieldNames(deriv->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::Contraction: {
    const auto *contract = static_cast<const ContractionIR *>(expr);
    collectFieldNames(contract->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    const auto *rename = static_cast<const IndexRenameIR *>(expr);
    collectFieldNames(rename->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    const auto *permute = static_cast<const IndexPermuteIR *>(expr);
    collectFieldNames(permute->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::Trace: {
    const auto *trace = static_cast<const TraceIR *>(expr);
    collectFieldNames(trace->in.get(), temps, out, visiting);
    return;
  }
  default:
    return;
  }
}

void collectDerivativeBaseFields(
    const ExprIR *expr,
    const std::unordered_map<std::string, const ExprIR *> &temps,
    std::unordered_set<std::string> &out,
    std::unordered_set<std::string> &visiting) {
  if (!expr)
    return;
  switch (expr->kind) {
  case ExprIR::Kind::Var: {
    const auto *var = static_cast<const VarIR *>(expr);
    if (var->vkind == VarKind::Local && visiting.insert(var->name).second) {
      auto it = temps.find(var->name);
      if (it != temps.end())
        collectDerivativeBaseFields(it->second, temps, out, visiting);
      visiting.erase(var->name);
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    const auto *bin = static_cast<const BinaryIR *>(expr);
    collectDerivativeBaseFields(bin->lhs.get(), temps, out, visiting);
    collectDerivativeBaseFields(bin->rhs.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::Call: {
    const auto *call = static_cast<const CallIR *>(expr);
    if (call->callee == "laplacian" && call->args.size() == 1) {
      std::unordered_set<std::string> fields;
      std::unordered_set<std::string> nested;
      collectFieldNames(call->args[0].get(), temps, fields, nested);
      out.insert(fields.begin(), fields.end());
      return;
    }
    for (const auto &arg : call->args)
      collectDerivativeBaseFields(arg.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    const auto *prod = static_cast<const TensorProductIR *>(expr);
    collectDerivativeBaseFields(prod->lhs.get(), temps, out, visiting);
    collectDerivativeBaseFields(prod->rhs.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    const auto *deriv = static_cast<const PartialDerivativeIR *>(expr);
    std::unordered_set<std::string> fields;
    std::unordered_set<std::string> nested;
    collectFieldNames(deriv->in.get(), temps, fields, nested);
    out.insert(fields.begin(), fields.end());
    return;
  }
  case ExprIR::Kind::Contraction: {
    const auto *contract = static_cast<const ContractionIR *>(expr);
    collectDerivativeBaseFields(contract->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    const auto *rename = static_cast<const IndexRenameIR *>(expr);
    collectDerivativeBaseFields(rename->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    const auto *permute = static_cast<const IndexPermuteIR *>(expr);
    collectDerivativeBaseFields(permute->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::Trace: {
    const auto *trace = static_cast<const TraceIR *>(expr);
    collectDerivativeBaseFields(trace->in.get(), temps, out, visiting);
    return;
  }
  default:
    return;
  }
}

std::optional<std::string>
scalarFieldForExpr(const ExprIR *expr,
                   const std::unordered_map<std::string, const ExprIR *> &temps,
                   std::unordered_set<std::string> &visiting) {
  if (!expr || expr->kind != ExprIR::Kind::Var)
    return std::nullopt;
  const auto *var = static_cast<const VarIR *>(expr);
  if (var->vkind == VarKind::Field)
    return var->name;
  if (var->vkind != VarKind::Local || !visiting.insert(var->name).second)
    return std::nullopt;
  const auto it = temps.find(var->name);
  const auto result = it == temps.end()
                          ? std::optional<std::string>{}
                          : scalarFieldForExpr(it->second, temps, visiting);
  visiting.erase(var->name);
  return result;
}

std::optional<std::string> firstDerivativeBaseField(
    const ExprIR *expr,
    const std::unordered_map<std::string, const ExprIR *> &temps,
    std::unordered_set<std::string> &visiting) {
  if (!expr)
    return std::nullopt;
  switch (expr->kind) {
  case ExprIR::Kind::Var: {
    const auto *var = static_cast<const VarIR *>(expr);
    if (var->vkind != VarKind::Local || !visiting.insert(var->name).second)
      return std::nullopt;
    const auto it = temps.find(var->name);
    const auto result =
        it == temps.end()
            ? std::optional<std::string>{}
            : firstDerivativeBaseField(it->second, temps, visiting);
    visiting.erase(var->name);
    return result;
  }
  case ExprIR::Kind::Binary: {
    const auto *bin = static_cast<const BinaryIR *>(expr);
    if (auto field = firstDerivativeBaseField(bin->lhs.get(), temps, visiting))
      return field;
    return firstDerivativeBaseField(bin->rhs.get(), temps, visiting);
  }
  case ExprIR::Kind::Call: {
    const auto *call = static_cast<const CallIR *>(expr);
    if (call->callee == "laplacian" && call->args.size() == 1) {
      std::unordered_set<std::string> scalarVisiting;
      if (auto field =
              scalarFieldForExpr(call->args[0].get(), temps, scalarVisiting))
        return field;
    }
    for (const auto &arg : call->args) {
      if (auto field = firstDerivativeBaseField(arg.get(), temps, visiting))
        return field;
    }
    return std::nullopt;
  }
  case ExprIR::Kind::TensorProduct: {
    const auto *product = static_cast<const TensorProductIR *>(expr);
    if (auto field =
            firstDerivativeBaseField(product->lhs.get(), temps, visiting))
      return field;
    return firstDerivativeBaseField(product->rhs.get(), temps, visiting);
  }
  case ExprIR::Kind::PartialDerivative: {
    const ExprIR *base =
        static_cast<const PartialDerivativeIR *>(expr)->in.get();
    while (base && base->kind == ExprIR::Kind::PartialDerivative)
      base = static_cast<const PartialDerivativeIR *>(base)->in.get();
    std::unordered_set<std::string> scalarVisiting;
    return scalarFieldForExpr(base, temps, scalarVisiting);
  }
  case ExprIR::Kind::Contraction:
    return firstDerivativeBaseField(
        static_cast<const ContractionIR *>(expr)->in.get(), temps, visiting);
  case ExprIR::Kind::IndexRename:
    return firstDerivativeBaseField(
        static_cast<const IndexRenameIR *>(expr)->in.get(), temps, visiting);
  case ExprIR::Kind::IndexPermute:
    return firstDerivativeBaseField(
        static_cast<const IndexPermuteIR *>(expr)->in.get(), temps, visiting);
  case ExprIR::Kind::Trace:
    return firstDerivativeBaseField(
        static_cast<const TraceIR *>(expr)->in.get(), temps, visiting);
  default:
    return std::nullopt;
  }
}

void collectParamNames(
    const ExprIR *expr,
    const std::unordered_map<std::string, const ExprIR *> &temps,
    llvm::StringSet<> &out, std::unordered_set<std::string> &visiting) {
  if (!expr)
    return;
  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return;
  case ExprIR::Kind::Var: {
    const auto *var = static_cast<const VarIR *>(expr);
    if (var->vkind == VarKind::Param)
      out.insert(var->name);
    if (var->vkind == VarKind::Local && visiting.insert(var->name).second) {
      auto it = temps.find(var->name);
      if (it != temps.end())
        collectParamNames(it->second, temps, out, visiting);
      visiting.erase(var->name);
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    const auto *bin = static_cast<const BinaryIR *>(expr);
    collectParamNames(bin->lhs.get(), temps, out, visiting);
    collectParamNames(bin->rhs.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::Call: {
    const auto *call = static_cast<const CallIR *>(expr);
    for (const auto &arg : call->args)
      collectParamNames(arg.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    const auto *prod = static_cast<const TensorProductIR *>(expr);
    collectParamNames(prod->lhs.get(), temps, out, visiting);
    collectParamNames(prod->rhs.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    const auto *deriv = static_cast<const PartialDerivativeIR *>(expr);
    collectParamNames(deriv->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::Contraction: {
    const auto *contract = static_cast<const ContractionIR *>(expr);
    collectParamNames(contract->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    const auto *rename = static_cast<const IndexRenameIR *>(expr);
    collectParamNames(rename->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    const auto *permute = static_cast<const IndexPermuteIR *>(expr);
    collectParamNames(permute->in.get(), temps, out, visiting);
    return;
  }
  case ExprIR::Kind::Trace: {
    const auto *trace = static_cast<const TraceIR *>(expr);
    collectParamNames(trace->in.get(), temps, out, visiting);
    return;
  }
  default:
    return;
  }
}

std::vector<std::string>
sortedParamNames(const ExprIR *expr,
                 const std::unordered_map<std::string, const ExprIR *> &temps) {
  llvm::StringSet<> seen;
  std::unordered_set<std::string> visiting;
  collectParamNames(expr, temps, seen, visiting);
  std::vector<std::string> out;
  out.reserve(seen.size());
  for (const auto &entry : seen)
    out.push_back(entry.getKey().str());
  std::sort(out.begin(), out.end());
  return out;
}

std::optional<SpectralCandidate>
classifySpectralCandidate(const ModuleIR &module, const EvolutionIR &evo,
                          const EquationIR &eq) {
  if (!module.hasResidualConstraints || !module.simulation ||
      module.simulation->spatial.scheme != SpatialScheme::Spectral)
    return std::nullopt;
  if (!eq.indices.empty() || !isScalarField(module, eq.fieldName))
    return std::nullopt;

  const auto temps = tempDefsFor(evo);
  std::unordered_set<std::string> fieldNames;
  std::unordered_set<std::string> visitingFields;
  collectFieldNames(eq.rhs.get(), temps, fieldNames, visitingFields);

  std::unordered_set<std::string> derivativeFields;
  std::unordered_set<std::string> visitingDerivatives;
  collectDerivativeBaseFields(eq.rhs.get(), temps, derivativeFields,
                              visitingDerivatives);

  if (derivativeFields.empty())
    return std::nullopt;
  std::unordered_set<std::string> visitingFirstDerivative;
  const auto firstDerivative =
      firstDerivativeBaseField(eq.rhs.get(), temps, visitingFirstDerivative);
  if (!firstDerivative || !derivativeFields.count(*firstDerivative))
    return std::nullopt;
  const std::string unknown = *firstDerivative;
  if (!isScalarField(module, unknown))
    return std::nullopt;
  if (!fieldNames.count(unknown))
    return std::nullopt;
  std::vector<std::string> auxiliaryFields;
  auxiliaryFields.reserve(fieldNames.size() - 1);
  for (const std::string &fieldName : fieldNames) {
    if (fieldName == unknown)
      continue;
    if (!isScalarField(module, fieldName))
      return std::nullopt;
    auxiliaryFields.push_back(fieldName);
  }
  std::sort(auxiliaryFields.begin(), auxiliaryFields.end());
  return SpectralCandidate{eq.fieldName, unknown, std::move(auxiliaryFields)};
}

class SpectralScalarEmitter {
public:
  SpectralScalarEmitter(
      mlir::OpBuilder &b, mlir::Location loc,
      const std::unordered_map<std::string, const ExprIR *> &temps,
      const llvm::StringMap<SpectralPointArgs> &fieldArgs,
      const SpectralPointArgs &coordinateArgs,
      const llvm::StringMap<mlir::Value> &paramArgs)
      : b(b), loc(loc), temps(temps), fieldArgs(fieldArgs),
        coordinateArgs(coordinateArgs), paramArgs(paramArgs) {}

  mlir::Value emit(const ExprIR *expr) {
    if (!expr)
      emitUnsupportedExprError(loc, "null spectral residual expression");

    switch (expr->kind) {
    case ExprIR::Kind::Number: {
      const auto *num = static_cast<const NumberIR *>(expr);
      return b
          .create<mlir::arith::ConstantFloatOp>(
              loc, llvm::APFloat(num->value),
              llvm::cast<mlir::FloatType>(b.getF64Type()))
          .getResult();
    }
    case ExprIR::Kind::Var:
      return emitVar(static_cast<const VarIR *>(expr));
    case ExprIR::Kind::Binary:
      return emitBinary(static_cast<const BinaryIR *>(expr));
    case ExprIR::Kind::TensorProduct:
      return emitTensorProduct(static_cast<const TensorProductIR *>(expr));
    case ExprIR::Kind::Call:
      return emitCall(static_cast<const CallIR *>(expr));
    case ExprIR::Kind::PartialDerivative:
      return emitDerivative(static_cast<const PartialDerivativeIR *>(expr));
    case ExprIR::Kind::Contraction:
      return emitContraction(static_cast<const ContractionIR *>(expr));
    case ExprIR::Kind::IndexRename:
      return emit(static_cast<const IndexRenameIR *>(expr)->in.get());
    case ExprIR::Kind::IndexPermute:
      return emit(static_cast<const IndexPermuteIR *>(expr)->in.get());
    case ExprIR::Kind::Trace:
      return emitTrace(static_cast<const TraceIR *>(expr));
    default:
      emitUnsupportedExprError(
          loc, "spectral point residual supports scalar arithmetic, params, "
               "coords, scalar field derivative bundles, gradient "
               "contractions, and laplacian() in this pass");
    }
  }

private:
  std::optional<std::string> fieldForExpr(const ExprIR *expr) const {
    std::unordered_set<std::string> visiting;
    return scalarFieldForExpr(expr, temps, visiting);
  }

  const SpectralPointArgs &bundleForField(const std::string &field) const {
    const auto it = fieldArgs.find(field);
    if (it == fieldArgs.end()) {
      emitUnsupportedExprError(
          loc, "spectral point residual references unsupported field '" +
                   field + "'");
    }
    return it->second;
  }

  mlir::Value emitVar(const VarIR *var) {
    if (var->vkind == VarKind::Field) {
      return bundleForField(var->name).value;
    }
    if (var->vkind == VarKind::Param) {
      auto it = paramArgs.find(var->name);
      if (it == paramArgs.end()) {
        emitUnsupportedExprError(loc, "missing spectral residual parameter '" +
                                          var->name + "'");
      }
      return it->second;
    }
    if (var->vkind == VarKind::Coord) {
      const unsigned axis =
          static_cast<unsigned>(var->coordIndex >= 0 ? var->coordIndex : 0);
      return axis == 0 ? coordinateArgs.x1
                       : (axis == 1 ? coordinateArgs.x2 : coordinateArgs.x3);
    }
    if (var->vkind == VarKind::Local) {
      auto cached = localValues.find(var->name);
      if (cached != localValues.end())
        return cached->second;
      auto it = temps.find(var->name);
      if (it == temps.end()) {
        emitUnsupportedExprError(loc, "unknown spectral residual temporary '" +
                                          var->name + "'");
      }
      mlir::Value value = emit(it->second);
      localValues[var->name] = value;
      return value;
    }
    emitUnsupportedExprError(loc, "unsupported spectral residual variable");
  }

  mlir::Value emitBinary(const BinaryIR *bin) {
    mlir::Value lhs = emit(bin->lhs.get());
    mlir::Value rhs = emit(bin->rhs.get());
    if (bin->op == "+")
      return b.create<mlir::arith::AddFOp>(loc, lhs, rhs).getResult();
    if (bin->op == "-")
      return b.create<mlir::arith::SubFOp>(loc, lhs, rhs).getResult();
    if (bin->op == "*")
      return b.create<mlir::arith::MulFOp>(loc, lhs, rhs).getResult();
    if (bin->op == "/")
      return b.create<mlir::arith::DivFOp>(loc, lhs, rhs).getResult();
    emitUnsupportedExprError(loc, "unsupported spectral residual binary op");
  }

  mlir::Value emitTensorProduct(const TensorProductIR *prod) {
    return b
        .create<mlir::arith::MulFOp>(loc, emit(prod->lhs.get()),
                                     emit(prod->rhs.get()))
        .getResult();
  }

  mlir::Value emitCall(const CallIR *call) {
    if (call->callee == "laplacian") {
      if (call->args.size() != 1) {
        emitUnsupportedExprError(
            loc, "spectral laplacian() expects one scalar field");
      }
      const auto field = fieldForExpr(call->args[0].get());
      if (!field)
        emitUnsupportedExprError(loc,
                                 "spectral laplacian() expects a scalar field");
      return laplacian(bundleForField(*field));
    }

    if (call->isExtern) {
      if (call->args.size() != 1) {
        emitUnsupportedExprError(
            loc, "spectral external scalar calls currently expect one arg");
      }
      mlir::Value arg = emit(call->args[0].get());
      if (call->callee == "sqrt")
        return b.create<mlir::math::SqrtOp>(loc, arg).getResult();
      if (call->callee == "sin")
        return b.create<mlir::math::SinOp>(loc, arg).getResult();
    }

    emitUnsupportedExprError(loc, "unsupported spectral residual call '" +
                                      call->callee + "'");
  }

  mlir::Value emitDerivative(const PartialDerivativeIR *deriv) {
    if (const auto field = fieldForExpr(deriv->in.get()))
      return firstDerivative(bundleForField(*field), deriv->coordIndex);

    if (deriv->in && deriv->in->kind == ExprIR::Kind::PartialDerivative) {
      const auto *inner =
          static_cast<const PartialDerivativeIR *>(deriv->in.get());
      if (const auto field = fieldForExpr(inner->in.get())) {
        return secondDerivative(bundleForField(*field), inner->coordIndex,
                                deriv->coordIndex);
      }
    }

    emitUnsupportedExprError(loc, "spectral derivative expects a scalar field");
  }

  mlir::Value emitContraction(const ContractionIR *contract) {
    if (const auto field = laplacianContractionField(contract))
      return laplacian(bundleForField(*field));
    if (const auto fields = gradientContractionFields(contract))
      return gradientDot(bundleForField(fields->first),
                         bundleForField(fields->second));
    emitUnsupportedExprError(
        loc, "spectral contraction supports scalar laplacians and contracted "
             "scalar gradients");
  }

  mlir::Value emitTrace(const TraceIR *trace) {
    if (const auto field = repeatedSecondDerivativeField(trace->in.get()))
      return laplacian(bundleForField(*field));
    emitUnsupportedExprError(
        loc, "spectral trace currently supports laplacian via contraction()");
  }

  std::optional<std::string>
  laplacianContractionField(const ContractionIR *contract) const {
    if (!contract)
      return std::nullopt;
    const auto field = repeatedSecondDerivativeField(contract->in.get());
    if (!field)
      return std::nullopt;
    const auto *outer =
        static_cast<const PartialDerivativeIR *>(contract->in.get());
    if (!contract->summedIndices.empty() &&
        std::find(contract->summedIndices.begin(),
                  contract->summedIndices.end(),
                  outer->coordIndex) == contract->summedIndices.end()) {
      return std::nullopt;
    }
    return field;
  }

  std::optional<std::string>
  repeatedSecondDerivativeField(const ExprIR *expr) const {
    if (!expr || expr->kind != ExprIR::Kind::PartialDerivative)
      return std::nullopt;
    const auto *outer = static_cast<const PartialDerivativeIR *>(expr);
    if (!outer->in || outer->in->kind != ExprIR::Kind::PartialDerivative)
      return std::nullopt;
    const auto *inner =
        static_cast<const PartialDerivativeIR *>(outer->in.get());
    if (outer->coordIndex != inner->coordIndex)
      return std::nullopt;
    return fieldForExpr(inner->in.get());
  }

  std::optional<std::pair<std::string, std::string>>
  gradientContractionFields(const ContractionIR *contract) const {
    if (!contract || !contract->in ||
        contract->in->kind != ExprIR::Kind::TensorProduct)
      return std::nullopt;
    const auto *product =
        static_cast<const TensorProductIR *>(contract->in.get());
    if (!product->lhs || !product->rhs ||
        product->lhs->kind != ExprIR::Kind::PartialDerivative ||
        product->rhs->kind != ExprIR::Kind::PartialDerivative)
      return std::nullopt;
    const auto *lhs =
        static_cast<const PartialDerivativeIR *>(product->lhs.get());
    const auto *rhs =
        static_cast<const PartialDerivativeIR *>(product->rhs.get());
    if (lhs->coordIndex != rhs->coordIndex)
      return std::nullopt;
    if (!contract->summedIndices.empty() &&
        std::find(contract->summedIndices.begin(),
                  contract->summedIndices.end(),
                  lhs->coordIndex) == contract->summedIndices.end())
      return std::nullopt;
    const auto lhsField = fieldForExpr(lhs->in.get());
    const auto rhsField = fieldForExpr(rhs->in.get());
    if (!lhsField || !rhsField)
      return std::nullopt;
    return std::make_pair(*lhsField, *rhsField);
  }

  mlir::Value laplacian(const SpectralPointArgs &args) const {
    mlir::Value sum =
        b.create<mlir::arith::AddFOp>(loc, args.d11, args.d22).getResult();
    return b.create<mlir::arith::AddFOp>(loc, sum, args.d33).getResult();
  }

  mlir::Value gradientDot(const SpectralPointArgs &lhs,
                          const SpectralPointArgs &rhs) const {
    mlir::Value d1 =
        b.create<mlir::arith::MulFOp>(loc, lhs.d1, rhs.d1).getResult();
    mlir::Value d2 =
        b.create<mlir::arith::MulFOp>(loc, lhs.d2, rhs.d2).getResult();
    mlir::Value d3 =
        b.create<mlir::arith::MulFOp>(loc, lhs.d3, rhs.d3).getResult();
    mlir::Value sum = b.create<mlir::arith::AddFOp>(loc, d1, d2).getResult();
    return b.create<mlir::arith::AddFOp>(loc, sum, d3).getResult();
  }

  mlir::Value firstDerivative(const SpectralPointArgs &args,
                              const std::string &coord) const {
    const unsigned axis = coordToAxis(coord);
    if (axis == 0)
      return args.d1;
    if (axis == 1)
      return args.d2;
    return args.d3;
  }

  mlir::Value secondDerivative(const SpectralPointArgs &args,
                               const std::string &lhs,
                               const std::string &rhs) const {
    unsigned a = coordToAxis(lhs);
    unsigned bAxis = coordToAxis(rhs);
    if (a > bAxis)
      std::swap(a, bAxis);
    if (a == 0 && bAxis == 0)
      return args.d11;
    if (a == 0 && bAxis == 1)
      return args.d12;
    if (a == 0 && bAxis == 2)
      return args.d13;
    if (a == 1 && bAxis == 1)
      return args.d22;
    if (a == 1 && bAxis == 2)
      return args.d23;
    return args.d33;
  }

  unsigned coordToAxis(const std::string &coord) const {
    if (coord == "x" || coord == "r" || coord == "rho" || coord == "i")
      return 0;
    if (coord == "y" || coord == "theta" || coord == "j")
      return 1;
    if (coord == "z" || coord == "phi" || coord == "k")
      return 2;
    emitUnsupportedExprError(
        loc, "unsupported spectral derivative coordinate '" + coord + "'");
  }

  mlir::OpBuilder &b;
  mlir::Location loc;
  const std::unordered_map<std::string, const ExprIR *> &temps;
  const llvm::StringMap<SpectralPointArgs> &fieldArgs;
  const SpectralPointArgs &coordinateArgs;
  const llvm::StringMap<mlir::Value> &paramArgs;
  llvm::StringMap<mlir::Value> localValues;
};

struct SpectralDualValue {
  mlir::Value primal;
  mlir::Value tangent;
  bool active = false;
};

class SpectralDualEmitter {
public:
  SpectralDualEmitter(
      mlir::OpBuilder &b, mlir::Location loc,
      const std::unordered_map<std::string, const ExprIR *> &temps,
      const llvm::StringMap<SpectralPointArgs> &fieldArgs,
      const llvm::StringMap<SpectralPointArgs> &directionFieldArgs,
      const SpectralPointArgs &coordinateArgs,
      const llvm::StringMap<mlir::Value> &paramArgs)
      : b(b), loc(loc), temps(temps), fieldArgs(fieldArgs),
        directionFieldArgs(directionFieldArgs), coordinateArgs(coordinateArgs),
        paramArgs(paramArgs) {
    zero = constant(0.0);
  }

  SpectralDualValue emit(const ExprIR *expr) {
    if (!expr)
      emitUnsupportedExprError(loc, "null spectral residual JVP expression");

    switch (expr->kind) {
    case ExprIR::Kind::Number: {
      const auto *num = static_cast<const NumberIR *>(expr);
      return {constant(num->value), zero, false};
    }
    case ExprIR::Kind::Var:
      return emitVar(static_cast<const VarIR *>(expr));
    case ExprIR::Kind::Binary:
      return emitBinary(static_cast<const BinaryIR *>(expr));
    case ExprIR::Kind::TensorProduct:
      return emitProduct(static_cast<const TensorProductIR *>(expr));
    case ExprIR::Kind::Call:
      return emitCall(static_cast<const CallIR *>(expr));
    case ExprIR::Kind::PartialDerivative:
      return emitDerivative(static_cast<const PartialDerivativeIR *>(expr));
    case ExprIR::Kind::Contraction:
      return emitContraction(static_cast<const ContractionIR *>(expr));
    case ExprIR::Kind::IndexRename:
      return emit(static_cast<const IndexRenameIR *>(expr)->in.get());
    case ExprIR::Kind::IndexPermute:
      return emit(static_cast<const IndexPermuteIR *>(expr)->in.get());
    case ExprIR::Kind::Trace:
      return emitTrace(static_cast<const TraceIR *>(expr));
    default:
      emitUnsupportedExprError(
          loc, "spectral residual JVP supports scalar arithmetic, params, "
               "coords, scalar field derivative bundles, gradient "
               "contractions, and laplacian() in this pass");
    }
  }

private:
  mlir::Value constant(double value) {
    return b
        .create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat(value),
            llvm::cast<mlir::FloatType>(b.getF64Type()))
        .getResult();
  }

  std::optional<std::string> fieldForExpr(const ExprIR *expr) const {
    std::unordered_set<std::string> visiting;
    return scalarFieldForExpr(expr, temps, visiting);
  }

  const SpectralPointArgs &
  bundleForField(const llvm::StringMap<SpectralPointArgs> &bundles,
                 const std::string &field) const {
    const auto it = bundles.find(field);
    if (it == bundles.end()) {
      emitUnsupportedExprError(
          loc,
          "spectral residual JVP references unsupported field '" + field + "'");
    }
    return it->second;
  }

  SpectralDualValue emitVar(const VarIR *var) {
    if (var->vkind == VarKind::Field) {
      return {bundleForField(fieldArgs, var->name).value,
              bundleForField(directionFieldArgs, var->name).value, true};
    }
    if (var->vkind == VarKind::Param) {
      auto it = paramArgs.find(var->name);
      if (it == paramArgs.end()) {
        emitUnsupportedExprError(
            loc, "missing spectral residual JVP parameter '" + var->name + "'");
      }
      return {it->second, zero, false};
    }
    if (var->vkind == VarKind::Coord) {
      const unsigned axis =
          static_cast<unsigned>(var->coordIndex >= 0 ? var->coordIndex : 0);
      return {axis == 0 ? coordinateArgs.x1
                        : (axis == 1 ? coordinateArgs.x2 : coordinateArgs.x3),
              zero, false};
    }
    if (var->vkind == VarKind::Local) {
      auto cached = localValues.find(var->name);
      if (cached != localValues.end())
        return cached->second;
      auto it = temps.find(var->name);
      if (it == temps.end()) {
        emitUnsupportedExprError(
            loc, "unknown spectral residual JVP temporary '" + var->name + "'");
      }
      SpectralDualValue value = emit(it->second);
      localValues[var->name] = value;
      return value;
    }
    emitUnsupportedExprError(loc, "unsupported spectral residual JVP variable");
  }

  SpectralDualValue emitBinary(const BinaryIR *bin) {
    SpectralDualValue lhs = emit(bin->lhs.get());
    SpectralDualValue rhs = emit(bin->rhs.get());
    if (bin->op == "+") {
      mlir::Value primal =
          b.create<mlir::arith::AddFOp>(loc, lhs.primal, rhs.primal);
      if (!lhs.active && !rhs.active)
        return {primal, zero, false};
      if (!lhs.active)
        return {primal, rhs.tangent, true};
      if (!rhs.active)
        return {primal, lhs.tangent, true};
      return {primal,
              b.create<mlir::arith::AddFOp>(loc, lhs.tangent, rhs.tangent),
              true};
    }
    if (bin->op == "-") {
      mlir::Value primal =
          b.create<mlir::arith::SubFOp>(loc, lhs.primal, rhs.primal);
      if (!lhs.active && !rhs.active)
        return {primal, zero, false};
      if (!rhs.active)
        return {primal, lhs.tangent, true};
      if (!lhs.active) {
        return {primal, b.create<mlir::arith::NegFOp>(loc, rhs.tangent), true};
      }
      return {primal,
              b.create<mlir::arith::SubFOp>(loc, lhs.tangent, rhs.tangent),
              true};
    }
    if (bin->op == "*")
      return multiply(lhs, rhs);
    if (bin->op == "/") {
      mlir::Value primal =
          b.create<mlir::arith::DivFOp>(loc, lhs.primal, rhs.primal);
      if (!lhs.active && !rhs.active)
        return {primal, zero, false};
      if (lhs.active && !rhs.active) {
        return {primal,
                b.create<mlir::arith::DivFOp>(loc, lhs.tangent, rhs.primal),
                true};
      }
      mlir::Value denominator =
          b.create<mlir::arith::MulFOp>(loc, rhs.primal, rhs.primal);
      if (!lhs.active) {
        mlir::Value numerator =
            b.create<mlir::arith::MulFOp>(loc, lhs.primal, rhs.tangent);
        mlir::Value quotient =
            b.create<mlir::arith::DivFOp>(loc, numerator, denominator);
        return {primal, b.create<mlir::arith::NegFOp>(loc, quotient), true};
      }
      mlir::Value left =
          b.create<mlir::arith::MulFOp>(loc, lhs.tangent, rhs.primal);
      mlir::Value right =
          b.create<mlir::arith::MulFOp>(loc, lhs.primal, rhs.tangent);
      mlir::Value numerator = b.create<mlir::arith::SubFOp>(loc, left, right);
      return {primal,
              b.create<mlir::arith::DivFOp>(loc, numerator, denominator), true};
    }
    emitUnsupportedExprError(loc,
                             "unsupported spectral residual JVP binary op");
  }

  SpectralDualValue multiply(SpectralDualValue lhs, SpectralDualValue rhs) {
    mlir::Value primal =
        b.create<mlir::arith::MulFOp>(loc, lhs.primal, rhs.primal);
    if (!lhs.active && !rhs.active)
      return {primal, zero, false};
    if (lhs.active && !rhs.active) {
      return {primal,
              b.create<mlir::arith::MulFOp>(loc, lhs.tangent, rhs.primal),
              true};
    }
    if (!lhs.active) {
      return {primal,
              b.create<mlir::arith::MulFOp>(loc, lhs.primal, rhs.tangent),
              true};
    }
    mlir::Value left =
        b.create<mlir::arith::MulFOp>(loc, lhs.tangent, rhs.primal);
    mlir::Value right =
        b.create<mlir::arith::MulFOp>(loc, lhs.primal, rhs.tangent);
    return {primal, b.create<mlir::arith::AddFOp>(loc, left, right), true};
  }

  SpectralDualValue emitProduct(const TensorProductIR *prod) {
    return multiply(emit(prod->lhs.get()), emit(prod->rhs.get()));
  }

  SpectralDualValue emitCall(const CallIR *call) {
    if (call->callee == "laplacian") {
      if (call->args.size() != 1) {
        emitUnsupportedExprError(
            loc, "spectral JVP laplacian() expects one scalar field");
      }
      const auto field = fieldForExpr(call->args[0].get());
      if (!field)
        emitUnsupportedExprError(
            loc, "spectral JVP laplacian() expects a scalar field");
      return laplacian(bundleForField(fieldArgs, *field),
                       bundleForField(directionFieldArgs, *field));
    }

    if (call->isExtern) {
      if (call->args.size() != 1) {
        emitUnsupportedExprError(
            loc, "spectral external scalar JVP calls currently expect one "
                 "arg");
      }
      SpectralDualValue arg = emit(call->args[0].get());
      if (call->callee == "sqrt") {
        mlir::Value primal = b.create<mlir::math::SqrtOp>(loc, arg.primal);
        if (!arg.active)
          return {primal, zero, false};
        mlir::Value denominator =
            b.create<mlir::arith::MulFOp>(loc, constant(2.0), primal);
        return {primal,
                b.create<mlir::arith::DivFOp>(loc, arg.tangent, denominator),
                true};
      }
      if (call->callee == "sin") {
        mlir::Value primal = b.create<mlir::math::SinOp>(loc, arg.primal);
        if (!arg.active)
          return {primal, zero, false};
        mlir::Value cosine = b.create<mlir::math::CosOp>(loc, arg.primal);
        return {primal, b.create<mlir::arith::MulFOp>(loc, cosine, arg.tangent),
                true};
      }
    }

    emitUnsupportedExprError(loc, "unsupported spectral residual JVP call '" +
                                      call->callee + "'");
  }

  SpectralDualValue emitDerivative(const PartialDerivativeIR *deriv) {
    if (const auto field = fieldForExpr(deriv->in.get())) {
      return firstDerivative(bundleForField(fieldArgs, *field),
                             bundleForField(directionFieldArgs, *field),
                             deriv->coordIndex);
    }
    if (deriv->in && deriv->in->kind == ExprIR::Kind::PartialDerivative) {
      const auto *inner =
          static_cast<const PartialDerivativeIR *>(deriv->in.get());
      if (const auto field = fieldForExpr(inner->in.get())) {
        return secondDerivative(bundleForField(fieldArgs, *field),
                                bundleForField(directionFieldArgs, *field),
                                inner->coordIndex, deriv->coordIndex);
      }
    }
    emitUnsupportedExprError(loc,
                             "spectral JVP derivative expects a scalar field");
  }

  SpectralDualValue emitContraction(const ContractionIR *contract) {
    if (const auto field = laplacianContractionField(contract)) {
      return laplacian(bundleForField(fieldArgs, *field),
                       bundleForField(directionFieldArgs, *field));
    }
    if (const auto fields = gradientContractionFields(contract)) {
      return gradientDot(bundleForField(fieldArgs, fields->first),
                         bundleForField(directionFieldArgs, fields->first),
                         bundleForField(fieldArgs, fields->second),
                         bundleForField(directionFieldArgs, fields->second));
    }
    emitUnsupportedExprError(
        loc, "spectral JVP contraction supports scalar laplacians and "
             "contracted scalar gradients");
  }

  SpectralDualValue emitTrace(const TraceIR *trace) {
    if (const auto field = repeatedSecondDerivativeField(trace->in.get())) {
      return laplacian(bundleForField(fieldArgs, *field),
                       bundleForField(directionFieldArgs, *field));
    }
    emitUnsupportedExprError(
        loc, "spectral JVP trace currently supports laplacian via "
             "contraction()");
  }

  SpectralDualValue laplacian(const SpectralPointArgs &primalArgs,
                              const SpectralPointArgs &tangentArgs) {
    mlir::Value primal12 =
        b.create<mlir::arith::AddFOp>(loc, primalArgs.d11, primalArgs.d22);
    mlir::Value tangent12 =
        b.create<mlir::arith::AddFOp>(loc, tangentArgs.d11, tangentArgs.d22);
    return {b.create<mlir::arith::AddFOp>(loc, primal12, primalArgs.d33),
            b.create<mlir::arith::AddFOp>(loc, tangent12, tangentArgs.d33),
            true};
  }

  std::optional<std::string>
  laplacianContractionField(const ContractionIR *contract) const {
    if (!contract)
      return std::nullopt;
    const auto field = repeatedSecondDerivativeField(contract->in.get());
    if (!field)
      return std::nullopt;
    const auto *outer =
        static_cast<const PartialDerivativeIR *>(contract->in.get());
    if (!contract->summedIndices.empty() &&
        std::find(contract->summedIndices.begin(),
                  contract->summedIndices.end(),
                  outer->coordIndex) == contract->summedIndices.end())
      return std::nullopt;
    return field;
  }

  std::optional<std::string>
  repeatedSecondDerivativeField(const ExprIR *expr) const {
    if (!expr || expr->kind != ExprIR::Kind::PartialDerivative)
      return std::nullopt;
    const auto *outer = static_cast<const PartialDerivativeIR *>(expr);
    if (!outer->in || outer->in->kind != ExprIR::Kind::PartialDerivative)
      return std::nullopt;
    const auto *inner =
        static_cast<const PartialDerivativeIR *>(outer->in.get());
    if (outer->coordIndex != inner->coordIndex)
      return std::nullopt;
    return fieldForExpr(inner->in.get());
  }

  std::optional<std::pair<std::string, std::string>>
  gradientContractionFields(const ContractionIR *contract) const {
    if (!contract || !contract->in ||
        contract->in->kind != ExprIR::Kind::TensorProduct)
      return std::nullopt;
    const auto *product =
        static_cast<const TensorProductIR *>(contract->in.get());
    if (!product->lhs || !product->rhs ||
        product->lhs->kind != ExprIR::Kind::PartialDerivative ||
        product->rhs->kind != ExprIR::Kind::PartialDerivative)
      return std::nullopt;
    const auto *lhs =
        static_cast<const PartialDerivativeIR *>(product->lhs.get());
    const auto *rhs =
        static_cast<const PartialDerivativeIR *>(product->rhs.get());
    if (lhs->coordIndex != rhs->coordIndex)
      return std::nullopt;
    if (!contract->summedIndices.empty() &&
        std::find(contract->summedIndices.begin(),
                  contract->summedIndices.end(),
                  lhs->coordIndex) == contract->summedIndices.end())
      return std::nullopt;
    const auto lhsField = fieldForExpr(lhs->in.get());
    const auto rhsField = fieldForExpr(rhs->in.get());
    if (!lhsField || !rhsField)
      return std::nullopt;
    return std::make_pair(*lhsField, *rhsField);
  }

  mlir::Value gradientDotValues(mlir::Value lhs1, mlir::Value lhs2,
                                mlir::Value lhs3, mlir::Value rhs1,
                                mlir::Value rhs2, mlir::Value rhs3) {
    mlir::Value d1 = b.create<mlir::arith::MulFOp>(loc, lhs1, rhs1);
    mlir::Value d2 = b.create<mlir::arith::MulFOp>(loc, lhs2, rhs2);
    mlir::Value d3 = b.create<mlir::arith::MulFOp>(loc, lhs3, rhs3);
    mlir::Value sum = b.create<mlir::arith::AddFOp>(loc, d1, d2);
    return b.create<mlir::arith::AddFOp>(loc, sum, d3);
  }

  SpectralDualValue gradientDot(const SpectralPointArgs &lhs,
                                const SpectralPointArgs &lhsTangent,
                                const SpectralPointArgs &rhs,
                                const SpectralPointArgs &rhsTangent) {
    mlir::Value primal =
        gradientDotValues(lhs.d1, lhs.d2, lhs.d3, rhs.d1, rhs.d2, rhs.d3);
    mlir::Value leftTangent = gradientDotValues(
        lhsTangent.d1, lhsTangent.d2, lhsTangent.d3, rhs.d1, rhs.d2, rhs.d3);
    mlir::Value rightTangent = gradientDotValues(
        lhs.d1, lhs.d2, lhs.d3, rhsTangent.d1, rhsTangent.d2, rhsTangent.d3);
    return {primal,
            b.create<mlir::arith::AddFOp>(loc, leftTangent, rightTangent),
            true};
  }

  SpectralDualValue firstDerivative(const SpectralPointArgs &primalArgs,
                                    const SpectralPointArgs &tangentArgs,
                                    const std::string &coord) const {
    const unsigned axis = coordToAxis(coord);
    if (axis == 0)
      return {primalArgs.d1, tangentArgs.d1, true};
    if (axis == 1)
      return {primalArgs.d2, tangentArgs.d2, true};
    return {primalArgs.d3, tangentArgs.d3, true};
  }

  SpectralDualValue secondDerivative(const SpectralPointArgs &primalArgs,
                                     const SpectralPointArgs &tangentArgs,
                                     const std::string &lhs,
                                     const std::string &rhs) const {
    unsigned a = coordToAxis(lhs);
    unsigned bAxis = coordToAxis(rhs);
    if (a > bAxis)
      std::swap(a, bAxis);
    if (a == 0 && bAxis == 0)
      return {primalArgs.d11, tangentArgs.d11, true};
    if (a == 0 && bAxis == 1)
      return {primalArgs.d12, tangentArgs.d12, true};
    if (a == 0 && bAxis == 2)
      return {primalArgs.d13, tangentArgs.d13, true};
    if (a == 1 && bAxis == 1)
      return {primalArgs.d22, tangentArgs.d22, true};
    if (a == 1 && bAxis == 2)
      return {primalArgs.d23, tangentArgs.d23, true};
    return {primalArgs.d33, tangentArgs.d33, true};
  }

  unsigned coordToAxis(const std::string &coord) const {
    if (coord == "x" || coord == "r" || coord == "rho" || coord == "i")
      return 0;
    if (coord == "y" || coord == "theta" || coord == "j")
      return 1;
    if (coord == "z" || coord == "phi" || coord == "k")
      return 2;
    emitUnsupportedExprError(
        loc, "unsupported spectral JVP derivative coordinate '" + coord + "'");
  }

  mlir::OpBuilder &b;
  mlir::Location loc;
  const std::unordered_map<std::string, const ExprIR *> &temps;
  const llvm::StringMap<SpectralPointArgs> &fieldArgs;
  const llvm::StringMap<SpectralPointArgs> &directionFieldArgs;
  const SpectralPointArgs &coordinateArgs;
  const llvm::StringMap<mlir::Value> &paramArgs;
  llvm::StringMap<SpectralDualValue> localValues;
  mlir::Value zero;
};

std::string spectralSymbolFor(const std::string &target) {
  return std::string(tensorium_mlir::abi::kSymbolSpectralResidualPrefix) +
         makeHostCIdentifier(target, "residual");
}

std::string spectralJvpSymbolFor(const std::string &target) {
  return std::string(tensorium_mlir::abi::kSymbolSpectralResidualJvpPrefix) +
         makeHostCIdentifier(target, "residual");
}

std::string spectralGridSymbolFor(const std::string &target) {
  return std::string(tensorium_mlir::abi::kSymbolSpectralResidualGridPrefix) +
         makeHostCIdentifier(target, "residual");
}

mlir::ArrayAttr makeI64ArrayAttr(mlir::OpBuilder &b,
                                 const std::vector<std::int64_t> &values) {
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  attrs.reserve(values.size());
  for (std::int64_t value : values)
    attrs.push_back(b.getI64IntegerAttr(value));
  return b.getArrayAttr(attrs);
}

void emitOneSpectralResidual(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::ModuleOp moduleOp, const ModuleIR &module,
                             const EvolutionIR &evo, const EquationIR &eq,
                             const SpectralCandidate &candidate) {
  const std::string symbol = spectralSymbolFor(candidate.target);
  if (moduleOp.lookupSymbol<mlir::func::FuncOp>(symbol))
    return;

  const auto temps = tempDefsFor(evo);
  const std::vector<std::string> params = sortedParamNames(eq.rhs.get(), temps);
  const std::vector<std::string> coords = spectralCoordNames(module);
  std::vector<std::string> fields;
  fields.reserve(1 + candidate.auxiliaryFields.size());
  fields.push_back(candidate.unknown);
  fields.insert(fields.end(), candidate.auxiliaryFields.begin(),
                candidate.auxiliaryFields.end());

  mlir::Type f64 = b.getF64Type();
  llvm::SmallVector<mlir::Type, 16> argTypes;
  const std::size_t fieldCount = fields.size();
  const std::size_t coordinateBase = 10 * fieldCount;
  for (std::size_t i = 0; i < coordinateBase + 3 + params.size(); ++i)
    argTypes.push_back(f64);

  auto fn = mlir::func::FuncOp::create(
      loc, symbol, b.getFunctionType(argTypes, mlir::TypeRange{f64}));
  setCommonABIAttrs(b, fn, tensorium_mlir::abi::kKindSpectralResidualPoint);
  fn->setAttr(tensorium_mlir::abi::kAttrFieldNames,
              makeStringArrayAttr(b, fields));
  fn->setAttr(tensorium_mlir::abi::kAttrOutputNames,
              makeStringArrayAttr(b, {candidate.target}));
  fn->setAttr(tensorium_mlir::abi::kAttrCoordNames,
              makeStringArrayAttr(b, coords));
  fn->setAttr(tensorium_mlir::abi::kAttrParamNames,
              makeStringArrayAttr(b, params));
  fn->setAttr(tensorium_mlir::abi::kAttrStencilRadius, b.getI64IntegerAttr(0));

  mlir::Block *entry = fn.addEntryBlock();
  b.setInsertionPointToEnd(entry);

  llvm::StringMap<SpectralPointArgs> fieldArgs;
  for (std::size_t field = 0; field < fieldCount; ++field) {
    const std::size_t base = 10 * field;
    fieldArgs[fields[field]] = SpectralPointArgs{entry->getArgument(base),
                                                 entry->getArgument(base + 1),
                                                 entry->getArgument(base + 2),
                                                 entry->getArgument(base + 3),
                                                 entry->getArgument(base + 4),
                                                 entry->getArgument(base + 5),
                                                 entry->getArgument(base + 6),
                                                 entry->getArgument(base + 7),
                                                 entry->getArgument(base + 8),
                                                 entry->getArgument(base + 9),
                                                 {},
                                                 {},
                                                 {}};
  }
  SpectralPointArgs coordinateArgs{};
  coordinateArgs.x1 = entry->getArgument(coordinateBase);
  coordinateArgs.x2 = entry->getArgument(coordinateBase + 1);
  coordinateArgs.x3 = entry->getArgument(coordinateBase + 2);

  llvm::StringMap<mlir::Value> paramArgs;
  const std::size_t paramBase = coordinateBase + 3;
  for (std::size_t i = 0; i < params.size(); ++i)
    paramArgs[params[i]] = entry->getArgument(paramBase + i);

  SpectralScalarEmitter emitter(b, loc, temps, fieldArgs, coordinateArgs,
                                paramArgs);
  mlir::Value value = emitter.emit(eq.rhs.get());
  b.create<mlir::func::ReturnOp>(loc, value);
  moduleOp.push_back(fn);
}

void emitOneSpectralResidualJvp(mlir::OpBuilder &b, mlir::Location loc,
                                mlir::ModuleOp moduleOp, const ModuleIR &module,
                                const EvolutionIR &evo, const EquationIR &eq,
                                const SpectralCandidate &candidate) {
  const std::string symbol = spectralJvpSymbolFor(candidate.target);
  if (moduleOp.lookupSymbol<mlir::func::FuncOp>(symbol))
    return;

  const auto temps = tempDefsFor(evo);
  const std::vector<std::string> params = sortedParamNames(eq.rhs.get(), temps);
  const std::vector<std::string> coords = spectralCoordNames(module);
  std::vector<std::string> fields;
  fields.reserve(1 + candidate.auxiliaryFields.size());
  fields.push_back(candidate.unknown);
  fields.insert(fields.end(), candidate.auxiliaryFields.begin(),
                candidate.auxiliaryFields.end());

  mlir::Type f64 = b.getF64Type();
  llvm::SmallVector<mlir::Type, 32> argTypes;
  const std::size_t fieldCount = fields.size();
  const std::size_t directionBase = 10 * fieldCount;
  const std::size_t coordinateBase = 20 * fieldCount;
  const std::size_t argumentCount = coordinateBase + 3 + params.size();
  for (std::size_t i = 0; i < argumentCount; ++i)
    argTypes.push_back(f64);

  auto fn = mlir::func::FuncOp::create(
      loc, symbol, b.getFunctionType(argTypes, mlir::TypeRange{f64}));
  setCommonABIAttrs(b, fn, tensorium_mlir::abi::kKindSpectralResidualJvpPoint);
  fn->setAttr(tensorium_mlir::abi::kAttrFieldNames,
              makeStringArrayAttr(b, fields));
  fn->setAttr(tensorium_mlir::abi::kAttrOutputNames,
              makeStringArrayAttr(b, {candidate.target}));
  fn->setAttr(tensorium_mlir::abi::kAttrCoordNames,
              makeStringArrayAttr(b, coords));
  fn->setAttr(tensorium_mlir::abi::kAttrParamNames,
              makeStringArrayAttr(b, params));
  fn->setAttr(tensorium_mlir::abi::kAttrStencilRadius, b.getI64IntegerAttr(0));

  mlir::Block *entry = fn.addEntryBlock();
  b.setInsertionPointToEnd(entry);

  llvm::StringMap<SpectralPointArgs> fieldArgs;
  llvm::StringMap<SpectralPointArgs> directionFieldArgs;
  for (std::size_t field = 0; field < fieldCount; ++field) {
    const std::size_t primal = 10 * field;
    const std::size_t tangent = directionBase + 10 * field;
    fieldArgs[fields[field]] = SpectralPointArgs{entry->getArgument(primal),
                                                 entry->getArgument(primal + 1),
                                                 entry->getArgument(primal + 2),
                                                 entry->getArgument(primal + 3),
                                                 entry->getArgument(primal + 4),
                                                 entry->getArgument(primal + 5),
                                                 entry->getArgument(primal + 6),
                                                 entry->getArgument(primal + 7),
                                                 entry->getArgument(primal + 8),
                                                 entry->getArgument(primal + 9),
                                                 {},
                                                 {},
                                                 {}};
    directionFieldArgs[fields[field]] =
        SpectralPointArgs{entry->getArgument(tangent),
                          entry->getArgument(tangent + 1),
                          entry->getArgument(tangent + 2),
                          entry->getArgument(tangent + 3),
                          entry->getArgument(tangent + 4),
                          entry->getArgument(tangent + 5),
                          entry->getArgument(tangent + 6),
                          entry->getArgument(tangent + 7),
                          entry->getArgument(tangent + 8),
                          entry->getArgument(tangent + 9),
                          {},
                          {},
                          {}};
  }
  SpectralPointArgs coordinateArgs{};
  coordinateArgs.x1 = entry->getArgument(coordinateBase);
  coordinateArgs.x2 = entry->getArgument(coordinateBase + 1);
  coordinateArgs.x3 = entry->getArgument(coordinateBase + 2);

  llvm::StringMap<mlir::Value> paramArgs;
  const std::size_t paramBase = coordinateBase + 3;
  for (std::size_t i = 0; i < params.size(); ++i)
    paramArgs[params[i]] = entry->getArgument(paramBase + i);

  SpectralDualEmitter emitter(b, loc, temps, fieldArgs, directionFieldArgs,
                              coordinateArgs, paramArgs);
  SpectralDualValue result = emitter.emit(eq.rhs.get());
  b.create<mlir::func::ReturnOp>(loc, result.tangent);
  moduleOp.push_back(fn);
}

void emitOneSpectralResidualGrid(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::ModuleOp moduleOp,
                                 const ModuleIR &module, const EvolutionIR &evo,
                                 const EquationIR &eq,
                                 const SpectralCandidate &candidate) {
  const std::string pointSymbol = spectralSymbolFor(candidate.target);
  const std::string gridSymbol = spectralGridSymbolFor(candidate.target);
  if (moduleOp.lookupSymbol<mlir::func::FuncOp>(gridSymbol))
    return;
  if (!moduleOp.lookupSymbol<mlir::func::FuncOp>(pointSymbol))
    return;

  const auto temps = tempDefsFor(evo);
  const std::vector<std::string> params = sortedParamNames(eq.rhs.get(), temps);
  const std::vector<std::string> coords = spectralCoordNames(module);
  std::vector<std::string> fields;
  fields.reserve(1 + candidate.auxiliaryFields.size());
  fields.push_back(candidate.unknown);
  fields.insert(fields.end(), candidate.auxiliaryFields.begin(),
                candidate.auxiliaryFields.end());

  mlir::Type indexType = b.getIndexType();
  mlir::Type f64 = b.getF64Type();
  mlir::Type memrefF64 =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, f64);
  llvm::SmallVector<mlir::Type, 24> argTypes;
  argTypes.push_back(indexType);
  for (std::size_t i = 0; i < params.size(); ++i)
    argTypes.push_back(f64);
  const std::size_t fieldCount = fields.size();
  for (std::size_t i = 0; i < 10 * fieldCount + coords.size() + 1; ++i)
    argTypes.push_back(memrefF64);

  auto fn = mlir::func::FuncOp::create(
      loc, gridSymbol, b.getFunctionType(argTypes, mlir::TypeRange{}));
  setCommonABIAttrs(b, fn, tensorium_mlir::abi::kKindSpectralResidualGrid);
  fn->setAttr(tensorium_mlir::abi::kAttrFieldNames,
              makeStringArrayAttr(b, fields));
  fn->setAttr(tensorium_mlir::abi::kAttrOutputNames,
              makeStringArrayAttr(b, {candidate.target}));
  fn->setAttr(tensorium_mlir::abi::kAttrCoordNames,
              makeStringArrayAttr(b, coords));
  fn->setAttr(tensorium_mlir::abi::kAttrParamNames,
              makeStringArrayAttr(b, params));
  fn->setAttr(tensorium_mlir::abi::kAttrStencilRadius, b.getI64IntegerAttr(0));

  const std::int64_t derivativeBase =
      1 + static_cast<std::int64_t>(params.size());
  const std::int64_t coordBase =
      derivativeBase + static_cast<std::int64_t>(10 * fieldCount);
  const std::int64_t outputArg =
      coordBase + static_cast<std::int64_t>(coords.size());
  std::vector<std::int64_t> readArgIndices;
  for (std::int64_t i = derivativeBase; i < outputArg; ++i)
    readArgIndices.push_back(i);
  fn->setAttr(tensorium_mlir::abi::kAttrReadArgIndices,
              makeI64ArrayAttr(b, readArgIndices));
  fn->setAttr(tensorium_mlir::abi::kAttrWriteArgIndices,
              makeI64ArrayAttr(b, {outputArg}));

  mlir::Block *entry = fn.addEntryBlock();
  b.setInsertionPointToEnd(entry);
  mlir::Value nPoints = entry->getArgument(0);

  llvm::SmallVector<mlir::Value, 8> paramValues;
  paramValues.reserve(params.size());
  for (std::size_t i = 0; i < params.size(); ++i)
    paramValues.push_back(entry->getArgument(1 + i));

  llvm::SmallVector<mlir::Value, 20> derivativeBuffers;
  for (std::size_t i = 0; i < 10 * fieldCount; ++i)
    derivativeBuffers.push_back(entry->getArgument(derivativeBase + i));

  llvm::SmallVector<mlir::Value, 3> coordBuffers;
  for (std::size_t i = 0; i < coords.size(); ++i)
    coordBuffers.push_back(entry->getArgument(coordBase + i));
  while (coordBuffers.size() < 3)
    coordBuffers.push_back(entry->getArgument(coordBase + coords.size() - 1));
  mlir::Value outputBuffer = entry->getArgument(outputArg);

  mlir::Value c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<mlir::scf::ForOp>(loc, c0, nPoints, c1);
  b.setInsertionPointToStart(loop.getBody());
  mlir::Value p = loop.getInductionVar();

  llvm::SmallVector<mlir::Value, 24> callArgs;
  for (mlir::Value buffer : derivativeBuffers)
    callArgs.push_back(
        b.create<mlir::memref::LoadOp>(loc, buffer, mlir::ValueRange{p}));
  for (std::size_t i = 0; i < 3; ++i)
    callArgs.push_back(b.create<mlir::memref::LoadOp>(loc, coordBuffers[i],
                                                      mlir::ValueRange{p}));
  callArgs.append(paramValues.begin(), paramValues.end());

  auto result = b.create<mlir::func::CallOp>(loc, pointSymbol,
                                             mlir::TypeRange{f64}, callArgs);
  b.create<mlir::memref::StoreOp>(loc, result.getResult(0), outputBuffer,
                                  mlir::ValueRange{p});

  b.setInsertionPointAfter(loop);
  b.create<mlir::func::ReturnOp>(loc);
  moduleOp.push_back(fn);
}

} // namespace

void emitSpectralResidualKernels(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::ModuleOp moduleOp,
                                 const ModuleIR &module) {
  if (!module.hasResidualConstraints || !module.simulation ||
      module.simulation->spatial.scheme != SpatialScheme::Spectral)
    return;

  for (const EvolutionIR &evo : module.evolutions) {
    for (const EquationIR &eq : evo.equations) {
      auto candidate = classifySpectralCandidate(module, evo, eq);
      if (!candidate)
        continue;
      emitOneSpectralResidual(b, loc, moduleOp, module, evo, eq, *candidate);
      emitOneSpectralResidualJvp(b, loc, moduleOp, module, evo, eq, *candidate);
      emitOneSpectralResidualGrid(b, loc, moduleOp, module, evo, eq,
                                  *candidate);
    }
  }
}

} // namespace tensorium_mlir
