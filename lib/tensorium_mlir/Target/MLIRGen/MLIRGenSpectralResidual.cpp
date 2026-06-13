#include "MLIRGenSpectralResidual.h"

#include "MLIRGenShared.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"
#include "tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
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

std::string spectralDerivedAuxName(const std::string &derivative,
                                   const std::string &field) {
  return "__spectral_deriv_" + derivative + "_" + field;
}

bool parseSpectralDerivedAuxName(const std::string &name,
                                 std::string &derivative,
                                 std::string &field) {
  const std::string prefix = "__spectral_deriv_";
  if (name.rfind(prefix, 0) != 0)
    return false;
  const std::size_t derivStart = prefix.size();
  const std::size_t split = name.find('_', derivStart);
  if (split == std::string::npos)
    return false;
  derivative = name.substr(derivStart, split - derivStart);
  field = name.substr(split + 1);
  return !derivative.empty() && !field.empty();
}

std::string spectralAuxBaseFieldName(const std::string &name) {
  std::string derivative;
  std::string field;
  if (parseSpectralDerivedAuxName(name, derivative, field))
    return field;
  return name;
}

bool parseYorkVectorLaplacianCall(const std::string &callee,
                                  std::string &base, int &component) {
  const std::string prefix = "york_vector_laplacian_";
  if (callee.rfind(prefix, 0) != 0)
    return false;
  const std::string rest = callee.substr(prefix.size());
  const std::size_t split = rest.rfind('_');
  if (split == std::string::npos || split + 1 >= rest.size())
    return false;
  base = rest.substr(0, split);
  try {
    component = std::stoi(rest.substr(split + 1));
  } catch (...) {
    return false;
  }
  return component >= 0 && component < 3 && !base.empty();
}

std::string secondDerivativeNameForAxes(int a, int b) {
  if (a > b)
    std::swap(a, b);
  if (a == 0 && b == 0)
    return "d11";
  if (a == 0 && b == 1)
    return "d12";
  if (a == 0 && b == 2)
    return "d13";
  if (a == 1 && b == 1)
    return "d22";
  if (a == 1 && b == 2)
    return "d23";
  return "d33";
}

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
  fn->setAttr(tensorium_mlir::abi::kAttrABIVersion,
              b.getI64IntegerAttr(
                  tensorium_mlir::abi::kGeneratedKernelABIVersion));
  fn->setAttr(tensorium_mlir::abi::kAttrABIKind, b.getStringAttr(kind));
  fn->setAttr(tensorium_mlir::abi::kAttrMemoryLayout,
              b.getStringAttr(
                  tensorium_mlir::abi::kMemLayoutSoAComponentMajor));
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

void collectFieldNames(const ExprIR *expr,
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
    const ExprIR *expr, const std::unordered_map<std::string, const ExprIR *> &temps,
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
    if ((call->callee == "laplacian" ||
         call->callee.rfind("york_vector_laplacian_diag_", 0) == 0) &&
        call->args.size() == 1) {
      std::unordered_set<std::string> fields;
      std::unordered_set<std::string> nested;
      collectFieldNames(call->args[0].get(), temps, fields, nested);
      out.insert(fields.begin(), fields.end());
      return;
    }
    std::string yorkBase;
    int yorkComponent = -1;
    if (parseYorkVectorLaplacianCall(call->callee, yorkBase, yorkComponent) &&
        call->args.size() == 1) {
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

void collectParamNames(const ExprIR *expr,
                       const std::unordered_map<std::string, const ExprIR *> &temps,
                       llvm::StringSet<> &out,
                       std::unordered_set<std::string> &visiting) {
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

void collectSpectralDerivedAuxNames(
    const ExprIR *expr, const std::unordered_map<std::string, const ExprIR *> &temps,
    std::vector<std::string> &out, std::unordered_set<std::string> &seen,
    std::unordered_set<std::string> &visiting) {
  if (!expr)
    return;
  switch (expr->kind) {
  case ExprIR::Kind::Var: {
    const auto *var = static_cast<const VarIR *>(expr);
    if (var->vkind == VarKind::Local && visiting.insert(var->name).second) {
      auto it = temps.find(var->name);
      if (it != temps.end())
        collectSpectralDerivedAuxNames(it->second, temps, out, seen, visiting);
      visiting.erase(var->name);
    }
    return;
  }
  case ExprIR::Kind::Binary: {
    const auto *bin = static_cast<const BinaryIR *>(expr);
    collectSpectralDerivedAuxNames(bin->lhs.get(), temps, out, seen, visiting);
    collectSpectralDerivedAuxNames(bin->rhs.get(), temps, out, seen, visiting);
    return;
  }
  case ExprIR::Kind::Call: {
    const auto *call = static_cast<const CallIR *>(expr);
    std::string base;
    int component = -1;
    if (parseYorkVectorLaplacianCall(call->callee, base, component)) {
      for (int sibling = 0; sibling < 3; ++sibling) {
        if (sibling == component)
          continue;
        const std::string name = spectralDerivedAuxName(
            secondDerivativeNameForAxes(component, sibling),
            base + std::to_string(sibling + 1));
        if (seen.insert(name).second)
          out.push_back(name);
      }
    }
    for (const auto &arg : call->args)
      collectSpectralDerivedAuxNames(arg.get(), temps, out, seen, visiting);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    const auto *prod = static_cast<const TensorProductIR *>(expr);
    collectSpectralDerivedAuxNames(prod->lhs.get(), temps, out, seen, visiting);
    collectSpectralDerivedAuxNames(prod->rhs.get(), temps, out, seen, visiting);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    const auto *deriv = static_cast<const PartialDerivativeIR *>(expr);
    collectSpectralDerivedAuxNames(deriv->in.get(), temps, out, seen, visiting);
    return;
  }
  case ExprIR::Kind::Contraction: {
    const auto *contract = static_cast<const ContractionIR *>(expr);
    collectSpectralDerivedAuxNames(contract->in.get(), temps, out, seen, visiting);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    const auto *rename = static_cast<const IndexRenameIR *>(expr);
    collectSpectralDerivedAuxNames(rename->in.get(), temps, out, seen, visiting);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    const auto *permute = static_cast<const IndexPermuteIR *>(expr);
    collectSpectralDerivedAuxNames(permute->in.get(), temps, out, seen, visiting);
    return;
  }
  case ExprIR::Kind::Trace: {
    const auto *trace = static_cast<const TraceIR *>(expr);
    collectSpectralDerivedAuxNames(trace->in.get(), temps, out, seen, visiting);
    return;
  }
  default:
    return;
  }
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

  std::string unknown;
  if (!eq.unknownFieldName.empty()) {
    unknown = eq.unknownFieldName;
    if (derivativeFields.empty()) {
      if (!fieldNames.count(unknown))
        return std::nullopt;
    } else if (!derivativeFields.count(unknown)) {
      return std::nullopt;
    }
    for (const std::string &fieldName : derivativeFields) {
      if (fieldName != unknown)
        return std::nullopt;
    }
  } else {
    if (derivativeFields.size() != 1)
      return std::nullopt;
    unknown = *derivativeFields.begin();
  }
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
  std::unordered_set<std::string> seenAux(auxiliaryFields.begin(),
                                          auxiliaryFields.end());
  std::unordered_set<std::string> visitingAux;
  collectSpectralDerivedAuxNames(eq.rhs.get(), temps, auxiliaryFields,
                                 seenAux, visitingAux);
  std::sort(auxiliaryFields.begin(), auxiliaryFields.end());
  for (const std::string &auxiliary : auxiliaryFields) {
    if (!isScalarField(module, spectralAuxBaseFieldName(auxiliary)))
      return std::nullopt;
  }
  return SpectralCandidate{eq.fieldName, unknown, std::move(auxiliaryFields)};
}

class SpectralScalarEmitter {
public:
  SpectralScalarEmitter(mlir::OpBuilder &b, mlir::Location loc,
                        std::string unknown,
                        const std::unordered_map<std::string, const ExprIR *> &temps,
                        const SpectralPointArgs &pointArgs,
                        const llvm::StringMap<mlir::Value> &auxiliaryArgs,
                        const llvm::StringMap<mlir::Value> &paramArgs)
      : b(b), loc(loc), unknown(std::move(unknown)), temps(temps),
        pointArgs(pointArgs), auxiliaryArgs(auxiliaryArgs),
        paramArgs(paramArgs) {}

  mlir::Value emit(const ExprIR *expr) {
    if (!expr)
      emitUnsupportedExprError(loc, "null spectral residual expression");

    switch (expr->kind) {
    case ExprIR::Kind::Number: {
      const auto *num = static_cast<const NumberIR *>(expr);
      return b.create<mlir::arith::ConstantFloatOp>(
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
               "coords, field value, d_*, and laplacian() in this pass");
    }
  }

private:
  bool isUnknownExpr(const ExprIR *expr) const {
    if (!expr)
      return false;
    if (expr->kind == ExprIR::Kind::Var) {
      const auto *var = static_cast<const VarIR *>(expr);
      if (var->vkind == VarKind::Field)
        return var->name == unknown;
      if (var->vkind == VarKind::Local) {
        auto it = temps.find(var->name);
        return it != temps.end() && isUnknownExpr(it->second);
      }
    }
    return false;
  }

  mlir::Value emitVar(const VarIR *var) {
    if (var->vkind == VarKind::Field) {
      if (var->name != unknown) {
        auto it = auxiliaryArgs.find(var->name);
        if (it == auxiliaryArgs.end()) {
          emitUnsupportedExprError(
              loc, "spectral point residual references unsupported field '" +
                       var->name + "'");
        }
        return it->second;
      }
      return pointArgs.value;
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
      const unsigned axis = static_cast<unsigned>(
          var->coordIndex >= 0 ? var->coordIndex : 0);
      return axis == 0 ? pointArgs.x1 : (axis == 1 ? pointArgs.x2 : pointArgs.x3);
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
    return b.create<mlir::arith::MulFOp>(loc, emit(prod->lhs.get()),
                                         emit(prod->rhs.get()))
        .getResult();
  }

  mlir::Value emitCall(const CallIR *call) {
    if (call->callee == "laplacian") {
      if (call->args.size() != 1 || !isUnknownExpr(call->args[0].get())) {
        emitUnsupportedExprError(
            loc, "spectral laplacian() currently expects the scalar unknown");
      }
      mlir::Value sum =
          b.create<mlir::arith::AddFOp>(loc, pointArgs.d11, pointArgs.d22)
              .getResult();
      return b.create<mlir::arith::AddFOp>(loc, sum, pointArgs.d33)
          .getResult();
    }

    if (call->callee.rfind("york_vector_laplacian_diag_", 0) == 0) {
      if (call->args.size() != 1 || !isUnknownExpr(call->args[0].get())) {
        emitUnsupportedExprError(
            loc, "spectral york_vector_laplacian_diag() expects the scalarized unknown");
      }
      const std::string axis = call->callee.substr(
          std::string("york_vector_laplacian_diag_").size());
      mlir::Value lap =
          b.create<mlir::arith::AddFOp>(
               loc,
               b.create<mlir::arith::AddFOp>(loc, pointArgs.d11, pointArgs.d22)
                   .getResult(),
               pointArgs.d33)
              .getResult();
      mlir::Value oneThird =
          b.create<mlir::arith::ConstantFloatOp>(
               loc, llvm::APFloat(1.0 / 3.0),
               llvm::cast<mlir::FloatType>(b.getF64Type()))
              .getResult();
      mlir::Value diagonal =
          b.create<mlir::arith::MulFOp>(
               loc, oneThird, secondDerivative(axis, axis))
              .getResult();
      return b.create<mlir::arith::AddFOp>(loc, lap, diagonal).getResult();
    }

    std::string yorkBase;
    int yorkComponent = -1;
    if (parseYorkVectorLaplacianCall(call->callee, yorkBase, yorkComponent)) {
      if (call->args.size() != 1 || !isUnknownExpr(call->args[0].get())) {
        emitUnsupportedExprError(
            loc, "spectral york_vector_laplacian() expects the scalarized unknown");
      }
      mlir::Value lap =
          b.create<mlir::arith::AddFOp>(
               loc,
               b.create<mlir::arith::AddFOp>(loc, pointArgs.d11, pointArgs.d22)
                   .getResult(),
               pointArgs.d33)
              .getResult();
      const std::string axisCoords[3] = {"i", "j", "k"};
      mlir::Value divGrad = secondDerivative(
          axisCoords[yorkComponent], axisCoords[yorkComponent]);
      for (int sibling = 0; sibling < 3; ++sibling) {
        if (sibling == yorkComponent)
          continue;
        const std::string auxName = spectralDerivedAuxName(
            secondDerivativeNameForAxes(yorkComponent, sibling),
            yorkBase + std::to_string(sibling + 1));
        auto it = auxiliaryArgs.find(auxName);
        if (it == auxiliaryArgs.end()) {
          emitUnsupportedExprError(
              loc, "missing spectral auxiliary derivative '" + auxName + "'");
        }
        divGrad = b.create<mlir::arith::AddFOp>(loc, divGrad, it->second)
                      .getResult();
      }
      mlir::Value oneThird =
          b.create<mlir::arith::ConstantFloatOp>(
               loc, llvm::APFloat(1.0 / 3.0),
               llvm::cast<mlir::FloatType>(b.getF64Type()))
              .getResult();
      mlir::Value scaled =
          b.create<mlir::arith::MulFOp>(loc, oneThird, divGrad).getResult();
      return b.create<mlir::arith::AddFOp>(loc, lap, scaled).getResult();
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
    if (isUnknownExpr(deriv->in.get()))
      return firstDerivative(deriv->coordIndex);

    if (deriv->in && deriv->in->kind == ExprIR::Kind::PartialDerivative) {
      const auto *inner =
          static_cast<const PartialDerivativeIR *>(deriv->in.get());
      if (isUnknownExpr(inner->in.get()))
        return secondDerivative(inner->coordIndex, deriv->coordIndex);
    }

    emitUnsupportedExprError(
        loc, "spectral derivative currently expects the scalar unknown");
  }

  mlir::Value emitContraction(const ContractionIR *contract) {
    if (isLaplacianContraction(contract)) {
      mlir::Value sum =
          b.create<mlir::arith::AddFOp>(loc, pointArgs.d11, pointArgs.d22)
              .getResult();
      return b.create<mlir::arith::AddFOp>(loc, sum, pointArgs.d33)
          .getResult();
    }
    emitUnsupportedExprError(
        loc, "spectral contraction currently supports scalar laplacian form");
  }

  mlir::Value emitTrace(const TraceIR *trace) {
    if (isRepeatedSecondDerivativeUnknown(trace->in.get())) {
      mlir::Value sum =
          b.create<mlir::arith::AddFOp>(loc, pointArgs.d11, pointArgs.d22)
              .getResult();
      return b.create<mlir::arith::AddFOp>(loc, sum, pointArgs.d33)
          .getResult();
    }
    emitUnsupportedExprError(
        loc, "spectral trace currently supports laplacian via contraction()");
  }

  bool isLaplacianContraction(const ContractionIR *contract) const {
    if (!contract || !isRepeatedSecondDerivativeUnknown(contract->in.get()))
      return false;
    const auto *outer =
        static_cast<const PartialDerivativeIR *>(contract->in.get());
    if (!contract->summedIndices.empty() &&
        std::find(contract->summedIndices.begin(), contract->summedIndices.end(),
                  outer->coordIndex) == contract->summedIndices.end()) {
      return false;
    }
    return true;
  }

  bool isRepeatedSecondDerivativeUnknown(const ExprIR *expr) const {
    if (!expr || expr->kind != ExprIR::Kind::PartialDerivative)
      return false;
    const auto *outer =
        static_cast<const PartialDerivativeIR *>(expr);
    if (!outer->in || outer->in->kind != ExprIR::Kind::PartialDerivative)
      return false;
    const auto *inner =
        static_cast<const PartialDerivativeIR *>(outer->in.get());
    if (outer->coordIndex != inner->coordIndex)
      return false;
    return isUnknownExpr(inner->in.get());
  }

  mlir::Value firstDerivative(const std::string &coord) const {
    const unsigned axis = coordToAxis(coord);
    if (axis == 0)
      return pointArgs.d1;
    if (axis == 1)
      return pointArgs.d2;
    return pointArgs.d3;
  }

  mlir::Value secondDerivative(const std::string &lhs,
                               const std::string &rhs) const {
    unsigned a = coordToAxis(lhs);
    unsigned bAxis = coordToAxis(rhs);
    if (a > bAxis)
      std::swap(a, bAxis);
    if (a == 0 && bAxis == 0)
      return pointArgs.d11;
    if (a == 0 && bAxis == 1)
      return pointArgs.d12;
    if (a == 0 && bAxis == 2)
      return pointArgs.d13;
    if (a == 1 && bAxis == 1)
      return pointArgs.d22;
    if (a == 1 && bAxis == 2)
      return pointArgs.d23;
    return pointArgs.d33;
  }

  unsigned coordToAxis(const std::string &coord) const {
    if (coord == "x" || coord == "r" || coord == "rho" || coord == "i")
      return 0;
    if (coord == "y" || coord == "theta" || coord == "j")
      return 1;
    if (coord == "z" || coord == "phi" || coord == "k")
      return 2;
    emitUnsupportedExprError(loc, "unsupported spectral derivative coordinate '" +
                                      coord + "'");
  }

  mlir::OpBuilder &b;
  mlir::Location loc;
  std::string unknown;
  const std::unordered_map<std::string, const ExprIR *> &temps;
  const SpectralPointArgs &pointArgs;
  const llvm::StringMap<mlir::Value> &auxiliaryArgs;
  const llvm::StringMap<mlir::Value> &paramArgs;
  llvm::StringMap<mlir::Value> localValues;
};

std::string spectralSymbolFor(const std::string &target) {
  return std::string(tensorium_mlir::abi::kSymbolSpectralResidualPrefix) +
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
  for (unsigned i = 0; i < 13 + candidate.auxiliaryFields.size() +
                               params.size(); ++i)
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

  SpectralPointArgs pointArgs{entry->getArgument(0), entry->getArgument(1),
                              entry->getArgument(2), entry->getArgument(3),
                              entry->getArgument(4), entry->getArgument(5),
                              entry->getArgument(6), entry->getArgument(7),
                              entry->getArgument(8), entry->getArgument(9),
                              entry->getArgument(
                                  10 + candidate.auxiliaryFields.size()),
                              entry->getArgument(
                                  11 + candidate.auxiliaryFields.size()),
                              entry->getArgument(
                                  12 + candidate.auxiliaryFields.size())};
  llvm::StringMap<mlir::Value> auxiliaryArgs;
  for (std::size_t i = 0; i < candidate.auxiliaryFields.size(); ++i)
    auxiliaryArgs[candidate.auxiliaryFields[i]] = entry->getArgument(10 + i);

  llvm::StringMap<mlir::Value> paramArgs;
  const std::size_t paramBase = 13 + candidate.auxiliaryFields.size();
  for (std::size_t i = 0; i < params.size(); ++i)
    paramArgs[params[i]] = entry->getArgument(paramBase + i);

  SpectralScalarEmitter emitter(b, loc, candidate.unknown, temps, pointArgs,
                                auxiliaryArgs, paramArgs);
  mlir::Value value = emitter.emit(eq.rhs.get());
  b.create<mlir::func::ReturnOp>(loc, value);
  moduleOp.push_back(fn);
}

void emitOneSpectralResidualGrid(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::ModuleOp moduleOp,
                                 const ModuleIR &module,
                                 const EvolutionIR &evo,
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
  for (std::size_t i = 0;
       i < 10 + candidate.auxiliaryFields.size() + coords.size() + 1; ++i)
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
  const std::int64_t auxiliaryBase = derivativeBase + 10;
  const std::int64_t coordBase =
      auxiliaryBase +
      static_cast<std::int64_t>(candidate.auxiliaryFields.size());
  const std::int64_t outputArg = coordBase + static_cast<std::int64_t>(coords.size());
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

  llvm::SmallVector<mlir::Value, 10> derivativeBuffers;
  for (std::size_t i = 0; i < 10; ++i)
    derivativeBuffers.push_back(entry->getArgument(derivativeBase + i));

  llvm::SmallVector<mlir::Value, 4> auxiliaryBuffers;
  for (std::size_t i = 0; i < candidate.auxiliaryFields.size(); ++i)
    auxiliaryBuffers.push_back(entry->getArgument(auxiliaryBase + i));

  llvm::SmallVector<mlir::Value, 3> coordBuffers;
  for (std::size_t i = 0; i < coords.size(); ++i)
    coordBuffers.push_back(entry->getArgument(coordBase + i));
  while (coordBuffers.size() < 3)
    coordBuffers.push_back(entry->getArgument(coordBase + coords.size() - 1));
  mlir::Value outputBuffer = entry->getArgument(outputArg);

  mlir::Value c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto parallel = b.create<mlir::scf::ParallelOp>(
      loc, llvm::SmallVector<mlir::Value, 1>{c0},
      llvm::SmallVector<mlir::Value, 1>{nPoints},
      llvm::SmallVector<mlir::Value, 1>{c1},
      [&](mlir::OpBuilder &ib, mlir::Location nestedLoc,
          mlir::ValueRange ivs) {
        mlir::Value p = ivs.front();

        llvm::SmallVector<mlir::Value, 24> callArgs;
        for (mlir::Value buffer : derivativeBuffers) {
          callArgs.push_back(ib.create<mlir::memref::LoadOp>(
              nestedLoc, buffer, mlir::ValueRange{p}));
        }
        for (mlir::Value buffer : auxiliaryBuffers) {
          callArgs.push_back(ib.create<mlir::memref::LoadOp>(
              nestedLoc, buffer, mlir::ValueRange{p}));
        }
        for (std::size_t i = 0; i < 3; ++i) {
          callArgs.push_back(ib.create<mlir::memref::LoadOp>(
              nestedLoc, coordBuffers[i], mlir::ValueRange{p}));
        }
        callArgs.append(paramValues.begin(), paramValues.end());

        auto result = ib.create<mlir::func::CallOp>(
            nestedLoc, pointSymbol, mlir::TypeRange{f64}, callArgs);
        ib.create<mlir::memref::StoreOp>(nestedLoc, result.getResult(0),
                                         outputBuffer, mlir::ValueRange{p});
      });

  b.setInsertionPointAfter(parallel);
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
      emitOneSpectralResidualGrid(b, loc, moduleOp, module, evo, eq,
                                  *candidate);
    }
  }
}

} // namespace tensorium_mlir
