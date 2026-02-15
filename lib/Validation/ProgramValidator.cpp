#include "tensorium/Core/IndexSet.h"
#include "tensorium/Validation/ProgramValidator.hpp"
#include <unordered_map>
#include <unordered_set>

using namespace tensorium;
using namespace tensorium::backend;
using namespace tensorium::validation;

static int fieldRank(const FieldIR &f) {
  return f.tensorType.rank();
}

static void validateLocalUseOrder(const ExprIR *expr,
                                  const std::unordered_set<std::string> &defined,
                                  ValidationResult &res,
                                  const std::string &evolutionName) {
  if (!expr)
    return;

  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return;

  case ExprIR::Kind::Var: {
    auto *var = static_cast<const VarIR *>(expr);
    if (var->vkind == VarKind::Local && !defined.count(var->name)) {
      res.diags.push_back(
          {Diagnostic::Kind::Error,
           "temporary '" + var->name + "' referenced before definition in "
           "evolution '" + evolutionName + "'"});
    }
    return;
  }

  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const BinaryIR *>(expr);
    validateLocalUseOrder(bin->lhs.get(), defined, res, evolutionName);
    validateLocalUseOrder(bin->rhs.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::Call: {
    auto *call = static_cast<const CallIR *>(expr);
    for (const auto &arg : call->args)
      validateLocalUseOrder(arg.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<const TensorProductIR *>(expr);
    validateLocalUseOrder(prod->lhs.get(), defined, res, evolutionName);
    validateLocalUseOrder(prod->rhs.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::Contraction: {
    auto *contract = static_cast<const ContractionIR *>(expr);
    validateLocalUseOrder(contract->in.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<const IndexRenameIR *>(expr);
    validateLocalUseOrder(rename->in.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::IndexPermute: {
    auto *permute = static_cast<const IndexPermuteIR *>(expr);
    validateLocalUseOrder(permute->in.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<const TraceIR *>(expr);
    validateLocalUseOrder(trace->in.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::PartialDerivative: {
    auto *partial = static_cast<const PartialDerivativeIR *>(expr);
    validateLocalUseOrder(partial->in.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::Gradient: {
    auto *gradient = static_cast<const GradientIR *>(expr);
    validateLocalUseOrder(gradient->in.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::CovariantDerivative: {
    auto *cov = static_cast<const CovariantDerivativeIR *>(expr);
    validateLocalUseOrder(cov->in.get(), defined, res, evolutionName);
    return;
  }

  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<const DivergenceIR *>(expr);
    validateLocalUseOrder(div->in.get(), defined, res, evolutionName);
    return;
  }
  }
}

ValidationResult validation::validateProgram(const ModuleIR &m) {
  ValidationResult res;

  std::unordered_map<std::string, const FieldIR *> fieldMap;
  for (auto &f : m.fields)
    fieldMap[f.name] = &f;

  for (auto &ev : m.evolutions) {
    std::unordered_set<std::string> definedTemporaries;
    for (const auto &temp : ev.temporaries) {
      validateLocalUseOrder(temp.rhs.get(), definedTemporaries, res, ev.name);
      definedTemporaries.insert(temp.name);
    }

    for (auto &eq : ev.equations) {

      auto it = fieldMap.find(eq.fieldName);
      if (it == fieldMap.end()) {
        res.diags.push_back(
            {Diagnostic::Kind::Error,
             "unknown field in dt lhs: " + eq.fieldName});
        continue;
      }

      const FieldIR &f = *it->second;
      int rank = fieldRank(f);

      if ((int)eq.indices.size() != rank) {
        res.diags.push_back(
            {Diagnostic::Kind::Error,
             "wrong number of indices on lhs for field '" + f.name + "'"});
      }

      for (auto &idx : eq.indices) {
        if (!core::isTensorIndexName(idx)) {
          res.diags.push_back(
              {Diagnostic::Kind::Error,
               "invalid index name: '" + idx + "'"});
        }
      }

      validateLocalUseOrder(eq.rhs.get(), definedTemporaries, res, ev.name);
    }
  }

  return res;
}
