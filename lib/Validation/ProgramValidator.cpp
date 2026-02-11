#include "tensorium/Core/IndexSet.h"
#include "tensorium/Validation/ProgramValidator.hpp"
#include <unordered_map>

using namespace tensorium;
using namespace tensorium::backend;
using namespace tensorium::validation;

static int fieldRank(const FieldIR &f) {
  return f.tensorType.rank();
}

ValidationResult validation::validateProgram(const ModuleIR &m) {
  ValidationResult res;

  std::unordered_map<std::string, const FieldIR *> fieldMap;
  for (auto &f : m.fields)
    fieldMap[f.name] = &f;

  for (auto &ev : m.evolutions) {
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
    }
  }

  return res;
}
