#include "tensorium/Sema/ProgramValidator.hpp"
#include <unordered_map>

using namespace tensorium;
using namespace tensorium::backend;
using namespace tensorium::sema;

static bool isValidIndexName(const std::string &s) {
  return s == "i" || s == "j" || s == "k" || s == "l" || s == "m" ||
         s == "n";
}

static int fieldRank(const FieldIR &f) {
  return f.up + f.down;
}

ValidationResult sema::validateProgram(const ModuleIR &m) {
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
        if (!isValidIndexName(idx)) {
          res.diags.push_back(
              {Diagnostic::Kind::Error,
               "invalid index name: '" + idx + "'"});
        }
      }
    }
  }

  return res;
}
