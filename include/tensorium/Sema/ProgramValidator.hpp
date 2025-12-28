#pragma once
#include "tensorium/Backend/DomainIR.hpp"
#include <string>
#include <vector>

namespace tensorium::sema {

struct Diagnostic {
  enum class Kind { Error, Warning };
  Kind kind;
  std::string message;
};

struct ValidationResult {
  std::vector<Diagnostic> diags;
  bool ok() const {
    for (auto &d : diags)
      if (d.kind == Diagnostic::Kind::Error)
        return false;
    return true;
  }
};

ValidationResult validateProgram(const backend::ModuleIR &m);

} // namespace tensorium::sema
