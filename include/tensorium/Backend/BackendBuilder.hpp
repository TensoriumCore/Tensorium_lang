
#pragma once
#include "tensorium/AST/AST.hpp"
#include "tensorium/AST/IndexedAST.hpp"
#include "tensorium/Backend/DomainIR.hpp"
#include "tensorium/Sema/Sema.hpp"

namespace tensorium::backend {

class BackendBuilder {
public:
  static ModuleIR build(const Program &prog, SemanticAnalyzer &sem);

private:
  static FieldKind lowerFieldKind(TensorKind k);
};

} // namespace tensorium::backend
