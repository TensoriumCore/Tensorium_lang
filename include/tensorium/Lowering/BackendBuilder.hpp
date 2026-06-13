#pragma once

#include "tensorium/AST/AST.hpp"
#include "tensorium/IR/DomainIR.hpp"
#include "tensorium/Lowering/SemanticAnalysis.hpp"

namespace tensorium::backend {

class BackendBuilder {
public:
  static ModuleIR build(const Program &prog,
                        lowering::SemanticAnalysis &semantics);

private:
  static FieldKind lowerFieldKind(TensorKind k);
};

} // namespace tensorium::backend
