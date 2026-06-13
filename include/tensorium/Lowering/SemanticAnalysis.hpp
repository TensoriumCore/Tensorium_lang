#pragma once

#include "tensorium/AST/AST.hpp"
#include "tensorium/AST/IndexedAST.hpp"

namespace tensorium::lowering {

class SemanticAnalysis {
public:
  virtual ~SemanticAnalysis() = default;

  virtual IndexedEvolution analyzeEvolution(const EvolutionDecl &evo) = 0;
  virtual IndexedEvolution analyzeConstraint(const ConstraintDecl &decl) = 0;
  virtual IndexedPrint analyzePrint(const PrintDecl &decl) = 0;
};

} // namespace tensorium::lowering
