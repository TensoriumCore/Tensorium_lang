#pragma once
#include "tensorium/AST/AST.hpp"
#include "tensorium/AST/IndexedAST.hpp" // Inclusion du fichier complet
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>

static const std::unordered_set<std::string> SPATIAL_INDICES = {"i", "j", "k",
                                                                "l", "m", "n"};
namespace tensorium {

class SemanticAnalyzer {
  const Program &prog;
  std::unordered_map<std::string, int> coordIndex;
  std::unordered_map<std::string, bool> locals;
  std::unordered_map<std::string, const FieldDecl *> fields;
  std::vector<FieldDecl> syntheticMetricFields;
  std::unordered_map<std::string, int> indexUseCount;
  std::unordered_set<std::string> lhsIndices;

  void validateSpatialIndex(const std::string &idx);
  int resolveIndex(const std::string &name);
  std::unique_ptr<IndexedExpr> transformExpr(const Expr *e);
  void validateSimulation(const SimulationConfig &sim);

public:
  explicit SemanticAnalyzer(const Program &p);
  IndexedMetric analyzeMetric(const MetricDecl &decl);
  IndexedEvolution analyzeEvolution(const EvolutionDecl &evo);
};
} // namespace tensorium
