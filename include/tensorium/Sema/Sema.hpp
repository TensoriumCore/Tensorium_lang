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

enum class CompilationMode { Executable, Symbolic };

class SemanticAnalyzer {
  const Program &prog;
  CompilationMode mode;
  std::unordered_map<std::string, int> coordIndex;
  std::unordered_map<std::string, bool> locals;
  std::unordered_map<std::string, const FieldDecl *> fields;
  std::unordered_map<std::string, const ExternDecl *> externSignatures;
  std::vector<FieldDecl> syntheticMetricFields;
  std::unordered_map<std::string, int> indexUseCount;
  std::unordered_set<std::string> lhsIndices;
  bool simulationMissing = false;

  void validateSpatialIndex(const std::string &idx);
  int resolveIndex(const std::string &name);
  std::unique_ptr<IndexedExpr> transformExpr(const Expr *e);
  void validateSimulation(const SimulationConfig &sim);

public:
  explicit SemanticAnalyzer(const Program &p,
                            CompilationMode mode = CompilationMode::Executable);
  bool hasSimulationMetadata() const { return !simulationMissing; }
  CompilationMode getMode() const { return mode; }
  IndexedMetric analyzeMetric(const MetricDecl &decl);
  IndexedEvolution analyzeEvolution(const EvolutionDecl &evo);
};
} // namespace tensorium
