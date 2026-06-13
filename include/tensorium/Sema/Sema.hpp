#pragma once
#include "tensorium/AST/AST.hpp"
#include "tensorium/AST/IndexedAST.hpp" // Inclusion du fichier complet
#include "tensorium/Core/CompilationMode.hpp"
#include "tensorium/Lowering/SemanticAnalysis.hpp"
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace tensorium {

class SemanticAnalyzer : public lowering::SemanticAnalysis {
  const Program &prog;
  CompilationMode mode;
  std::unordered_map<std::string, int> coordIndex;
  // Active local scope for the block currently being analyzed (metric/evolution).
  std::unordered_map<std::string, TensorTypeDesc> locals;
  // Metric scalar aliases collected from metric assignments with scalar LHS.
  std::unordered_map<std::string, TensorTypeDesc> metricScalarLocals;
  std::unordered_set<std::string> params;
  std::unordered_map<std::string, const FieldDecl *> fields;
  std::unordered_map<std::string, const ExternDecl *> externSignatures;
  std::vector<FieldDecl> syntheticMetricFields;
  std::unordered_map<std::string, int> indexUseCount;
  std::unordered_set<std::string> lhsIndices;
  bool simulationMissing = false;
  std::vector<std::string> warnings;
  int metricFieldCount = 0;
  int inverseMetricFieldCount = 0;
  bool hasConnectionTensor = false;

  void validateSpatialIndex(const std::string &idx);
  int resolveIndex(const std::string &name);
  std::unique_ptr<IndexedExpr> transformExpr(const Expr *e);
  std::unique_ptr<IndexedExpr> transformNablaCall(const CallExpr &call,
                                                  bool isContravariant);
  void validateSimulation(const SimulationConfig &sim);
  void validateInitialData(const InitialDataDecl &init);
  void validateInitialDataExpr(const Expr *expr, const std::string &context);
  void enforceMetricFieldRules(const FieldDecl &field);
  bool containsExplicitMetricAntisymmetry(const IndexedExpr *expr) const;
  bool isSimpleIndexSwap(const IndexedExpr *lhs, const IndexedExpr *rhs) const;
  bool isNegatedSwap(const IndexedExpr *lhs, const IndexedExpr *rhs) const;

public:
  explicit SemanticAnalyzer(const Program &p,
                            CompilationMode mode = CompilationMode::Executable);
  bool hasSimulationMetadata() const { return !simulationMissing; }
  CompilationMode getMode() const { return mode; }
  const std::vector<std::string> &getWarnings() const { return warnings; }
  IndexedMetric analyzeMetric(const MetricDecl &decl);
  IndexedEvolution analyzeEvolution(const EvolutionDecl &evo) override;
  IndexedEvolution analyzeConstraint(const ConstraintDecl &decl) override;
  IndexedPrint analyzePrint(const PrintDecl &decl) override;
};
} // namespace tensorium
