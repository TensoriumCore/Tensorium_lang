#pragma once
#include "tensorium/AST/AST.hpp"
#include "tensorium/Lex/Lexer.hpp"

namespace tensorium {
class Parser {
  Lexer &lex;
  Token cur;
  void advance();
  void expect(TokenType type);
  [[noreturn]] void syntaxError(const std::string &msg);

  std::unique_ptr<Expr> parseExpr();
  std::unique_ptr<Expr> parseAddExpr();
  std::unique_ptr<Expr> parseMulExpr();
  std::unique_ptr<Expr> parsePowExpr();
  std::unique_ptr<Expr> parseUnaryExpr();
  std::unique_ptr<Expr> parsePrimary();
  std::vector<std::unique_ptr<Expr>> parseExprList();

  TensorAccess parseLHS();
  Assignment parseAssignment();
  TensorTypeDesc parseTensorTypeDesc();
  std::vector<std::string> parseParamsBlock();
  ExternDecl parseExternDecl();
  FieldDecl parseFieldDecl();
  MetricDecl parseMetric();
  InitialDataDecl parseInitialData();
  SpectralDomainDecl parseSpectralDomain();
  ConstraintUnknownDecl parseConstraintUnknown();
  ConstraintEquationDecl parseConstraintEquation();
  ConstraintBoundaryDecl parseConstraintBoundary();
  ConstraintInterfaceDecl parseConstraintInterface();
  ConstraintCttReconstructionDecl parseConstraintCttReconstruction();
  ConstraintSolveConfig parseConstraintSolve();
  EvolutionEq parseEvolutionEq();
  EvolutionDecl parseEvolution();
  SimulationConfig parseSimulation();
  TimeConfig parseTimeBlock();
  SpatialConfig parseSpatialBlock();
  std::vector<std::unique_ptr<Expr>>
  parseExprVectorLiteral(size_t expectedSize, const std::string &label);
  std::vector<std::vector<std::unique_ptr<Expr>>>
  parseExprMatrixLiteral(size_t rows, size_t cols, const std::string &label);

public:
  explicit Parser(Lexer &l);
  Program parseProgram();
};
} // namespace tensorium
