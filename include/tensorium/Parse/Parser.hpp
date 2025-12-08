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
  FieldDecl parseFieldDecl();
  MetricDecl parseMetric();
  EvolutionEq parseEvolutionEq();
  EvolutionDecl parseEvolution();

public:
  explicit Parser(Lexer &l);
  Program parseProgram();
};
} // namespace tensorium
