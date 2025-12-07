#pragma once
#include "ast.hpp"
#include "lexer.hpp"
#include "semantics/indexed_ast.hpp"
#include "tokens.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class Parser {
  Lexer &lex;
  Token cur;

  void advance() { cur = lex.next(); }

  [[noreturn]] void syntaxError(const std::string &msg) {
    throw std::runtime_error(
        "Syntax error at line " + std::to_string(cur.line) + ", col " +
        std::to_string(cur.column) + ": " + msg + " (got '" + cur.text + "')");
  }

  void expect(TokenType type) {
    if (cur.type != type) {
      syntaxError("expected token type " + std::to_string((int)type));
    }
    advance();
  }

  std::unique_ptr<Expr> parseExpr() { return parseAddExpr(); }

  std::unique_ptr<Expr> parseAddExpr() {
    auto left = parseMulExpr();
    while (cur.type == TokenType::Plus || cur.type == TokenType::Minus) {
      char op = cur.text[0];
      advance();
      auto right = parseMulExpr();
      left =
          std::make_unique<BinaryExpr>(std::move(left), op, std::move(right));
    }
    return left;
  }

  std::unique_ptr<Expr> parseMulExpr() {
    auto left = parsePowExpr();
    while (cur.type == TokenType::Star || cur.type == TokenType::Slash) {
      char op = cur.text[0];
      advance();
      auto right = parsePowExpr();
      left =
          std::make_unique<BinaryExpr>(std::move(left), op, std::move(right));
    }
    return left;
  }

  std::unique_ptr<Expr> parsePowExpr() {
    auto base = parseUnaryExpr();
    if (cur.type == TokenType::Caret) {
      advance();
      auto exponent = parsePowExpr();
      return std::make_unique<BinaryExpr>(std::move(base), '^',
                                          std::move(exponent));
    }
    return base;
  }

  std::unique_ptr<Expr> parseUnaryExpr() {
    if (cur.type == TokenType::Plus) {
      advance();
      return parseUnaryExpr();
    }
    if (cur.type == TokenType::Minus) {
      advance();
      auto inner = parseUnaryExpr();
      return std::make_unique<BinaryExpr>(std::make_unique<NumberExpr>(0.0),
                                          '-', std::move(inner));
    }
    return parsePrimary();
  }

  std::unique_ptr<Expr> parsePrimary() {
    if (cur.type == TokenType::Number) {
      double value = std::stod(cur.text);
      advance();
      return std::make_unique<NumberExpr>(value);
    }

    if (cur.type == TokenType::Identifier) {
      std::string name = cur.text;
      advance();

      if (cur.type == TokenType::LParen) {
        advance();
        auto args = parseExprList();
        expect(TokenType::RParen);

        auto call = std::make_unique<CallExpr>();
        call->callee = name;
        call->args = std::move(args);
        return call;
      }

      if (cur.type == TokenType::LBracket) {
        advance();
        std::vector<std::string> idx;

        while (cur.type == TokenType::Identifier) {
          idx.push_back(cur.text);
          advance();
          if (cur.type == TokenType::Comma) {
            advance();
            continue;
          }
          break;
        }

        expect(TokenType::RBracket);

        return std::make_unique<IndexedVarExpr>(name, std::move(idx));
      }

      return std::make_unique<VarExpr>(name);
    }

    if (cur.type == TokenType::LParen) {
      advance();
      auto e = parseExpr();
      expect(TokenType::RParen);
      return std::make_unique<ParenExpr>(std::move(e));
    }

    syntaxError("unexpected token in expression");
  }

  std::vector<std::unique_ptr<Expr>> parseExprList() {
    std::vector<std::unique_ptr<Expr>> list;

    if (cur.type == TokenType::RParen)
      return list;

    list.push_back(parseExpr());
    while (cur.type == TokenType::Comma) {
      advance();
      list.push_back(parseExpr());
    }
    return list;
  }

  TensorAccess parseLHS() {
    TensorAccess lhs;

    if (cur.type != TokenType::Identifier)
      syntaxError("expected identifier on LHS");

    lhs.base = cur.text;
    advance();

    TokenType open =
        (cur.type == TokenType::LBracket || cur.type == TokenType::LParen)
            ? cur.type
            : TokenType::Unknown;
    if (open != TokenType::Unknown) {
      TokenType close = (open == TokenType::LBracket) ? TokenType::RBracket
                                                      : TokenType::RParen;
      advance();
      while (cur.type == TokenType::Identifier) {
        lhs.indices.push_back(cur.text);
        advance();
        if (cur.type == TokenType::Comma) {
          advance();
          continue;
        }
        break;
      }
      expect(close);
    }

    return lhs;
  }

  Assignment parseAssignment() {
    Assignment a;
    a.lhs = parseLHS();
    expect(TokenType::Equals);
    a.rhs = parseExpr();
    return a;
  }

  FieldDecl parseFieldDecl() {
    expect(TokenType::KwField);

    TensorKind kind;
    int up = 0, down = 0;

    if (cur.type == TokenType::KwScalar) {
      kind = TensorKind::Scalar;
      advance();
    } else if (cur.type == TokenType::KwVector) {
      kind = TensorKind::Vector;
      up = 1;
      advance();
    } else if (cur.type == TokenType::KwCovector) {
      kind = TensorKind::Covector;
      down = 1;
      advance();
    } else if (cur.type == TokenType::KwCovTensor2) {
      kind = TensorKind::CovTensor2;
      down = 2;
      advance();
    } else if (cur.type == TokenType::KwConTensor2) {
      kind = TensorKind::ConTensor2;
      up = 2;
      advance();
    } else
      syntaxError("expected tensor type after 'field'");

    if (cur.type != TokenType::Identifier)
      syntaxError("expected field name after type");

    FieldDecl f;
    f.kind = kind;
    f.up = up;
    f.down = down;
    f.name = cur.text;
    advance();

    if (cur.type == TokenType::LBracket) {
      advance();
      while (cur.type == TokenType::Identifier) {
        f.indices.push_back(cur.text);
        advance();
        if (cur.type == TokenType::Comma)
          advance();
        else
          break;
      }
      expect(TokenType::RBracket);
    }

    // NEW: consume optional line break / semicolon-like behavior
    while (cur.type == TokenType::Unknown || cur.text == "\n")
      advance();

    return f;
  }

  MetricDecl parseMetric() {
    if (cur.type != TokenType::KwMetric)
      syntaxError("expected 'metric' keyword");

    advance();

    if (cur.type != TokenType::Identifier)
      syntaxError("expected metric name");

    MetricDecl decl;
    decl.name = cur.text;
    advance();

    expect(TokenType::LParen);

    while (cur.type == TokenType::Identifier) {
      decl.indices.push_back(cur.text);
      advance();
      if (cur.type == TokenType::Comma)
        advance();
      else
        break;
    }

    expect(TokenType::RParen);
    expect(TokenType::LBrace);

    while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
      if (cur.type == TokenType::Identifier) {
        decl.entries.push_back(parseAssignment());
      } else {
        syntaxError("unexpected token in metric body");
      }
    }

    expect(TokenType::RBrace);
    return decl;
  }

  EvolutionEq parseEvolutionEq() {
    if (cur.type != TokenType::KwDt)
      syntaxError("expected 'dt' at start of evolution equation");
    advance();

    if (cur.type != TokenType::Identifier)
      syntaxError("expected field name after 'dt'");

    EvolutionEq eq;
    eq.fieldName = cur.text;
    advance();

    if (cur.type == TokenType::LBracket || cur.type == TokenType::LParen) {
      TokenType closing = (cur.type == TokenType::LBracket)
                              ? TokenType::RBracket
                              : TokenType::RParen;
      advance();
      while (cur.type == TokenType::Identifier) {
        eq.indices.push_back(cur.text);
        advance();
        if (cur.type == TokenType::Comma) {
          advance();
          continue;
        }
        break;
      }
      expect(closing);
    }

    expect(TokenType::Equals);
    eq.rhs = parseExpr();
    return eq;
  }

  EvolutionDecl parseEvolution() {
    expect(TokenType::KwEvolution);

    if (cur.type != TokenType::Identifier)
      syntaxError("expected evolution name (e.g. BSSN)");

    EvolutionDecl evo;
    evo.name = cur.text;
    advance();

    expect(TokenType::LBrace);

    while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {

      if (cur.type == TokenType::KwDt) {
        evo.equations.push_back(parseEvolutionEq());
        continue;
      }

      if (cur.type == TokenType::Identifier) {
        Assignment a = parseAssignment();
        evo.tempAssignments.push_back(std::move(a));
        continue;
      }

      syntaxError("expected 'dt' or assignment inside evolution block");
    }

    expect(TokenType::RBrace);
    return evo;
  }

public:
  explicit Parser(Lexer &l) : lex(l) { advance(); }

  Program parseProgram() {
    Program prog;

    while (cur.type != TokenType::End) {
      if (cur.type == TokenType::KwField) {
        prog.fields.push_back(parseFieldDecl());
        continue;
      }
      if (cur.type == TokenType::KwMetric) {
        prog.metrics.push_back(parseMetric());
        continue;
      }
      if (cur.type == TokenType::KwEvolution) {
        prog.evolutions.push_back(parseEvolution());
        continue;
      }

      syntaxError("unexpected token at top-level");
    }

    return prog;
  }
};
