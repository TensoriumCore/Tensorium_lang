#include "tensorium/Parse/Parser.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensorium {

std::unique_ptr<Expr> Parser::parseExpr() { return parseAddExpr(); }

std::unique_ptr<Expr> Parser::parseAddExpr() {
  auto left = parseMulExpr();
  while (cur.type == TokenType::Plus || cur.type == TokenType::Minus) {
    char op = cur.text[0];
    advance();
    left = std::make_unique<BinaryExpr>(std::move(left), op, parseMulExpr());
  }
  return left;
}

std::unique_ptr<Expr> Parser::parseMulExpr() {
  auto left = parsePowExpr();
  while (cur.type == TokenType::Star || cur.type == TokenType::Slash) {
    char op = cur.text[0];
    advance();
    left = std::make_unique<BinaryExpr>(std::move(left), op, parsePowExpr());
  }
  return left;
}

std::unique_ptr<Expr> Parser::parsePowExpr() {
  auto base = parseUnaryExpr();
  if (cur.type == TokenType::Caret) {
    advance();
    return std::make_unique<BinaryExpr>(std::move(base), '^', parsePowExpr());
  }
  return base;
}

std::unique_ptr<Expr> Parser::parseUnaryExpr() {
  if (cur.type == TokenType::Plus) {
    advance();
    return parseUnaryExpr();
  }
  if (cur.type == TokenType::Minus) {
    advance();
    return std::make_unique<BinaryExpr>(std::make_unique<NumberExpr>(0.0), '-',
                                        parseUnaryExpr());
  }
  return parsePrimary();
}

std::unique_ptr<Expr> Parser::parsePrimary() {
  if (cur.type == TokenType::Number) {
    double v = std::stod(cur.text);
    advance();
    return std::make_unique<NumberExpr>(v);
  }
  if (cur.type == TokenType::Identifier) {
    std::string n = cur.text;
    advance();
    if (n == "nabla" && cur.type == TokenType::Caret) {
      advance();
      if (cur.type != TokenType::Identifier)
        syntaxError("expected index after nabla^");
      std::string idx = cur.text;
      if (idx.size() != 1)
        syntaxError("nabla^ expects a single index name");
      advance();
      expect(TokenType::LParen);
      auto args = parseExprList();
      expect(TokenType::RParen);
      auto c = std::make_unique<CallExpr>();
      c->callee = "nabla^" + idx;
      c->args = std::move(args);
      return c;
    }
    if (cur.type == TokenType::LParen) {
      advance();
      auto args = parseExprList();
      expect(TokenType::RParen);
      auto c = std::make_unique<CallExpr>();
      c->callee = n;
      c->args = std::move(args);
      return c;
    }
    if (cur.type == TokenType::LBracket) {
      advance();
      std::vector<std::string> idx;
      std::vector<int> offs;
      while (cur.type == TokenType::Identifier) {
        idx.push_back(cur.text);
        advance();
        int off = 0;
        if (cur.type == TokenType::Plus || cur.type == TokenType::Minus) {
          bool neg = (cur.type == TokenType::Minus);
          advance();
          if (cur.type != TokenType::Number)
            syntaxError("index offset expects an integer literal");
          if (cur.text.find('.') != std::string::npos)
            syntaxError("index offset expects an integer literal");
          off = std::stoi(cur.text);
          if (neg)
            off = -off;
          advance();
        }
        offs.push_back(off);
        if (cur.type == TokenType::Comma) {
          advance();
          continue;
        }
        break;
      }
      expect(TokenType::RBracket);
      return std::make_unique<IndexedVarExpr>(n, std::move(idx),
                                              std::move(offs));
    }
    return std::make_unique<VarExpr>(n);
  }
  if (cur.type == TokenType::LParen) {
    advance();
    auto e = parseExpr();
    expect(TokenType::RParen);
    return std::make_unique<ParenExpr>(std::move(e));
  }
  syntaxError("Unexpected token in expr");
}

std::vector<std::unique_ptr<Expr>> Parser::parseExprList() {
  std::vector<std::unique_ptr<Expr>> l;
  if (cur.type == TokenType::RParen)
    return l;
  l.push_back(parseExpr());
  while (cur.type == TokenType::Comma) {
    advance();
    l.push_back(parseExpr());
  }
  return l;
}

TensorAccess Parser::parseLHS() {
  TensorAccess lhs;
  if (cur.type != TokenType::Identifier)
    syntaxError("Expected ID on LHS");
  lhs.base = cur.text;
  advance();

  TokenType close = TokenType::Unknown;
  if (cur.type == TokenType::LBracket)
    close = TokenType::RBracket;
  else if (cur.type == TokenType::LParen)
    close = TokenType::RParen;

  if (close != TokenType::Unknown) {
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

Assignment Parser::parseAssignment() {
  Assignment a;
  a.lhs = parseLHS();
  expect(TokenType::Equals);
  a.rhs = parseExpr();
  return a;
}

} // namespace tensorium
