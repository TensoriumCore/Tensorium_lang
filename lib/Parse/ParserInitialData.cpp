#include "tensorium/Parse/Parser.hpp"

#include <string>
#include <utility>

namespace tensorium {

std::vector<std::unique_ptr<Expr>>
Parser::parseExprVectorLiteral(size_t expectedSize, const std::string &label) {
  expect(TokenType::LBracket);
  std::vector<std::unique_ptr<Expr>> out;

  if (cur.type != TokenType::RBracket) {
    while (true) {
      out.push_back(parseExpr());
      if (cur.type == TokenType::Comma) {
        advance();
        continue;
      }
      break;
    }
  }

  expect(TokenType::RBracket);

  if (expectedSize != 0 && out.size() != expectedSize) {
    syntaxError(label + " expects " + std::to_string(expectedSize) +
                " entries, got " + std::to_string(out.size()));
  }

  return out;
}

std::vector<std::vector<std::unique_ptr<Expr>>>
Parser::parseExprMatrixLiteral(size_t rows, size_t cols,
                               const std::string &label) {
  expect(TokenType::LBracket);
  std::vector<std::vector<std::unique_ptr<Expr>>> out;

  for (size_t r = 0; r < rows; ++r) {
    out.push_back(parseExprVectorLiteral(cols, label + " row"));
    if (r + 1 < rows)
      expect(TokenType::Comma);
  }

  expect(TokenType::RBracket);
  return out;
}

InitialDataDecl Parser::parseInitialData() {
  expect(TokenType::KwInitialData);
  expect(TokenType::LBrace);

  InitialDataDecl init;

  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::KwMetric4) {
      if (init.hasMetric4)
        syntaxError("duplicate metric4 entry in initial_data");
      if (init.hasDecomposed)
        syntaxError("initial_data must use either metric4 or alpha/beta/gamma");

      init.hasMetric4 = true;
      advance();

      if (cur.type != TokenType::Identifier)
        syntaxError("metric4 expects metric symbol name");
      init.metric4.name = cur.text;
      advance();

      expect(TokenType::LBracket);
      if (cur.type != TokenType::Identifier)
        syntaxError("metric4 expects first index name");
      init.metric4.indices.push_back(cur.text);
      advance();
      expect(TokenType::Comma);
      if (cur.type != TokenType::Identifier)
        syntaxError("metric4 expects second index name");
      init.metric4.indices.push_back(cur.text);
      advance();
      expect(TokenType::RBracket);

      expect(TokenType::Equals);
      init.metric4.components =
          parseExprMatrixLiteral(4, 4, "metric4 4x4 matrix");
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "enforce_symmetry") {
      advance();
      expect(TokenType::Equals);
      if (cur.type != TokenType::Identifier ||
          (cur.text != "true" && cur.text != "false")) {
        syntaxError("enforce_symmetry expects true or false");
      }
      init.enforceSymmetry = (cur.text == "true");
      advance();
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "alpha") {
      if (init.hasMetric4)
        syntaxError(
            "initial_data must use either metric4 or alpha/beta/gamma/gammaU");
      init.hasDecomposed = true;
      advance();
      expect(TokenType::Equals);
      init.decomposed.alpha = parseExpr();
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "beta") {
      if (init.hasMetric4)
        syntaxError(
            "initial_data must use either metric4 or alpha/beta/gamma/gammaU");
      init.hasDecomposed = true;
      advance();
      if (cur.type == TokenType::LBracket) {
        advance();
        if (cur.type != TokenType::Identifier)
          syntaxError("beta expects index symbol");
        advance();
        expect(TokenType::RBracket);
      }
      expect(TokenType::Equals);
      init.decomposed.beta = parseExprVectorLiteral(3, "beta");
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "gamma") {
      if (init.hasMetric4)
        syntaxError(
            "initial_data must use either metric4 or alpha/beta/gamma/gammaU");
      init.hasDecomposed = true;
      advance();
      if (cur.type == TokenType::LBracket) {
        advance();
        if (cur.type != TokenType::Identifier)
          syntaxError("gamma expects first index symbol");
        advance();
        expect(TokenType::Comma);
        if (cur.type != TokenType::Identifier)
          syntaxError("gamma expects second index symbol");
        advance();
        expect(TokenType::RBracket);
      }
      expect(TokenType::Equals);
      init.decomposed.gamma = parseExprMatrixLiteral(3, 3, "gamma 3x3 matrix");
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "gammaU") {
      if (init.hasMetric4)
        syntaxError(
            "initial_data must use either metric4 or alpha/beta/gamma/gammaU");
      init.hasDecomposed = true;
      advance();
      if (cur.type == TokenType::LBracket) {
        advance();
        if (cur.type != TokenType::Identifier)
          syntaxError("gammaU expects first index symbol");
        advance();
        expect(TokenType::Comma);
        if (cur.type != TokenType::Identifier)
          syntaxError("gammaU expects second index symbol");
        advance();
        expect(TokenType::RBracket);
      }
      expect(TokenType::Equals);
      init.decomposed.gammaU =
          parseExprMatrixLiteral(3, 3, "gammaU 3x3 matrix");
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "split_3p1") {
      if (init.split3p1.enabled)
        syntaxError("duplicate split_3p1 block in initial_data");
      init.split3p1.enabled = true;
      advance();
      expect(TokenType::LBrace);

      while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
        if (cur.type != TokenType::Identifier)
          syntaxError("split_3p1 expects mapping key");
        std::string key = cur.text;
        advance();
        expect(TokenType::Arrow);
        TensorAccess target = parseLHS();

        if (key == "alpha") {
          init.split3p1.hasAlpha = true;
          init.split3p1.alphaTarget = std::move(target);
        } else if (key == "beta") {
          init.split3p1.hasBeta = true;
          init.split3p1.betaTarget = std::move(target);
        } else if (key == "gamma") {
          init.split3p1.hasGamma = true;
          init.split3p1.gammaTarget = std::move(target);
        } else if (key == "gammaU") {
          init.split3p1.hasGammaU = true;
          init.split3p1.gammaUTarget = std::move(target);
        } else {
          syntaxError("unknown split_3p1 key '" + key + "'");
        }

        if (cur.type == TokenType::Semicolon)
          advance();
      }

      expect(TokenType::RBrace);
      continue;
    }

    syntaxError("unexpected entry in initial_data block");
  }

  expect(TokenType::RBrace);

  if (!init.hasMetric4 && !init.hasDecomposed) {
    syntaxError(
        "initial_data requires metric4 or alpha/beta/gamma/gammaU definitions");
  }
  if (init.hasDecomposed &&
      (!init.decomposed.alpha || init.decomposed.beta.empty() ||
       init.decomposed.gamma.empty())) {
    syntaxError("alpha, beta and gamma must all be defined in initial_data");
  }

  return init;
}

} // namespace tensorium
