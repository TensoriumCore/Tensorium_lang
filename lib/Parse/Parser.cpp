#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Basic/Diagnostics.hpp"
#include "tensorium/Basic/Token.hpp"
#include <algorithm>
#include <stdexcept>

namespace tensorium {
Parser::Parser(Lexer &l) : lex(l) { advance(); }
void Parser::advance() { cur = lex.next(); }
void Parser::expect(TokenType type) {
  if (cur.type != type) {
    std::string got;
    if (cur.type == TokenType::End) {
      got = "end of file";
    } else if (cur.type == TokenType::Unknown) {
      got = "unknown token '" + cur.text + "'";
    } else {
      got = "'" + cur.text + "'";
    }
    syntaxError("expected " + std::string(tokenTypeName(type)) + ", got " +
                got);
  }
  advance();
}
void Parser::syntaxError(const std::string &msg) {
  SourceLocation loc;
  loc.line = std::max(1, cur.line);
  loc.column = std::max(1, cur.column);
  loc.length = std::max<int>(1, static_cast<int>(cur.text.size()));
  throw DiagnosticError(DiagnosticLevel::Error, msg, loc, "E0001");
}

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
      return std::make_unique<IndexedVarExpr>(n, std::move(idx));
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

TensorTypeDesc Parser::parseTensorTypeDesc() {
  TensorTypeDesc desc;
  auto simple = [&](TensorKind kind, int up, int down) {
    desc.kind = kind;
    desc.up = up;
    desc.down = down;
    advance();
    return desc;
  };

  switch (cur.type) {
  case TokenType::KwScalar:
    return simple(TensorKind::Scalar, 0, 0);
  case TokenType::KwVector:
    return simple(TensorKind::Vector, 1, 0);
  case TokenType::KwCovector:
    return simple(TensorKind::Covector, 0, 1);
  case TokenType::KwCovTensor2:
    return simple(TensorKind::CovTensor2, 0, 2);
  case TokenType::KwConTensor2:
    return simple(TensorKind::ConTensor2, 2, 0);
  case TokenType::KwCovTensor3:
    return simple(TensorKind::CovTensor3, 0, 3);
  case TokenType::KwConTensor3:
    return simple(TensorKind::ConTensor3, 3, 0);
  case TokenType::KwCovTensor4:
    return simple(TensorKind::CovTensor4, 0, 4);
  case TokenType::KwConTensor4:
    return simple(TensorKind::ConTensor4, 4, 0);
  default:
    break;
  }

  if (cur.type == TokenType::Identifier && cur.text == "mixed_tensor") {
    desc.kind = TensorKind::MixedTensor;
    advance();
    expect(TokenType::LParen);
    bool haveUp = false;
    bool haveDown = false;
    while (cur.type != TokenType::RParen) {
      if (cur.type != TokenType::Identifier)
        syntaxError("expected mixed_tensor attribute");
      const std::string attr = cur.text;
      advance();
      expect(TokenType::Equals);
      if (cur.type != TokenType::Number)
        syntaxError("mixed_tensor attribute expects integer");
      int value = std::stoi(cur.text);
      if (attr == "up") {
        if (haveUp)
          syntaxError("duplicate up attribute in mixed_tensor");
        desc.up = value;
        haveUp = true;
      } else if (attr == "down") {
        if (haveDown)
          syntaxError("duplicate down attribute in mixed_tensor");
        desc.down = value;
        haveDown = true;
      } else {
        syntaxError("unknown mixed_tensor attribute");
      }
      advance();
      if (cur.type == TokenType::Comma) {
        advance();
        continue;
      }
      if (cur.type != TokenType::RParen)
        syntaxError("expected ',' or ')' in mixed_tensor");
    }
    expect(TokenType::RParen);
    if (!haveUp && !haveDown)
      syntaxError("mixed_tensor requires up or down attribute");
    return desc;
  }

  syntaxError("Expected tensor type");
  return desc;
}

std::vector<std::string> Parser::parseParamsBlock() {
  expect(TokenType::KwParams);
  expect(TokenType::LBrace);

  std::vector<std::string> out;
  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type != TokenType::Identifier)
      syntaxError("params block expects parameter identifier");
    out.push_back(cur.text);
    advance();

    if (cur.type == TokenType::Comma || cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }
    if (cur.type != TokenType::RBrace) {
      syntaxError("expected ',' or '}' in params block");
    }
  }

  expect(TokenType::RBrace);
  return out;
}

ExternDecl Parser::parseExternDecl() {
  expect(TokenType::KwExtern);
  ExternDecl decl;
  decl.returnType = parseTensorTypeDesc();
  if (cur.type != TokenType::Identifier)
    syntaxError("Expected extern function name");
  decl.name = cur.text;
  advance();

  expect(TokenType::LParen);
  if (cur.type != TokenType::RParen) {
    while (true) {
      decl.params.push_back(parseTensorTypeDesc());
      if (cur.type == TokenType::Comma) {
        advance();
        continue;
      }
      break;
    }
  }
  expect(TokenType::RParen);
  decl.paramCount = decl.params.size();
  return decl;
}

FieldDecl Parser::parseFieldDecl() {
  expect(TokenType::KwField);

  TensorKind k;
  int u = 0, d = 0;
  bool consumedType = false;

  if (cur.type == TokenType::Identifier && cur.text == "mixed_tensor") {
    k = TensorKind::MixedTensor;
    advance();
    expect(TokenType::LParen);
    bool haveUp = false;
    bool haveDown = false;
    while (cur.type != TokenType::RParen) {
      if (cur.type != TokenType::Identifier)
        syntaxError("expected mixed_tensor attribute");
      const std::string attr = cur.text;
      advance();
      expect(TokenType::Equals);
      if (cur.type != TokenType::Number)
        syntaxError("mixed_tensor attribute expects integer");
      int value = std::stoi(cur.text);
      if (attr == "up") {
        if (haveUp)
          syntaxError("duplicate up attribute in mixed_tensor");
        u = value;
        haveUp = true;
      } else if (attr == "down") {
        if (haveDown)
          syntaxError("duplicate down attribute in mixed_tensor");
        d = value;
        haveDown = true;
      } else {
        syntaxError("unknown mixed_tensor attribute");
      }
      advance();
      if (cur.type == TokenType::Comma) {
        advance();
        continue;
      }
      if (cur.type != TokenType::RParen)
        syntaxError("expected ',' or ')' in mixed_tensor");
    }
    expect(TokenType::RParen);
    if (!haveUp && !haveDown)
      syntaxError("mixed_tensor requires up or down attribute");
    consumedType = true;
  } else if (cur.type == TokenType::KwScalar) {
    k = TensorKind::Scalar;
  } else if (cur.type == TokenType::KwVector) {
    k = TensorKind::Vector;
    u = 1;
  } else if (cur.type == TokenType::KwCovector) {
    k = TensorKind::Covector;
    d = 1;
  } else if (cur.type == TokenType::KwCovTensor2) {
    k = TensorKind::CovTensor2;
    d = 2;
  } else if (cur.type == TokenType::KwConTensor2) {
    k = TensorKind::ConTensor2;
    u = 2;
  } else if (cur.type == TokenType::KwCovTensor3) {
    k = TensorKind::CovTensor3;
    d = 3;
  } else if (cur.type == TokenType::KwConTensor3) {
    k = TensorKind::ConTensor3;
    u = 3;
  } else if (cur.type == TokenType::KwCovTensor4) {
    k = TensorKind::CovTensor4;
    d = 4;
  } else if (cur.type == TokenType::KwConTensor4) {
    k = TensorKind::ConTensor4;
    u = 4;
  } else if (cur.type == TokenType::KwMetric) {
    k = TensorKind::Metric;
    d = 2;
  } else if (cur.type == TokenType::KwInverseMetric) {
    k = TensorKind::InverseMetric;
    u = 2;
  } else {
    syntaxError("Unknown field type '" + cur.text + "'");
  }

  if (!consumedType)
    advance();

  if (cur.type != TokenType::Identifier)
    syntaxError("Expected field name");
  FieldDecl f;
  f.kind = k;
  f.up = u;
  f.down = d;
  f.isMetric = (k == TensorKind::Metric);
  f.isInverseMetric = (k == TensorKind::InverseMetric);
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
  return f;
}

MetricDecl Parser::parseMetric() {
  expect(TokenType::KwMetric);
  if (cur.type != TokenType::Identifier)
    syntaxError("Metric name");
  MetricDecl m;
  m.name = cur.text;
  advance();
  expect(TokenType::LParen);
  while (cur.type == TokenType::Identifier) {
    m.indices.push_back(cur.text);
    advance();
    if (cur.type == TokenType::Comma)
      advance();
    else
      break;
  }
  expect(TokenType::RParen);
  expect(TokenType::LBrace);
  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::Identifier)
      m.entries.push_back(parseAssignment());
    else
      syntaxError("Unexpected in metric");
  }
  expect(TokenType::RBrace);
  return m;
}

InitialDataDecl Parser::parseInitialData() {
  expect(TokenType::KwInitialData);

  std::string problemName;
  if (cur.type == TokenType::Identifier) {
    problemName = cur.text;
    advance();
  }
  expect(TokenType::LBrace);

  InitialDataDecl init;
  init.constraintProblem.name =
      problemName.empty() ? "constraints" : std::move(problemName);

  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "domain") {
      init.hasConstraintProblem = true;
      init.constraintProblem.domains.push_back(parseSpectralDomain());
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "unknown") {
      init.hasConstraintProblem = true;
      init.constraintProblem.unknowns.push_back(parseConstraintUnknown());
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "equation") {
      init.hasConstraintProblem = true;
      init.constraintProblem.equations.push_back(parseConstraintEquation());
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "boundary") {
      init.hasConstraintProblem = true;
      init.constraintProblem.boundaries.push_back(parseConstraintBoundary());
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "interface") {
      init.hasConstraintProblem = true;
      init.constraintProblem.interfaces.push_back(parseConstraintInterface());
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "seed") {
      init.hasConstraintProblem = true;
      advance();
      init.constraintProblem.seeds.push_back(parseAssignment());
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "solve") {
      if (init.constraintProblem.hasSolve)
        syntaxError("duplicate solve block in initial_data");
      init.hasConstraintProblem = true;
      init.constraintProblem.solve = parseConstraintSolve();
      init.constraintProblem.hasSolve = true;
      continue;
    }

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

  if (!init.hasMetric4 && !init.hasDecomposed && !init.hasConstraintProblem) {
    syntaxError("initial_data requires analytic data or a constrained problem");
  }
  if (init.hasDecomposed &&
      (!init.decomposed.alpha || init.decomposed.beta.empty() ||
       init.decomposed.gamma.empty())) {
    syntaxError("alpha, beta and gamma must all be defined in initial_data");
  }

  if (init.hasConstraintProblem) {
    const auto &problem = init.constraintProblem;
    if (problem.domains.empty())
      syntaxError("constrained initial_data requires at least one domain");
    if (problem.unknowns.empty())
      syntaxError("constrained initial_data requires at least one unknown");
    if (problem.equations.empty())
      syntaxError("constrained initial_data requires at least one equation");
    if (problem.boundaries.empty())
      syntaxError("constrained initial_data requires at least one boundary");
    if (!problem.hasSolve)
      syntaxError("constrained initial_data requires a solve block");
  }

  return init;
}

SpectralDomainDecl Parser::parseSpectralDomain() {
  if (cur.type != TokenType::Identifier || cur.text != "domain")
    syntaxError("expected domain block");
  advance();

  if (cur.type != TokenType::Identifier)
    syntaxError("domain expects a name");
  SpectralDomainDecl domain;
  domain.name = cur.text;
  advance();
  expect(TokenType::LBrace);

  bool hasCoordinates = false;
  bool hasTopology = false;
  bool hasResolution = false;
  bool hasBasis = false;
  bool hasBounds = false;
  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }
    if (cur.type != TokenType::Identifier)
      syntaxError("domain expects a property name");

    const std::string key = cur.text;
    advance();
    expect(TokenType::Equals);

    if (key == "resolution" || key == "bounds") {
      bool &alreadySeen = key == "resolution" ? hasResolution : hasBounds;
      if (alreadySeen)
        syntaxError("duplicate " + key + " in domain");
      expect(TokenType::LBracket);
      while (cur.type == TokenType::Number) {
        if (key == "resolution") {
          if (cur.text.find_first_of(".eE") != std::string::npos)
            syntaxError("domain resolution expects integers");
          domain.resolution.push_back(std::stoi(cur.text));
        } else {
          domain.bounds.push_back(std::stod(cur.text));
        }
        advance();
        if (cur.type == TokenType::Comma) {
          advance();
          continue;
        }
        break;
      }
      expect(TokenType::RBracket);
      alreadySeen = true;
      continue;
    }

    if (cur.type != TokenType::Identifier)
      syntaxError("domain property '" + key + "' expects an identifier");
    const std::string value = cur.text;
    advance();
    if (key == "coordinates") {
      if (hasCoordinates)
        syntaxError("duplicate coordinates in domain");
      domain.coordinates = value;
      hasCoordinates = true;
    } else if (key == "topology") {
      if (hasTopology)
        syntaxError("duplicate topology in domain");
      domain.topology = value;
      hasTopology = true;
    } else if (key == "basis") {
      if (hasBasis)
        syntaxError("duplicate basis in domain");
      domain.basis = value;
      hasBasis = true;
    } else {
      syntaxError("unknown domain property '" + key + "'");
    }
  }

  expect(TokenType::RBrace);
  if (!hasCoordinates || !hasTopology || !hasResolution || !hasBasis)
    syntaxError("domain requires coordinates, topology, resolution and basis");
  return domain;
}

ConstraintUnknownDecl Parser::parseConstraintUnknown() {
  if (cur.type != TokenType::Identifier || cur.text != "unknown")
    syntaxError("expected unknown declaration");
  advance();

  ConstraintUnknownDecl unknown;
  unknown.type = parseTensorTypeDesc();
  if (cur.type != TokenType::Identifier)
    syntaxError("unknown declaration expects a name");
  unknown.name = cur.text;
  advance();
  if (cur.type == TokenType::LBracket) {
    advance();
    while (cur.type == TokenType::Identifier) {
      unknown.indices.push_back(cur.text);
      advance();
      if (cur.type == TokenType::Comma) {
        advance();
        continue;
      }
      break;
    }
    expect(TokenType::RBracket);
  }
  return unknown;
}

ConstraintEquationDecl Parser::parseConstraintEquation() {
  if (cur.type != TokenType::Identifier || cur.text != "equation")
    syntaxError("expected equation declaration");
  advance();

  ConstraintEquationDecl equation;
  equation.type = parseTensorTypeDesc();
  if (cur.type != TokenType::Identifier)
    syntaxError("equation expects a name");
  equation.name = cur.text;
  advance();
  if (cur.type == TokenType::LBracket) {
    advance();
    while (cur.type == TokenType::Identifier) {
      equation.indices.push_back(cur.text);
      advance();
      if (cur.type == TokenType::Comma) {
        advance();
        continue;
      }
      break;
    }
    expect(TokenType::RBracket);
  }
  expect(TokenType::Equals);
  equation.residual = parseExpr();
  return equation;
}

ConstraintBoundaryDecl Parser::parseConstraintBoundary() {
  if (cur.type != TokenType::Identifier || cur.text != "boundary")
    syntaxError("expected boundary block");
  advance();

  if (cur.type != TokenType::Identifier)
    syntaxError("boundary expects a region name");
  ConstraintBoundaryDecl boundary;
  boundary.region = cur.text;
  advance();
  expect(TokenType::LBrace);
  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }
    if (cur.type != TokenType::Identifier)
      syntaxError("boundary expects an unknown assignment");
    boundary.conditions.push_back(parseAssignment());
  }
  expect(TokenType::RBrace);
  if (boundary.conditions.empty())
    syntaxError("boundary block requires at least one condition");
  return boundary;
}

ConstraintInterfaceDecl Parser::parseConstraintInterface() {
  if (cur.type != TokenType::Identifier || cur.text != "interface")
    syntaxError("expected interface declaration");
  advance();

  if (cur.type != TokenType::Identifier)
    syntaxError("interface expects an inner domain name");
  ConstraintInterfaceDecl interface;
  interface.innerDomain = cur.text;
  advance();
  expect(TokenType::Arrow);
  if (cur.type != TokenType::Identifier)
    syntaxError("interface expects an outer domain name");
  interface.outerDomain = cur.text;
  advance();
  return interface;
}

ConstraintSolveConfig Parser::parseConstraintSolve() {
  if (cur.type != TokenType::Identifier || cur.text != "solve")
    syntaxError("expected solve block");
  advance();
  expect(TokenType::LBrace);

  ConstraintSolveConfig solve;
  bool hasNonlinear = false;
  bool hasLinear = false;
  bool hasTolerance = false;
  bool hasMaxIterations = false;
  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }
    if (cur.type != TokenType::Identifier)
      syntaxError("solve expects a property name");
    const std::string key = cur.text;
    advance();
    expect(TokenType::Equals);

    if (key == "tolerance") {
      if (hasTolerance)
        syntaxError("duplicate tolerance in solve block");
      if (cur.type != TokenType::Number)
        syntaxError("tolerance expects a number");
      solve.tolerance = std::stod(cur.text);
      advance();
      hasTolerance = true;
      continue;
    }
    if (key == "max_iterations") {
      if (hasMaxIterations)
        syntaxError("duplicate max_iterations in solve block");
      if (cur.type != TokenType::Number)
        syntaxError("max_iterations expects an integer");
      if (cur.text.find_first_of(".eE") != std::string::npos)
        syntaxError("max_iterations expects an integer");
      solve.maxIterations = std::stoi(cur.text);
      advance();
      hasMaxIterations = true;
      continue;
    }
    if (cur.type != TokenType::Identifier)
      syntaxError("solve property '" + key + "' expects an identifier");
    if (key == "nonlinear") {
      if (hasNonlinear)
        syntaxError("duplicate nonlinear in solve block");
      solve.nonlinear = cur.text;
      hasNonlinear = true;
    } else if (key == "linear") {
      if (hasLinear)
        syntaxError("duplicate linear in solve block");
      solve.linear = cur.text;
      hasLinear = true;
    } else {
      syntaxError("unknown solve property '" + key + "'");
    }
    advance();
  }
  expect(TokenType::RBrace);
  if (!hasNonlinear || !hasLinear || !hasTolerance || !hasMaxIterations)
    syntaxError(
        "solve requires nonlinear, linear, tolerance and max_iterations");
  return solve;
}

EvolutionEq Parser::parseEvolutionEq() {
  expect(TokenType::KwDt);
  if (cur.type != TokenType::Identifier)
    syntaxError("Field name after dt");
  EvolutionEq eq;
  eq.fieldName = cur.text;
  advance();

  TokenType close = TokenType::Unknown;
  if (cur.type == TokenType::LBracket)
    close = TokenType::RBracket;
  else if (cur.type == TokenType::LParen)
    close = TokenType::RParen;

  if (close != TokenType::Unknown) {
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
    expect(close);
  }
  expect(TokenType::Equals);
  eq.rhs = parseExpr();
  return eq;
}

TimeConfig Parser::parseTimeBlock() {
  expect(TokenType::KwTime);
  expect(TokenType::LBrace);

  TimeConfig cfg;
  bool hasDt = false;
  bool hasIntegrator = false;

  while (cur.type != TokenType::RBrace) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }

    if (cur.text == "dt") {
      if (hasDt)
        syntaxError("duplicate 'dt' entry in time block");
      advance();
      expect(TokenType::Equals);
      if (cur.type != TokenType::Number)
        syntaxError("dt expects a number");
      cfg.dt = std::stod(cur.text);
      hasDt = true;
      advance();
      continue;
    }

    if (cur.text == "integrator") {
      if (hasIntegrator)
        syntaxError("duplicate 'integrator' entry in time block");
      advance();
      expect(TokenType::Equals);

      if (cur.text == "euler")
        cfg.integrator = TimeIntegrator::Euler;
      else if (cur.text == "rk3")
        cfg.integrator = TimeIntegrator::RK3;
      else if (cur.text == "rk4")
        cfg.integrator = TimeIntegrator::RK4;
      else
        syntaxError("unknown time integrator");

      hasIntegrator = true;
      advance();
      continue;
    }

    syntaxError("unexpected entry in time block");
  }

  expect(TokenType::RBrace);
  if (!hasDt)
    syntaxError("time block requires 'dt = <number>'");
  if (!hasIntegrator)
    syntaxError("time block requires 'integrator = euler|rk3|rk4'");
  return cfg;
}

SpatialConfig Parser::parseSpatialBlock() {
  expect(TokenType::KwSpatial);
  expect(TokenType::LBrace);

  SpatialConfig cfg;
  bool hasScheme = false;
  bool hasDerivative = false;
  bool hasOrder = false;

  while (cur.type != TokenType::RBrace) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }

    if (cur.text == "scheme") {
      if (hasScheme)
        syntaxError("duplicate 'scheme' entry in spatial block");
      advance();
      expect(TokenType::Equals);

      if (cur.text == "fd")
        cfg.scheme = SpatialScheme::FiniteDifference;
      else if (cur.text == "spectral")
        cfg.scheme = SpatialScheme::Spectral;
      else
        syntaxError("unknown spatial scheme");

      hasScheme = true;
      advance();
      continue;
    }

    if (cur.text == "derivative") {
      if (hasDerivative)
        syntaxError("duplicate 'derivative' entry in spatial block");
      advance();
      expect(TokenType::Equals);

      if (cur.text == "centered")
        cfg.derivative = DerivativeScheme::Centered;
      else if (cur.text == "upwind")
        cfg.derivative = DerivativeScheme::Upwind;
      else
        syntaxError("unknown derivative scheme");

      hasDerivative = true;
      advance();
      continue;
    }

    if (cur.text == "order") {
      if (hasOrder)
        syntaxError("duplicate 'order' entry in spatial block");
      advance();
      expect(TokenType::Equals);

      if (cur.type != TokenType::Number)
        syntaxError("order expects an integer");

      cfg.order = std::stoi(cur.text);
      hasOrder = true;
      advance();
      continue;
    }

    syntaxError("unexpected entry in spatial block");
  }

  expect(TokenType::RBrace);
  if (!hasScheme)
    syntaxError("spatial block requires 'scheme = fd|spectral'");
  if (!hasDerivative)
    syntaxError("spatial block requires 'derivative = centered|upwind'");
  if (!hasOrder)
    syntaxError("spatial block requires 'order = <int>'");
  return cfg;
}

EvolutionDecl Parser::parseEvolution() {
  expect(TokenType::KwEvolution);
  if (cur.type != TokenType::Identifier)
    syntaxError("Evo name");
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
      evo.tempAssignments.push_back(parseAssignment());
      continue;
    }
    syntaxError("Expected dt or assign");
  }
  expect(TokenType::RBrace);
  return evo;
}

SimulationConfig Parser::parseSimulation() {
  expect(TokenType::KwSimulation);
  expect(TokenType::LBrace);

  SimulationConfig cfg;
  bool hasCoordinates = false;
  bool hasDimension = false;
  bool hasResolution = false;
  bool hasTime = false;
  bool hasSpatial = false;

  while (cur.type != TokenType::RBrace) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }

    if (cur.text == "coordinates") {
      if (hasCoordinates)
        syntaxError("duplicate 'coordinates' entry in simulation block");
      advance();
      expect(TokenType::Equals);

      if (cur.text == "cartesian")
        cfg.coordinates = CoordinateSystem::Cartesian;
      else if (cur.text == "spherical")
        cfg.coordinates = CoordinateSystem::Spherical;
      else if (cur.text == "cylindrical")
        cfg.coordinates = CoordinateSystem::Cylindrical;
      else
        syntaxError("unknown coordinate system");

      hasCoordinates = true;
      advance();
      continue;
    }

    if (cur.text == "dimension") {
      if (hasDimension)
        syntaxError("duplicate 'dimension' entry in simulation block");
      advance();
      expect(TokenType::Equals);
      cfg.dimension = std::stoi(cur.text);
      expect(TokenType::Number);
      hasDimension = true;
      continue;
    }

    if (cur.text == "resolution") {
      if (hasResolution)
        syntaxError("duplicate 'resolution' entry in simulation block");
      advance();
      expect(TokenType::Equals);
      expect(TokenType::LBracket);

      cfg.resolution.clear();
      while (cur.type == TokenType::Number) {
        cfg.resolution.push_back(std::stoi(cur.text));
        advance();
        if (cur.type == TokenType::Comma)
          advance();
        else
          break;
      }

      expect(TokenType::RBracket);
      hasResolution = true;
      continue;
    }

    if (cur.type == TokenType::KwTime) {
      if (hasTime)
        syntaxError("duplicate 'time' block in simulation");
      cfg.time = parseTimeBlock();
      hasTime = true;
      continue;
    }

    if (cur.type == TokenType::KwSpatial) {
      if (hasSpatial)
        syntaxError("duplicate 'spatial' block in simulation");
      cfg.spatial = parseSpatialBlock();
      hasSpatial = true;
      continue;
    }

    syntaxError("unexpected entry in simulation block");
  }

  expect(TokenType::RBrace);
  if (!hasDimension)
    syntaxError("simulation block requires 'dimension = <int>'");
  if (!hasResolution)
    syntaxError("simulation block requires 'resolution = [..]'");
  if (!hasTime)
    syntaxError(
        "simulation block requires 'time { dt = ... integrator = ... }'");
  if (!hasSpatial)
    syntaxError("simulation block requires 'spatial { scheme = ... derivative "
                "= ... order = ... }'");
  return cfg;
}

Program Parser::parseProgram() {
  Program p;
  while (cur.type != TokenType::End) {
    if (cur.type == TokenType::KwParams) {
      auto params = parseParamsBlock();
      p.params.insert(p.params.end(), params.begin(), params.end());
      continue;
    }
    if (cur.type == TokenType::KwField) {
      p.fields.push_back(parseFieldDecl());
      continue;
    }
    if (cur.type == TokenType::KwExtern) {
      p.externs.push_back(parseExternDecl());
      continue;
    }
    if (cur.type == TokenType::KwMetric) {
      p.metrics.push_back(parseMetric());
      continue;
    }
    if (cur.type == TokenType::KwEvolution) {
      p.evolutions.push_back(parseEvolution());
      continue;
    }
    if (cur.type == TokenType::KwSimulation) {
      if (p.simulation)
        syntaxError("Multiple simulation blocks not allowed");
      p.simulation = std::make_unique<SimulationConfig>(parseSimulation());
      continue;
    }
    if (cur.type == TokenType::KwInitialData) {
      if (p.initialData)
        syntaxError("Multiple initial_data blocks not allowed");
      p.initialData = std::make_unique<InitialDataDecl>(parseInitialData());
      continue;
    }
    syntaxError("Unexpected top level");
  }
  return p;
}
} // namespace tensorium
