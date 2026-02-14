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
    syntaxError("expected " + std::string(tokenTypeName(type)) + ", got " + got);
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
        syntaxError("initial_data must use either metric4 or alpha/beta/gamma/gammaU");
      init.hasDecomposed = true;
      advance();
      expect(TokenType::Equals);
      init.decomposed.alpha = parseExpr();
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "beta") {
      if (init.hasMetric4)
        syntaxError("initial_data must use either metric4 or alpha/beta/gamma/gammaU");
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
        syntaxError("initial_data must use either metric4 or alpha/beta/gamma/gammaU");
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
        syntaxError("initial_data must use either metric4 or alpha/beta/gamma/gammaU");
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
      init.decomposed.gammaU = parseExprMatrixLiteral(3, 3, "gammaU 3x3 matrix");
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
    syntaxError("initial_data requires metric4 or alpha/beta/gamma/gammaU definitions");
  }
  if (init.hasDecomposed &&
      (!init.decomposed.alpha || init.decomposed.beta.empty() ||
       init.decomposed.gamma.empty())) {
    syntaxError("alpha, beta and gamma must all be defined in initial_data");
  }

  return init;
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

  while (cur.type != TokenType::RBrace) {

    if (cur.text == "dt") {
      advance();
      expect(TokenType::Equals);
      if (cur.type != TokenType::Number)
        syntaxError("dt expects a number");
      cfg.dt = std::stod(cur.text);
      advance();
      continue;
    }

    if (cur.text == "integrator") {
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

      advance();
      continue;
    }

    syntaxError("unexpected entry in time block");
  }

  expect(TokenType::RBrace);
  return cfg;
}

SpatialConfig Parser::parseSpatialBlock() {
  expect(TokenType::KwSpatial);
  expect(TokenType::LBrace);

  SpatialConfig cfg;

  while (cur.type != TokenType::RBrace) {

    if (cur.text == "scheme") {
      advance();
      expect(TokenType::Equals);

      if (cur.text == "fd")
        cfg.scheme = SpatialScheme::FiniteDifference;
      else if (cur.text == "spectral")
        cfg.scheme = SpatialScheme::Spectral;
      else
        syntaxError("unknown spatial scheme");

      advance();
      continue;
    }

    if (cur.text == "derivative") {
      advance();
      expect(TokenType::Equals);

      if (cur.text == "centered")
        cfg.derivative = DerivativeScheme::Centered;
      else if (cur.text == "upwind")
        cfg.derivative = DerivativeScheme::Upwind;
      else
        syntaxError("unknown derivative scheme");

      advance();
      continue;
    }

    if (cur.text == "order") {
      advance();
      expect(TokenType::Equals);

      if (cur.type != TokenType::Number)
        syntaxError("order expects an integer");

      cfg.order = std::stoi(cur.text);
      advance();
      continue;
    }

    syntaxError("unexpected entry in spatial block");
  }

  expect(TokenType::RBrace);
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

  while (cur.type != TokenType::RBrace) {

    if (cur.text == "coordinates") {
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

      advance();
      continue;
    }

    if (cur.text == "dimension") {
      advance();
      expect(TokenType::Equals);
      cfg.dimension = std::stoi(cur.text);
      expect(TokenType::Number);
      continue;
    }

    if (cur.text == "resolution") {
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
      continue;
    }

    if (cur.type == TokenType::KwTime) {
      cfg.time = parseTimeBlock();
      continue;
    }

    if (cur.type == TokenType::KwSpatial) {
      cfg.spatial = parseSpatialBlock();
      continue;
    }

    syntaxError("unexpected entry in simulation block");
  }

  expect(TokenType::RBrace);
  return cfg;
}

Program Parser::parseProgram() {
  Program p;
  while (cur.type != TokenType::End) {
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
