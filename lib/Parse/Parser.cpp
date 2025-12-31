#include "tensorium/Parse/Parser.hpp"
#include <stdexcept>

namespace tensorium {
Parser::Parser(Lexer &l) : lex(l) { advance(); }
void Parser::advance() { cur = lex.next(); }
void Parser::expect(TokenType type) {
  if (cur.type != type)
    syntaxError("Expected " + std::to_string((int)type));
  advance();
}
void Parser::syntaxError(const std::string &msg) {
  throw std::runtime_error("Syntax: " + msg + " at " +
                           std::to_string(cur.line));
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

FieldDecl Parser::parseFieldDecl() {
  expect(TokenType::KwField);

  TensorKind k;
  int u = 0, d = 0;

  if (cur.type == TokenType::KwScalar) {
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
  } else {
    syntaxError("Unknown field type '" + cur.text + "'");
  }

  advance();

  if (cur.type != TokenType::Identifier)
    syntaxError("Expected field name");
  FieldDecl f;
  f.kind = k;
  f.up = u;
  f.down = d;
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
    syntaxError("Unexpected top level");
  }
  return p;
}
} // namespace tensorium
