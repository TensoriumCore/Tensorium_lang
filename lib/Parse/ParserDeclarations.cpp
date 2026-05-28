#include "tensorium/Parse/Parser.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensorium {

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
      if (cur.text.find('.') != std::string::npos)
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

ConstraintEq Parser::parseConstraintEq() {
  expect(TokenType::KwResidual);
  if (cur.type != TokenType::Identifier)
    syntaxError("Field name after residual");
  ConstraintEq eq;
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

ConstraintDecl Parser::parseConstraints() {
  expect(TokenType::KwConstraints);
  if (cur.type != TokenType::Identifier)
    syntaxError("Constraints block name");
  ConstraintDecl constraints;
  constraints.name = cur.text;
  advance();
  expect(TokenType::LBrace);
  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::KwResidual) {
      constraints.residuals.push_back(parseConstraintEq());
      continue;
    }
    if (cur.type == TokenType::Identifier) {
      constraints.tempAssignments.push_back(parseAssignment());
      continue;
    }
    syntaxError("Expected residual or assign");
  }
  expect(TokenType::RBrace);
  return constraints;
}

PrintDecl Parser::parsePrint() {
  expect(TokenType::KwPrint);
  expect(TokenType::LParen);
  PrintDecl out;
  out.expr = parseExpr();
  expect(TokenType::RParen);
  expect(TokenType::Semicolon);
  return out;
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
    if (cur.type == TokenType::KwConstraints) {
      p.constraints.push_back(parseConstraints());
      continue;
    }
    if (cur.type == TokenType::KwPrint) {
      p.prints.push_back(parsePrint());
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
