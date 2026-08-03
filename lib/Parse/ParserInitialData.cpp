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

} // namespace tensorium
