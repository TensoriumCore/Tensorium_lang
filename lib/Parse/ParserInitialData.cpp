#include "tensorium/Parse/Parser.hpp"

#include <set>
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
  init.name = problemName.empty() ? "initial_data" : problemName;
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

    if (cur.type == TokenType::Identifier && cur.text == "spectral") {
      if (init.hasSpectralProblem)
        syntaxError("duplicate spectral block in initial_data");
      init.hasSpectralProblem = true;
      init.spectralProblem = parseSpectralInitialData();
      continue;
    }

    if (cur.type == TokenType::Identifier && cur.text == "geometry") {
      if (init.constraintProblem.geometry.enabled)
        syntaxError("duplicate geometry block in initial_data");
      init.hasConstraintProblem = true;
      init.constraintProblem.geometry = parseConstraintGeometry();
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

    if (cur.type == TokenType::Identifier && cur.text == "reconstruct") {
      if (init.constraintProblem.cttReconstruction.enabled)
        syntaxError("duplicate reconstruct block in initial_data");
      init.hasConstraintProblem = true;
      init.constraintProblem.cttReconstruction =
          parseConstraintCttReconstruction();
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

  if (!init.hasMetric4 && !init.hasDecomposed && !init.hasConstraintProblem &&
      !init.hasSpectralProblem) {
    syntaxError(
        "initial_data requires analytic data, a radial constraint problem, "
        "or a spectral problem");
  }
  const int modeCount = static_cast<int>(init.hasMetric4 || init.hasDecomposed) +
                        static_cast<int>(init.hasConstraintProblem) +
                        static_cast<int>(init.hasSpectralProblem);
  if (modeCount > 1) {
    syntaxError("initial_data analytic, radial constraint, and spectral modes "
                "cannot be mixed");
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

  if (init.hasSpectralProblem) {
    const auto &problem = init.spectralProblem;
    if (problem.system.empty())
      syntaxError("spectral initial_data requires a residual system");
    if (problem.resolution.size() != 3)
      syntaxError("spectral initial_data resolution requires 3 entries");
    if (problem.basis.size() != 3)
      syntaxError("spectral initial_data basis requires 3 entries");
    if (!problem.hasSolve)
      syntaxError("spectral initial_data requires a solve block");
  }

  return init;
}

SpectralInitialDataDecl Parser::parseSpectralInitialData() {
  if (cur.type != TokenType::Identifier || cur.text != "spectral")
    syntaxError("expected spectral block");
  advance();
  expect(TokenType::LBrace);

  SpectralInitialDataDecl out;
  out.enabled = true;
  bool hasSystem = false;
  bool hasResolution = false;
  bool hasBasis = false;
  bool hasCoordinateMap = false;
  bool hasCoordinateParameters = false;
  bool hasUnknownMap = false;
  bool hasUnknownMapParameters = false;
  bool hasProjector = false;
  bool hasReconstruction = false;

  auto parseIdentifierValue = [&]() {
    if (cur.type != TokenType::Identifier)
      syntaxError("spectral initial_data property expects an identifier");
    std::string value = cur.text;
    advance();
    return value;
  };
  auto parseSignedNumber = [&]() {
    double sign = 1.0;
    if (cur.type == TokenType::Minus) {
      sign = -1.0;
      advance();
    } else if (cur.type == TokenType::Plus) {
      advance();
    }
    if (cur.type != TokenType::Number)
      syntaxError("spectral initial_data numeric property expects a number");
    const double value = sign * std::stod(cur.text);
    advance();
    return value;
  };
  auto parseIdentifierList = [&]() {
    std::vector<std::string> values;
    expect(TokenType::LBracket);
    while (cur.type != TokenType::RBracket) {
      values.push_back(parseIdentifierValue());
      if (cur.type == TokenType::Comma) {
        advance();
        continue;
      }
      if (cur.type != TokenType::RBracket)
        syntaxError("expected ',' or ']' in identifier list");
    }
    expect(TokenType::RBracket);
    return values;
  };
  auto parseNumberList = [&]() {
    std::vector<double> values;
    expect(TokenType::LBracket);
    while (cur.type != TokenType::RBracket) {
      values.push_back(parseSignedNumber());
      if (cur.type == TokenType::Comma) {
        advance();
        continue;
      }
      if (cur.type != TokenType::RBracket)
        syntaxError("expected ',' or ']' in numeric list");
    }
    expect(TokenType::RBracket);
    return values;
  };

  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }
    if (cur.type != TokenType::Identifier)
      syntaxError("spectral initial_data expects a property name");

    const std::string key = cur.text;
    if (key == "solve") {
      if (out.hasSolve)
        syntaxError("duplicate solve block in spectral initial_data");
      out.solve = parseConstraintSolve();
      out.hasSolve = true;
      continue;
    }
    if (key == "parameter") {
      advance();
      if (cur.type != TokenType::Identifier)
        syntaxError("parameter binding expects a parameter name");
      SpectralParameterBindingDecl binding;
      binding.name = cur.text;
      advance();
      expect(TokenType::Equals);
      binding.value = parseSignedNumber();
      out.parameters.push_back(std::move(binding));
      continue;
    }

    advance();
    expect(TokenType::Equals);
    if (key == "system") {
      if (hasSystem)
        syntaxError("duplicate system in spectral initial_data");
      out.system = parseIdentifierValue();
      hasSystem = true;
    } else if (key == "coordinate_map") {
      if (hasCoordinateMap)
        syntaxError("duplicate coordinate_map in spectral initial_data");
      out.coordinateMap = parseIdentifierValue();
      hasCoordinateMap = true;
    } else if (key == "resolution") {
      if (hasResolution)
        syntaxError("duplicate resolution in spectral initial_data");
      const auto numbers = parseNumberList();
      for (double number : numbers) {
        if (number != static_cast<double>(static_cast<int>(number)))
          syntaxError("spectral resolution expects integers");
        out.resolution.push_back(static_cast<int>(number));
      }
      hasResolution = true;
    } else if (key == "basis") {
      if (hasBasis)
        syntaxError("duplicate basis in spectral initial_data");
      out.basis = parseIdentifierList();
      hasBasis = true;
    } else if (key == "coordinate_parameters") {
      if (hasCoordinateParameters)
        syntaxError(
            "duplicate coordinate_parameters in spectral initial_data");
      out.coordinateParameters = parseIdentifierList();
      hasCoordinateParameters = true;
    } else if (key == "unknown_map") {
      if (hasUnknownMap)
        syntaxError("duplicate unknown_map in spectral initial_data");
      out.unknownMap = parseIdentifierValue();
      hasUnknownMap = true;
    } else if (key == "unknown_map_parameters") {
      if (hasUnknownMapParameters)
        syntaxError(
            "duplicate unknown_map_parameters in spectral initial_data");
      out.unknownMapParameters = parseNumberList();
      hasUnknownMapParameters = true;
    } else if (key == "field_projector") {
      if (hasProjector)
        syntaxError("duplicate field_projector in spectral initial_data");
      out.fieldProjector = parseIdentifierValue();
      hasProjector = true;
    } else if (key == "reconstruction") {
      if (hasReconstruction)
        syntaxError("duplicate reconstruction in spectral initial_data");
      out.reconstruction = parseIdentifierValue();
      hasReconstruction = true;
    } else {
      syntaxError("unknown spectral initial_data property '" + key + "'");
    }
  }

  expect(TokenType::RBrace);
  return out;
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
  if (cur.type == TokenType::Identifier && cur.text == "symmetric") {
    unknown.symmetric = true;
    advance();
  }
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

ConstraintGeometryDecl Parser::parseConstraintGeometry() {
  if (cur.type != TokenType::Identifier || cur.text != "geometry")
    syntaxError("expected geometry block");
  advance();

  if (cur.type != TokenType::Identifier)
    syntaxError("geometry expects a geometry kind");
  ConstraintGeometryDecl geometry;
  geometry.enabled = true;
  geometry.kind = cur.text;
  advance();
  expect(TokenType::LBrace);

  bool hasMetric = false;
  bool hasInverseMetric = false;
  bool hasRadialScale = false;
  bool hasTangentialScale = false;
  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }
    if (cur.type != TokenType::Identifier && cur.type != TokenType::KwMetric &&
        cur.type != TokenType::KwInverseMetric)
      syntaxError("geometry expects a property name");
    const std::string key = cur.text;
    advance();
    expect(TokenType::Equals);

    if (key == "metric" || key == "inverse_metric") {
      if (cur.type != TokenType::Identifier)
        syntaxError("geometry " + key + " expects a symbol name");
      if (key == "metric") {
        if (hasMetric)
          syntaxError("duplicate geometry metric property");
        geometry.metricName = cur.text;
        hasMetric = true;
      } else {
        if (hasInverseMetric)
          syntaxError("duplicate geometry inverse_metric property");
        geometry.inverseMetricName = cur.text;
        hasInverseMetric = true;
      }
      advance();
      continue;
    }

    if (key == "radial_scale") {
      if (hasRadialScale)
        syntaxError("duplicate geometry radial_scale property");
      geometry.radialScale = parseExpr();
      hasRadialScale = true;
      continue;
    }
    if (key == "tangential_scale") {
      if (hasTangentialScale)
        syntaxError("duplicate geometry tangential_scale property");
      geometry.tangentialScale = parseExpr();
      hasTangentialScale = true;
      continue;
    }
    syntaxError("unknown geometry property '" + key + "'");
  }
  expect(TokenType::RBrace);

  if (!hasMetric || !hasInverseMetric || !hasRadialScale ||
      !hasTangentialScale) {
    syntaxError("geometry requires metric, inverse_metric, radial_scale and "
                "tangential_scale");
  }
  return geometry;
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

ConstraintCttReconstructionDecl Parser::parseConstraintCttReconstruction() {
  if (cur.type != TokenType::Identifier || cur.text != "reconstruct")
    syntaxError("expected reconstruct block");
  advance();
  if (cur.type != TokenType::Identifier || cur.text != "ctt")
    syntaxError("reconstruct currently expects the 'ctt' formulation");
  advance();
  expect(TokenType::LBrace);

  ConstraintCttReconstructionDecl reconstruction;
  reconstruction.enabled = true;
  bool hasConformalFactor = false;
  bool hasRadialVector = false;
  bool hasConformalElectricRadial = false;
  bool hasMeanCurvature = false;
  while (cur.type != TokenType::RBrace && cur.type != TokenType::End) {
    if (cur.type == TokenType::Semicolon) {
      advance();
      continue;
    }
    if (cur.type != TokenType::Identifier)
      syntaxError("reconstruct ctt expects a property name");
    const std::string key = cur.text;
    advance();
    expect(TokenType::Equals);

    if (key == "conformal_factor" || key == "radial_vector" ||
        key == "conformal_electric_radial") {
      if (cur.type != TokenType::Identifier)
        syntaxError("reconstruct ctt property '" + key +
                    "' expects an unknown name");
      if (key == "conformal_factor") {
        if (hasConformalFactor)
          syntaxError("duplicate conformal_factor in reconstruct ctt");
        reconstruction.conformalFactor = cur.text;
        hasConformalFactor = true;
      } else if (key == "radial_vector") {
        if (hasRadialVector)
          syntaxError("duplicate radial_vector in reconstruct ctt");
        reconstruction.radialVectorPotential = cur.text;
        hasRadialVector = true;
      } else {
        if (hasConformalElectricRadial)
          syntaxError("duplicate conformal_electric_radial in reconstruct ctt");
        reconstruction.conformalElectricRadial = cur.text;
        hasConformalElectricRadial = true;
      }
      advance();
      continue;
    }
    if (key == "mean_curvature") {
      if (hasMeanCurvature)
        syntaxError("duplicate mean_curvature in reconstruct ctt");
      reconstruction.meanCurvature = parseExpr();
      hasMeanCurvature = true;
      continue;
    }
    syntaxError("unknown reconstruct ctt property '" + key + "'");
  }
  expect(TokenType::RBrace);
  if (!hasConformalFactor || !hasMeanCurvature) {
    syntaxError("reconstruct ctt requires conformal_factor and mean_curvature");
  }
  return reconstruction;
}

ConstraintSolveConfig Parser::parseConstraintSolve() {
  if (cur.type != TokenType::Identifier || cur.text != "solve")
    syntaxError("expected solve block");
  advance();
  expect(TokenType::LBrace);

  ConstraintSolveConfig solve;
  std::set<std::string> seen;
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

    if (!seen.insert(key).second)
      syntaxError("duplicate " + key + " in solve block");

    if (key == "tolerance" || key == "linear_tolerance" ||
        key == "linear_relative_tolerance" ||
        key == "jvp_relative_step" || key == "jvp_absolute_step") {
      if (cur.type != TokenType::Number)
        syntaxError(key + " expects a number");
      const double value = std::stod(cur.text);
      advance();
      if (key == "tolerance")
        solve.tolerance = value;
      else if (key == "linear_tolerance")
        solve.linearTolerance = value;
      else if (key == "linear_relative_tolerance")
        solve.linearRelativeTolerance = value;
      else if (key == "jvp_relative_step")
        solve.jvpRelativeStep = value;
      else
        solve.jvpAbsoluteStep = value;
      continue;
    }
    if (key == "max_iterations" || key == "max_linear_iterations" ||
        key == "restart" || key == "preconditioner_sweeps") {
      if (cur.type != TokenType::Number)
        syntaxError(key + " expects an integer");
      if (cur.text.find_first_of(".eE") != std::string::npos)
        syntaxError(key + " expects an integer");
      const int value = std::stoi(cur.text);
      advance();
      if (key == "max_iterations")
        solve.maxIterations = value;
      else if (key == "max_linear_iterations")
        solve.maxLinearIterations = value;
      else if (key == "restart")
        solve.restart = value;
      else
        solve.preconditionerSweeps = value;
      continue;
    }
    if (cur.type != TokenType::Identifier)
      syntaxError("solve property '" + key + "' expects an identifier");
    if (key == "nonlinear") {
      solve.nonlinear = cur.text;
    } else if (key == "linear") {
      solve.linear = cur.text;
    } else if (key == "preconditioner") {
      solve.preconditioner = cur.text;
    } else {
      syntaxError("unknown solve property '" + key + "'");
    }
    advance();
  }
  expect(TokenType::RBrace);
  if (!seen.count("nonlinear") || !seen.count("linear") ||
      !seen.count("tolerance") || !seen.count("max_iterations"))
    syntaxError(
        "solve requires nonlinear, linear, tolerance and max_iterations");
  return solve;
}

} // namespace tensorium
