#include "tensorium/Parse/Parser.hpp"

#include <string>

namespace tensorium {

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
      if (cur.text.find('.') != std::string::npos)
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
      if (cur.type != TokenType::Number)
        syntaxError("dimension expects an integer");
      if (cur.text.find('.') != std::string::npos)
        syntaxError("dimension expects an integer");
      cfg.dimension = std::stoi(cur.text);
      hasDimension = true;
      advance();
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
        if (cur.text.find('.') != std::string::npos)
          syntaxError("resolution expects integer values");
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
    syntaxError("simulation block requires 'time { dt = ... integrator = ... }'");
  if (!hasSpatial)
    syntaxError("simulation block requires "
                "'spatial { scheme = ... derivative = ... order = ... }'");
  return cfg;
}

} // namespace tensorium
