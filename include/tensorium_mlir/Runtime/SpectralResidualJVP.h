#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualAssembly.h"

namespace tensorium_mlir::runtime {

inline double spectralSystemMaxAbs(std::span<const std::vector<double>> fields) {
  double out = 0.0;
  for (const auto &field : fields) {
    const double fieldMax = spectralVectorMaxAbs(field);
    if (!std::isfinite(fieldMax))
      return fieldMax;
    out = std::max(out, fieldMax);
  }
  return out;
}

inline void validateSpectralSystemFieldBundle(
    const SpectralGrid3D &grid, std::span<const std::vector<double>> fields,
    const char *label) {
  if (fields.empty())
    throw std::runtime_error(std::string("spectral residual system ") + label +
                             " bundle is empty");
  for (const auto &field : fields) {
    if (field.size() != grid.size()) {
      throw std::runtime_error(std::string("spectral residual system ") +
                               label + " size mismatch");
    }
  }
}

inline double spectralSystemJacobianVectorProductStep(
    const SpectralGrid3D &grid, std::span<const std::vector<double>> values,
    std::span<const std::vector<double>> directions,
    const SpectralJacobianVectorProductOptions &options) {
  validateSpectralSystemFieldBundle(grid, values, "state");
  validateSpectralSystemFieldBundle(grid, directions, "direction");
  if (values.size() != directions.size())
    throw std::runtime_error(
        "spectral residual system state/direction count mismatch");
  if (!(options.relativeStep > 0.0) || !std::isfinite(options.relativeStep))
    throw std::runtime_error("spectral system JVP relative step must be positive");
  if (options.absoluteStep < 0.0 || !std::isfinite(options.absoluteStep))
    throw std::runtime_error("spectral system JVP absolute step must be finite");

  const double stateMax = spectralSystemMaxAbs(values);
  const double directionMax = spectralSystemMaxAbs(directions);
  if (!std::isfinite(stateMax) || !std::isfinite(directionMax))
    throw std::runtime_error("spectral system JVP state/direction must be finite");
  if (directionMax == 0.0)
    return 0.0;
  return std::max(options.absoluteStep,
                  options.relativeStep * std::max(1.0, stateMax) /
                      directionMax);
}

inline std::vector<std::vector<double>> perturbSpectralSystemUnknowns(
    std::span<const std::vector<double>> values,
    std::span<const std::vector<double>> directions, double scale) {
  if (values.size() != directions.size())
    throw std::runtime_error(
        "spectral residual system perturbation count mismatch");
  std::vector<std::vector<double>> out;
  out.reserve(values.size());
  for (std::size_t field = 0; field < values.size(); ++field) {
    if (values[field].size() != directions[field].size())
      throw std::runtime_error(
          "spectral residual system perturbation size mismatch");
    out.push_back(values[field]);
    for (std::size_t p = 0; p < out.back().size(); ++p)
      out.back()[p] += scale * directions[field][p];
  }
  return out;
}

inline SpectralResidualSystemJacobianVectorProductResult
makeSpectralResidualSystemJacobianVectorProductResult(std::vector<double> values,
                                                      double step,
                                                      bool usedGridKernels,
                                                      bool usedJvpKernels = false) {
  SpectralResidualSystemJacobianVectorProductResult result;
  result.values = std::move(values);
  result.step = step;
  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = spectralVectorIsFinite(result.values);
  result.usedGeneratedGridKernels = usedGridKernels;
  result.usedGeneratedJvpKernels = usedJvpKernels;
  return result;
}

inline SpectralDerivatives3D mapSpectralJvpDerivatives(
    const SpectralResidualProblem &problem,
    const SpectralDerivatives3D &derivatives) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  SpectralDerivatives3D mapped = derivatives;
  if (problem.unknownMap.transform) {
    mapped = applySpectralUnknownMap(grid, mapped, problem.unknownMap,
                                     problem.unknownMapParams);
  }
  if (problem.fieldProjector.projectDerivatives) {
    problem.fieldProjector.projectDerivatives(
        &grid, &mapped, problem.fieldProjector.userData);
    validateSpectralDerivativeBundle(grid, mapped);
  }
  if (problem.derivativeMap.transform) {
    mapped = applySpectralDerivativeMap(
        grid, mapped, problem.derivativeMap, problem.coordinateParams);
  }
  return mapped;
}

inline std::vector<double> evaluateGeneratedSpectralJacobianVectorProduct(
    const SpectralResidualProblem &problem,
    const SpectralDerivatives3D &stateDerivatives,
    const SpectralDerivatives3D &directionDerivatives,
    std::span<const std::vector<double>> auxiliaryFields,
    std::span<const std::vector<double>> auxiliaryDirections = {}) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  if (!problem.kernel.evaluateJvp)
    throw std::runtime_error("spectral residual JVP kernel callback is null");
  validateSpectralDerivativeBundle(grid, stateDerivatives);
  validateSpectralDerivativeBundle(grid, directionDerivatives);
  if (!auxiliaryDirections.empty() &&
      auxiliaryDirections.size() != auxiliaryFields.size()) {
    throw std::runtime_error("spectral auxiliary JVP field count mismatch");
  }
  for (const auto &field : auxiliaryFields) {
    if (field.size() != grid.size())
      throw std::runtime_error("spectral auxiliary JVP field size mismatch");
  }
  for (const auto &field : auxiliaryDirections) {
    if (field.size() != grid.size()) {
      throw std::runtime_error(
          "spectral auxiliary JVP direction size mismatch");
    }
  }

  const SpectralDerivatives3D state =
      mapSpectralJvpDerivatives(problem, stateDerivatives);
  const SpectralDerivatives3D direction =
      mapSpectralJvpDerivatives(problem, directionDerivatives);
  std::vector<double> out(grid.size(), 0.0);
  std::vector<double> pointAuxiliary(auxiliaryFields.size(), 0.0);
  std::vector<double> directionAuxiliary(auxiliaryFields.size(), 0.0);
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const std::size_t index = grid.index(i, j, k);
        for (std::size_t auxiliary = 0; auxiliary < auxiliaryFields.size();
             ++auxiliary) {
          pointAuxiliary[auxiliary] = auxiliaryFields[auxiliary][index];
          directionAuxiliary[auxiliary] =
              auxiliaryDirections.empty()
                  ? 0.0
                  : auxiliaryDirections[auxiliary][index];
        }
        const auto point = makeSpectralResidualPoint(
            grid, state, i, j, k, problem.coordinateMap,
            problem.coordinateParams, pointAuxiliary);
        const auto tangent = makeSpectralResidualPoint(
            grid, direction, i, j, k, problem.coordinateMap,
            problem.coordinateParams, directionAuxiliary);
        out[index] = problem.kernel.evaluateJvp(
            &point, &tangent, problem.params.data(),
            static_cast<std::int64_t>(problem.params.size()),
            problem.kernel.userData);
      }
    }
  }
  return out;
}

inline bool spectralResidualSystemHasGeneratedJvpKernels(
    const SpectralResidualSystemProblem &system) {
  return !system.equations.empty() &&
         std::all_of(system.equations.begin(), system.equations.end(),
                     [](const SpectralResidualSystemEquation &equation) {
                       return equation.problem.kernel.evaluateJvp != nullptr;
                     });
}

inline SpectralResidualSystemJacobianVectorProductResult
evaluateGeneratedSpectralResidualSystemJacobianVectorProduct(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> values,
    std::span<const std::vector<double>> directions, double diagnosticStep) {
  const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
  if (!spectralResidualSystemHasGeneratedJvpKernels(system))
    throw std::runtime_error("spectral residual system JVP kernel is missing");

  struct UnknownMapBinding {
    SpectralUnknownMap map;
    std::span<const double> params{};
    bool bound = false;
  };
  std::vector<UnknownMapBinding> bindings(values.size());
  for (const auto &equation : system.equations) {
    if (equation.unknownIndex >= values.size())
      throw std::runtime_error(
          "spectral residual system JVP unknown index out of range");
    if (!equation.problem.unknownMap.transform)
      continue;
    auto &binding = bindings[equation.unknownIndex];
    if (binding.bound) {
      const bool sameMap =
          binding.map.transform == equation.problem.unknownMap.transform &&
          binding.map.userData == equation.problem.unknownMap.userData;
      const bool sameParams =
          binding.params.size() == equation.problem.unknownMapParams.size() &&
          std::equal(binding.params.begin(), binding.params.end(),
                     equation.problem.unknownMapParams.begin());
      if (!sameMap || !sameParams) {
        throw std::runtime_error(
            "spectral residual system JVP has conflicting unknown maps");
      }
    } else {
      binding.map = equation.problem.unknownMap;
      binding.params = equation.problem.unknownMapParams;
      binding.bound = true;
    }
  }
  for (const auto &equation : system.equations) {
    if (bindings[equation.unknownIndex].bound &&
        !equation.problem.unknownMap.transform) {
      throw std::runtime_error(
          "spectral residual system JVP equation is missing its unknown map");
    }
  }

  std::vector<SpectralDerivatives3D> stateDerivatives;
  std::vector<SpectralDerivatives3D> directionDerivatives;
  stateDerivatives.reserve(values.size());
  directionDerivatives.reserve(directions.size());
  for (std::size_t unknown = 0; unknown < values.size(); ++unknown) {
    stateDerivatives.push_back(grid.derivatives(values[unknown]));
    directionDerivatives.push_back(grid.derivatives(directions[unknown]));
  }

  std::vector<std::vector<double>> physicalValues(values.size());
  std::vector<std::vector<double>> physicalDirections(values.size());
  for (std::size_t unknown = 0; unknown < values.size(); ++unknown) {
    if (!bindings[unknown].bound)
      continue;
    physicalValues[unknown] =
        applySpectralUnknownMap(grid, stateDerivatives[unknown],
                                bindings[unknown].map,
                                bindings[unknown].params)
            .value;
    physicalDirections[unknown] =
        applySpectralUnknownMap(grid, directionDerivatives[unknown],
                                bindings[unknown].map,
                                bindings[unknown].params)
            .value;
  }

  std::vector<double> out;
  out.reserve(system.equations.size() * grid.size());
  for (const auto &equation : system.equations) {
    SpectralResidualProblem problem = equation.problem;
    if (!problem.grid)
      problem.grid = &grid;
    if (problem.grid != &grid)
      throw std::runtime_error("spectral residual system JVP grid mismatch");

    std::vector<std::vector<double>> auxiliaryFields;
    std::vector<std::vector<double>> auxiliaryDirections;
    auxiliaryFields.reserve(problem.auxiliaryFields.size());
    auxiliaryDirections.reserve(problem.auxiliaryFields.size());
    if (!equation.auxiliaryUnknownIndices.empty() &&
        equation.auxiliaryUnknownIndices.size() !=
            problem.auxiliaryFields.size()) {
      throw std::runtime_error(
          "spectral residual system auxiliary JVP map size mismatch");
    }
    for (std::size_t auxiliary = 0;
         auxiliary < problem.auxiliaryFields.size(); ++auxiliary) {
      const SpectralAuxiliaryUnknownIndex mappedUnknown =
          equation.auxiliaryUnknownIndices.empty()
              ? kSpectralStaticAuxiliary
              : equation.auxiliaryUnknownIndices[auxiliary];
      if (mappedUnknown == kSpectralStaticAuxiliary) {
        auxiliaryFields.push_back(problem.auxiliaryFields[auxiliary]);
        auxiliaryDirections.emplace_back(grid.size(), 0.0);
        continue;
      }
      if (mappedUnknown < 0 ||
          static_cast<std::size_t>(mappedUnknown) >= values.size()) {
        throw std::runtime_error(
            "spectral residual system auxiliary JVP unknown out of range");
      }
      const std::size_t unknown = static_cast<std::size_t>(mappedUnknown);
      auxiliaryFields.push_back(bindings[unknown].bound
                                    ? physicalValues[unknown]
                                    : values[unknown]);
      auxiliaryDirections.push_back(bindings[unknown].bound
                                        ? physicalDirections[unknown]
                                        : directions[unknown]);
    }

    const auto equationJvp = evaluateGeneratedSpectralJacobianVectorProduct(
        problem, stateDerivatives[equation.unknownIndex],
        directionDerivatives[equation.unknownIndex], auxiliaryFields,
        auxiliaryDirections);
    out.insert(out.end(), equationJvp.begin(), equationJvp.end());
  }
  return makeSpectralResidualSystemJacobianVectorProductResult(
      std::move(out), diagnosticStep, false, true);
}

inline SpectralResidualSystemJacobianVectorProductResult
evaluateSpectralResidualSystemJacobianVectorProduct(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> values,
    std::span<const std::vector<double>> directions,
    const SpectralJacobianVectorProductOptions &options = {}) {
  const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
  const double step =
      spectralSystemJacobianVectorProductStep(grid, values, directions, options);
  if (step == 0.0) {
    return makeSpectralResidualSystemJacobianVectorProductResult(
        std::vector<double>(system.equations.size() * grid.size(), 0.0), step,
        false, spectralResidualSystemHasGeneratedJvpKernels(system));
  }

  if (spectralResidualSystemHasGeneratedJvpKernels(system)) {
    return evaluateGeneratedSpectralResidualSystemJacobianVectorProduct(
        system, values, directions, step);
  }

  const auto plus = perturbSpectralSystemUnknowns(values, directions, step);
  const auto plusResidual = assembleSpectralResidualSystem(
      system, std::span<const std::vector<double>>(plus.data(), plus.size()));

  std::vector<double> out(plusResidual.values.size(), 0.0);
  bool usedGridKernels = plusResidual.usedGeneratedGridKernels;
  if (options.centeredDifference) {
    const auto minus = perturbSpectralSystemUnknowns(values, directions, -step);
    const auto minusResidual = assembleSpectralResidualSystem(
        system,
        std::span<const std::vector<double>>(minus.data(), minus.size()));
    if (minusResidual.values.size() != out.size())
      throw std::runtime_error("spectral residual system JVP size mismatch");
    usedGridKernels =
        usedGridKernels && minusResidual.usedGeneratedGridKernels;
    const double scale = 0.5 / step;
    for (std::size_t p = 0; p < out.size(); ++p)
      out[p] = (plusResidual.values[p] - minusResidual.values[p]) * scale;
  } else {
    const auto baseResidual = assembleSpectralResidualSystem(system, values);
    if (baseResidual.values.size() != out.size())
      throw std::runtime_error("spectral residual system JVP size mismatch");
    usedGridKernels =
        usedGridKernels && baseResidual.usedGeneratedGridKernels;
    const double scale = 1.0 / step;
    for (std::size_t p = 0; p < out.size(); ++p)
      out[p] = (plusResidual.values[p] - baseResidual.values[p]) * scale;
  }

  return makeSpectralResidualSystemJacobianVectorProductResult(
      std::move(out), step, usedGridKernels);
}

inline double spectralJacobianVectorProductStep(
    const SpectralGrid3D &grid, std::span<const double> values,
    std::span<const double> direction,
    const SpectralJacobianVectorProductOptions &options) {
  if (values.size() != grid.size())
    throw std::runtime_error("spectral residual state size mismatch");
  if (direction.size() != grid.size())
    throw std::runtime_error("spectral residual direction size mismatch");
  if (!(options.relativeStep > 0.0) || !std::isfinite(options.relativeStep))
    throw std::runtime_error("spectral JVP relative step must be positive");
  if (options.absoluteStep < 0.0 || !std::isfinite(options.absoluteStep))
    throw std::runtime_error("spectral JVP absolute step must be finite");

  const double stateMax = spectralVectorMaxAbs(values);
  const double directionMax = spectralVectorMaxAbs(direction);
  if (!std::isfinite(stateMax) || !std::isfinite(directionMax))
    throw std::runtime_error("spectral JVP state and direction must be finite");
  if (directionMax == 0.0)
    return 0.0;
  return std::max(options.absoluteStep,
                  options.relativeStep * std::max(1.0, stateMax) /
                      directionMax);
}

inline SpectralJacobianVectorProductResult
evaluateSpectralJacobianVectorProduct(
    const SpectralResidualProblem &problem, const std::vector<double> &values,
    const std::vector<double> &direction,
    const SpectralJacobianVectorProductOptions &options = {}) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  const double step =
      spectralJacobianVectorProductStep(grid, values, direction, options);
  if (step == 0.0)
    return makeSpectralJacobianVectorProductResult(
        std::vector<double>(grid.size(), 0.0), step,
        problem.kernel.evaluateJvp != nullptr);

  if (problem.kernel.evaluateJvp) {
    return makeSpectralJacobianVectorProductResult(
        evaluateGeneratedSpectralJacobianVectorProduct(
            problem, grid.derivatives(values), grid.derivatives(direction),
            problem.auxiliaryFields),
        step, true);
  }

  std::vector<double> plus(values.size(), 0.0);
  for (std::size_t p = 0; p < values.size(); ++p)
    plus[p] = values[p] + step * direction[p];
  const auto plusResidual = assembleSpectralResidual(problem, plus);

  std::vector<double> out(values.size(), 0.0);
  if (options.centeredDifference) {
    std::vector<double> minus(values.size(), 0.0);
    for (std::size_t p = 0; p < values.size(); ++p)
      minus[p] = values[p] - step * direction[p];
    const auto minusResidual = assembleSpectralResidual(problem, minus);
    const double scale = 0.5 / step;
    for (std::size_t p = 0; p < values.size(); ++p)
      out[p] = (plusResidual.values[p] - minusResidual.values[p]) * scale;
  } else {
    const auto baseResidual = assembleSpectralResidual(problem, values);
    const double scale = 1.0 / step;
    for (std::size_t p = 0; p < values.size(); ++p)
      out[p] = (plusResidual.values[p] - baseResidual.values[p]) * scale;
  }

  return makeSpectralJacobianVectorProductResult(std::move(out), step);
}

inline SpectralJacobianVectorProductResult
evaluateSpectralJacobianVectorProduct(
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const std::vector<double> &direction, const SpectralResidualKernel &kernel,
    std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields = {},
    const SpectralJacobianVectorProductOptions &options = {},
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  const SpectralResidualProblem problem{&grid, kernel, params, auxiliaryFields,
                                        coordinateMap, coordinateParams};
  return evaluateSpectralJacobianVectorProduct(problem, values, direction,
                                               options);
}

inline double spectralResidualRatio(double initialResidualL2,
                                    double residualL2) {
  if (initialResidualL2 > 0.0)
    return residualL2 / initialResidualL2;
  return residualL2 == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

inline bool reachedSpectralResidualTarget(
    const SpectralEllipticSolveResult &result,
    const SpectralEllipticSolveOptions &options) {
  if (!result.residualIsFinite())
    return false;
  if (options.residualTolerance > 0.0 &&
      result.finalResidualL2 <= options.residualTolerance)
    return true;
  if (options.residualRatioTarget > 0.0 &&
      result.residualRatio <= options.residualRatioTarget)
    return true;
  return false;
}

} // namespace tensorium_mlir::runtime
