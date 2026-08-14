#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualAssembly.h"

namespace tensorium_mlir::runtime {

inline double
spectralSystemMaxAbs(std::span<const std::vector<double>> fields) {
  double out = 0.0;
  for (const auto &field : fields) {
    const double fieldMax = spectralVectorMaxAbs(field);
    if (!std::isfinite(fieldMax))
      return fieldMax;
    out = std::max(out, fieldMax);
  }
  return out;
}

inline void
validateSpectralSystemFieldBundle(const SpectralGrid3D &grid,
                                  std::span<const std::vector<double>> fields,
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
    throw std::runtime_error(
        "spectral system JVP relative step must be positive");
  if (options.absoluteStep < 0.0 || !std::isfinite(options.absoluteStep))
    throw std::runtime_error(
        "spectral system JVP absolute step must be finite");

  const double stateMax = spectralSystemMaxAbs(values);
  const double directionMax = spectralSystemMaxAbs(directions);
  if (!std::isfinite(stateMax) || !std::isfinite(directionMax))
    throw std::runtime_error(
        "spectral system JVP state/direction must be finite");
  if (directionMax == 0.0)
    return 0.0;
  return std::max(options.absoluteStep, options.relativeStep *
                                            std::max(1.0, stateMax) /
                                            directionMax);
}

inline std::vector<std::vector<double>>
perturbSpectralSystemUnknowns(std::span<const std::vector<double>> values,
                              std::span<const std::vector<double>> directions,
                              double scale) {
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
makeSpectralResidualSystemJacobianVectorProductResult(
    std::vector<double> values, double step, bool usedGridKernels,
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

inline SpectralDerivatives3D
mapSpectralJvpDerivatives(const SpectralResidualProblem &problem,
                          const SpectralDerivatives3D &derivatives) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  SpectralDerivatives3D mapped = derivatives;
  if (problem.unknownMap.transform) {
    mapped = applySpectralUnknownMap(grid, mapped, problem.unknownMap,
                                     problem.unknownMapParams);
  }
  if (problem.fieldProjector.projectDerivatives) {
    problem.fieldProjector.projectDerivatives(&grid, &mapped,
                                              problem.fieldProjector.userData);
    validateSpectralDerivativeBundle(grid, mapped);
  }
  if (problem.derivativeMap.transform) {
    mapped = applySpectralDerivativeMap(grid, mapped, problem.derivativeMap,
                                        problem.coordinateParams);
  }
  return mapped;
}

inline std::vector<double> evaluateGeneratedSpectralJacobianVectorProduct(
    const SpectralResidualProblem &problem,
    const SpectralDerivatives3D &stateDerivatives,
    const SpectralDerivatives3D &directionDerivatives,
    std::span<const std::vector<double>> auxiliaryFields,
    std::span<const std::vector<double>> auxiliaryDirections = {},
    std::span<const SpectralDerivatives3D> auxiliaryDerivativeFields = {},
    std::span<const SpectralDerivatives3D> auxiliaryDirectionDerivativeFields =
        {}) {
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
  if (!auxiliaryDerivativeFields.empty() &&
      auxiliaryDerivativeFields.size() != auxiliaryFields.size()) {
    throw std::runtime_error(
        "spectral auxiliary JVP derivative count mismatch");
  }
  if (!auxiliaryDirectionDerivativeFields.empty() &&
      auxiliaryDirectionDerivativeFields.size() != auxiliaryFields.size()) {
    throw std::runtime_error(
        "spectral auxiliary JVP direction derivative count mismatch");
  }

  std::vector<SpectralDerivatives3D> computedAuxiliaryDerivatives;
  if (auxiliaryDerivativeFields.empty() && !auxiliaryFields.empty()) {
    computedAuxiliaryDerivatives.reserve(auxiliaryFields.size());
    for (const auto &field : auxiliaryFields) {
      SpectralDerivatives3D derivatives = grid.derivatives(field);
      if (problem.derivativeMap.transform) {
        derivatives = applySpectralDerivativeMap(
            grid, derivatives, problem.derivativeMap, problem.coordinateParams);
      }
      computedAuxiliaryDerivatives.push_back(std::move(derivatives));
    }
    auxiliaryDerivativeFields = std::span<const SpectralDerivatives3D>(
        computedAuxiliaryDerivatives.data(),
        computedAuxiliaryDerivatives.size());
  }
  std::vector<SpectralDerivatives3D> computedAuxiliaryDirectionDerivatives;
  if (auxiliaryDirectionDerivativeFields.empty() && !auxiliaryFields.empty()) {
    computedAuxiliaryDirectionDerivatives.reserve(auxiliaryFields.size());
    for (std::size_t auxiliary = 0; auxiliary < auxiliaryFields.size();
         ++auxiliary) {
      SpectralDerivatives3D derivatives =
          auxiliaryDirections.empty()
              ? SpectralDerivatives3D{std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0),
                                      std::vector<double>(grid.size(), 0.0)}
              : grid.derivatives(auxiliaryDirections[auxiliary]);
      if (problem.derivativeMap.transform) {
        derivatives = applySpectralDerivativeMap(
            grid, derivatives, problem.derivativeMap, problem.coordinateParams);
      }
      computedAuxiliaryDirectionDerivatives.push_back(std::move(derivatives));
    }
    auxiliaryDirectionDerivativeFields = std::span<const SpectralDerivatives3D>(
        computedAuxiliaryDirectionDerivatives.data(),
        computedAuxiliaryDirectionDerivatives.size());
  }
  for (const auto &derivatives : auxiliaryDerivativeFields)
    validateSpectralDerivativeBundle(grid, derivatives);
  for (const auto &derivatives : auxiliaryDirectionDerivativeFields)
    validateSpectralDerivativeBundle(grid, derivatives);

  const SpectralDerivatives3D state =
      mapSpectralJvpDerivatives(problem, stateDerivatives);
  const SpectralDerivatives3D direction =
      mapSpectralJvpDerivatives(problem, directionDerivatives);
  std::vector<double> out(grid.size(), 0.0);
  std::vector<double> pointAuxiliary(auxiliaryFields.size(), 0.0);
  std::vector<double> directionAuxiliary(auxiliaryFields.size(), 0.0);
  std::vector<tensorium_spectral_residual_derivatives>
      pointAuxiliaryDerivatives(auxiliaryFields.size());
  std::vector<tensorium_spectral_residual_derivatives>
      directionAuxiliaryDerivatives(auxiliaryFields.size());
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
          pointAuxiliaryDerivatives[auxiliary] =
              spectralResidualDerivativePoint(grid.pointDerivatives(
                  auxiliaryDerivativeFields[auxiliary], index));
          directionAuxiliaryDerivatives[auxiliary] =
              spectralResidualDerivativePoint(grid.pointDerivatives(
                  auxiliaryDirectionDerivativeFields[auxiliary], index));
        }
        const auto point = makeSpectralResidualPoint(
            grid, state, i, j, k, problem.coordinateMap,
            problem.coordinateParams, pointAuxiliary,
            pointAuxiliaryDerivatives);
        const auto tangent = makeSpectralResidualPoint(
            grid, direction, i, j, k, problem.coordinateMap,
            problem.coordinateParams, directionAuxiliary,
            directionAuxiliaryDerivatives);
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

  struct UnknownRepresentationBinding {
    SpectralUnknownMap map;
    std::span<const double> params{};
    SpectralFieldProjector projector;
    bool bound = false;
  };
  std::vector<UnknownRepresentationBinding> bindings(values.size());
  for (const auto &equation : system.equations) {
    if (equation.unknownIndex >= values.size())
      throw std::runtime_error(
          "spectral residual system JVP unknown index out of range");
    auto &binding = bindings[equation.unknownIndex];
    if (binding.bound) {
      const bool sameMap =
          binding.map.transform == equation.problem.unknownMap.transform &&
          binding.map.userData == equation.problem.unknownMap.userData;
      const bool sameParams =
          binding.params.size() == equation.problem.unknownMapParams.size() &&
          std::equal(binding.params.begin(), binding.params.end(),
                     equation.problem.unknownMapParams.begin());
      const bool sameProjector =
          binding.projector.project ==
              equation.problem.fieldProjector.project &&
          binding.projector.projectDerivatives ==
              equation.problem.fieldProjector.projectDerivatives &&
          binding.projector.userData ==
              equation.problem.fieldProjector.userData;
      if (!sameMap || !sameParams || !sameProjector) {
        throw std::runtime_error(
            "spectral residual system JVP has conflicting unknown "
            "representations");
      }
    } else {
      binding.map = equation.problem.unknownMap;
      binding.params = equation.problem.unknownMapParams;
      binding.projector = equation.problem.fieldProjector;
      binding.bound = true;
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

  std::vector<SpectralDerivatives3D> representedStateDerivatives;
  std::vector<SpectralDerivatives3D> representedDirectionDerivatives;
  representedStateDerivatives.reserve(values.size());
  representedDirectionDerivatives.reserve(values.size());
  for (std::size_t unknown = 0; unknown < values.size(); ++unknown) {
    SpectralDerivatives3D state = stateDerivatives[unknown];
    SpectralDerivatives3D direction = directionDerivatives[unknown];
    if (bindings[unknown].bound && bindings[unknown].map.transform) {
      state = applySpectralUnknownMap(grid, state, bindings[unknown].map,
                                      bindings[unknown].params);
      direction = applySpectralUnknownMap(
          grid, direction, bindings[unknown].map, bindings[unknown].params);
    }
    if (bindings[unknown].bound &&
        bindings[unknown].projector.projectDerivatives) {
      bindings[unknown].projector.projectDerivatives(
          &grid, &state, bindings[unknown].projector.userData);
      bindings[unknown].projector.projectDerivatives(
          &grid, &direction, bindings[unknown].projector.userData);
      validateSpectralDerivativeBundle(grid, state);
      validateSpectralDerivativeBundle(grid, direction);
    }
    representedStateDerivatives.push_back(std::move(state));
    representedDirectionDerivatives.push_back(std::move(direction));
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
    std::vector<SpectralDerivatives3D> auxiliaryDerivatives;
    std::vector<SpectralDerivatives3D> auxiliaryDirectionDerivatives;
    auxiliaryFields.reserve(problem.auxiliaryFields.size());
    auxiliaryDirections.reserve(problem.auxiliaryFields.size());
    auxiliaryDerivatives.reserve(problem.auxiliaryFields.size());
    auxiliaryDirectionDerivatives.reserve(problem.auxiliaryFields.size());
    if (!equation.auxiliaryUnknownIndices.empty() &&
        equation.auxiliaryUnknownIndices.size() !=
            problem.auxiliaryFields.size()) {
      throw std::runtime_error(
          "spectral residual system auxiliary JVP map size mismatch");
    }
    for (std::size_t auxiliary = 0; auxiliary < problem.auxiliaryFields.size();
         ++auxiliary) {
      const SpectralAuxiliaryUnknownIndex mappedUnknown =
          equation.auxiliaryUnknownIndices.empty()
              ? kSpectralStaticAuxiliary
              : equation.auxiliaryUnknownIndices[auxiliary];
      if (mappedUnknown == kSpectralStaticAuxiliary) {
        SpectralDerivatives3D state =
            grid.derivatives(problem.auxiliaryFields[auxiliary]);
        SpectralDerivatives3D direction =
            grid.derivatives(std::vector<double>(grid.size(), 0.0));
        if (problem.derivativeMap.transform) {
          state = applySpectralDerivativeMap(grid, state, problem.derivativeMap,
                                             problem.coordinateParams);
          direction = applySpectralDerivativeMap(
              grid, direction, problem.derivativeMap, problem.coordinateParams);
        }
        auxiliaryFields.push_back(state.value);
        auxiliaryDirections.push_back(direction.value);
        auxiliaryDerivatives.push_back(std::move(state));
        auxiliaryDirectionDerivatives.push_back(std::move(direction));
        continue;
      }
      if (mappedUnknown < 0 ||
          static_cast<std::size_t>(mappedUnknown) >= values.size()) {
        throw std::runtime_error(
            "spectral residual system auxiliary JVP unknown out of range");
      }
      const std::size_t unknown = static_cast<std::size_t>(mappedUnknown);
      SpectralDerivatives3D state = representedStateDerivatives[unknown];
      SpectralDerivatives3D direction =
          representedDirectionDerivatives[unknown];
      if (problem.derivativeMap.transform) {
        state = applySpectralDerivativeMap(grid, state, problem.derivativeMap,
                                           problem.coordinateParams);
        direction = applySpectralDerivativeMap(
            grid, direction, problem.derivativeMap, problem.coordinateParams);
      }
      auxiliaryFields.push_back(state.value);
      auxiliaryDirections.push_back(direction.value);
      auxiliaryDerivatives.push_back(std::move(state));
      auxiliaryDirectionDerivatives.push_back(std::move(direction));
    }

    const auto equationJvp = evaluateGeneratedSpectralJacobianVectorProduct(
        problem, stateDerivatives[equation.unknownIndex],
        directionDerivatives[equation.unknownIndex], auxiliaryFields,
        auxiliaryDirections, auxiliaryDerivatives,
        auxiliaryDirectionDerivatives);
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
  const double step = spectralSystemJacobianVectorProductStep(
      grid, values, directions, options);
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
    usedGridKernels = usedGridKernels && minusResidual.usedGeneratedGridKernels;
    const double scale = 0.5 / step;
    for (std::size_t p = 0; p < out.size(); ++p)
      out[p] = (plusResidual.values[p] - minusResidual.values[p]) * scale;
  } else {
    const auto baseResidual = assembleSpectralResidualSystem(system, values);
    if (baseResidual.values.size() != out.size())
      throw std::runtime_error("spectral residual system JVP size mismatch");
    usedGridKernels = usedGridKernels && baseResidual.usedGeneratedGridKernels;
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
  return std::max(options.absoluteStep, options.relativeStep *
                                            std::max(1.0, stateMax) /
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
  const SpectralResidualProblem problem{
      &grid, kernel, params, auxiliaryFields, coordinateMap, coordinateParams};
  return evaluateSpectralJacobianVectorProduct(problem, values, direction,
                                               options);
}

inline double spectralResidualRatio(double initialResidualL2,
                                    double residualL2) {
  if (initialResidualL2 > 0.0)
    return residualL2 / initialResidualL2;
  return residualL2 == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

inline bool
reachedSpectralResidualTarget(const SpectralEllipticSolveResult &result,
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
