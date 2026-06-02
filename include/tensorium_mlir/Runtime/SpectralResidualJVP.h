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
                                                      bool usedGridKernels) {
  SpectralResidualSystemJacobianVectorProductResult result;
  result.values = std::move(values);
  result.step = step;
  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = spectralVectorIsFinite(result.values);
  result.usedGeneratedGridKernels = usedGridKernels;
  return result;
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
        false);
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
        std::vector<double>(grid.size(), 0.0), step);

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
