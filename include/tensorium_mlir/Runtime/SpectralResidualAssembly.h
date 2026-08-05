#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualTypes.h"

namespace tensorium_mlir::runtime {

inline SpectralResidualKernel
spectralResidualKernelFromDesc(const tensorium_spectral_residual_kernel_desc &desc) {
  if (!desc.symbol_name || desc.symbol_name[0] == '\0')
    throw std::runtime_error("spectral residual kernel symbol is empty");
  if (!desc.evaluate)
    throw std::runtime_error("spectral residual kernel callback is null");
  return SpectralResidualKernel{desc.symbol_name, desc.evaluate,
                                desc.user_data};
}

inline SpectralResidualGridKernel spectralResidualGridKernelFromDesc(
    const tensorium_spectral_residual_grid_kernel_desc &desc) {
  if (!desc.symbol_name || desc.symbol_name[0] == '\0')
    throw std::runtime_error("spectral residual grid kernel symbol is empty");
  if (!desc.evaluate)
    throw std::runtime_error("spectral residual grid kernel callback is null");
  return SpectralResidualGridKernel{desc.symbol_name, desc.evaluate,
                                    desc.user_data};
}

inline SpectralCoordinateMap
spectralCoordinateMapFromDesc(const tensorium_spectral_coordinate_map_desc &desc) {
  if (!desc.symbol_name || desc.symbol_name[0] == '\0')
    throw std::runtime_error("spectral coordinate map symbol is empty");
  if (!desc.map)
    throw std::runtime_error("spectral coordinate map callback is null");
  return SpectralCoordinateMap{desc.symbol_name, desc.map, desc.user_data};
}

inline SpectralGeneratedResidualSystem makeSpectralResidualSystemFromDesc(
    const tensorium_spectral_residual_system_desc &desc,
    const SpectralGrid3D &grid,
    const tensorium_spectral_residual_kernel_desc *pointKernelDescs,
    std::size_t pointKernelCount,
    const tensorium_spectral_residual_grid_kernel_desc *gridKernelDescs,
    std::size_t gridKernelCount,
    std::span<const SpectralGeneratedResidualSystemEquationInputs> inputs) {
  if (!desc.symbol_name || !desc.equations || desc.equation_count <= 0 ||
      desc.unknown_count <= 0) {
    throw std::runtime_error("spectral residual system descriptor is invalid");
  }
  if (!pointKernelDescs)
    throw std::runtime_error("spectral residual system point kernels are null");
  if (inputs.size() != static_cast<std::size_t>(desc.equation_count)) {
    throw std::runtime_error(
        "spectral residual system input count mismatch");
  }

  SpectralGeneratedResidualSystem out;
  out.grid = &grid;
  out.symbolName = desc.symbol_name;
  out.equations.reserve(static_cast<std::size_t>(desc.equation_count));

  for (std::int64_t i = 0; i < desc.equation_count; ++i) {
    const auto &equationDesc = desc.equations[i];
    if (!equationDesc.residual_name || !equationDesc.unknown_name ||
        equationDesc.unknown_index < 0 ||
        equationDesc.unknown_index >= desc.unknown_count ||
        equationDesc.point_kernel_index < 0 ||
        static_cast<std::size_t>(equationDesc.point_kernel_index) >=
            pointKernelCount) {
      throw std::runtime_error(
          "spectral residual system equation descriptor is invalid");
    }
    if (equationDesc.param_count < 0 || equationDesc.auxiliary_count < 0)
      throw std::runtime_error(
          "spectral residual system equation descriptor count is invalid");
    const auto &input = inputs[static_cast<std::size_t>(i)];
    if (input.params.size() !=
        static_cast<std::size_t>(equationDesc.param_count)) {
      throw std::runtime_error(
          "spectral residual system parameter count mismatch");
    }
    if (input.auxiliaryFields.size() !=
        static_cast<std::size_t>(equationDesc.auxiliary_count)) {
      throw std::runtime_error(
          "spectral residual system auxiliary count mismatch");
    }
    if (equationDesc.auxiliary_count > 0 &&
        !equationDesc.auxiliary_unknown_indices) {
      throw std::runtime_error(
          "spectral residual system auxiliary map is null");
    }

    SpectralResidualProblem problem{
        &grid,
        spectralResidualKernelFromDesc(
            pointKernelDescs[equationDesc.point_kernel_index]),
        input.params,
        input.auxiliaryFields};
    if (equationDesc.grid_kernel_index >= 0) {
      if (!gridKernelDescs ||
          static_cast<std::size_t>(equationDesc.grid_kernel_index) >=
              gridKernelCount) {
        throw std::runtime_error(
            "spectral residual system grid kernel index out of range");
      }
      problem.gridKernel = spectralResidualGridKernelFromDesc(
          gridKernelDescs[equationDesc.grid_kernel_index]);
    }

    out.equations.push_back(SpectralResidualSystemEquation{
        problem,
        static_cast<std::size_t>(equationDesc.unknown_index),
        equationDesc.residual_name,
        std::span<const SpectralAuxiliaryUnknownIndex>(
            equationDesc.auxiliary_unknown_indices,
            static_cast<std::size_t>(equationDesc.auxiliary_count))});
  }
  return out;
}

inline void validateSpectralDerivativeBundle(const SpectralGrid3D &grid,
                                             const SpectralDerivatives3D &derivs) {
  const std::size_t size = grid.size();
  if (derivs.value.size() != size || derivs.d1.size() != size ||
      derivs.d2.size() != size || derivs.d3.size() != size ||
      derivs.d11.size() != size || derivs.d12.size() != size ||
      derivs.d13.size() != size || derivs.d22.size() != size ||
      derivs.d23.size() != size || derivs.d33.size() != size) {
    throw std::runtime_error("spectral derivative bundle size mismatch");
  }
}

inline SpectralDerivatives3D applySpectralDerivativeMap(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &logicalDerivatives,
    const SpectralDerivativeMap &derivativeMap,
    std::span<const double> coordinateParams = {}) {
  validateSpectralDerivativeBundle(grid, logicalDerivatives);
  if (!derivativeMap.transform)
    return logicalDerivatives;

  SpectralDerivatives3D physicalDerivatives;
  physicalDerivatives.value.resize(grid.size());
  physicalDerivatives.d1.resize(grid.size());
  physicalDerivatives.d2.resize(grid.size());
  physicalDerivatives.d3.resize(grid.size());
  physicalDerivatives.d11.resize(grid.size());
  physicalDerivatives.d12.resize(grid.size());
  physicalDerivatives.d13.resize(grid.size());
  physicalDerivatives.d22.resize(grid.size());
  physicalDerivatives.d23.resize(grid.size());
  physicalDerivatives.d33.resize(grid.size());

  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const SpectralPoint3D point = grid.point(i, j, k);
        const double logical[3] = {point.x1, point.x2, point.x3};
        const SpectralPointDerivatives3D logicalPoint =
            grid.pointDerivatives(logicalDerivatives, point.index);
        SpectralPointDerivatives3D physicalPoint{};
        derivativeMap.transform(
            logical, &logicalPoint, &physicalPoint, coordinateParams.data(),
            static_cast<std::int64_t>(coordinateParams.size()),
            derivativeMap.userData);

        physicalDerivatives.value[point.index] = physicalPoint.value;
        physicalDerivatives.d1[point.index] = physicalPoint.d1;
        physicalDerivatives.d2[point.index] = physicalPoint.d2;
        physicalDerivatives.d3[point.index] = physicalPoint.d3;
        physicalDerivatives.d11[point.index] = physicalPoint.d11;
        physicalDerivatives.d12[point.index] = physicalPoint.d12;
        physicalDerivatives.d13[point.index] = physicalPoint.d13;
        physicalDerivatives.d22[point.index] = physicalPoint.d22;
        physicalDerivatives.d23[point.index] = physicalPoint.d23;
        physicalDerivatives.d33[point.index] = physicalPoint.d33;
      }
    }
  }
  return physicalDerivatives;
}

inline SpectralDerivatives3D applySpectralUnknownMap(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &solverDerivatives,
    const SpectralUnknownMap &unknownMap,
    std::span<const double> unknownMapParams = {}) {
  const SpectralDerivativeMap derivativeTransform{
      unknownMap.symbolName, unknownMap.transform, unknownMap.userData};
  return applySpectralDerivativeMap(grid, solverDerivatives,
                                    derivativeTransform, unknownMapParams);
}

inline double spectralVectorMaxAbs(std::span<const double> values) {
  double out = 0.0;
  for (double value : values) {
    if (!std::isfinite(value))
      return value;
    out = std::max(out, std::fabs(value));
  }
  return out;
}

inline double spectralVectorL2Norm(std::span<const double> values) {
  if (values.empty())
    return 0.0;
  double sum = 0.0;
  for (double value : values) {
    if (!std::isfinite(value))
      return value;
    sum += value * value;
  }
  return std::sqrt(sum / static_cast<double>(values.size()));
}

inline bool spectralVectorIsFinite(std::span<const double> values) {
  for (double value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

inline SpectralResidualAssemblyResult
makeSpectralResidualAssemblyResult(std::vector<double> values,
                                   bool usedGeneratedGridKernel = false) {
  SpectralResidualAssemblyResult result;
  result.values = std::move(values);
  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = spectralVectorIsFinite(result.values);
  result.usedGeneratedGridKernel = usedGeneratedGridKernel;
  return result;
}

inline SpectralJacobianVectorProductResult
makeSpectralJacobianVectorProductResult(std::vector<double> values,
                                        double step) {
  SpectralJacobianVectorProductResult result;
  result.values = std::move(values);
  result.step = step;
  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = spectralVectorIsFinite(result.values);
  return result;
}

inline const SpectralGrid3D &
requireSpectralResidualGrid(const SpectralResidualProblem &problem) {
  if (!problem.grid)
    throw std::runtime_error("spectral residual problem grid is null");
  return *problem.grid;
}

inline tensorium_spectral_residual_point makeSpectralResidualPoint(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    std::size_t i, std::size_t j, std::size_t k,
    const SpectralCoordinateMap &coordinateMap,
    std::span<const double> coordinateParams,
    std::span<const double> auxiliaryValues = {}) {
  const SpectralPoint3D point = grid.point(i, j, k);
  const SpectralPointDerivatives3D u =
      grid.pointDerivatives(derivs, point.index);

  tensorium_spectral_residual_point out{};
  out.i = static_cast<std::int64_t>(point.i);
  out.j = static_cast<std::int64_t>(point.j);
  out.k = static_cast<std::int64_t>(point.k);
  out.index = static_cast<std::int64_t>(point.index);
  out.logical[0] = point.x1;
  out.logical[1] = point.x2;
  out.logical[2] = point.x3;
  out.physical[0] = point.x1;
  out.physical[1] = point.x2;
  out.physical[2] = point.x3;
  if (coordinateMap.map) {
    coordinateMap.map(out.logical, out.physical, coordinateParams.data(),
                      static_cast<std::int64_t>(coordinateParams.size()),
                      coordinateMap.userData);
  }
  out.value = u.value;
  out.d1 = u.d1;
  out.d2 = u.d2;
  out.d3 = u.d3;
  out.d11 = u.d11;
  out.d12 = u.d12;
  out.d13 = u.d13;
  out.d22 = u.d22;
  out.d23 = u.d23;
  out.d33 = u.d33;
  out.aux_values = auxiliaryValues.data();
  out.aux_count = static_cast<std::int64_t>(auxiliaryValues.size());
  return out;
}

inline std::vector<double> evaluateSpectralResidualWithAuxFields(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  validateSpectralDerivativeBundle(grid, derivs);
  if (!kernel.evaluate)
    throw std::runtime_error("spectral residual kernel callback is null");
  for (const auto &field : auxiliaryFields) {
    if (field.size() != grid.size())
      throw std::runtime_error("spectral auxiliary field size mismatch");
  }

  std::vector<double> out(grid.size(), 0.0);
  std::vector<double> pointAux(auxiliaryFields.size(), 0.0);
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const std::size_t pointIndex = grid.index(i, j, k);
        for (std::size_t aux = 0; aux < auxiliaryFields.size(); ++aux)
          pointAux[aux] = auxiliaryFields[aux][pointIndex];
        const auto point = makeSpectralResidualPoint(
            grid, derivs, i, j, k, coordinateMap, coordinateParams, pointAux);
        out[static_cast<std::size_t>(point.index)] =
            kernel.evaluate(&point, params.data(),
                            static_cast<std::int64_t>(params.size()),
                            kernel.userData);
      }
    }
  }
  return out;
}

inline std::vector<double> evaluateSpectralResidual(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  return evaluateSpectralResidualWithAuxFields(grid, derivs, kernel, params, {},
                                               coordinateMap, coordinateParams);
}

inline std::vector<double> evaluateSpectralResidual(
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  return evaluateSpectralResidual(grid, grid.derivatives(values), kernel,
                                  params, coordinateMap, coordinateParams);
}

inline std::vector<double> evaluateSpectralResidualWithAuxFields(
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  return evaluateSpectralResidualWithAuxFields(
      grid, grid.derivatives(values), kernel, params, auxiliaryFields,
      coordinateMap, coordinateParams);
}

inline std::array<std::vector<double>, 3> makeSpectralPhysicalCoordinateBuffers(
    const SpectralGrid3D &grid, const SpectralCoordinateMap &coordinateMap,
    std::span<const double> coordinateParams) {
  std::array<std::vector<double>, 3> coords = {
      std::vector<double>(grid.size(), 0.0),
      std::vector<double>(grid.size(), 0.0),
      std::vector<double>(grid.size(), 0.0)};

  double logical[3] = {0.0, 0.0, 0.0};
  double physical[3] = {0.0, 0.0, 0.0};
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const SpectralPoint3D point = grid.point(i, j, k);
        logical[0] = point.x1;
        logical[1] = point.x2;
        logical[2] = point.x3;
        physical[0] = point.x1;
        physical[1] = point.x2;
        physical[2] = point.x3;
        if (coordinateMap.map) {
          coordinateMap.map(logical, physical, coordinateParams.data(),
                            static_cast<std::int64_t>(
                                coordinateParams.size()),
                            coordinateMap.userData);
        }
        coords[0][point.index] = physical[0];
        coords[1][point.index] = physical[1];
        coords[2][point.index] = physical[2];
      }
    }
  }
  return coords;
}

inline std::vector<double> evaluateSpectralResidualWithGridKernel(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    const SpectralResidualGridKernel &kernel, std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  validateSpectralDerivativeBundle(grid, derivs);
  if (!kernel.evaluate)
    throw std::runtime_error("spectral residual grid kernel callback is null");
  for (const auto &field : auxiliaryFields) {
    if (field.size() != grid.size())
      throw std::runtime_error("spectral auxiliary field size mismatch");
  }

  std::vector<const double *> auxiliaryPointers;
  auxiliaryPointers.reserve(auxiliaryFields.size());
  for (const auto &field : auxiliaryFields)
    auxiliaryPointers.push_back(field.data());

  const auto coords =
      makeSpectralPhysicalCoordinateBuffers(grid, coordinateMap,
                                            coordinateParams);
  std::vector<double> out(grid.size(), 0.0);
  const int status = kernel.evaluate(
      static_cast<std::int64_t>(grid.size()), params.data(),
      static_cast<std::int64_t>(params.size()), derivs.value.data(),
      derivs.d1.data(), derivs.d2.data(), derivs.d3.data(), derivs.d11.data(),
      derivs.d12.data(), derivs.d13.data(), derivs.d22.data(),
      derivs.d23.data(), derivs.d33.data(),
      auxiliaryPointers.empty() ? nullptr : auxiliaryPointers.data(),
      static_cast<std::int64_t>(auxiliaryPointers.size()), coords[0].data(),
      coords[1].data(), coords[2].data(), out.data(), kernel.userData);
  if (status != 0)
    throw std::runtime_error("spectral residual grid kernel failed: " +
                             std::to_string(status));
  return out;
}

inline SpectralResidualAssemblyResult assembleSpectralResidual(
    const SpectralResidualProblem &problem,
    const SpectralDerivatives3D &derivs) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  SpectralDerivatives3D physicalUnknownDerivatives;
  const SpectralDerivatives3D *unknownDerivatives = &derivs;
  if (problem.unknownMap.transform) {
    physicalUnknownDerivatives = applySpectralUnknownMap(
        grid, derivs, problem.unknownMap, problem.unknownMapParams);
    unknownDerivatives = &physicalUnknownDerivatives;
  }
  SpectralDerivatives3D mappedDerivatives;
  const SpectralDerivatives3D *assembledDerivatives = unknownDerivatives;
  if (problem.derivativeMap.transform) {
    mappedDerivatives = applySpectralDerivativeMap(
        grid, *unknownDerivatives, problem.derivativeMap,
        problem.coordinateParams);
    assembledDerivatives = &mappedDerivatives;
  }
  if (problem.gridKernel.evaluate) {
    return makeSpectralResidualAssemblyResult(
        evaluateSpectralResidualWithGridKernel(
            grid, *assembledDerivatives, problem.gridKernel, problem.params,
            problem.auxiliaryFields, problem.coordinateMap,
            problem.coordinateParams),
        true);
  }
  return makeSpectralResidualAssemblyResult(evaluateSpectralResidualWithAuxFields(
      grid, *assembledDerivatives, problem.kernel, problem.params,
      problem.auxiliaryFields,
      problem.coordinateMap, problem.coordinateParams));
}

inline SpectralResidualAssemblyResult assembleSpectralResidual(
    const SpectralResidualProblem &problem, const std::vector<double> &values) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  if (values.size() != grid.size())
    throw std::runtime_error("spectral residual state size mismatch");
  return assembleSpectralResidual(problem, grid.derivatives(values));
}

inline SpectralResidualAssemblyResult assembleSpectralResidual(
    const SpectralGrid3D &grid, const std::vector<double> &values,
    const SpectralResidualKernel &kernel, std::span<const double> params,
    std::span<const std::vector<double>> auxiliaryFields = {},
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  const SpectralResidualProblem problem{&grid, kernel, params, auxiliaryFields,
                                        coordinateMap, coordinateParams};
  return assembleSpectralResidual(problem, values);
}

inline const SpectralGrid3D &requireSpectralResidualSystemGrid(
    const SpectralResidualSystemProblem &system) {
  if (!system.grid)
    throw std::runtime_error("spectral residual system grid is null");
  return *system.grid;
}

inline SpectralResidualSystemAssemblyResult assembleSpectralResidualSystem(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> unknownFields) {
  const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
  if (system.equations.empty())
    throw std::runtime_error("spectral residual system has no equations");
  for (const auto &field : unknownFields) {
    if (field.size() != grid.size())
      throw std::runtime_error("spectral residual system unknown size mismatch");
  }

  struct UnknownMapBinding {
    SpectralUnknownMap map;
    std::span<const double> params{};
    bool bound = false;
  };
  std::vector<UnknownMapBinding> unknownMapBindings(unknownFields.size());
  for (const auto &equation : system.equations) {
    if (equation.unknownIndex >= unknownFields.size())
      throw std::runtime_error(
          "spectral residual system equation unknown index out of range");
    if (!equation.problem.unknownMap.transform)
      continue;
    auto &binding = unknownMapBindings[equation.unknownIndex];
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
            "spectral residual system has conflicting unknown maps");
      }
      continue;
    }
    binding.map = equation.problem.unknownMap;
    binding.params = equation.problem.unknownMapParams;
    binding.bound = true;
  }
  for (const auto &equation : system.equations) {
    if (unknownMapBindings[equation.unknownIndex].bound &&
        !equation.problem.unknownMap.transform) {
      throw std::runtime_error(
          "spectral residual system equation is missing its unknown map");
    }
  }

  std::vector<bool> physicalUnknownValueNeeded(unknownFields.size(), false);
  for (const auto &equation : system.equations) {
    for (SpectralAuxiliaryUnknownIndex mappedUnknown :
         equation.auxiliaryUnknownIndices) {
      if (mappedUnknown >= 0 &&
          static_cast<std::size_t>(mappedUnknown) < unknownFields.size()) {
        physicalUnknownValueNeeded[static_cast<std::size_t>(mappedUnknown)] =
            true;
      }
    }
  }

  std::vector<std::vector<double>> physicalUnknownValues(unknownFields.size());
  for (std::size_t unknown = 0; unknown < unknownFields.size(); ++unknown) {
    const auto &binding = unknownMapBindings[unknown];
    if (!binding.bound || !physicalUnknownValueNeeded[unknown])
      continue;
    auto mapped = applySpectralUnknownMap(
        grid, grid.derivatives(unknownFields[unknown]), binding.map,
        binding.params);
    physicalUnknownValues[unknown] = std::move(mapped.value);
  }

  SpectralResidualSystemAssemblyResult result;
  result.equationCount = system.equations.size();
  result.pointsPerEquation = grid.size();
  result.values.reserve(result.equationCount * result.pointsPerEquation);
  result.equationResults.reserve(result.equationCount);
  result.usedGeneratedGridKernels = true;

  for (const auto &equation : system.equations) {
    if (equation.unknownIndex >= unknownFields.size())
      throw std::runtime_error(
          "spectral residual system equation unknown index out of range");
    SpectralResidualProblem problem = equation.problem;
    if (!problem.grid)
      problem.grid = &grid;
    if (problem.grid != &grid)
      throw std::runtime_error("spectral residual system grid mismatch");
    std::vector<std::vector<double>> resolvedAuxiliaryFields;
    if (!equation.auxiliaryUnknownIndices.empty()) {
      if (equation.auxiliaryUnknownIndices.size() !=
          problem.auxiliaryFields.size()) {
        throw std::runtime_error(
            "spectral residual system auxiliary map size mismatch");
      }
      resolvedAuxiliaryFields.reserve(problem.auxiliaryFields.size());
      for (std::size_t i = 0; i < problem.auxiliaryFields.size(); ++i) {
        const SpectralAuxiliaryUnknownIndex mappedUnknown =
            equation.auxiliaryUnknownIndices[i];
        if (mappedUnknown == kSpectralStaticAuxiliary) {
          resolvedAuxiliaryFields.push_back(problem.auxiliaryFields[i]);
          continue;
        }
        if (mappedUnknown < 0 ||
            static_cast<std::size_t>(mappedUnknown) >= unknownFields.size()) {
          throw std::runtime_error(
              "spectral residual system auxiliary unknown index out of range");
        }
        const std::size_t unknown = static_cast<std::size_t>(mappedUnknown);
        resolvedAuxiliaryFields.push_back(
            unknownMapBindings[unknown].bound ? physicalUnknownValues[unknown]
                                              : unknownFields[unknown]);
      }
      problem.auxiliaryFields = std::span<const std::vector<double>>(
          resolvedAuxiliaryFields.data(), resolvedAuxiliaryFields.size());
    }

    const auto residual =
        assembleSpectralResidual(problem, unknownFields[equation.unknownIndex]);
    result.usedGeneratedGridKernels =
        result.usedGeneratedGridKernels && residual.usedGeneratedGridKernel;
    result.finite = result.finite && residual.finite;
    result.values.insert(result.values.end(), residual.values.begin(),
                         residual.values.end());
    result.equationResults.push_back(std::move(residual));
  }

  result.l2Norm = spectralVectorL2Norm(result.values);
  result.maxAbs = spectralVectorMaxAbs(result.values);
  result.finite = result.finite && spectralVectorIsFinite(result.values);
  return result;
}

} // namespace tensorium_mlir::runtime
