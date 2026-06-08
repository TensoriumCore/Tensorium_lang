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

inline SpectralBoundaryFace
spectralBoundaryFaceFromName(std::string_view name) {
  if (name == "lower_x1")
    return SpectralBoundaryFace::LowerX1;
  if (name == "upper_x1")
    return SpectralBoundaryFace::UpperX1;
  if (name == "lower_x2")
    return SpectralBoundaryFace::LowerX2;
  if (name == "upper_x2")
    return SpectralBoundaryFace::UpperX2;
  if (name == "lower_x3")
    return SpectralBoundaryFace::LowerX3;
  if (name == "upper_x3")
    return SpectralBoundaryFace::UpperX3;
  throw std::runtime_error("unknown spectral boundary face: " +
                           std::string(name));
}

inline SpectralBoundaryConditionKind
spectralBoundaryConditionKindFromName(std::string_view name) {
  if (name == "dirichlet")
    return SpectralBoundaryConditionKind::Dirichlet;
  if (name == "robin")
    return SpectralBoundaryConditionKind::Robin;
  throw std::runtime_error("unknown spectral boundary condition kind: " +
                           std::string(name));
}

inline SpectralBoundaryCondition spectralBoundaryConditionFromDesc(
    const tensorium_spectral_boundary_condition_desc &desc) {
  if (!desc.face || !desc.kind)
    throw std::runtime_error("spectral boundary condition descriptor is invalid");
  return SpectralBoundaryCondition{
      spectralBoundaryFaceFromName(desc.face),
      spectralBoundaryConditionKindFromName(desc.kind),
      desc.value_coefficient,
      desc.normal_derivative_coefficient,
      desc.target_value,
      desc.derivative_kind ? desc.derivative_kind : "normal",
      desc.value_coefficient_coordinate ? desc.value_coefficient_coordinate : "",
      desc.normal_derivative_coefficient_coordinate
          ? desc.normal_derivative_coefficient_coordinate
          : "",
      desc.target_value_coordinate ? desc.target_value_coordinate : ""};
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
  out.boundaryConditions.resize(static_cast<std::size_t>(desc.equation_count));

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
    if (equationDesc.boundary_condition_count < 0)
      throw std::runtime_error(
          "spectral residual system boundary count is invalid");
    if (equationDesc.boundary_condition_count > 0 &&
        !equationDesc.boundary_conditions) {
      throw std::runtime_error(
          "spectral residual system boundary descriptor is null");
    }
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
    auto &equationBoundaries =
        out.boundaryConditions[static_cast<std::size_t>(i)];
    equationBoundaries.reserve(
        static_cast<std::size_t>(equationDesc.boundary_condition_count));
    for (std::int64_t boundaryIndex = 0;
         boundaryIndex < equationDesc.boundary_condition_count;
         ++boundaryIndex) {
      equationBoundaries.push_back(spectralBoundaryConditionFromDesc(
          equationDesc.boundary_conditions[boundaryIndex]));
    }
    problem.boundaryConditions =
        std::span<const SpectralBoundaryCondition>(equationBoundaries.data(),
                                                   equationBoundaries.size());

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

inline std::size_t spectralBoundaryFaceDimension(SpectralBoundaryFace face) {
  switch (face) {
  case SpectralBoundaryFace::LowerX1:
  case SpectralBoundaryFace::UpperX1:
    return 0;
  case SpectralBoundaryFace::LowerX2:
  case SpectralBoundaryFace::UpperX2:
    return 1;
  case SpectralBoundaryFace::LowerX3:
  case SpectralBoundaryFace::UpperX3:
    return 2;
  }
  throw std::runtime_error("unknown spectral boundary face");
}

inline bool spectralBoundaryFaceIsLower(SpectralBoundaryFace face) {
  return face == SpectralBoundaryFace::LowerX1 ||
         face == SpectralBoundaryFace::LowerX2 ||
         face == SpectralBoundaryFace::LowerX3;
}

inline bool spectralPointIsOnBoundaryFace(const SpectralGrid3D &grid,
                                          SpectralBoundaryFace face,
                                          std::size_t i, std::size_t j,
                                          std::size_t k) {
  switch (face) {
  case SpectralBoundaryFace::LowerX1:
    return i == grid.n1() - 1;
  case SpectralBoundaryFace::UpperX1:
    return i == 0;
  case SpectralBoundaryFace::LowerX2:
    return j == grid.n2() - 1;
  case SpectralBoundaryFace::UpperX2:
    return j == 0;
  case SpectralBoundaryFace::LowerX3:
    return k == 0;
  case SpectralBoundaryFace::UpperX3:
    return k == grid.n3() - 1;
  }
  throw std::runtime_error("unknown spectral boundary face");
}

inline double spectralBoundaryNormalDerivative(
    const SpectralDerivatives3D &derivs, SpectralBoundaryFace face,
    std::size_t pointIndex) {
  double derivative = 0.0;
  switch (spectralBoundaryFaceDimension(face)) {
  case 0:
    derivative = derivs.d1[pointIndex];
    break;
  case 1:
    derivative = derivs.d2[pointIndex];
    break;
  case 2:
    derivative = derivs.d3[pointIndex];
    break;
  default:
    throw std::runtime_error("unknown spectral boundary dimension");
  }
  return spectralBoundaryFaceIsLower(face) ? -derivative : derivative;
}

inline double spectralBoundaryRadialDerivative(
    const SpectralDerivatives3D &derivs, std::size_t pointIndex,
    const std::array<double, 3> &physical) {
  const double radius =
      std::sqrt(physical[0] * physical[0] + physical[1] * physical[1] +
                physical[2] * physical[2]);
  if (!(radius > 0.0) || !std::isfinite(radius))
    throw std::runtime_error(
        "radial spectral boundary derivative is undefined at radius zero");
  return (physical[0] * derivs.d1[pointIndex] +
          physical[1] * derivs.d2[pointIndex] +
          physical[2] * derivs.d3[pointIndex]) /
         radius;
}

inline std::array<double, 3> spectralBoundaryPhysicalCoordinates(
    const SpectralGrid3D &grid, std::size_t i, std::size_t j, std::size_t k,
    const SpectralCoordinateMap &coordinateMap,
    std::span<const double> coordinateParams) {
  const SpectralPoint3D point = grid.point(i, j, k);
  std::array<double, 3> logical{point.x1, point.x2, point.x3};
  std::array<double, 3> physical = logical;
  if (coordinateMap.map) {
    coordinateMap.map(logical.data(), physical.data(), coordinateParams.data(),
                      static_cast<std::int64_t>(coordinateParams.size()),
                      coordinateMap.userData);
  }
  return physical;
}

inline double spectralBoundaryCoordinateValue(
    std::string_view coordinate, const std::array<double, 3> &physical) {
  if (coordinate == "x1" || coordinate == "x")
    return physical[0];
  if (coordinate == "x2" || coordinate == "y")
    return physical[1];
  if (coordinate == "x3" || coordinate == "z")
    return physical[2];
  if (coordinate == "r" || coordinate == "radius") {
    return std::sqrt(physical[0] * physical[0] + physical[1] * physical[1] +
                     physical[2] * physical[2]);
  }
  throw std::runtime_error("unknown spectral boundary coordinate: " +
                           std::string(coordinate));
}

inline double spectralBoundaryCoefficient(
    double constant, const std::string &coordinate,
    const std::array<double, 3> &physical) {
  if (coordinate.empty())
    return constant;
  return spectralBoundaryCoordinateValue(coordinate, physical);
}

inline double spectralBoundarySelectedDerivative(
    const SpectralBoundaryCondition &condition,
    const SpectralDerivatives3D &derivs, std::size_t pointIndex,
    const std::array<double, 3> &physical) {
  if (condition.derivativeKind.empty() || condition.derivativeKind == "normal")
    return spectralBoundaryNormalDerivative(derivs, condition.face, pointIndex);
  if (condition.derivativeKind == "radial")
    return spectralBoundaryRadialDerivative(derivs, pointIndex, physical);
  throw std::runtime_error("unknown spectral boundary derivative kind: " +
                           condition.derivativeKind);
}

inline double evaluateSpectralBoundaryCondition(
    const SpectralBoundaryCondition &condition,
    const SpectralDerivatives3D &derivs, std::size_t pointIndex,
    const std::array<double, 3> &physical) {
  if (condition.kind == SpectralBoundaryConditionKind::Dirichlet) {
    return derivs.value[pointIndex] -
           spectralBoundaryCoefficient(condition.targetValue,
                                       condition.targetValueCoordinate,
                                       physical);
  }
  if (condition.kind == SpectralBoundaryConditionKind::Robin) {
    return spectralBoundaryCoefficient(condition.valueCoefficient,
                                       condition.valueCoefficientCoordinate,
                                       physical) *
               derivs.value[pointIndex] +
           spectralBoundaryCoefficient(
               condition.normalDerivativeCoefficient,
               condition.normalDerivativeCoefficientCoordinate, physical) *
               spectralBoundarySelectedDerivative(condition, derivs, pointIndex,
                                                  physical) -
           spectralBoundaryCoefficient(condition.targetValue,
                                       condition.targetValueCoordinate,
                                       physical);
  }
  throw std::runtime_error("unknown spectral boundary condition kind");
}

inline double evaluateSpectralBoundaryConditionLinearization(
    const SpectralBoundaryCondition &condition,
    const SpectralDerivatives3D &derivs, std::size_t pointIndex,
    const std::array<double, 3> &physical) {
  if (condition.kind == SpectralBoundaryConditionKind::Dirichlet) {
    return derivs.value[pointIndex];
  }
  if (condition.kind == SpectralBoundaryConditionKind::Robin) {
    return spectralBoundaryCoefficient(condition.valueCoefficient,
                                       condition.valueCoefficientCoordinate,
                                       physical) *
               derivs.value[pointIndex] +
           spectralBoundaryCoefficient(
               condition.normalDerivativeCoefficient,
               condition.normalDerivativeCoefficientCoordinate, physical) *
               spectralBoundarySelectedDerivative(condition, derivs, pointIndex,
                                                  physical);
  }
  throw std::runtime_error("unknown spectral boundary condition kind");
}

inline void applySpectralBoundaryConditions(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    std::span<const SpectralBoundaryCondition> conditions,
    std::vector<double> &residual,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  if (conditions.empty())
    return;
  validateSpectralDerivativeBundle(grid, derivs);
  if (residual.size() != grid.size())
    throw std::runtime_error("spectral boundary residual size mismatch");

  for (const auto &condition : conditions) {
    const std::size_t dim = spectralBoundaryFaceDimension(condition.face);
    if (grid.axis(dim).basis == SpectralBasis::FourierPeriodic)
      throw std::runtime_error(
          "cannot impose a boundary condition on a periodic spectral axis");

#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          if (!spectralPointIsOnBoundaryFace(grid, condition.face, i, j, k))
            continue;
          const std::size_t pointIndex = grid.index(i, j, k);
          const auto physical = spectralBoundaryPhysicalCoordinates(
              grid, i, j, k, coordinateMap, coordinateParams);
          residual[pointIndex] = evaluateSpectralBoundaryCondition(
              condition, derivs, pointIndex, physical);
        }
      }
    }
  }
}

inline void applySpectralBoundaryConditionLinearizations(
    const SpectralGrid3D &grid, const SpectralDerivatives3D &derivs,
    std::span<const SpectralBoundaryCondition> conditions,
    std::vector<double> &residualColumn,
    const SpectralCoordinateMap &coordinateMap = {},
    std::span<const double> coordinateParams = {}) {
  if (conditions.empty())
    return;
  validateSpectralDerivativeBundle(grid, derivs);
  if (residualColumn.size() != grid.size())
    throw std::runtime_error(
        "spectral boundary linearization column size mismatch");

  for (const auto &condition : conditions) {
    const std::size_t dim = spectralBoundaryFaceDimension(condition.face);
    if (grid.axis(dim).basis == SpectralBasis::FourierPeriodic)
      throw std::runtime_error(
          "cannot impose a boundary condition on a periodic spectral axis");

#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      for (std::size_t j = 0; j < grid.n2(); ++j) {
        for (std::size_t i = 0; i < grid.n1(); ++i) {
          if (!spectralPointIsOnBoundaryFace(grid, condition.face, i, j, k))
            continue;
          const std::size_t pointIndex = grid.index(i, j, k);
          const auto physical = spectralBoundaryPhysicalCoordinates(
              grid, i, j, k, coordinateMap, coordinateParams);
          residualColumn[pointIndex] =
              evaluateSpectralBoundaryConditionLinearization(condition, derivs,
                                                             pointIndex,
                                                             physical);
        }
      }
    }
  }
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
#pragma omp parallel
  {
    std::vector<double> pointAux(auxiliaryFields.size(), 0.0);
#pragma omp for collapse(3) schedule(static)
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

#pragma omp parallel for collapse(3) schedule(static)
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const SpectralPoint3D point = grid.point(i, j, k);
        double logical[3] = {point.x1, point.x2, point.x3};
        double physical[3] = {point.x1, point.x2, point.x3};
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
  std::vector<double> values;
  bool usedGeneratedGridKernel = false;
  if (problem.gridKernel.evaluate) {
    values = evaluateSpectralResidualWithGridKernel(
        grid, derivs, problem.gridKernel, problem.params,
        problem.auxiliaryFields, problem.coordinateMap,
        problem.coordinateParams);
    usedGeneratedGridKernel = true;
  } else {
    values = evaluateSpectralResidualWithAuxFields(
        grid, derivs, problem.kernel, problem.params, problem.auxiliaryFields,
        problem.coordinateMap, problem.coordinateParams);
  }
  applySpectralBoundaryConditions(grid, derivs, problem.boundaryConditions,
                                  values, problem.coordinateMap,
                                  problem.coordinateParams);
  return makeSpectralResidualAssemblyResult(std::move(values),
                                            usedGeneratedGridKernel);
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
        resolvedAuxiliaryFields.push_back(
            unknownFields[static_cast<std::size_t>(mappedUnknown)]);
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
