#pragma once

#include "tensorium_mlir/Runtime/SpectralEllipticSolver.h"
#include "tensorium_mlir/Runtime/SpectralResidualAssembly.h"
#include "tensorium_mlir/Runtime/SpectralUnknownMaps.h"
#include "tensorium_mlir/Runtime/TwoPunctureMap.h"
#include "tensorium_mlir/Runtime/TwoPunctureSymmetry.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tensorium_mlir::runtime {

struct GeneratedSpectralInitialDataSolution {
  const tensorium_spectral_initial_data_desc *descriptor = nullptr;
  const tensorium_spectral_residual_system_desc *systemDescriptor = nullptr;
  std::unique_ptr<SpectralGrid3D> grid;
  std::unordered_map<std::string, double> parameters;
  std::vector<std::vector<double>> equationParameters;
  std::vector<std::vector<std::vector<double>>> auxiliaryStorage;
  std::vector<double> coordinateParameters;
  std::vector<double> unknownMapParameters;
  SpectralGeneratedResidualSystem generatedSystem;
  std::vector<std::vector<double>> fields;
  SpectralEllipticSolveOptions options;
  SpectralEllipticSolveResult solveResult;
  SpectralResidualSystemAssemblyResult residual;

  bool converged() const {
    return solveResult.converged() && residual.finite;
  }

  SpectralResidualSystemProblem system() const {
    return generatedSystem.view();
  }
};

inline void requireGeneratedInitialDataString(const char *value,
                                              std::string_view label) {
  if (!value || value[0] == '\0')
    throw std::runtime_error("generated spectral initial_data " +
                             std::string(label) + " is empty");
}

inline std::unordered_map<std::string, double>
generatedInitialDataParameters(
    const tensorium_spectral_initial_data_desc &desc,
    const std::unordered_map<std::string, double> &overrides = {}) {
  if (desc.parameter_count < 0 ||
      (desc.parameter_count > 0 &&
       (!desc.parameter_names || !desc.parameter_values))) {
    throw std::runtime_error(
        "generated spectral initial_data parameter metadata is invalid");
  }

  std::unordered_map<std::string, double> parameters;
  for (std::int64_t i = 0; i < desc.parameter_count; ++i) {
    requireGeneratedInitialDataString(desc.parameter_names[i],
                                      "parameter name");
    if (!std::isfinite(desc.parameter_values[i]))
      throw std::runtime_error(
          "generated spectral initial_data parameter is not finite");
    const auto inserted =
        parameters.emplace(desc.parameter_names[i], desc.parameter_values[i]);
    if (!inserted.second)
      throw std::runtime_error(
          "generated spectral initial_data parameter is duplicated");
  }
  for (const auto &[name, value] : overrides) {
    if (!parameters.count(name))
      throw std::runtime_error("unknown spectral initial_data override '" +
                               name + "'");
    if (!std::isfinite(value))
      throw std::runtime_error("spectral initial_data override '" + name +
                               "' is not finite");
    parameters[name] = value;
  }
  return parameters;
}

inline SpectralAxis generatedInitialDataAxis(const char *basis,
                                             std::size_t resolution) {
  requireGeneratedInitialDataString(basis, "basis");
  if (std::string_view(basis) == "chebyshev")
    return SpectralAxis::chebyshevZeros(resolution);
  if (std::string_view(basis) == "fourier")
    return SpectralAxis::fourierPeriodic(resolution);
  throw std::runtime_error("unsupported generated spectral basis '" +
                           std::string(basis) + "'");
}

inline SpectralLinearSolveKind
generatedInitialDataLinearSolver(const char *name) {
  requireGeneratedInitialDataString(name, "linear solver");
  if (std::string_view(name) == "direct")
    return SpectralLinearSolveKind::DenseJacobian;
  if (std::string_view(name) == "gmres")
    return SpectralLinearSolveKind::MatrixFreeGMRES;
  throw std::runtime_error("unsupported generated spectral linear solver '" +
                           std::string(name) + "'");
}

inline SpectralPreconditionerKind
generatedInitialDataPreconditioner(const char *name) {
  requireGeneratedInitialDataString(name, "preconditioner");
  const std::string_view value(name);
  if (value == "none")
    return SpectralPreconditionerKind::None;
  if (value == "diagonal_jvp")
    return SpectralPreconditionerKind::DiagonalJVP;
  if (value == "dense_laplacian_shift")
    return SpectralPreconditionerKind::DenseLaplacianShift;
  if (value == "modal_laplacian_shift")
    return SpectralPreconditionerKind::ModalLaplacianShift;
  if (value == "mapped_fd_laplacian_shift")
    return SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift;
  if (value == "mapped_fd_multigrid")
    return SpectralPreconditionerKind::MappedFiniteDifferenceMultigrid;
  throw std::runtime_error("unsupported generated spectral preconditioner '" +
                           std::string(name) + "'");
}

inline SpectralEllipticSolveOptions generatedInitialDataSolveOptions(
    const tensorium_spectral_initial_data_desc &desc) {
  requireGeneratedInitialDataString(desc.nonlinear_solver,
                                    "nonlinear solver");
  if (std::string_view(desc.nonlinear_solver) != "newton")
    throw std::runtime_error(
        "generated spectral initial_data requires nonlinear = newton");

  SpectralEllipticSolveOptions options;
  options.maxNewtonSteps = static_cast<int>(desc.max_newton_steps);
  options.residualTolerance = desc.residual_tolerance;
  options.residualRatioTarget = 0.0;
  options.linearSolver = generatedInitialDataLinearSolver(desc.linear_solver);
  options.denseJacobianMaxUnknowns =
      options.linearSolver == SpectralLinearSolveKind::DenseJacobian
          ? std::numeric_limits<std::size_t>::max()
          : 1;
  options.gmresMaxIterations =
      static_cast<int>(desc.max_linear_iterations);
  options.gmresRestart = static_cast<int>(desc.restart);
  options.gmresTolerance = desc.linear_tolerance;
  options.gmresRelativeTolerance = desc.linear_relative_tolerance;
  options.gmresPreconditioner =
      generatedInitialDataPreconditioner(desc.preconditioner);
  options.preconditionerRelaxationSweeps =
      static_cast<int>(desc.preconditioner_sweeps);
  options.preconditionerMultigridPreSweeps =
      static_cast<int>(desc.preconditioner_sweeps);
  options.preconditionerMultigridPostSweeps =
      static_cast<int>(desc.preconditioner_sweeps);
  options.jvpOptions.relativeStep = desc.jvp_relative_step;
  options.jvpOptions.absoluteStep = desc.jvp_absolute_step;
  options.linearPivotTolerance = 1.0e-13;
  return options;
}

inline const tensorium_spectral_residual_system_desc &
findGeneratedInitialDataSystem(
    const tensorium_spectral_initial_data_desc &initialData,
    const tensorium_spectral_residual_system_desc *systems,
    std::size_t systemCount) {
  requireGeneratedInitialDataString(initialData.system_name, "system name");
  if (!systems || systemCount == 0)
    throw std::runtime_error(
        "generated spectral initial_data has no residual systems");
  for (std::size_t i = 0; i < systemCount; ++i) {
    if (systems[i].symbol_name &&
        std::string_view(systems[i].symbol_name) == initialData.system_name)
      return systems[i];
  }
  throw std::runtime_error("generated spectral residual system '" +
                           std::string(initialData.system_name) +
                           "' was not found");
}

inline GeneratedSpectralInitialDataSolution solveGeneratedSpectralInitialData(
    const tensorium_spectral_initial_data_desc &desc,
    const tensorium_spectral_residual_system_desc *systems,
    std::size_t systemCount,
    const tensorium_spectral_residual_kernel_desc *pointKernels,
    std::size_t pointKernelCount,
    const tensorium_spectral_residual_grid_kernel_desc *gridKernels,
    std::size_t gridKernelCount,
    const std::unordered_map<std::string, double> &parameterOverrides = {},
    const char *preconditionerOverride = nullptr,
    int preconditionerSweepsOverride = 0) {
  if (desc.abi_version !=
      tensorium_mlir::abi::kGeneratedKernelABIVersion)
    throw std::runtime_error(
        "unsupported generated spectral initial_data ABI version");
  requireGeneratedInitialDataString(desc.symbol_name, "name");
  requireGeneratedInitialDataString(desc.coordinate_map, "coordinate map");
  requireGeneratedInitialDataString(desc.unknown_map, "unknown map");
  requireGeneratedInitialDataString(desc.field_projector, "field projector");
  requireGeneratedInitialDataString(desc.reconstruction, "reconstruction");
  if (desc.dimension != 3 || !desc.resolution || !desc.basis)
    throw std::runtime_error(
        "generated spectral initial_data requires a three-dimensional grid");

  GeneratedSpectralInitialDataSolution solution;
  solution.descriptor = &desc;
  solution.systemDescriptor =
      &findGeneratedInitialDataSystem(desc, systems, systemCount);
  solution.parameters =
      generatedInitialDataParameters(desc, parameterOverrides);

  std::array<SpectralAxis, 3> axes;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (desc.resolution[axis] < 3)
      throw std::runtime_error(
          "generated spectral initial_data resolution must be >= 3");
    axes[axis] = generatedInitialDataAxis(
        desc.basis[axis], static_cast<std::size_t>(desc.resolution[axis]));
  }
  solution.grid = std::make_unique<SpectralGrid3D>(
      std::move(axes[0]), std::move(axes[1]), std::move(axes[2]));

  if (desc.coordinate_parameter_count < 0 ||
      (desc.coordinate_parameter_count > 0 &&
       !desc.coordinate_parameter_names))
    throw std::runtime_error(
        "generated spectral coordinate parameter metadata is invalid");
  for (std::int64_t i = 0; i < desc.coordinate_parameter_count; ++i) {
    requireGeneratedInitialDataString(desc.coordinate_parameter_names[i],
                                      "coordinate parameter name");
    const auto found =
        solution.parameters.find(desc.coordinate_parameter_names[i]);
    if (found == solution.parameters.end())
      throw std::runtime_error("missing generated coordinate parameter '" +
                               std::string(desc.coordinate_parameter_names[i]) +
                               "'");
    solution.coordinateParameters.push_back(found->second);
  }
  if (desc.unknown_map_parameter_count < 0 ||
      (desc.unknown_map_parameter_count > 0 &&
       !desc.unknown_map_parameters))
    throw std::runtime_error(
        "generated spectral unknown map metadata is invalid");
  if (desc.unknown_map_parameter_count > 0) {
    solution.unknownMapParameters.assign(
        desc.unknown_map_parameters,
        desc.unknown_map_parameters + desc.unknown_map_parameter_count);
  }

  const auto &systemDesc = *solution.systemDescriptor;
  if (systemDesc.equation_count != systemDesc.unknown_count)
    throw std::runtime_error(
        "generated spectral solver currently requires one equation per "
        "unknown");
  solution.equationParameters.resize(
      static_cast<std::size_t>(systemDesc.equation_count));
  solution.auxiliaryStorage.resize(
      static_cast<std::size_t>(systemDesc.equation_count));
  std::vector<SpectralGeneratedResidualSystemEquationInputs> inputs(
      static_cast<std::size_t>(systemDesc.equation_count));
  for (std::int64_t equationIndex = 0;
       equationIndex < systemDesc.equation_count; ++equationIndex) {
    const auto &equation = systemDesc.equations[equationIndex];
    auto &parameterValues = solution.equationParameters[equationIndex];
    for (std::int64_t parameterIndex = 0;
         parameterIndex < equation.param_count; ++parameterIndex) {
      requireGeneratedInitialDataString(equation.param_names[parameterIndex],
                                        "residual parameter name");
      const auto found =
          solution.parameters.find(equation.param_names[parameterIndex]);
      if (found == solution.parameters.end())
        throw std::runtime_error("missing generated residual parameter '" +
                                 std::string(
                                     equation.param_names[parameterIndex]) +
                                 "'");
      parameterValues.push_back(found->second);
    }
    auto &auxiliaries = solution.auxiliaryStorage[equationIndex];
    auxiliaries.resize(static_cast<std::size_t>(equation.auxiliary_count));
    for (std::int64_t auxiliary = 0; auxiliary < equation.auxiliary_count;
         ++auxiliary) {
      if (!equation.auxiliary_unknown_indices ||
          equation.auxiliary_unknown_indices[auxiliary] ==
              kSpectralStaticAuxiliary) {
        throw std::runtime_error(
            "static auxiliary fields require explicit runtime bindings");
      }
      auxiliaries[auxiliary].assign(solution.grid->size(), 0.0);
    }
    inputs[equationIndex] = SpectralGeneratedResidualSystemEquationInputs{
        parameterValues, auxiliaries};
  }

  solution.generatedSystem = makeSpectralResidualSystemFromDesc(
      systemDesc, *solution.grid, pointKernels, pointKernelCount, gridKernels,
      gridKernelCount, inputs);

  SpectralCoordinateMap coordinateMap;
  SpectralDerivativeMap derivativeMap;
  if (std::string_view(desc.coordinate_map) == "two_puncture") {
    coordinateMap = makeTwoPunctureCoordinateMap();
    derivativeMap = makeTwoPunctureDerivativeMap();
  } else if (std::string_view(desc.coordinate_map) != "identity") {
    throw std::runtime_error("unsupported generated coordinate map '" +
                             std::string(desc.coordinate_map) + "'");
  }

  SpectralUnknownMap unknownMap;
  if (std::string_view(desc.unknown_map) == "linear_boundary")
    unknownMap = makeLinearBoundaryFactorUnknownMap();
  else if (std::string_view(desc.unknown_map) != "identity")
    throw std::runtime_error("unsupported generated unknown map '" +
                             std::string(desc.unknown_map) + "'");

  SpectralFieldProjector projector;
  if (std::string_view(desc.field_projector) ==
      "two_puncture_inversion_even")
    projector = makeTwoPunctureInversionEvenFieldProjector();
  else if (std::string_view(desc.field_projector) != "none")
    throw std::runtime_error("unsupported generated field projector '" +
                             std::string(desc.field_projector) + "'");

  for (auto &equation : solution.generatedSystem.equations) {
    equation.problem.coordinateMap = coordinateMap;
    equation.problem.coordinateParams = solution.coordinateParameters;
    equation.problem.derivativeMap = derivativeMap;
    equation.problem.unknownMap = unknownMap;
    equation.problem.unknownMapParams = solution.unknownMapParameters;
    equation.problem.fieldProjector = projector;
  }

  solution.fields.assign(
      static_cast<std::size_t>(systemDesc.unknown_count),
      std::vector<double>(solution.grid->size(), 0.0));
  solution.options = generatedInitialDataSolveOptions(desc);
  if (preconditionerOverride && preconditionerOverride[0] != '\0') {
    solution.options.gmresPreconditioner =
        generatedInitialDataPreconditioner(preconditionerOverride);
  }
  if (preconditionerSweepsOverride < 0)
    throw std::runtime_error(
        "generated initial_data preconditioner sweeps override is invalid");
  if (preconditionerSweepsOverride > 0) {
    solution.options.preconditionerRelaxationSweeps =
        preconditionerSweepsOverride;
    solution.options.preconditionerMultigridPreSweeps =
        preconditionerSweepsOverride;
    solution.options.preconditionerMultigridPostSweeps =
        preconditionerSweepsOverride;
  }
  solution.solveResult = solveSpectralNewton(
      solution.generatedSystem.view(),
      std::span<std::vector<double>>(solution.fields.data(),
                                     solution.fields.size()),
      solution.options);
  solution.residual = assembleSpectralResidualSystem(
      solution.generatedSystem.view(),
      std::span<const std::vector<double>>(solution.fields.data(),
                                           solution.fields.size()));
  projectSpectralResidualSystem(solution.generatedSystem.view(),
                                solution.residual);
  return solution;
}

} // namespace tensorium_mlir::runtime
