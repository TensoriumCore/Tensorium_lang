#pragma once

#include "tensorium_mlir/Runtime/SpectralEllipticSolver.h"
#include "tensorium_mlir/Runtime/SpectralResidualAssembly.h"
#include "tensorium_mlir/Runtime/SpectralUnknownMaps.h"
#include "tensorium_mlir/Runtime/TwoPunctureMap.h"
#include "tensorium_mlir/Runtime/TwoPunctureSymmetry.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tensorium_mlir::runtime {

struct GeneratedSpectralInitialDataContinuationStage {
  std::array<std::size_t, 3> resolution{};
  SpectralEllipticSolveResult solveResult;
  double residualL2 = 0.0;
  double residualMaxAbs = 0.0;
  double rawResidualL2 = 0.0;
  double rawResidualMaxAbs = 0.0;
  double solveWallSeconds = 0.0;
};

struct GeneratedSpectralInitialDataSeed {
  const SpectralGrid3D *grid = nullptr;
  std::span<const std::vector<double>> fields;
};

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
  SpectralResidualSystemAssemblyResult rawResidual;
  SpectralResidualSystemAssemblyResult residual;
  double projectedOutResidualL2 = 0.0;
  double projectedOutResidualMaxAbs = 0.0;
  std::size_t projectedOutResidualMaxIndex = 0;
  double solveWallSeconds = 0.0;
  std::vector<GeneratedSpectralInitialDataContinuationStage> continuationStages;

  bool converged() const {
    return solveResult.converged() && rawResidual.finite && residual.finite;
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

inline std::unordered_map<std::string, double> generatedInitialDataParameters(
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
  requireGeneratedInitialDataString(desc.nonlinear_solver, "nonlinear solver");
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
  options.gmresMaxIterations = static_cast<int>(desc.max_linear_iterations);
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
    int preconditionerSweepsOverride = 0,
    const std::array<std::size_t, 3> &resolutionOverride = {},
    const GeneratedSpectralInitialDataSeed *initialSeed = nullptr) {
  if (desc.abi_version != tensorium_mlir::abi::kGeneratedKernelABIVersion)
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

  const bool hasResolutionOverride =
      std::any_of(resolutionOverride.begin(), resolutionOverride.end(),
                  [](std::size_t resolution) { return resolution != 0; });
  if (hasResolutionOverride &&
      std::any_of(resolutionOverride.begin(), resolutionOverride.end(),
                  [](std::size_t resolution) { return resolution < 3; })) {
    throw std::runtime_error(
        "generated spectral initial_data resolution override must contain "
        "three values >= 3");
  }
  std::array<SpectralAxis, 3> axes;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const std::size_t resolution =
        hasResolutionOverride ? resolutionOverride[axis]
                              : static_cast<std::size_t>(desc.resolution[axis]);
    if (desc.resolution[axis] < 3 || resolution < 3)
      throw std::runtime_error(
          "generated spectral initial_data resolution must be >= 3");
    axes[axis] = generatedInitialDataAxis(desc.basis[axis], resolution);
  }
  solution.grid = std::make_unique<SpectralGrid3D>(
      std::move(axes[0]), std::move(axes[1]), std::move(axes[2]));

  if (desc.coordinate_parameter_count < 0 ||
      (desc.coordinate_parameter_count > 0 && !desc.coordinate_parameter_names))
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
      (desc.unknown_map_parameter_count > 0 && !desc.unknown_map_parameters))
    throw std::runtime_error(
        "generated spectral unknown map metadata is invalid");
  if (desc.unknown_map_parameter_count > 0) {
    solution.unknownMapParameters.assign(desc.unknown_map_parameters,
                                         desc.unknown_map_parameters +
                                             desc.unknown_map_parameter_count);
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
    for (std::int64_t parameterIndex = 0; parameterIndex < equation.param_count;
         ++parameterIndex) {
      requireGeneratedInitialDataString(equation.param_names[parameterIndex],
                                        "residual parameter name");
      const auto found =
          solution.parameters.find(equation.param_names[parameterIndex]);
      if (found == solution.parameters.end())
        throw std::runtime_error(
            "missing generated residual parameter '" +
            std::string(equation.param_names[parameterIndex]) + "'");
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
  if (std::string_view(desc.field_projector) == "two_puncture_inversion_even")
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

  solution.fields.assign(static_cast<std::size_t>(systemDesc.unknown_count),
                         std::vector<double>(solution.grid->size(), 0.0));
  if (initialSeed) {
    if (!initialSeed->grid)
      throw std::runtime_error(
          "generated spectral initial_data seed grid is missing");
    if (initialSeed->fields.size() != solution.fields.size())
      throw std::runtime_error(
          "generated spectral initial_data seed field count mismatch");
    for (std::size_t field = 0; field < solution.fields.size(); ++field) {
      if (initialSeed->fields[field].size() != initialSeed->grid->size())
        throw std::runtime_error(
            "generated spectral initial_data seed field size mismatch");
      solution.fields[field] = interpolateSpectralField(
          *initialSeed->grid, *solution.grid, initialSeed->fields[field]);
    }
  }
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
  solution.solveResult =
      solveSpectralNewton(solution.generatedSystem.view(),
                          std::span<std::vector<double>>(
                              solution.fields.data(), solution.fields.size()),
                          solution.options);
  solution.rawResidual = assembleSpectralResidualSystem(
      solution.generatedSystem.view(),
      std::span<const std::vector<double>>(solution.fields.data(),
                                           solution.fields.size()));
  solution.residual = solution.rawResidual;
  projectSpectralResidualSystem(solution.generatedSystem.view(),
                                solution.residual);
  double projectedOutSquaredNorm = 0.0;
  for (std::size_t point = 0; point < solution.residual.values.size();
       ++point) {
    const double rejected =
        solution.rawResidual.values[point] - solution.residual.values[point];
    projectedOutSquaredNorm += rejected * rejected;
    const double rejectedAbs = std::fabs(rejected);
    if (rejectedAbs > solution.projectedOutResidualMaxAbs) {
      solution.projectedOutResidualMaxAbs = rejectedAbs;
      solution.projectedOutResidualMaxIndex = point;
    }
  }
  if (!solution.residual.values.empty()) {
    solution.projectedOutResidualL2 =
        std::sqrt(projectedOutSquaredNorm /
                  static_cast<double>(solution.residual.values.size()));
  }
  return solution;
}

inline GeneratedSpectralInitialDataSolution
solveGeneratedSpectralInitialDataContinuation(
    const tensorium_spectral_initial_data_desc &desc,
    const tensorium_spectral_residual_system_desc *systems,
    std::size_t systemCount,
    const tensorium_spectral_residual_kernel_desc *pointKernels,
    std::size_t pointKernelCount,
    const tensorium_spectral_residual_grid_kernel_desc *gridKernels,
    std::size_t gridKernelCount,
    std::span<const std::array<std::size_t, 3>> resolutions,
    const std::unordered_map<std::string, double> &parameterOverrides = {},
    const char *preconditionerOverride = nullptr,
    int preconditionerSweepsOverride = 0) {
  if (resolutions.empty())
    throw std::runtime_error(
        "generated spectral initial_data continuation is empty");
  for (std::size_t stage = 0; stage < resolutions.size(); ++stage) {
    if (std::any_of(resolutions[stage].begin(), resolutions[stage].end(),
                    [](std::size_t value) { return value < 3; })) {
      throw std::runtime_error(
          "generated spectral initial_data continuation resolutions must be "
          ">= 3");
    }
    if (stage == 0)
      continue;
    bool refines = false;
    for (std::size_t dim = 0; dim < 3; ++dim) {
      if (resolutions[stage][dim] < resolutions[stage - 1][dim])
        throw std::runtime_error(
            "generated spectral initial_data continuation must be "
            "coarse-to-fine");
      refines =
          refines || resolutions[stage][dim] > resolutions[stage - 1][dim];
    }
    if (!refines)
      throw std::runtime_error(
          "generated spectral initial_data continuation contains a repeated "
          "resolution");
  }

  GeneratedSpectralInitialDataSolution current;
  std::vector<GeneratedSpectralInitialDataContinuationStage> reports;
  reports.reserve(resolutions.size());
  bool hasCurrent = false;
  for (const auto &resolution : resolutions) {
    GeneratedSpectralInitialDataSeed seed;
    const GeneratedSpectralInitialDataSeed *seedPointer = nullptr;
    if (hasCurrent) {
      seed.grid = current.grid.get();
      seed.fields = std::span<const std::vector<double>>(current.fields.data(),
                                                         current.fields.size());
      seedPointer = &seed;
    }

    const auto stageStart = std::chrono::steady_clock::now();
    auto next = solveGeneratedSpectralInitialData(
        desc, systems, systemCount, pointKernels, pointKernelCount, gridKernels,
        gridKernelCount, parameterOverrides, preconditionerOverride,
        preconditionerSweepsOverride, resolution, seedPointer);
    const auto stageEnd = std::chrono::steady_clock::now();
    const double stageSeconds =
        std::chrono::duration<double>(stageEnd - stageStart).count();
    reports.push_back(GeneratedSpectralInitialDataContinuationStage{
        resolution, next.solveResult, next.residual.l2Norm,
        next.residual.maxAbs, next.rawResidual.l2Norm, next.rawResidual.maxAbs,
        stageSeconds});
    current = std::move(next);
    hasCurrent = true;
    if (!current.converged() ||
        current.residual.l2Norm > current.options.residualTolerance)
      break;
  }
  current.continuationStages = std::move(reports);
  return current;
}

} // namespace tensorium_mlir::runtime
