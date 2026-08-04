#include "tensorium/Solver/ConstraintHandoff.h"

#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Solver/ConstraintSolver.hpp"
#include "tensorium/Validation/IRCanonicalize.hpp"
#include "tensorium/Validation/IRVerifier.hpp"
#include "tensorium/Validation/ProgramValidator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

struct tensorium_constraint_solution_v1 {
  tensorium::solver::ConstraintSolution value;
};

namespace {

using Status = tensorium_constraint_status_v1;

void clearError(char *errorMessage, int64_t errorCapacity) {
  if (errorMessage && errorCapacity > 0)
    errorMessage[0] = '\0';
}

Status report(Status status, std::string_view message, char *errorMessage,
              int64_t errorCapacity) {
  if (errorMessage && errorCapacity > 0) {
    const auto capacity = static_cast<std::size_t>(errorCapacity);
    const std::size_t count = std::min(message.size(), capacity - 1);
    std::memcpy(errorMessage, message.data(), count);
    errorMessage[count] = '\0';
  }
  return status;
}

bool validErrorBuffer(char *errorMessage, int64_t errorCapacity) {
  return errorCapacity >= 0 && (errorCapacity == 0 || errorMessage != nullptr);
}

template <typename T> bool validStruct(const T *value) {
  return value && value->struct_size >= sizeof(T);
}

std::string readFile(const char *path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open constraint DSL file: " +
                             std::string(path));
  std::ostringstream stream;
  stream << input.rdbuf();
  if (!input.good() && !input.eof())
    throw std::runtime_error("failed to read constraint DSL file: " +
                             std::string(path));
  return stream.str();
}

void appendValidationErrors(
    const tensorium::validation::ValidationResult &result,
    std::vector<std::string> &errors) {
  for (const auto &diagnostic : result.diags) {
    if (diagnostic.kind == tensorium::validation::Diagnostic::Kind::Error)
      errors.push_back(diagnostic.message);
  }
}

void throwValidationErrors(const char *stage,
                           const std::vector<std::string> &errors) {
  if (errors.empty())
    return;
  std::ostringstream message;
  message << stage << " failed:";
  for (const auto &error : errors)
    message << "\n  - " << error;
  throw std::runtime_error(message.str());
}

tensorium::backend::ModuleIR buildModule(std::string_view source) {
  std::string ownedSource(source);
  tensorium::Lexer lexer(ownedSource.c_str());
  tensorium::Parser parser(lexer);
  tensorium::Program program = parser.parseProgram();
  tensorium::SemanticAnalyzer semanticAnalyzer(
      program, tensorium::CompilationMode::Executable);
  tensorium::backend::ModuleIR module =
      tensorium::backend::BackendBuilder::build(program, semanticAnalyzer);
  tensorium::validation::canonicalizeDifferentialIR(module);
  tensorium::validation::canonicalizeEinsteinIR(module);

  std::vector<std::string> errors;
  appendValidationErrors(tensorium::validation::verifyIR(module), errors);
  throwValidationErrors("IR verification", errors);
  errors.clear();
  appendValidationErrors(tensorium::validation::validateProgram(module),
                         errors);
  throwValidationErrors("program validation", errors);
  return module;
}

bool buildRequest(const tensorium_constraint_parameter_v1 *parameters,
                  int64_t parameterCount,
                  tensorium::solver::ConstraintSolveRequest &request,
                  std::string &error) {
  if (parameterCount < 0) {
    error = "parameter_count must be non-negative";
    return false;
  }
  if (parameterCount > 0 && !parameters) {
    error = "parameters is null while parameter_count is non-zero";
    return false;
  }

  std::unordered_set<std::string> names;
  for (int64_t i = 0; i < parameterCount; ++i) {
    const auto &parameter = parameters[i];
    if (parameter.struct_size < sizeof(tensorium_constraint_parameter_v1)) {
      error = "constraint parameter struct_size is incompatible with ABI v1";
      return false;
    }
    if (!parameter.name || parameter.name[0] == '\0') {
      error = "constraint parameter name is empty";
      return false;
    }
    if (!std::isfinite(parameter.value)) {
      error = "constraint parameter '" + std::string(parameter.name) +
              "' must be finite";
      return false;
    }
    if (!names.insert(parameter.name).second) {
      error = "duplicate constraint parameter '" +
              std::string(parameter.name) + "'";
      return false;
    }
    request.parameters.emplace(parameter.name, parameter.value);
  }
  return true;
}

Status solveSource(const char *source,
                   const tensorium_constraint_parameter_v1 *parameters,
                   int64_t parameterCount,
                   tensorium_constraint_solution_v1 **solutionOut,
                   char *errorMessage, int64_t errorCapacity) {
  if (!validErrorBuffer(errorMessage, errorCapacity))
    return TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT;
  clearError(errorMessage, errorCapacity);
  if (!solutionOut)
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "solution_out is null", errorMessage, errorCapacity);
  *solutionOut = nullptr;
  if (!source)
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "constraint DSL source is null", errorMessage,
                  errorCapacity);

  tensorium::solver::ConstraintSolveRequest request;
  try {
    std::string parameterError;
    if (!buildRequest(parameters, parameterCount, request, parameterError))
      return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                    parameterError, errorMessage, errorCapacity);
  } catch (const std::exception &exception) {
    return report(TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR, exception.what(),
                  errorMessage, errorCapacity);
  } catch (...) {
    return report(TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "unknown parameter preparation failure", errorMessage,
                  errorCapacity);
  }

  tensorium::backend::ModuleIR module;
  try {
    module = buildModule(source);
  } catch (const std::exception &exception) {
    return report(TENSORIUM_CONSTRAINT_STATUS_FRONTEND_ERROR, exception.what(),
                  errorMessage, errorCapacity);
  } catch (...) {
    return report(TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "unknown frontend failure", errorMessage, errorCapacity);
  }

  try {
    auto solution =
        tensorium::solver::solveRadialConstraintProblem(module, request);
    if (!solution.converged) {
      std::ostringstream message;
      message << "constraint solve did not converge after "
              << solution.iterations
              << " iterations; residual_inf=" << solution.residualNorm;
      return report(TENSORIUM_CONSTRAINT_STATUS_SOLVER_ERROR, message.str(),
                    errorMessage, errorCapacity);
    }
    auto handle = std::make_unique<tensorium_constraint_solution_v1>();
    handle->value = std::move(solution);
    *solutionOut = handle.release();
    return TENSORIUM_CONSTRAINT_STATUS_OK;
  } catch (const std::exception &exception) {
    return report(TENSORIUM_CONSTRAINT_STATUS_SOLVER_ERROR, exception.what(),
                  errorMessage, errorCapacity);
  } catch (...) {
    return report(TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "unknown constraint solver failure", errorMessage,
                  errorCapacity);
  }
}

Status validateCall(const tensorium_constraint_solution_v1 *solution,
                    char *errorMessage, int64_t errorCapacity) {
  if (!validErrorBuffer(errorMessage, errorCapacity))
    return TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT;
  clearError(errorMessage, errorCapacity);
  if (!solution)
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "constraint solution is null", errorMessage, errorCapacity);
  return TENSORIUM_CONSTRAINT_STATUS_OK;
}

Status convertTarget(const tensorium_ctt_target_grid_v1 *input,
                     tensorium::solver::CttTargetGrid &output,
                     char *errorMessage, int64_t errorCapacity) {
  if (!validStruct(input))
    return report(TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH,
                  "target grid struct_size is incompatible with ABI v1",
                  errorMessage, errorCapacity);
  if (input->point_count <= 0)
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "target grid point_count must be positive", errorMessage,
                  errorCapacity);
  if (static_cast<uint64_t>(input->point_count) >
      std::numeric_limits<std::size_t>::max()) {
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "target grid point_count is too large", errorMessage,
                  errorCapacity);
  }
  if (input->coordinates == TENSORIUM_CTT_COORDINATES_SPHERICAL) {
    output.coordinates = tensorium::solver::CttTargetCoordinates::Spherical;
  } else if (input->coordinates == TENSORIUM_CTT_COORDINATES_CARTESIAN) {
    output.coordinates = tensorium::solver::CttTargetCoordinates::Cartesian;
  } else {
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "target grid coordinate system is invalid", errorMessage,
                  errorCapacity);
  }
  output.pointCount = static_cast<std::size_t>(input->point_count);
  for (std::size_t i = 0; i < output.coordinateComponents.size(); ++i)
    output.coordinateComponents[i] = input->coordinate_components[i];
  return TENSORIUM_CONSTRAINT_STATUS_OK;
}

} // namespace

extern "C" {

int64_t tensorium_constraint_handoff_abi_version(void) {
  return TENSORIUM_CONSTRAINT_HANDOFF_ABI_VERSION;
}

tensorium_constraint_status_v1 tensorium_solve_radial_constraints_source_v1(
    const char *source, const tensorium_constraint_parameter_v1 *parameters,
    int64_t parameterCount, tensorium_constraint_solution_v1 **solutionOut,
    char *errorMessage, int64_t errorCapacity) {
  return solveSource(source, parameters, parameterCount, solutionOut,
                     errorMessage, errorCapacity);
}

tensorium_constraint_status_v1 tensorium_solve_radial_constraints_file_v1(
    const char *path, const tensorium_constraint_parameter_v1 *parameters,
    int64_t parameterCount, tensorium_constraint_solution_v1 **solutionOut,
    char *errorMessage, int64_t errorCapacity) {
  if (!validErrorBuffer(errorMessage, errorCapacity))
    return TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT;
  clearError(errorMessage, errorCapacity);
  if (!path || path[0] == '\0')
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "constraint DSL path is empty", errorMessage, errorCapacity);
  try {
    const std::string source = readFile(path);
    return solveSource(source.c_str(), parameters, parameterCount, solutionOut,
                       errorMessage, errorCapacity);
  } catch (const std::exception &exception) {
    if (solutionOut)
      *solutionOut = nullptr;
    return report(TENSORIUM_CONSTRAINT_STATUS_IO_ERROR, exception.what(),
                  errorMessage, errorCapacity);
  } catch (...) {
    if (solutionOut)
      *solutionOut = nullptr;
    return report(TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "unknown constraint DSL I/O failure", errorMessage,
                  errorCapacity);
  }
}

tensorium_constraint_status_v1 tensorium_constraint_solution_info_get_v1(
    const tensorium_constraint_solution_v1 *solution,
    tensorium_constraint_solution_info_v1 *info, char *errorMessage,
    int64_t errorCapacity) {
  Status status = validateCall(solution, errorMessage, errorCapacity);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return status;
  if (!validStruct(info))
    return report(TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH,
                  "solution info struct_size is incompatible with ABI v1",
                  errorMessage, errorCapacity);

  const auto &value = solution->value;
  info->converged = value.converged ? 1 : 0;
  info->iterations = static_cast<int64_t>(value.iterations);
  info->residual_norm = value.residualNorm;
  info->source_point_count =
      static_cast<int64_t>(value.coordinates.size());
  info->domain_count = static_cast<int64_t>(value.domains.size());
  info->has_physical_ctt = value.physicalCtt ? 1 : 0;
  return TENSORIUM_CONSTRAINT_STATUS_OK;
}

tensorium_constraint_status_v1 tensorium_interpolate_radial_ctt_v1(
    const tensorium_constraint_solution_v1 *solution,
    const tensorium_ctt_target_grid_v1 *target,
    const tensorium_ctt_physical_buffers_v1 *outputs, char *errorMessage,
    int64_t errorCapacity) {
  Status status = validateCall(solution, errorMessage, errorCapacity);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return status;
  tensorium::solver::CttTargetGrid cppTarget;
  status = convertTarget(target, cppTarget, errorMessage, errorCapacity);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return status;
  if (!validStruct(outputs))
    return report(TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH,
                  "physical output struct_size is incompatible with ABI v1",
                  errorMessage, errorCapacity);
  if (outputs->point_count != target->point_count)
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "physical output point_count does not match target grid",
                  errorMessage, errorCapacity);

  tensorium::solver::CttEvolutionBuffers cppOutputs;
  for (std::size_t i = 0; i < 9; ++i) {
    cppOutputs.spatialMetric[i] = outputs->spatial_metric[i];
    cppOutputs.inverseSpatialMetric[i] = outputs->inverse_spatial_metric[i];
    cppOutputs.extrinsicCurvature[i] = outputs->extrinsic_curvature[i];
  }
  cppOutputs.meanCurvature = outputs->mean_curvature;
  try {
    tensorium::solver::interpolateRadialCttToGrid(solution->value, cppTarget,
                                                   cppOutputs);
    return TENSORIUM_CONSTRAINT_STATUS_OK;
  } catch (const std::exception &exception) {
    return report(TENSORIUM_CONSTRAINT_STATUS_SOLVER_ERROR, exception.what(),
                  errorMessage, errorCapacity);
  } catch (...) {
    return report(TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "unknown CTT interpolation failure", errorMessage,
                  errorCapacity);
  }
}

tensorium_constraint_status_v1 tensorium_interpolate_radial_electromagnetic_v1(
    const tensorium_constraint_solution_v1 *solution,
    const tensorium_ctt_target_grid_v1 *target,
    const tensorium_electromagnetic_buffers_v1 *outputs, char *errorMessage,
    int64_t errorCapacity) {
  Status status = validateCall(solution, errorMessage, errorCapacity);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return status;
  tensorium::solver::CttTargetGrid cppTarget;
  status = convertTarget(target, cppTarget, errorMessage, errorCapacity);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return status;
  if (!validStruct(outputs))
    return report(
        TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH,
        "electromagnetic output struct_size is incompatible with ABI v1",
        errorMessage, errorCapacity);
  if (outputs->point_count != target->point_count)
    return report(
        TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
        "electromagnetic output point_count does not match target grid",
        errorMessage, errorCapacity);

  tensorium::solver::ElectromagneticEvolutionBuffers cppOutputs;
  for (std::size_t component = 0; component < 3; ++component) {
    cppOutputs.electricField[component] = outputs->electric_field[component];
    cppOutputs.magneticField[component] = outputs->magnetic_field[component];
  }
  try {
    tensorium::solver::interpolateRadialElectromagneticToGrid(
        solution->value, cppTarget, cppOutputs);
    return TENSORIUM_CONSTRAINT_STATUS_OK;
  } catch (const std::exception &exception) {
    return report(TENSORIUM_CONSTRAINT_STATUS_SOLVER_ERROR, exception.what(),
                  errorMessage, errorCapacity);
  } catch (...) {
    return report(TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "unknown electromagnetic interpolation failure", errorMessage,
                  errorCapacity);
  }
}

tensorium_constraint_status_v1 tensorium_initialize_bssn_from_radial_ctt_v1(
    const tensorium_constraint_solution_v1 *solution,
    const tensorium_ctt_target_grid_v1 *target,
    const tensorium_ctt_bssn_buffers_v1 *outputs,
    const tensorium_bssn_gauge_seed_v1 *gauge, char *errorMessage,
    int64_t errorCapacity) {
  Status status = validateCall(solution, errorMessage, errorCapacity);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return status;
  tensorium::solver::CttTargetGrid cppTarget;
  status = convertTarget(target, cppTarget, errorMessage, errorCapacity);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return status;
  if (!validStruct(outputs))
    return report(TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH,
                  "BSSN output struct_size is incompatible with ABI v1",
                  errorMessage, errorCapacity);
  if (outputs->point_count != target->point_count)
    return report(TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT,
                  "BSSN output point_count does not match target grid",
                  errorMessage, errorCapacity);
  if (gauge && !validStruct(gauge))
    return report(TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH,
                  "gauge seed struct_size is incompatible with ABI v1",
                  errorMessage, errorCapacity);

  tensorium::solver::CttBssnBuffers cppOutputs;
  cppOutputs.chi = outputs->chi;
  cppOutputs.meanCurvature = outputs->mean_curvature;
  cppOutputs.lapse = outputs->lapse;
  for (std::size_t i = 0; i < 9; ++i) {
    cppOutputs.conformalMetric[i] = outputs->conformal_metric[i];
    cppOutputs.inverseConformalMetric[i] =
        outputs->inverse_conformal_metric[i];
    cppOutputs.traceFreeExtrinsicCurvature[i] =
        outputs->trace_free_extrinsic_curvature[i];
  }
  for (std::size_t i = 0; i < 3; ++i)
    cppOutputs.shift[i] = outputs->shift[i];

  tensorium::solver::BssnGaugeSeed cppGauge;
  if (gauge) {
    cppGauge.lapse = gauge->lapse;
    for (std::size_t i = 0; i < 3; ++i)
      cppGauge.shift[i] = gauge->shift[i];
  }
  try {
    tensorium::solver::initializeBssnFromRadialCtt(
        solution->value, cppTarget, cppOutputs, cppGauge);
    return TENSORIUM_CONSTRAINT_STATUS_OK;
  } catch (const std::exception &exception) {
    return report(TENSORIUM_CONSTRAINT_STATUS_SOLVER_ERROR, exception.what(),
                  errorMessage, errorCapacity);
  } catch (...) {
    return report(TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "unknown CTT-to-BSSN initialization failure", errorMessage,
                  errorCapacity);
  }
}

void tensorium_constraint_solution_destroy_v1(
    tensorium_constraint_solution_v1 *solution) {
  delete solution;
}

} // extern "C"
