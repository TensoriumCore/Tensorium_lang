#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralEllipticSolveStatus;
using tensorium_mlir::runtime::SpectralEllipticSolveOptions;
using tensorium_mlir::runtime::SpectralGeneratedResidualSystemEquationInputs;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralLinearSolveKind;
using tensorium_mlir::runtime::SpectralPreconditionerKind;
using tensorium_mlir::runtime::assembleGeneratedSpectralResidualSystem;
using tensorium_mlir::runtime::makeGeneratedSpectralResidualSystem;
using tensorium_mlir::runtime::solveGeneratedSpectralEllipticSystem;

struct PunctureParams {
  double eps2 = 0.08;
  double mass = 0.35;
  double px = 0.08;
  double x0 = 0.12;
  double y0 = -0.08;
  double z0 = 0.0;
};

struct ContinuationStage {
  std::string name;
  PunctureParams params{};
  double residualTolerance = 0.0;
  double requiredRatio = 1.0;
  double minimumShiftMagnitude = 0.0;
  bool requiresNewtonUpdate = false;
};

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> splitCsvLine(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ','))
    out.push_back(trim(item));
  return out;
}

double parseDoubleField(const std::vector<std::string> &fields,
                        std::size_t index, const char *name) {
  if (index >= fields.size() || fields[index].empty())
    throw std::runtime_error(std::string("missing continuation field: ") + name);
  std::size_t consumed = 0;
  const double value = std::stod(fields[index], &consumed);
  if (consumed != fields[index].size())
    throw std::runtime_error(std::string("invalid continuation field: ") + name);
  return value;
}

bool parseBoolField(const std::vector<std::string> &fields, std::size_t index,
                    const char *name) {
  if (index >= fields.size())
    throw std::runtime_error(std::string("missing continuation field: ") + name);
  const std::string value = trim(fields[index]);
  if (value == "true" || value == "1")
    return true;
  if (value == "false" || value == "0")
    return false;
  throw std::runtime_error(std::string("invalid continuation bool field: ") +
                           name);
}

std::size_t getenvSize(const char *name, std::size_t defaultValue) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
    return defaultValue;
  std::size_t consumed = 0;
  const auto parsed = std::stoull(raw, &consumed);
  if (consumed != std::string(raw).size() || parsed == 0)
    throw std::runtime_error(std::string("invalid positive integer env var: ") +
                             name);
  return static_cast<std::size_t>(parsed);
}

ContinuationStage parseContinuationStage(const std::vector<std::string> &fields,
                                         std::size_t lineNumber) {
  if (fields.size() != 11) {
    throw std::runtime_error("Bowen-York continuation stage line " +
                             std::to_string(lineNumber) +
                             " must have 11 CSV fields");
  }
  ContinuationStage stage;
  stage.name = fields[0];
  if (stage.name.empty())
    throw std::runtime_error("Bowen-York continuation stage name is empty");
  stage.params.eps2 = parseDoubleField(fields, 1, "eps2");
  stage.params.mass = parseDoubleField(fields, 2, "mass");
  stage.params.px = parseDoubleField(fields, 3, "px");
  stage.params.x0 = parseDoubleField(fields, 4, "x0");
  stage.params.y0 = parseDoubleField(fields, 5, "y0");
  stage.params.z0 = parseDoubleField(fields, 6, "z0");
  stage.residualTolerance =
      parseDoubleField(fields, 7, "residual_tolerance");
  stage.requiredRatio = parseDoubleField(fields, 8, "required_ratio");
  stage.minimumShiftMagnitude =
      parseDoubleField(fields, 9, "minimum_shift_magnitude");
  stage.requiresNewtonUpdate =
      parseBoolField(fields, 10, "requires_newton_update");
  return stage;
}

std::vector<ContinuationStage> defaultContinuationStages() {
  return {
      ContinuationStage{"wide-easy",
                        PunctureParams{0.22, 0.30, 0.02, 0.12, -0.08, 0.0},
                        2e-4,
                        0.999,
                        0.0,
                        false},
      ContinuationStage{"wide",
                        PunctureParams{0.16, 0.32, 0.04, 0.12, -0.08, 0.0},
                        0.0,
                        0.999,
                        0.02,
                        true},
  };
}

std::vector<ContinuationStage> loadContinuationStages() {
  const char *path = std::getenv("TENSORIUM_BY_CONTINUATION_STAGES");
  if (!path || path[0] == '\0')
    return defaultContinuationStages();

  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("failed to open Bowen-York continuation stages: " +
                             std::string(path));

  std::vector<ContinuationStage> stages;
  std::string line;
  std::size_t lineNumber = 0;
  while (std::getline(in, line)) {
    ++lineNumber;
    const std::string stripped = trim(line);
    if (stripped.empty() || stripped[0] == '#')
      continue;
    stages.push_back(parseContinuationStage(splitCsvLine(stripped), lineNumber));
  }
  if (stages.empty())
    throw std::runtime_error("Bowen-York continuation stage file is empty");
  return stages;
}

double bowenYorkA2(double x, double y, double z, const PunctureParams &params) {
  const double dx = x - params.x0;
  const double dy = y - params.y0;
  const double dz = z - params.z0;
  const double r2 = dx * dx + dy * dy + dz * dz + params.eps2;
  const double invR2 = 1.0 / r2;
  const double invR4 = invR2 * invR2;
  const double p2 = params.px * params.px;
  const double pn2 = p2 * dx * dx * invR2;
  return 4.5 * (p2 + 2.0 * pn2) * invR4;
}

double psiSingular(double x, double y, double z, const PunctureParams &params) {
  const double dx = x - params.x0;
  const double dy = y - params.y0;
  const double dz = z - params.z0;
  const double r2 = dx * dx + dy * dy + dz * dz + params.eps2;
  return 1.0 + 0.5 * params.mass / std::sqrt(r2);
}

double minPsi(const SpectralGrid3D &grid, std::span<const double> u,
              const PunctureParams &params) {
  double out = std::numeric_limits<double>::infinity();
#pragma omp parallel for collapse(3) reduction(min : out) schedule(static)
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const double z = grid.axis(2).points[k];
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      const double y = grid.axis(1).points[j];
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const double x = grid.axis(0).points[i];
        const std::size_t p = grid.index(i, j, k);
        out = std::min(out, psiSingular(x, y, z, params) + u[p]);
      }
    }
  }
  return out;
}

std::size_t nearestAxisPoint(const SpectralAxis &axis, double target) {
  std::size_t best = 0;
  double bestDistance = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < axis.points.size(); ++i) {
    const double distance = std::abs(axis.points[i] - target);
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}

void exportBowenYorkSliceCsv(const std::string &path, const SpectralGrid3D &grid,
                             std::span<const double> u,
                             std::span<const double> residual,
                             const PunctureParams &params) {
  if (u.size() != grid.size() || residual.size() != grid.size())
    throw std::runtime_error("Bowen-York slice export size mismatch");
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("failed to open Bowen-York slice CSV: " + path);

  const std::size_t k = nearestAxisPoint(grid.axis(2), params.z0);
  out << std::setprecision(17);
  out << "i,j,k,x,y,z,u,psi_singular,psi,residual,r_puncture\n";
  for (std::size_t j = 0; j < grid.n2(); ++j) {
    const double y = grid.axis(1).points[j];
    for (std::size_t i = 0; i < grid.n1(); ++i) {
      const double x = grid.axis(0).points[i];
      const double z = grid.axis(2).points[k];
      const std::size_t p = grid.index(i, j, k);
      const double dx = x - params.x0;
      const double dy = y - params.y0;
      const double dz = z - params.z0;
      const double rPuncture = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double singular = psiSingular(x, y, z, params);
      out << i << "," << j << "," << k << "," << x << "," << y << "," << z
          << "," << u[p] << "," << singular << "," << singular + u[p] << ","
          << residual[p] << "," << rPuncture << "\n";
    }
  }
}

double estimateBowenYorkJacobianShift(const SpectralGrid3D &grid,
                                      std::span<const double> u,
                                      const PunctureParams &params) {
  double sum = 0.0;
  double maxAbs = 0.0;
#pragma omp parallel for collapse(3) reduction(+ : sum) reduction(max : maxAbs) schedule(static)
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const double z = grid.axis(2).points[k];
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      const double y = grid.axis(1).points[j];
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const double x = grid.axis(0).points[i];
        const std::size_t p = grid.index(i, j, k);
        const double psi = psiSingular(x, y, z, params) + u[p];
        if (!(psi > 0.0) || !std::isfinite(psi))
          continue;
        const double psi2 = psi * psi;
        const double psi4 = psi2 * psi2;
        const double psi8 = psi4 * psi4;
        const double reaction = -0.875 * bowenYorkA2(x, y, z, params) / psi8;
        if (std::isfinite(reaction)) {
          sum += reaction;
          maxAbs = std::max(maxAbs, std::abs(reaction));
        }
      }
    }
  }
  const double mean = sum / static_cast<double>(grid.size());
  const double biased = mean - 0.25 * maxAbs;
  return std::clamp(biased, -1.0, -1.0e-4);
}

SpectralEllipticSolveOptions makeOptions(double ratioTarget,
                                         double residualTolerance,
                                         double initialResidualL2,
                                         double preconditionerShift) {
  SpectralEllipticSolveOptions options;
  options.maxNewtonSteps = 8;
  options.residualTolerance = residualTolerance;
  options.residualRatioTarget = ratioTarget;
  options.linearSolver = SpectralLinearSolveKind::MatrixFreeGMRES;
  options.denseJacobianMaxUnknowns = 1;
  options.gmresMaxIterations = 384;
  options.gmresTolerance =
      std::min(3e-3, std::max(1e-7, 0.9 * initialResidualL2));
  options.gmresRelativeTolerance = 0.0;
  options.gmresPreconditioner =
      SpectralPreconditionerKind::DenseLaplacianShift;
  options.preconditionerLaplacianShift = preconditionerShift;
  options.jvpOptions.relativeStep = 1e-6;
  options.linearPivotTolerance = 1e-13;
  options.preconditionerPivotTolerance = 1e-12;
  return options;
}

const char *solveStatusName(SpectralEllipticSolveStatus status) {
  switch (status) {
  case SpectralEllipticSolveStatus::MaxSteps:
    return "max_steps";
  case SpectralEllipticSolveStatus::Converged:
    return "converged";
  case SpectralEllipticSolveStatus::InvalidResidual:
    return "invalid_residual";
  case SpectralEllipticSolveStatus::LinearSolveFailed:
    return "linear_solve_failed";
  case SpectralEllipticSolveStatus::LineSearchFailed:
    return "line_search_failed";
  case SpectralEllipticSolveStatus::InvalidInput:
    return "invalid_input";
  }
  return "unknown";
}

} // namespace

int main() {
  try {
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT >= 1,
                  "expected at least one generated spectral point kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT >= 1,
                  "expected at least one generated spectral grid kernel");
    static_assert(TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT >= 1,
                  "expected at least one generated spectral residual system");

    const std::vector<ContinuationStage> stages = loadContinuationStages();

    const std::size_t gridN = getenvSize("TENSORIUM_BY_GRID_N", 5);
    SpectralGrid3D grid(SpectralAxis::chebyshevLobatto(gridN, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(gridN, -1.0, 1.0),
                        SpectralAxis::chebyshevLobatto(gridN, -1.0, 1.0));
    std::array<std::vector<double>, 1> solutionFields{
        std::vector<double>(grid.size(), 0.0)};

    double firstResidualL2 = 0.0;
    double lastResidualL2 = 0.0;
    double lastResidualMax = 0.0;
    double lastMinPsi = 0.0;
    PunctureParams lastParams{};
    std::vector<double> lastResidualValues;
    bool usedGeneratedGridKernel = false;
    bool usedGMRES = false;
    bool usedPreconditioner = false;

    for (std::size_t stageIndex = 0; stageIndex < stages.size();
         ++stageIndex) {
      const auto &stage = stages[stageIndex];
      const double rawParams[] = {stage.params.eps2, stage.params.mass,
                                  stage.params.px,   stage.params.x0,
                                  stage.params.y0,   stage.params.z0};
      const std::array<SpectralGeneratedResidualSystemEquationInputs, 1>
          systemInputs{{
              SpectralGeneratedResidualSystemEquationInputs{
                  std::span<const double>(rawParams, 6),
                  std::span<const std::vector<double>>()},
          }};
      const auto generatedSystem = makeGeneratedSpectralResidualSystem(
          tensorium_spectral_residual_systems,
          TENSORIUM_SPECTRAL_RESIDUAL_SYSTEM_COUNT,
          "SpectralBowenYorkRegularizedPuncture3D", grid,
          tensorium_spectral_residual_kernels,
          TENSORIUM_SPECTRAL_RESIDUAL_KERNEL_COUNT,
          tensorium_spectral_residual_grid_kernels,
          TENSORIUM_SPECTRAL_RESIDUAL_GRID_KERNEL_COUNT,
          std::span<const SpectralGeneratedResidualSystemEquationInputs>(
              systemInputs.data(), systemInputs.size()),
          1, 1);
      const auto initialResidual = assembleGeneratedSpectralResidualSystem(
          generatedSystem,
          std::span<const std::vector<double>>(solutionFields.data(),
                                               solutionFields.size()));
      if (stageIndex == 0)
        firstResidualL2 = initialResidual.l2Norm;

      const double initialMinPsi = minPsi(grid, solutionFields[0], stage.params);
      const double estimatedShift = estimateBowenYorkJacobianShift(
          grid, solutionFields[0], stage.params);
      const double preconditionerShift =
          std::min(estimatedShift, -stage.minimumShiftMagnitude);
      const auto options =
          makeOptions(stage.requiredRatio, stage.residualTolerance,
                      initialResidual.l2Norm, preconditionerShift);
      const auto run = solveGeneratedSpectralEllipticSystem(
          generatedSystem,
          std::span<std::vector<double>>(solutionFields.data(),
                                         solutionFields.size()),
          options);
      const auto &solveResult = run.solveResult;
      const auto &finalResidual = run.finalResidual;
      lastResidualL2 = solveResult.finalResidualL2;
      lastResidualMax = finalResidual.maxAbs;
      lastMinPsi = minPsi(grid, solutionFields[0], stage.params);
      lastParams = stage.params;
      lastResidualValues = finalResidual.values;
      usedGeneratedGridKernel =
          usedGeneratedGridKernel || initialResidual.usedGeneratedGridKernels ||
          solveResult.usedGeneratedGridKernel ||
          finalResidual.usedGeneratedGridKernels;
      usedGMRES = usedGMRES || solveResult.usedMatrixFreeGMRES;
      usedPreconditioner =
          usedPreconditioner || solveResult.usedPreconditioner;

      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s initial l2 = %.17g max = %.17g\n",
                  stage.name.c_str(), initialResidual.l2Norm,
                  initialResidual.maxAbs);
      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s estimated shift = %.17g preconditioner shift = %.17g\n",
                  stage.name.c_str(), estimatedShift, preconditionerShift);
      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s options residual_tolerance = %.17g required_ratio = %.17g gmres_max_iterations = %d gmres_tolerance = %.17g min_damping = %.17g\n",
                  stage.name.c_str(), options.residualTolerance,
                  options.residualRatioTarget, options.gmresMaxIterations,
                  options.gmresTolerance, options.minDamping);
      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s steps = %d/%d status = %s (%d) line_search_damping = %.17g\n",
                  stage.name.c_str(), solveResult.steps,
                  solveResult.maxSteps, solveStatusName(solveResult.status),
                  static_cast<int>(solveResult.status),
                  solveResult.lastDamping);
      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s final l2 = %.17g max = %.17g ratio = %.17g\n",
                  stage.name.c_str(), solveResult.finalResidualL2,
                  finalResidual.maxAbs, solveResult.residualRatio);
      std::printf("[generated-spectral-bowen-york-puncture-continuation] stage %s linear iterations = %d residual = %.17g min psi initial = %.17g final = %.17g\n",
                  stage.name.c_str(), solveResult.linearIterations,
                  solveResult.finalLinearResidualL2, initialMinPsi,
                  lastMinPsi);

      const bool solvedByInitialTolerance =
          solveResult.converged() && solveResult.steps == 0 &&
          stage.residualTolerance > 0.0 &&
          initialResidual.l2Norm <= stage.residualTolerance;
      const bool solvedByNewton =
          solveResult.converged() && solveResult.usedGeneratedGridKernel &&
          solveResult.usedMatrixFreeGMRES && solveResult.usedPreconditioner &&
          solveResult.finalResidualL2 < initialResidual.l2Norm;
      const bool madeLinearProgress =
          solveResult.usedMatrixFreeGMRES && solveResult.usedPreconditioner &&
          solveResult.linearIterations > 0 &&
          solveResult.finalLinearResidualL2 < initialResidual.l2Norm;
      const bool initialResidualOk =
          initialResidual.finite && initialResidual.usedGeneratedGridKernels;
      const bool finalResidualOk =
          finalResidual.finite && finalResidual.usedGeneratedGridKernels;
      const bool progressAccepted =
          solvedByInitialTolerance || solvedByNewton || madeLinearProgress;
      const bool psiPositive = lastMinPsi > 0.0;
      if (!initialResidualOk || (stage.requiresNewtonUpdate && !solvedByNewton) ||
          !progressAccepted || !finalResidualOk || !psiPositive) {
        std::fprintf(
            stderr,
            "[generated-spectral-bowen-york-puncture-continuation] failure diagnostics stage=%s status=%s steps=%d/%d gmres_iterations=%d gmres_residual=%.17g line_search_damping=%.17g residual_ratio=%.17g required_ratio=%.17g initial_l2=%.17g final_l2=%.17g final_max=%.17g min_psi_initial=%.17g min_psi_final=%.17g estimated_shift=%.17g preconditioner_shift=%.17g\n",
            stage.name.c_str(), solveStatusName(solveResult.status),
            solveResult.steps, solveResult.maxSteps,
            solveResult.linearIterations, solveResult.finalLinearResidualL2,
            solveResult.lastDamping, solveResult.residualRatio,
            stage.requiredRatio, initialResidual.l2Norm,
            solveResult.finalResidualL2, finalResidual.maxAbs,
            initialMinPsi, lastMinPsi, estimatedShift, preconditionerShift);
        std::fprintf(
            stderr,
            "[generated-spectral-bowen-york-puncture-continuation] failure predicates initial_residual_ok=%d final_residual_ok=%d solved_by_initial_tolerance=%d solved_by_newton=%d made_linear_progress=%d requires_newton_update=%d psi_positive=%d used_grid_kernel=%d used_gmres=%d used_preconditioner=%d\n",
            initialResidualOk ? 1 : 0, finalResidualOk ? 1 : 0,
            solvedByInitialTolerance ? 1 : 0, solvedByNewton ? 1 : 0,
            madeLinearProgress ? 1 : 0, stage.requiresNewtonUpdate ? 1 : 0,
            psiPositive ? 1 : 0, solveResult.usedGeneratedGridKernel ? 1 : 0,
            solveResult.usedMatrixFreeGMRES ? 1 : 0,
            solveResult.usedPreconditioner ? 1 : 0);
        std::fprintf(stderr,
                     "generated Bowen-York puncture continuation stage failed\n");
        return 3;
      }
    }

    std::printf("[generated-spectral-bowen-york-puncture-continuation] first residual l2 = %.17g\n",
                firstResidualL2);
    std::printf("[generated-spectral-bowen-york-puncture-continuation] target residual l2 = %.17g max = %.17g\n",
                lastResidualL2, lastResidualMax);

    const char *sliceCsvPath = std::getenv("TENSORIUM_BY_SLICE_CSV");
    if (sliceCsvPath && sliceCsvPath[0] != '\0') {
      exportBowenYorkSliceCsv(sliceCsvPath, grid, solutionFields[0],
                              lastResidualValues, lastParams);
      std::printf("[generated-spectral-bowen-york-puncture-continuation] wrote slice csv = %s\n",
                  sliceCsvPath);
    }

    if (!usedGeneratedGridKernel || !usedGMRES || !usedPreconditioner ||
        !(lastResidualL2 > 0.0) || !(lastMinPsi > 0.0)) {
      std::fprintf(stderr,
                   "generated Bowen-York puncture continuation mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr,
                 "generated Bowen-York puncture continuation runner failed: %s\n",
                 ex.what());
    return 2;
  }
  return 0;
}
