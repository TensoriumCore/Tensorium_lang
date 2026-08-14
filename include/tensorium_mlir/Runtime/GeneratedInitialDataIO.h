#pragma once

#include "tensorium_mlir/Runtime/GeneratedInitialData.h"
#include "tensorium_mlir/Runtime/TwoPunctureDiagnostics.h"
#include "tensorium_mlir/Runtime/TwoPunctureHandoff.h"
#include "tensorium_mlir/Runtime/TwoPunctureMassCalibration.h"
#include "tensorium_mlir/Runtime/TwoPunctureRegularity.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tensorium_mlir::runtime {

struct GeneratedInitialDataSliceOptions {
  std::size_t resolution = 129;
  double halfWidth = 8.0;
};

struct GeneratedInitialDataExportReport {
  std::string csvPath;
  std::string metadataPath;
  std::size_t pointCount = 0;
  double admEnergy = 0.0;
  std::array<double, 3> admLinearMomentum{};
  std::array<double, 3> admAngularMomentum{};
  std::array<double, 2> punctureAdmMasses{};
  double regularityError = 0.0;
  double bssnTraceError = 0.0;
  double minChi = 0.0;
  double maxChi = 0.0;
};

inline SpectralPoint3D generatedInitialDataProjectedOutMaxPoint(
    const GeneratedSpectralInitialDataSolution &solution) {
  if (!solution.grid || solution.grid->size() == 0)
    throw std::runtime_error("generated initial_data residual grid is empty");
  const std::size_t pointIndex =
      solution.projectedOutResidualMaxIndex % solution.grid->size();
  const std::size_t i = pointIndex % solution.grid->n1();
  const std::size_t line = pointIndex / solution.grid->n1();
  const std::size_t j = line % solution.grid->n2();
  const std::size_t k = line / solution.grid->n2();
  return solution.grid->point(i, j, k);
}

inline void writeGeneratedInitialDataContinuationMetadata(
    std::ostream &output,
    const GeneratedSpectralInitialDataSolution &solution) {
  output << "[";
  for (std::size_t stage = 0; stage < solution.continuationStages.size();
       ++stage) {
    const auto &report = solution.continuationStages[stage];
    if (stage != 0)
      output << ", ";
    output << "{\"resolution\": [" << report.resolution[0] << ", "
           << report.resolution[1] << ", " << report.resolution[2]
           << "], \"status\": " << static_cast<int>(report.solveResult.status)
           << ", \"initial_residual_l2\": "
           << report.solveResult.initialResidualL2
           << ", \"newton_steps\": " << report.solveResult.steps
           << ", \"linear_iterations\": " << report.solveResult.linearIterations
           << ", \"residual_l2\": " << report.residualL2
           << ", \"residual_max\": " << report.residualMaxAbs
           << ", \"raw_residual_l2\": " << report.rawResidualL2
           << ", \"raw_residual_max\": " << report.rawResidualMaxAbs
           << ", \"solve_wall_seconds\": " << report.solveWallSeconds << "}";
  }
  output << "]";
}

inline GeneratedInitialDataExportReport
exportGeneratedInitialDataCollocationCsv(
    const GeneratedSpectralInitialDataSolution &solution,
    const std::string &outputPath) {
  if (!solution.descriptor || !solution.systemDescriptor || !solution.grid ||
      solution.generatedSystem.equations.empty() || solution.fields.empty())
    throw std::runtime_error("generated initial_data solution is incomplete");
  if (!solution.converged())
    throw std::runtime_error(
        "generated initial_data must converge before export");

  std::vector<std::vector<double>> physicalFields = solution.fields;
  for (std::size_t unknown = 0; unknown < physicalFields.size(); ++unknown) {
    const SpectralResidualProblem *problem = nullptr;
    for (const auto &equation : solution.generatedSystem.equations) {
      if (equation.unknownIndex == unknown) {
        problem = &equation.problem;
        break;
      }
    }
    if (!problem)
      throw std::runtime_error(
          "generated initial_data unknown has no residual equation");
    if (problem->unknownMap.transform) {
      physicalFields[unknown] =
          applySpectralUnknownMap(
              *solution.grid,
              solution.grid->derivatives(solution.fields[unknown]),
              problem->unknownMap, problem->unknownMapParams)
              .value;
    }
  }

  std::ofstream csv(outputPath);
  if (!csv)
    throw std::runtime_error("cannot open generated initial_data CSV: " +
                             outputPath);
  csv << std::setprecision(17) << "i,j,k,q1,q2,q3,x,y,z";
  for (std::int64_t unknown = 0;
       unknown < solution.systemDescriptor->unknown_count; ++unknown) {
    csv << ',' << solution.systemDescriptor->unknown_names[unknown];
  }
  csv << '\n';

  const auto &coordinateProblem =
      solution.generatedSystem.equations.front().problem;
  for (std::size_t k = 0; k < solution.grid->n3(); ++k) {
    for (std::size_t j = 0; j < solution.grid->n2(); ++j) {
      for (std::size_t i = 0; i < solution.grid->n1(); ++i) {
        const auto point = solution.grid->point(i, j, k);
        const double logical[3] = {point.x1, point.x2, point.x3};
        double physical[3] = {point.x1, point.x2, point.x3};
        if (coordinateProblem.coordinateMap.map) {
          coordinateProblem.coordinateMap.map(
              logical, physical, coordinateProblem.coordinateParams.data(),
              static_cast<std::int64_t>(
                  coordinateProblem.coordinateParams.size()),
              coordinateProblem.coordinateMap.userData);
        }
        csv << i << ',' << j << ',' << k << ',' << logical[0] << ','
            << logical[1] << ',' << logical[2] << ',' << physical[0] << ','
            << physical[1] << ',' << physical[2];
        for (const auto &field : physicalFields)
          csv << ',' << field[point.index];
        csv << '\n';
      }
    }
  }
  csv.close();
  if (!csv)
    throw std::runtime_error("failed while writing generated initial_data CSV");

  const std::string metadataPath = outputPath + ".json";
  const auto projectedOutMaxPoint =
      generatedInitialDataProjectedOutMaxPoint(solution);
  std::ofstream metadata(metadataPath);
  if (!metadata)
    throw std::runtime_error("cannot open generated initial_data metadata: " +
                             metadataPath);
  metadata << std::setprecision(17) << "{\n"
           << "  \"case\": \"" << solution.descriptor->symbol_name << "\",\n"
           << "  \"reconstruction\": \"none\",\n"
           << "  \"spectral_resolution\": [" << solution.grid->n1() << ", "
           << solution.grid->n2() << ", " << solution.grid->n3() << "],\n"
           << "  \"continuation_stages\": ";
  writeGeneratedInitialDataContinuationMetadata(metadata, solution);
  metadata << ",\n"
           << "  \"newton_steps\": " << solution.solveResult.steps << ",\n"
           << "  \"linear_iterations\": "
           << solution.solveResult.linearIterations << ",\n"
           << "  \"solve_wall_seconds\": " << solution.solveWallSeconds << ",\n"
           << "  \"residual_l2\": " << solution.residual.l2Norm << ",\n"
           << "  \"residual_max\": " << solution.residual.maxAbs << ",\n"
           << "  \"projected_residual_l2\": " << solution.residual.l2Norm
           << ",\n"
           << "  \"projected_residual_max\": " << solution.residual.maxAbs
           << ",\n"
           << "  \"raw_residual_l2\": " << solution.rawResidual.l2Norm << ",\n"
           << "  \"raw_residual_max\": " << solution.rawResidual.maxAbs << ",\n"
           << "  \"projected_out_residual_l2\": "
           << solution.projectedOutResidualL2 << ",\n"
           << "  \"projected_out_residual_max\": "
           << solution.projectedOutResidualMaxAbs << ",\n"
           << "  \"projected_out_residual_max_logical\": ["
           << projectedOutMaxPoint.x1 << ", " << projectedOutMaxPoint.x2 << ", "
           << projectedOutMaxPoint.x3 << "],\n"
           << "  \"layout\": \"spectral collocation, i-fastest\"\n"
           << "}\n";
  metadata.close();
  if (!metadata)
    throw std::runtime_error(
        "failed while writing generated initial_data metadata");

  GeneratedInitialDataExportReport report;
  report.csvPath = outputPath;
  report.metadataPath = metadataPath;
  report.pointCount = solution.grid->size();
  return report;
}

inline double generatedInitialDataParameter(
    const GeneratedSpectralInitialDataSolution &solution,
    std::string_view name) {
  const auto found = solution.parameters.find(std::string(name));
  if (found == solution.parameters.end())
    throw std::runtime_error("generated initial_data parameter '" +
                             std::string(name) + "' is unavailable");
  return found->second;
}

inline double generatedInitialDataParameter(
    const GeneratedSpectralInitialDataSolution &solution,
    std::string_view canonicalName, std::string_view descriptiveName) {
  const auto canonical = solution.parameters.find(std::string(canonicalName));
  const auto descriptive =
      solution.parameters.find(std::string(descriptiveName));
  if (canonical != solution.parameters.end() &&
      descriptive != solution.parameters.end() &&
      canonicalName != descriptiveName) {
    throw std::runtime_error(
        "generated initial_data parameter is ambiguous: '" +
        std::string(canonicalName) + "' and '" + std::string(descriptiveName) +
        "' are both available");
  }
  if (canonical != solution.parameters.end())
    return canonical->second;
  if (descriptive != solution.parameters.end())
    return descriptive->second;
  throw std::runtime_error("generated initial_data parameter '" +
                           std::string(canonicalName) + "' or '" +
                           std::string(descriptiveName) + "' is unavailable");
}

template <std::size_t Components>
inline std::array<double *, Components> generatedInitialDataBuffers(
    std::array<std::vector<double>, Components> &storage,
    std::size_t pointCount) {
  std::array<double *, Components> pointers{};
  for (std::size_t component = 0; component < Components; ++component) {
    storage[component].assign(pointCount, 0.0);
    pointers[component] = storage[component].data();
  }
  return pointers;
}

inline GeneratedInitialDataExportReport exportGeneratedInitialDataBssnSlice(
    const GeneratedSpectralInitialDataSolution &solution,
    const std::string &outputPath,
    const GeneratedInitialDataSliceOptions &sliceOptions = {}) {
  if (!solution.descriptor || !solution.grid ||
      solution.generatedSystem.equations.empty() || solution.fields.empty())
    throw std::runtime_error("generated initial_data solution is incomplete");
  if (!solution.converged())
    throw std::runtime_error(
        "generated initial_data must converge before reconstruction");
  requireGeneratedInitialDataString(solution.descriptor->reconstruction,
                                    "reconstruction");
  if (std::string_view(solution.descriptor->reconstruction) !=
      "two_puncture_bssn") {
    throw std::runtime_error("unsupported generated BSSN reconstruction '" +
                             std::string(solution.descriptor->reconstruction) +
                             "'");
  }
  if (sliceOptions.resolution < 3 || !(sliceOptions.halfWidth > 0.0) ||
      !std::isfinite(sliceOptions.halfWidth))
    throw std::runtime_error("invalid generated initial_data slice geometry");
  if (sliceOptions.resolution >
      std::numeric_limits<std::size_t>::max() / sliceOptions.resolution)
    throw std::runtime_error("generated initial_data slice is too large");

  TwoPuncturePhysicalParameters physical;
  physical.halfSeparation = generatedInitialDataParameter(solution, "b");
  physical.bareMasses = {generatedInitialDataParameter(solution, "m1"),
                         generatedInitialDataParameter(solution, "m2")};
  physical.momenta = {{
      {generatedInitialDataParameter(solution, "p1x", "P1_x"),
       generatedInitialDataParameter(solution, "p1y", "P1_y"),
       generatedInitialDataParameter(solution, "p1z", "P1_z")},
      {generatedInitialDataParameter(solution, "p2x", "P2_x"),
       generatedInitialDataParameter(solution, "p2y", "P2_y"),
       generatedInitialDataParameter(solution, "p2z", "P2_z")},
  }};
  physical.spins = {{
      {generatedInitialDataParameter(solution, "s1x", "S1_x"),
       generatedInitialDataParameter(solution, "s1y", "S1_y"),
       generatedInitialDataParameter(solution, "s1z", "S1_z")},
      {generatedInitialDataParameter(solution, "s2x", "S2_x"),
       generatedInitialDataParameter(solution, "s2y", "S2_y"),
       generatedInitialDataParameter(solution, "s2z", "S2_z")},
  }};

  const auto &field = solution.fields.front();
  const double vInfinity = solution.grid->interpolate(field, 1.0, 0.0, 0.0);
  const auto adm = makeTwoPunctureAdmDiagnostics(
      physical.halfSeparation, physical.bareMasses[0], physical.bareMasses[1],
      vInfinity, physical.momenta[0], physical.momenta[1], physical.spins[0],
      physical.spins[1]);
  const auto punctureSample =
      sampleTwoPunctureRegularField(*solution.grid, field);
  const auto localMasses = makeTwoPunctureLocalMassDiagnostics(
      physical.halfSeparation, physical.bareMasses[0], physical.bareMasses[1],
      punctureSample.values[0], punctureSample.values[1]);
  const auto regularity =
      measureTwoPunctureScalarRegularity(*solution.grid, field);
  if (regularity.maxPhiVariation() > 1.0e-3)
    throw std::runtime_error(
        "generated two-puncture axis regularity check failed");

  const std::size_t sliceN = sliceOptions.resolution;
  const std::size_t pointCount = sliceN * sliceN;
  std::vector<double> x(pointCount, 0.0);
  std::vector<double> y(pointCount, 0.0);
  std::vector<double> z(pointCount, 0.0);
  const double spacing =
      2.0 * sliceOptions.halfWidth / static_cast<double>(sliceN - 1);
  for (std::size_t i = 0; i < sliceN; ++i) {
    for (std::size_t j = 0; j < sliceN; ++j) {
      const std::size_t point = i * sliceN + j;
      x[point] = -sliceOptions.halfWidth + spacing * static_cast<double>(i);
      y[point] = -sliceOptions.halfWidth + spacing * static_cast<double>(j);
    }
  }

  std::vector<double> chi(pointCount, 0.0);
  std::vector<double> meanCurvature(pointCount, 0.0);
  std::vector<double> correction(pointCount, 0.0);
  std::vector<double> psi(pointCount, 0.0);
  std::array<std::vector<double>, 9> conformalMetric;
  std::array<std::vector<double>, 9> inverseConformalMetric;
  std::array<std::vector<double>, 9> traceFreeExtrinsicCurvature;
  std::array<std::vector<double>, 3> conformalConnection;
  std::array<std::vector<double>, 3> shift;
  TwoPunctureBssnGridBuffers outputs;
  outputs.chi = chi.data();
  outputs.meanCurvature = meanCurvature.data();
  outputs.regularCorrection = correction.data();
  outputs.conformalFactor = psi.data();
  outputs.conformalMetric =
      generatedInitialDataBuffers(conformalMetric, pointCount);
  outputs.inverseConformalMetric =
      generatedInitialDataBuffers(inverseConformalMetric, pointCount);
  outputs.traceFreeExtrinsicCurvature =
      generatedInitialDataBuffers(traceFreeExtrinsicCurvature, pointCount);
  outputs.conformalConnection =
      generatedInitialDataBuffers(conformalConnection, pointCount);
  outputs.shift = generatedInitialDataBuffers(shift, pointCount);

  interpolateTwoPunctureBssnToCartesianGrid(
      solution.generatedSystem.equations.front().problem, field, physical,
      TwoPunctureCartesianGridView{pointCount, {x.data(), y.data(), z.data()}},
      outputs);

  std::vector<double> lapse(pointCount, 0.0);
  double maxTrace = 0.0;
  double minChi = std::numeric_limits<double>::infinity();
  double maxChi = 0.0;
  for (std::size_t point = 0; point < pointCount; ++point) {
    lapse[point] = std::sqrt(chi[point]);
    minChi = std::min(minChi, chi[point]);
    maxChi = std::max(maxChi, chi[point]);
    const double trace = traceFreeExtrinsicCurvature[0][point] +
                         traceFreeExtrinsicCurvature[4][point] +
                         traceFreeExtrinsicCurvature[8][point];
    maxTrace = std::max(maxTrace, std::abs(trace));
    if (!std::isfinite(chi[point]) || !std::isfinite(lapse[point]) ||
        !std::isfinite(trace))
      throw std::runtime_error(
          "generated BSSN reconstruction produced non-finite fields");
  }

  std::ofstream csv(outputPath);
  if (!csv)
    throw std::runtime_error("cannot open generated initial_data CSV: " +
                             outputPath);
  csv << std::setprecision(17)
      << "i,j,x,y,z,u,psi,chi,alpha,gammatilde_xx,gammatilde_xy,"
         "gammatilde_xz,gammatilde_yy,gammatilde_yz,gammatilde_zz,"
         "Atilde_xx,Atilde_xy,Atilde_xz,Atilde_yy,Atilde_yz,Atilde_zz,K,"
         "Gamma_x,Gamma_y,Gamma_z,beta_x,beta_y,beta_z\n";
  for (std::size_t i = 0; i < sliceN; ++i) {
    for (std::size_t j = 0; j < sliceN; ++j) {
      const std::size_t point = i * sliceN + j;
      csv << i << ',' << j << ',' << x[point] << ',' << y[point] << ','
          << z[point] << ',' << correction[point] << ',' << psi[point] << ','
          << chi[point] << ',' << lapse[point] << ','
          << conformalMetric[0][point] << ',' << conformalMetric[1][point]
          << ',' << conformalMetric[2][point] << ','
          << conformalMetric[4][point] << ',' << conformalMetric[5][point]
          << ',' << conformalMetric[8][point] << ','
          << traceFreeExtrinsicCurvature[0][point] << ','
          << traceFreeExtrinsicCurvature[1][point] << ','
          << traceFreeExtrinsicCurvature[2][point] << ','
          << traceFreeExtrinsicCurvature[4][point] << ','
          << traceFreeExtrinsicCurvature[5][point] << ','
          << traceFreeExtrinsicCurvature[8][point] << ','
          << meanCurvature[point] << ',' << conformalConnection[0][point] << ','
          << conformalConnection[1][point] << ','
          << conformalConnection[2][point] << ',' << shift[0][point] << ','
          << shift[1][point] << ',' << shift[2][point] << '\n';
    }
  }
  csv.close();
  if (!csv)
    throw std::runtime_error("failed while writing generated initial_data CSV");

  const std::string metadataPath = outputPath + ".json";
  const auto projectedOutMaxPoint =
      generatedInitialDataProjectedOutMaxPoint(solution);
  std::ofstream metadata(metadataPath);
  if (!metadata)
    throw std::runtime_error("cannot open generated initial_data metadata: " +
                             metadataPath);
  metadata << std::setprecision(17) << "{\n"
           << "  \"case\": \"" << solution.descriptor->symbol_name << "\",\n"
           << "  \"formulation\": \"Bowen-York puncture / BSSN\",\n"
           << "  \"half_separation\": " << physical.halfSeparation << ",\n"
           << "  \"bare_masses\": [" << physical.bareMasses[0] << ", "
           << physical.bareMasses[1] << "],\n"
           << "  \"momenta\": [[" << physical.momenta[0][0] << ", "
           << physical.momenta[0][1] << ", " << physical.momenta[0][2] << "], ["
           << physical.momenta[1][0] << ", " << physical.momenta[1][1] << ", "
           << physical.momenta[1][2] << "]],\n"
           << "  \"spins\": [[" << physical.spins[0][0] << ", "
           << physical.spins[0][1] << ", " << physical.spins[0][2] << "], ["
           << physical.spins[1][0] << ", " << physical.spins[1][1] << ", "
           << physical.spins[1][2] << "]],\n"
           << "  \"spectral_resolution\": [" << solution.grid->n1() << ", "
           << solution.grid->n2() << ", " << solution.grid->n3() << "],\n"
           << "  \"continuation_stages\": ";
  writeGeneratedInitialDataContinuationMetadata(metadata, solution);
  metadata << ",\n"
           << "  \"slice_resolution\": [" << sliceN << ", " << sliceN
           << ", 1],\n"
           << "  \"slice_half_width\": " << sliceOptions.halfWidth << ",\n"
           << "  \"slice_spacing\": " << spacing << ",\n"
           << "  \"newton_steps\": " << solution.solveResult.steps << ",\n"
           << "  \"linear_iterations\": "
           << solution.solveResult.linearIterations << ",\n"
           << "  \"solve_wall_seconds\": " << solution.solveWallSeconds << ",\n"
           << "  \"residual_l2\": " << solution.residual.l2Norm << ",\n"
           << "  \"residual_max\": " << solution.residual.maxAbs << ",\n"
           << "  \"projected_residual_l2\": " << solution.residual.l2Norm
           << ",\n"
           << "  \"projected_residual_max\": " << solution.residual.maxAbs
           << ",\n"
           << "  \"raw_residual_l2\": " << solution.rawResidual.l2Norm << ",\n"
           << "  \"raw_residual_max\": " << solution.rawResidual.maxAbs << ",\n"
           << "  \"projected_out_residual_l2\": "
           << solution.projectedOutResidualL2 << ",\n"
           << "  \"projected_out_residual_max\": "
           << solution.projectedOutResidualMaxAbs << ",\n"
           << "  \"projected_out_residual_max_logical\": ["
           << projectedOutMaxPoint.x1 << ", " << projectedOutMaxPoint.x2 << ", "
           << projectedOutMaxPoint.x3 << "],\n"
           << "  \"adm_energy\": " << adm.energy << ",\n"
           << "  \"adm_linear_momentum\": [" << adm.linearMomentum[0] << ", "
           << adm.linearMomentum[1] << ", " << adm.linearMomentum[2] << "],\n"
           << "  \"adm_angular_momentum\": [" << adm.angularMomentum[0] << ", "
           << adm.angularMomentum[1] << ", " << adm.angularMomentum[2] << "],\n"
           << "  \"puncture_adm_masses\": [" << localMasses.admMasses[0] << ", "
           << localMasses.admMasses[1] << "],\n"
           << "  \"axis_regularity_error\": " << regularity.maxPhiVariation()
           << ",\n"
           << "  \"bssn_trace_error\": " << maxTrace << ",\n"
           << "  \"gauge\": {\"lapse\": \"psi^-2\", \"shift\": [0, 0, 0]},\n"
           << "  \"fields\": [\"chi\", \"alpha\", \"gammatilde_ij\", "
              "\"Atilde_ij\", \"K\", \"Gamma_i\", \"beta_i\"],\n"
           << "  \"layout\": \"Cartesian z=0 slice, row-major i*slice_n+j\"\n"
           << "}\n";
  metadata.close();
  if (!metadata)
    throw std::runtime_error(
        "failed while writing generated initial_data metadata");

  GeneratedInitialDataExportReport report;
  report.csvPath = outputPath;
  report.metadataPath = metadataPath;
  report.pointCount = pointCount;
  report.admEnergy = adm.energy;
  report.admLinearMomentum = adm.linearMomentum;
  report.admAngularMomentum = adm.angularMomentum;
  report.punctureAdmMasses = localMasses.admMasses;
  report.regularityError = regularity.maxPhiVariation();
  report.bssnTraceError = maxTrace;
  report.minChi = minChi;
  report.maxChi = maxChi;
  return report;
}

} // namespace tensorium_mlir::runtime
