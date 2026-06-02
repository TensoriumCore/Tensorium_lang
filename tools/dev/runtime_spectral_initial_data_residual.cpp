#include "tensorium_mlir/Runtime/SpectralResidualKernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

using tensorium_mlir::runtime::kSpectralPi;
using tensorium_mlir::runtime::SpectralAxis;
using tensorium_mlir::runtime::SpectralCoordinateMap;
using tensorium_mlir::runtime::SpectralGrid3D;
using tensorium_mlir::runtime::SpectralResidualKernel;
using tensorium_mlir::runtime::evaluateSpectralResidual;

struct GenericResidualParams {
  double alpha = 0.0;
  double beta = 0.0;
  double gamma = 0.0;
  double phiOrigin = 0.0;
  double phiAngularWave = 0.0;
};

double maxAbs(const std::vector<double> &values) {
  double out = 0.0;
  for (double value : values)
    out = std::max(out, std::abs(value));
  return out;
}

GenericResidualParams readGenericParams(const double *params,
                                        std::int64_t paramCount) {
  if (paramCount != 5)
    throw std::runtime_error("generic spectral residual expects 5 params");
  return GenericResidualParams{params[0], params[1], params[2], params[3],
                               params[4]};
}

double genericField(double x, double y, double phi,
                    const GenericResidualParams &params) {
  const double theta = params.phiAngularWave * (phi - params.phiOrigin);
  return x * x * x + 0.5 * x * y + y * y + 0.2 * std::sin(theta) +
         0.07 * x * y * std::cos(theta);
}

double genericFieldD1(double x, double y, double phi,
                      const GenericResidualParams &params) {
  const double theta = params.phiAngularWave * (phi - params.phiOrigin);
  return 3.0 * x * x + 0.5 * y + 0.07 * y * std::cos(theta);
}

double genericFieldD23(double x, double phi,
                       const GenericResidualParams &params) {
  const double theta = params.phiAngularWave * (phi - params.phiOrigin);
  return -0.07 * x * params.phiAngularWave * std::sin(theta);
}

double genericFieldLaplacian(double x, double y, double phi,
                             const GenericResidualParams &params) {
  const double theta = params.phiAngularWave * (phi - params.phiOrigin);
  const double angular2 = params.phiAngularWave * params.phiAngularWave;
  return 6.0 * x + 2.0 -
         angular2 * (0.2 * std::sin(theta) +
                     0.07 * x * y * std::cos(theta));
}

double genericResidualKernel(const tensorium_spectral_residual_point *point,
                             const double *rawParams, std::int64_t paramCount,
                             void *) {
  const GenericResidualParams params =
      readGenericParams(rawParams, paramCount);
  const double x = point->physical[0];
  const double y = point->physical[1];
  const double phi = point->physical[2];
  const double analyticU = genericField(x, y, phi, params);
  const double analyticD1 = genericFieldD1(x, y, phi, params);
  const double analyticD23 = genericFieldD23(x, phi, params);
  const double analyticLap = genericFieldLaplacian(x, y, phi, params);
  const double source =
      -(analyticLap + params.alpha * analyticU + params.beta * analyticD1 +
        params.gamma * analyticD23);
  return point->d11 + point->d22 + point->d33 + params.alpha * point->value +
         params.beta * point->d1 + params.gamma * point->d23 + source;
}

double hamiltonianToyResidualKernel(
    const tensorium_spectral_residual_point *point, const double *params,
    std::int64_t paramCount, void *) {
  if (paramCount != 2)
    throw std::runtime_error("Hamiltonian toy spectral residual expects 2 params");
  const double amplitude = params[0];
  const double phiMode = params[1];
  const double x = point->physical[0];
  const double y = point->physical[1];
  const double phi = point->physical[2];
  const double t2x = 2.0 * x * x - 1.0;
  const double t2y = 2.0 * y * y - 1.0;
  const double analyticLap =
      amplitude * (4.0 * t2y + 4.0 * t2x - phiMode * phiMode * t2x * t2y) *
      std::cos(phiMode * phi);
  const double psi = 1.0 + point->value;
  const double psi7 = std::pow(psi, 7.0);
  const double a2 = -8.0 * analyticLap * psi7;
  return point->d11 + point->d22 + point->d33 + 0.125 * a2 / psi7;
}

struct LayoutVisitContext {
  std::size_t n1 = 0;
  std::size_t n2 = 0;
  std::vector<int> *visits = nullptr;
};

double layoutVisitKernel(const tensorium_spectral_residual_point *point,
                         const double *, std::int64_t, void *userData) {
  auto *context = static_cast<LayoutVisitContext *>(userData);
  const std::size_t expectedIndex =
      static_cast<std::size_t>(point->i) +
      context->n1 * (static_cast<std::size_t>(point->j) +
                     context->n2 * static_cast<std::size_t>(point->k));
  if (static_cast<std::size_t>(point->index) != expectedIndex)
    return 1.0;
  (*context->visits)[static_cast<std::size_t>(point->index)] += 1;
  return 0.0;
}

double coordinateMapProbeKernel(const tensorium_spectral_residual_point *point,
                                const double *params, std::int64_t paramCount,
                                void *) {
  if (paramCount != 3)
    throw std::runtime_error("coordinate map probe expects 3 params");
  return (point->physical[0] - point->logical[0] - params[0]) +
         10.0 * (point->physical[1] - point->logical[1] - params[1]) +
         100.0 * (point->physical[2] - point->logical[2] - params[2]);
}

void shiftedCoordinateMap(const double *logical, double *physical,
                          const double *params, std::int64_t paramCount,
                          void *) {
  if (paramCount != 3)
    throw std::runtime_error("shifted spectral coordinate map expects 3 params");
  physical[0] = logical[0] + params[0];
  physical[1] = logical[1] + params[1];
  physical[2] = logical[2] + params[2];
}

bool testGenericParameterizedResidual() {
  const double phiPeriod = 4.0 * kSpectralPi;
  GenericResidualParams params;
  params.alpha = -0.35;
  params.beta = 0.125;
  params.gamma = -0.75;
  params.phiOrigin = -kSpectralPi;
  params.phiAngularWave = 2.0 * kSpectralPi * 2.0 / phiPeriod;

  SpectralGrid3D grid(SpectralAxis::chebyshevZeros(9, -0.5, 1.25),
                      SpectralAxis::chebyshevZeros(8, -1.5, 0.75),
                      SpectralAxis::fourierPeriodic(16, phiPeriod,
                                                    params.phiOrigin));
  std::vector<double> values(grid.size(), 0.0);
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const double phi = grid.axis(2).points[k];
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      const double y = grid.axis(1).points[j];
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const double x = grid.axis(0).points[i];
        values[grid.index(i, j, k)] = genericField(x, y, phi, params);
      }
    }
  }

  const auto derivs = grid.derivatives(values);
  const double residualParams[] = {params.alpha, params.beta, params.gamma,
                                   params.phiOrigin, params.phiAngularWave};
  const SpectralResidualKernel kernel{"tensorium_spectral_generic_residual",
                                      &genericResidualKernel, nullptr};
  const auto residual = evaluateSpectralResidual(
      grid, derivs, kernel, std::span<const double>(residualParams, 5));

  const double error = maxAbs(residual);
  std::printf("[spectral-initial-data] generic residual max = %.17g\n", error);
  if (error > 4e-10) {
    std::fprintf(stderr, "generic spectral residual is not analytically zero\n");
    return false;
  }
  return true;
}

bool testHamiltonianToyResidual() {
  const double amplitude = 0.04;
  const double phiMode = 2.0;

  SpectralGrid3D grid(SpectralAxis::chebyshevZeros(8),
                      SpectralAxis::chebyshevZeros(7),
                      SpectralAxis::fourierPeriodic(14));
  std::vector<double> values(grid.size(), 0.0);
  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const double phi = grid.axis(2).points[k];
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      const double y = grid.axis(1).points[j];
      const double t2y = 2.0 * y * y - 1.0;
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const double x = grid.axis(0).points[i];
        const double t2x = 2.0 * x * x - 1.0;
        values[grid.index(i, j, k)] =
            amplitude * t2x * t2y * std::cos(phiMode * phi);
      }
    }
  }

  const auto derivs = grid.derivatives(values);
  const double params[] = {amplitude, phiMode};
  const SpectralResidualKernel kernel{
      "tensorium_spectral_hamiltonian_toy_residual",
      &hamiltonianToyResidualKernel, nullptr};
  const auto residual = evaluateSpectralResidual(
      grid, derivs, kernel, std::span<const double>(params, 2));

  const double error = maxAbs(residual);
  std::printf("[spectral-initial-data] Hamiltonian toy max = %.17g\n", error);
  if (error > 5e-11) {
    std::fprintf(stderr,
                 "Hamiltonian toy spectral residual is not analytically zero\n");
    return false;
  }
  return true;
}

bool testPointwiseLayoutVisit() {
  SpectralGrid3D grid(SpectralAxis::chebyshevZeros(4),
                      SpectralAxis::chebyshevZeros(3),
                      SpectralAxis::fourierPeriodic(5));
  std::vector<double> values(grid.size(), 1.0);
  const auto derivs = grid.derivatives(values);
  std::vector<int> visits(grid.size(), 0);
  LayoutVisitContext context{grid.n1(), grid.n2(), &visits};
  const SpectralResidualKernel kernel{"tensorium_spectral_layout_visit",
                                      &layoutVisitKernel, &context};

  const auto residual =
      evaluateSpectralResidual(grid, derivs, kernel, std::span<const double>());

  if (maxAbs(residual) != 0.0) {
    std::fprintf(stderr, "spectral pointwise callback index layout mismatch\n");
    return false;
  }
  for (int visitCount : visits) {
    if (visitCount != 1) {
      std::fprintf(stderr, "spectral pointwise callback visit mismatch\n");
      return false;
    }
  }
  std::printf("[spectral-initial-data] pointwise layout visits = %zu\n",
              visits.size());
  return true;
}

bool testCoordinateMapFeedsPhysicalCoordinates() {
  SpectralGrid3D grid(SpectralAxis::chebyshevZeros(7, -1.0, 1.0),
                      SpectralAxis::chebyshevZeros(6, -1.0, 1.0),
                      SpectralAxis::fourierPeriodic(10));
  const double shifts[] = {0.25, -0.125, 0.5};
  std::vector<double> values(grid.size(), 1.0);
  const SpectralResidualKernel kernel{"tensorium_spectral_coordinate_map_probe",
                                      &coordinateMapProbeKernel, nullptr};
  const SpectralCoordinateMap map{"tensorium_spectral_shifted_map",
                                  &shiftedCoordinateMap, nullptr};
  const auto residual = evaluateSpectralResidual(
      grid, grid.derivatives(values), kernel, std::span<const double>(shifts, 3),
      map,
      std::span<const double>(shifts, 3));

  double maxValue = 0.0;
  for (double value : residual)
    maxValue = std::max(maxValue, std::abs(value));

  if (maxValue > 1e-14) {
    std::fprintf(stderr,
                 "spectral coordinate map did not feed physical coordinates\n");
    return false;
  }
  std::printf("[spectral-initial-data] shifted map probe max = %.17g\n",
              maxValue);
  return true;
}

} // namespace

int main() {
  try {
    if (!testGenericParameterizedResidual())
      return 3;
    if (!testHamiltonianToyResidual())
      return 3;
    if (!testPointwiseLayoutVisit())
      return 3;
    if (!testCoordinateMapFeedsPhysicalCoordinates())
      return 3;
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "spectral initial-data runtime test failed: %s\n",
                 ex.what());
    return 2;
  }

  return 0;
}
