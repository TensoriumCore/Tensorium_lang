#pragma once

#include "tensorium_mlir/Runtime/SpectralGrid.h"
#include "tensorium_mlir/Runtime/TwoPunctureDiagnostics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tensorium_mlir::runtime {

struct TwoPunctureRegularFieldSample {
  std::array<double, 2> values{};
  std::array<double, 2> maxPhiVariation{};
};

// Interpolate the physical regular correction
// u = scale * (A - boundary) * v at the puncture ends. The two entries are
// ordered as x=+b (B=-1) and x=-b (B=+1). Averaging the endpoint trace over
// phi isolates the regular m=0 value; maxPhiVariation exposes unresolved axis
// behavior instead of silently discarding it.
inline TwoPunctureRegularFieldSample sampleTwoPunctureRegularField(
    const SpectralGrid3D &grid, const std::vector<double> &solverField,
    double boundary = 1.0, double scale = 1.0) {
  if (solverField.size() != grid.size() || !std::isfinite(boundary) ||
      !std::isfinite(scale)) {
    throw std::runtime_error("invalid two-puncture regular-field sample input");
  }

  TwoPunctureRegularFieldSample out;
  std::array<std::vector<double>, 2> phiValues{
      std::vector<double>(grid.n3(), 0.0),
      std::vector<double>(grid.n3(), 0.0)};
  const double punctureFactor = scale * (-1.0 - boundary);
  for (std::size_t puncture = 0; puncture < 2; ++puncture) {
    const double B = puncture == 0 ? -1.0 : 1.0;
    for (std::size_t k = 0; k < grid.n3(); ++k) {
      phiValues[puncture][k] =
          punctureFactor *
          grid.interpolate(solverField, -1.0, B, grid.axis(2).points[k]);
      out.values[puncture] += phiValues[puncture][k];
    }
    out.values[puncture] /= static_cast<double>(grid.n3());
    for (double value : phiValues[puncture]) {
      out.maxPhiVariation[puncture] =
          std::max(out.maxPhiVariation[puncture],
                   std::fabs(value - out.values[puncture]));
    }
  }
  return out;
}

enum class TwoPunctureMassCalibrationStatus {
  MaxIterations,
  Converged,
  InvalidInput,
  BackendFailed,
  InvalidMassUpdate,
};

struct TwoPunctureMassCalibrationOptions {
  int maxIterations = 8;
  double absoluteTolerance = 1.0e-10;
  double relativeTolerance = 1.0e-10;
  double updateDamping = 1.0;
};

struct TwoPunctureMassCalibrationResult {
  TwoPunctureMassCalibrationStatus status =
      TwoPunctureMassCalibrationStatus::MaxIterations;
  int iterations = 0;
  std::array<double, 2> targetMasses{};
  std::array<double, 2> bareMasses{};
  std::array<double, 2> regularFieldAtPunctures{};
  std::array<double, 2> localAdmMasses{};
  double maxMassError = std::numeric_limits<double>::infinity();

  bool converged() const {
    return status == TwoPunctureMassCalibrationStatus::Converged;
  }
};

// Invert the two local ADM mass equations while holding u_+ and u_- fixed.
// Re-solving the elliptic backend after each update accounts for their bare-
// mass dependence.
inline std::array<double, 2> updateTwoPunctureBareMasses(
    double halfSeparation, const std::array<double, 2> &targetMasses,
    const std::array<double, 2> &regularFieldAtPunctures) {
  if (!std::isfinite(halfSeparation) || halfSeparation <= 0.0 ||
      !std::isfinite(targetMasses[0]) || targetMasses[0] <= 0.0 ||
      !std::isfinite(targetMasses[1]) || targetMasses[1] <= 0.0 ||
      !std::isfinite(regularFieldAtPunctures[0]) ||
      !std::isfinite(regularFieldAtPunctures[1])) {
    throw std::runtime_error("invalid two-puncture bare-mass update input");
  }

  const double plusFactor = 1.0 + regularFieldAtPunctures[0];
  const double minusFactor = 1.0 + regularFieldAtPunctures[1];
  if (!(plusFactor > 0.0) || !(minusFactor > 0.0))
    throw std::runtime_error("two-puncture regular field gives invalid mass");

  const double coupledFactor = 4.0 * halfSeparation * plusFactor * minusFactor;
  const double difference =
      -targetMasses[1] + targetMasses[0] + coupledFactor;
  const double radicand =
      16.0 * halfSeparation * targetMasses[1] * plusFactor * minusFactor +
      difference * difference;
  if (!std::isfinite(radicand) || radicand < 0.0)
    throw std::runtime_error("two-puncture bare-mass update is not real");

  const double shared = -coupledFactor + std::sqrt(radicand);
  const std::array<double, 2> updated = {
      (shared + targetMasses[0] - targetMasses[1]) / (2.0 * plusFactor),
      (shared - targetMasses[0] + targetMasses[1]) / (2.0 * minusFactor)};
  if (!std::isfinite(updated[0]) || updated[0] <= 0.0 ||
      !std::isfinite(updated[1]) || updated[1] <= 0.0) {
    throw std::runtime_error("two-puncture bare-mass update is invalid");
  }
  return updated;
}

template <typename EvaluateRegularFieldFn>
inline TwoPunctureMassCalibrationResult calibrateTwoPunctureBareMasses(
    double halfSeparation, const std::array<double, 2> &targetMasses,
    const std::array<double, 2> &initialBareMasses,
    EvaluateRegularFieldFn &&evaluateRegularField,
    const TwoPunctureMassCalibrationOptions &options = {}) {
  TwoPunctureMassCalibrationResult result;
  result.targetMasses = targetMasses;
  result.bareMasses = initialBareMasses;
  if (!std::isfinite(halfSeparation) || halfSeparation <= 0.0 ||
      !std::isfinite(targetMasses[0]) || targetMasses[0] <= 0.0 ||
      !std::isfinite(targetMasses[1]) || targetMasses[1] <= 0.0 ||
      !std::isfinite(initialBareMasses[0]) || initialBareMasses[0] <= 0.0 ||
      !std::isfinite(initialBareMasses[1]) || initialBareMasses[1] <= 0.0 ||
      options.maxIterations <= 0 || options.absoluteTolerance < 0.0 ||
      options.relativeTolerance < 0.0 ||
      !std::isfinite(options.absoluteTolerance) ||
      !std::isfinite(options.relativeTolerance) ||
      !std::isfinite(options.updateDamping) ||
      !(options.updateDamping > 0.0 && options.updateDamping <= 1.0)) {
    result.status = TwoPunctureMassCalibrationStatus::InvalidInput;
    return result;
  }

  for (int iteration = 1; iteration <= options.maxIterations; ++iteration) {
    std::array<double, 2> regularField{};
    if (!evaluateRegularField(result.bareMasses, regularField)) {
      result.status = TwoPunctureMassCalibrationStatus::BackendFailed;
      return result;
    }
    if (!std::isfinite(regularField[0]) ||
        !std::isfinite(regularField[1])) {
      result.status = TwoPunctureMassCalibrationStatus::InvalidMassUpdate;
      return result;
    }
    TwoPunctureLocalMassDiagnostics local;
    try {
      local = makeTwoPunctureLocalMassDiagnostics(
          halfSeparation, result.bareMasses[0], result.bareMasses[1],
          regularField[0], regularField[1]);
    } catch (const std::runtime_error &) {
      result.status = TwoPunctureMassCalibrationStatus::InvalidMassUpdate;
      return result;
    }
    result.iterations = iteration;
    result.regularFieldAtPunctures = regularField;
    result.localAdmMasses = local.admMasses;
    result.maxMassError = 0.0;
    bool withinTolerance = true;
    for (std::size_t puncture = 0; puncture < 2; ++puncture) {
      const double error =
          std::fabs(result.localAdmMasses[puncture] - targetMasses[puncture]);
      const double tolerance =
          std::max(options.absoluteTolerance,
                   options.relativeTolerance * targetMasses[puncture]);
      result.maxMassError = std::max(result.maxMassError, error);
      withinTolerance = withinTolerance && error <= tolerance;
    }
    if (withinTolerance) {
      result.status = TwoPunctureMassCalibrationStatus::Converged;
      return result;
    }
    if (iteration == options.maxIterations)
      break;

    std::array<double, 2> updated{};
    try {
      updated = updateTwoPunctureBareMasses(
          halfSeparation, targetMasses, result.regularFieldAtPunctures);
    } catch (const std::runtime_error &) {
      result.status = TwoPunctureMassCalibrationStatus::InvalidMassUpdate;
      return result;
    }
    for (std::size_t puncture = 0; puncture < 2; ++puncture) {
      result.bareMasses[puncture] +=
          options.updateDamping *
          (updated[puncture] - result.bareMasses[puncture]);
    }
  }
  result.status = TwoPunctureMassCalibrationStatus::MaxIterations;
  return result;
}

} // namespace tensorium_mlir::runtime
