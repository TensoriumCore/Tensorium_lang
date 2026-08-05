#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualTypes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tensorium_mlir::runtime {

struct TwoPuncturePhysicalParameters {
  double halfSeparation = 0.0;
  std::array<double, 2> bareMasses{};
  std::array<std::array<double, 3>, 2> momenta{};
  std::array<std::array<double, 3>, 2> spins{};
};

struct TwoPunctureLogicalPoint {
  double A = 0.0;
  double B = 0.0;
  double phi = 0.0;
  double distancePlus = 0.0;
  double distanceMinus = 0.0;
};

struct TwoPunctureBssnPoint {
  TwoPunctureLogicalPoint logical;
  double regularCorrection = 0.0;
  double conformalFactor = 1.0;
  double chi = 1.0;
  std::array<double, 9> conformalMetric{};
  std::array<double, 9> inverseConformalMetric{};
  std::array<double, 9> traceFreeExtrinsicCurvature{};
  std::array<double, 3> conformalConnection{};
  double meanCurvature = 0.0;
};

struct TwoPunctureCartesianGridView {
  std::size_t pointCount = 0;
  std::array<const double *, 3> coordinates{};
};

struct TwoPunctureBssnGridBuffers {
  double *chi = nullptr;
  std::array<double *, 9> conformalMetric{};
  std::array<double *, 9> inverseConformalMetric{};
  std::array<double *, 9> traceFreeExtrinsicCurvature{};
  double *meanCurvature = nullptr;
  std::array<double *, 3> conformalConnection{};

  // Optional diagnostic/interchange fields.
  double *regularCorrection = nullptr;
  double *conformalFactor = nullptr;

  // Optional gauge fields. The constraints do not determine these values.
  double *lapse = nullptr;
  std::array<double *, 3> shift{};
};

struct TwoPunctureGaugeSeed {
  double lapse = 1.0;
  std::array<double, 3> shift{};
};

inline void requireTwoPuncturePhysicalParameters(
    const TwoPuncturePhysicalParameters &parameters) {
  if (!(parameters.halfSeparation > 0.0) ||
      !std::isfinite(parameters.halfSeparation)) {
    throw std::runtime_error(
        "two-puncture handoff requires a finite positive half-separation");
  }
  for (double mass : parameters.bareMasses) {
    if (!(mass > 0.0) || !std::isfinite(mass))
      throw std::runtime_error(
          "two-puncture handoff requires finite positive bare masses");
  }
  for (std::size_t puncture = 0; puncture < 2; ++puncture) {
    for (std::size_t component = 0; component < 3; ++component) {
      if (!std::isfinite(parameters.momenta[puncture][component]) ||
          !std::isfinite(parameters.spins[puncture][component])) {
        throw std::runtime_error("two-puncture handoff vectors must be finite");
      }
    }
  }
}

inline TwoPunctureLogicalPoint
invertTwoPunctureCoordinates(double x, double y, double z,
                             double halfSeparation) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
      !(halfSeparation > 0.0) || !std::isfinite(halfSeparation)) {
    throw std::runtime_error("invalid two-puncture inverse-map input");
  }

  TwoPunctureLogicalPoint out;
  out.distancePlus = std::hypot(x - halfSeparation, y, z);
  out.distanceMinus = std::hypot(x + halfSeparation, y, z);
  if (!std::isfinite(out.distancePlus) || !std::isfinite(out.distanceMinus)) {
    throw std::runtime_error("two-puncture inverse-map distance overflowed");
  }

  const double coshX = std::max(1.0, (out.distancePlus + out.distanceMinus) /
                                         (2.0 * halfSeparation));
  const double cosR = std::clamp((out.distanceMinus - out.distancePlus) /
                                     (2.0 * halfSeparation),
                                 -1.0, 1.0);
  const double X = std::acosh(coshX);
  const double R = std::acos(cosR);
  out.A = 2.0 * std::tanh(0.5 * X) - 1.0;
  out.B = std::tan(0.5 * (R - 0.5 * std::acos(-1.0)));
  out.A = std::clamp(out.A, -1.0, 1.0);
  out.B = std::clamp(out.B, -1.0, 1.0);
  out.phi = std::atan2(z, y);
  if (out.phi < 0.0)
    out.phi += 2.0 * std::acos(-1.0);
  return out;
}

inline std::array<double, 9> evaluateTwoPunctureBowenYorkTensor(
    double x, double y, double z,
    const TwoPuncturePhysicalParameters &parameters) {
  requireTwoPuncturePhysicalParameters(parameters);
  const std::array<double, 3> point = {x, y, z};
  std::array<double, 9> tensor{};
  for (std::size_t puncture = 0; puncture < 2; ++puncture) {
    const double center =
        puncture == 0 ? parameters.halfSeparation : -parameters.halfSeparation;
    const std::array<double, 3> displacement = {point[0] - center, point[1],
                                                point[2]};
    const double radius =
        std::hypot(displacement[0], displacement[1], displacement[2]);
    if (!(radius > 0.0) || !std::isfinite(radius))
      throw std::runtime_error("Bowen-York tensor is singular at a puncture");
    std::array<double, 3> normal{};
    for (std::size_t i = 0; i < 3; ++i)
      normal[i] = displacement[i] / radius;
    const auto &momentum = parameters.momenta[puncture];
    const auto &spin = parameters.spins[puncture];
    double momentumNormal = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
      momentumNormal += momentum[i] * normal[i];
    const std::array<double, 3> spinCrossNormal = {
        spin[1] * normal[2] - spin[2] * normal[1],
        spin[2] * normal[0] - spin[0] * normal[2],
        spin[0] * normal[1] - spin[1] * normal[0]};
    const double momentumFactor = 1.5 / (radius * radius);
    const double spinFactor = momentumFactor * 2.0 / radius;
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        const double delta = i == j ? 1.0 : 0.0;
        tensor[3 * i + j] +=
            momentumFactor *
                (momentum[i] * normal[j] + momentum[j] * normal[i] -
                 (delta - normal[i] * normal[j]) * momentumNormal) +
            spinFactor * (spinCrossNormal[i] * normal[j] +
                          spinCrossNormal[j] * normal[i]);
      }
    }
  }
  return tensor;
}

inline double interpolateTwoPunctureRegularCorrection(
    const SpectralResidualProblem &problem,
    const std::vector<double> &solverField,
    const TwoPunctureLogicalPoint &logical) {
  if (!problem.grid || solverField.size() != problem.grid->size())
    throw std::runtime_error("two-puncture handoff solver-field size mismatch");
  SpectralPointDerivatives3D solverBundle;
  solverBundle.value =
      problem.grid->interpolate(solverField, logical.A, logical.B, logical.phi);
  SpectralPointDerivatives3D physicalBundle = solverBundle;
  if (problem.unknownMap.transform) {
    const double coordinates[3] = {logical.A, logical.B, logical.phi};
    problem.unknownMap.transform(
        coordinates, &solverBundle, &physicalBundle,
        problem.unknownMapParams.data(),
        static_cast<std::int64_t>(problem.unknownMapParams.size()),
        problem.unknownMap.userData);
  }
  if (!std::isfinite(physicalBundle.value))
    throw std::runtime_error(
        "two-puncture handoff interpolated a non-finite correction");
  return physicalBundle.value;
}

inline TwoPunctureBssnPoint
evaluateTwoPunctureBssnPoint(const SpectralResidualProblem &problem,
                             const std::vector<double> &solverField,
                             const TwoPuncturePhysicalParameters &parameters,
                             double x, double y, double z) {
  requireTwoPuncturePhysicalParameters(parameters);
  if (problem.coordinateParams.empty() ||
      std::abs(problem.coordinateParams.front() - parameters.halfSeparation) >
          16.0 * std::numeric_limits<double>::epsilon() *
              std::max({1.0, std::abs(problem.coordinateParams.front()),
                        parameters.halfSeparation})) {
    throw std::runtime_error(
        "two-puncture handoff half-separation does not match the spectral "
        "coordinate map");
  }
  TwoPunctureBssnPoint out;
  out.logical =
      invertTwoPunctureCoordinates(x, y, z, parameters.halfSeparation);
  out.regularCorrection = interpolateTwoPunctureRegularCorrection(
      problem, solverField, out.logical);

  const bool atPuncture =
      out.logical.distancePlus == 0.0 || out.logical.distanceMinus == 0.0;
  if (atPuncture) {
    out.conformalFactor = std::numeric_limits<double>::infinity();
    out.chi = 0.0;
  } else {
    out.conformalFactor =
        1.0 + 0.5 * parameters.bareMasses[0] / out.logical.distancePlus +
        0.5 * parameters.bareMasses[1] / out.logical.distanceMinus +
        out.regularCorrection;
    if (!(out.conformalFactor > 0.0) || !std::isfinite(out.conformalFactor)) {
      throw std::runtime_error(
          "two-puncture handoff produced an invalid conformal factor");
    }
    const double inversePsi = 1.0 / out.conformalFactor;
    const double inversePsi2 = inversePsi * inversePsi;
    out.chi = inversePsi2 * inversePsi2;
    const double inversePsi6 = out.chi * inversePsi2;
    const auto bowenYork =
        evaluateTwoPunctureBowenYorkTensor(x, y, z, parameters);
    for (std::size_t component = 0; component < 9; ++component)
      out.traceFreeExtrinsicCurvature[component] =
          inversePsi6 * bowenYork[component];
  }

  for (std::size_t i = 0; i < 3; ++i) {
    out.conformalMetric[3 * i + i] = 1.0;
    out.inverseConformalMetric[3 * i + i] = 1.0;
  }
  return out;
}

inline void interpolateTwoPunctureBssnToCartesianGrid(
    const SpectralResidualProblem &problem,
    const std::vector<double> &solverField,
    const TwoPuncturePhysicalParameters &parameters,
    const TwoPunctureCartesianGridView &target,
    const TwoPunctureBssnGridBuffers &outputs,
    const TwoPunctureGaugeSeed &gauge = {}) {
  if (target.pointCount == 0)
    throw std::runtime_error(
        "two-puncture handoff target grid must not be empty");
  for (const double *coordinate : target.coordinates) {
    if (!coordinate)
      throw std::runtime_error(
          "two-puncture handoff target has a null coordinate");
  }
  if (!outputs.chi || !outputs.meanCurvature)
    throw std::runtime_error(
        "two-puncture BSSN handoff has a null scalar output");
  for (std::size_t component = 0; component < 9; ++component) {
    if (!outputs.conformalMetric[component] ||
        !outputs.inverseConformalMetric[component] ||
        !outputs.traceFreeExtrinsicCurvature[component]) {
      throw std::runtime_error(
          "two-puncture BSSN handoff has a null tensor output");
    }
  }
  const auto requireAllOrNone = [](const auto &pointers,
                                   const char *description) {
    const bool any =
        std::any_of(pointers.begin(), pointers.end(),
                    [](const double *value) { return value != nullptr; });
    const bool all =
        std::all_of(pointers.begin(), pointers.end(),
                    [](const double *value) { return value != nullptr; });
    if (any && !all)
      throw std::runtime_error(description);
    return all;
  };
  const bool writeConnection = requireAllOrNone(
      outputs.conformalConnection,
      "two-puncture BSSN conformal connection requires all components");
  const bool writeShift = requireAllOrNone(
      outputs.shift, "two-puncture BSSN shift output requires all components");
  if (!std::isfinite(gauge.lapse))
    throw std::runtime_error("two-puncture BSSN lapse seed must be finite");
  for (double shift : gauge.shift) {
    if (!std::isfinite(shift))
      throw std::runtime_error("two-puncture BSSN shift seed must be finite");
  }

  for (std::size_t point = 0; point < target.pointCount; ++point) {
    const auto value = evaluateTwoPunctureBssnPoint(
        problem, solverField, parameters, target.coordinates[0][point],
        target.coordinates[1][point], target.coordinates[2][point]);
    outputs.chi[point] = value.chi;
    outputs.meanCurvature[point] = value.meanCurvature;
    if (outputs.regularCorrection)
      outputs.regularCorrection[point] = value.regularCorrection;
    if (outputs.conformalFactor)
      outputs.conformalFactor[point] = value.conformalFactor;
    for (std::size_t component = 0; component < 9; ++component) {
      outputs.conformalMetric[component][point] =
          value.conformalMetric[component];
      outputs.inverseConformalMetric[component][point] =
          value.inverseConformalMetric[component];
      outputs.traceFreeExtrinsicCurvature[component][point] =
          value.traceFreeExtrinsicCurvature[component];
    }
    if (writeConnection) {
      for (std::size_t component = 0; component < 3; ++component)
        outputs.conformalConnection[component][point] =
            value.conformalConnection[component];
    }
    if (outputs.lapse)
      outputs.lapse[point] = gauge.lapse;
    if (writeShift) {
      for (std::size_t component = 0; component < 3; ++component)
        outputs.shift[component][point] = gauge.shift[component];
    }
  }
}

} // namespace tensorium_mlir::runtime
