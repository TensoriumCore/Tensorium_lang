#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace tensorium_mlir::runtime {

struct TwoPunctureAdmDiagnostics {
  double energy = 0.0;
  std::array<double, 3> linearMomentum{};
  std::array<double, 3> angularMomentum{};
};

struct TwoPunctureLocalMassDiagnostics {
  std::array<double, 2> regularFieldAtPunctures{};
  std::array<double, 2> admMasses{};
};

inline std::array<double, 3>
twoPunctureCrossProduct(const std::array<double, 3> &lhs,
                        const std::array<double, 3> &rhs) {
  return {lhs[1] * rhs[2] - lhs[2] * rhs[1], lhs[2] * rhs[0] - lhs[0] * rhs[2],
          lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

inline TwoPunctureAdmDiagnostics makeTwoPunctureAdmDiagnostics(
    double halfSeparation, double bareMass1, double bareMass2,
    double regularFieldAtInfinity, const std::array<double, 3> &momentum1,
    const std::array<double, 3> &momentum2, const std::array<double, 3> &spin1,
    const std::array<double, 3> &spin2) {
  if (!std::isfinite(halfSeparation) || halfSeparation <= 0.0 ||
      !std::isfinite(bareMass1) || bareMass1 <= 0.0 ||
      !std::isfinite(bareMass2) || bareMass2 <= 0.0 ||
      !std::isfinite(regularFieldAtInfinity)) {
    throw std::runtime_error("invalid two-puncture ADM diagnostic input");
  }
  for (std::size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(momentum1[i]) || !std::isfinite(momentum2[i]) ||
        !std::isfinite(spin1[i]) || !std::isfinite(spin2[i])) {
      throw std::runtime_error("invalid two-puncture ADM vector input");
    }
  }

  TwoPunctureAdmDiagnostics out;
  out.energy =
      bareMass1 + bareMass2 - 4.0 * halfSeparation * regularFieldAtInfinity;
  const std::array<double, 3> center1 = {halfSeparation, 0.0, 0.0};
  const std::array<double, 3> center2 = {-halfSeparation, 0.0, 0.0};
  const auto orbital1 = twoPunctureCrossProduct(center1, momentum1);
  const auto orbital2 = twoPunctureCrossProduct(center2, momentum2);
  for (std::size_t i = 0; i < 3; ++i) {
    out.linearMomentum[i] = momentum1[i] + momentum2[i];
    out.angularMomentum[i] = spin1[i] + spin2[i] + orbital1[i] + orbital2[i];
  }
  if (!std::isfinite(out.energy))
    throw std::runtime_error("two-puncture ADM energy overflowed");
  for (std::size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(out.linearMomentum[i]) ||
        !std::isfinite(out.angularMomentum[i])) {
      throw std::runtime_error("two-puncture ADM vector overflowed");
    }
  }
  return out;
}

inline TwoPunctureLocalMassDiagnostics makeTwoPunctureLocalMassDiagnostics(
    double halfSeparation, double bareMassPlus, double bareMassMinus,
    double regularFieldPlus, double regularFieldMinus) {
  if (!std::isfinite(halfSeparation) || halfSeparation <= 0.0 ||
      !std::isfinite(bareMassPlus) || bareMassPlus <= 0.0 ||
      !std::isfinite(bareMassMinus) || bareMassMinus <= 0.0 ||
      !std::isfinite(regularFieldPlus) ||
      !std::isfinite(regularFieldMinus)) {
    throw std::runtime_error("invalid two-puncture local-mass input");
  }

  const double interaction =
      bareMassPlus * bareMassMinus / (4.0 * halfSeparation);
  TwoPunctureLocalMassDiagnostics out;
  out.regularFieldAtPunctures = {regularFieldPlus, regularFieldMinus};
  out.admMasses = {
      (1.0 + regularFieldPlus) * bareMassPlus + interaction,
      (1.0 + regularFieldMinus) * bareMassMinus + interaction};
  if (!std::isfinite(out.admMasses[0]) ||
      !std::isfinite(out.admMasses[1])) {
    throw std::runtime_error("two-puncture local ADM mass overflowed");
  }
  return out;
}

} // namespace tensorium_mlir::runtime
