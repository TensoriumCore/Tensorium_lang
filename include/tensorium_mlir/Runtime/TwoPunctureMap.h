#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualTypes.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tensorium_mlir::runtime {

struct TwoPunctureCoordinates {
  double x = 0.0;
  double rho = 0.0;
  double y = 0.0;
  double z = 0.0;
  double X = 0.0;
  double R = 0.0;
};

inline double requireTwoPunctureHalfSeparation(const double *params,
                                               std::int64_t paramCount) {
  if (!params || paramCount != 1 || !std::isfinite(params[0]) ||
      params[0] <= 0.0) {
    throw std::runtime_error(
        "two-puncture map requires one positive finite half-separation");
  }
  return params[0];
}

inline TwoPunctureCoordinates mapTwoPunctureCoordinates(double A, double B,
                                                        double phi, double b) {
  if (!std::isfinite(A) || !std::isfinite(B) || !std::isfinite(phi) ||
      !std::isfinite(b) || b <= 0.0 || A <= -1.0 || A >= 1.0) {
    throw std::runtime_error("invalid two-puncture logical coordinate");
  }

  const double compactA = 0.5 * (A + 1.0);
  const double X = 2.0 * std::atanh(compactA);
  const double R = 0.5 * kSpectralPi + 2.0 * std::atan(B);
  // Evaluate sin(R) and cos(R) from B directly. Besides avoiding two
  // transcendental calls, these rational forms preserve x(A,-B)=-x(A,B)
  // and rho(A,-B)=rho(A,B) exactly in floating-point arithmetic.
  const double bDenominator = 1.0 + B * B;
  const double cosR = -2.0 * B / bDenominator;
  const double sinR = (1.0 - B * B) / bDenominator;
  const double x = b * std::cosh(X) * cosR;
  const double rho = b * std::sinh(X) * sinR;
  return TwoPunctureCoordinates{
      x, rho, rho * std::cos(phi), rho * std::sin(phi), X, R};
}

inline void twoPunctureCoordinateMap(const double *logical, double *physical,
                                     const double *params,
                                     std::int64_t paramCount, void *) {
  if (!logical || !physical)
    throw std::runtime_error("two-puncture coordinate map received null data");
  const double b = requireTwoPunctureHalfSeparation(params, paramCount);
  const TwoPunctureCoordinates point =
      mapTwoPunctureCoordinates(logical[0], logical[1], logical[2], b);
  physical[0] = point.x;
  physical[1] = point.y;
  physical[2] = point.z;
}

inline void
twoPunctureDerivativeMap(const double logical[3],
                         const SpectralPointDerivatives3D *logicalDerivatives,
                         SpectralPointDerivatives3D *physicalDerivatives,
                         const double *params, std::int64_t paramCount,
                         void *) {
  if (!logical || !logicalDerivatives || !physicalDerivatives) {
    throw std::runtime_error("two-puncture derivative map received null data");
  }
  const double b = requireTwoPunctureHalfSeparation(params, paramCount);
  const double A = logical[0];
  const double B = logical[1];
  const double phi = logical[2];
  const TwoPunctureCoordinates point = mapTwoPunctureCoordinates(A, B, phi, b);

  const double compactA = 0.5 * (A + 1.0);
  const double A_X = 1.0 - compactA * compactA;
  const double A_XX = -compactA * A_X;
  const double B_R = 0.5 * (1.0 + B * B);
  const double B_RR = B * B_R;

  SpectralPointDerivatives3D xr{};
  xr.value = logicalDerivatives->value;
  xr.d1 = A_X * logicalDerivatives->d1;
  xr.d2 = B_R * logicalDerivatives->d2;
  xr.d3 = logicalDerivatives->d3;
  xr.d11 = A_X * A_X * logicalDerivatives->d11 + A_XX * logicalDerivatives->d1;
  xr.d12 = A_X * B_R * logicalDerivatives->d12;
  xr.d13 = A_X * logicalDerivatives->d13;
  xr.d22 = B_R * B_R * logicalDerivatives->d22 + B_RR * logicalDerivatives->d2;
  xr.d23 = B_R * logicalDerivatives->d23;
  xr.d33 = logicalDerivatives->d33;

  using Complex = std::complex<double>;
  const double bDenominator = 1.0 + B * B;
  const double cosR = -2.0 * B / bDenominator;
  const double sinR = (1.0 - B * B) / bDenominator;
  const Complex c(point.x, point.rho);
  const Complex c_C(b * std::sinh(point.X) * cosR,
                    b * std::cosh(point.X) * sinR);
  const double scale2 = std::norm(c_C);
  const double minScale2 =
      64.0 * std::numeric_limits<double>::epsilon() * b * b;
  if (!std::isfinite(scale2) || scale2 <= minScale2 ||
      !std::isfinite(point.rho) ||
      std::abs(point.rho) <=
          64.0 * std::numeric_limits<double>::epsilon() * b) {
    throw std::runtime_error(
        "two-puncture derivative map is singular on the puncture or axis");
  }

  const Complex C_c = 1.0 / c_C;
  const Complex C_cc = -C_c * C_c * C_c * c;
  const Complex U_C(0.5 * xr.d1, -0.5 * xr.d2);
  const Complex U_CC(0.25 * (xr.d11 - xr.d22), -0.5 * xr.d12);
  const double U_CB = 0.25 * (xr.d11 + xr.d22);
  const Complex U_c = U_C * C_c;
  const Complex U_cc = C_cc * U_C + C_c * C_c * U_CC;
  const double U_cb = U_CB * std::norm(C_c);
  const Complex U_Cphi(0.5 * xr.d13, -0.5 * xr.d23);
  const Complex U_cphi = U_Cphi * C_c;

  const double u_x = 2.0 * U_c.real();
  const double u_rho = -2.0 * U_c.imag();
  const double u_phi = xr.d3;
  const double u_xx = 2.0 * (U_cb + U_cc.real());
  const double u_xrho = -2.0 * U_cc.imag();
  const double u_xphi = 2.0 * U_cphi.real();
  const double u_rhorho = 2.0 * (U_cb - U_cc.real());
  const double u_rhophi = -2.0 * U_cphi.imag();
  const double u_phiphi = xr.d33;

  const double sinPhi = std::sin(phi);
  const double cosPhi = std::cos(phi);
  const double sinPhi2 = sinPhi * sinPhi;
  const double cosPhi2 = cosPhi * cosPhi;
  const double sin2Phi = 2.0 * sinPhi * cosPhi;
  const double cos2Phi = cosPhi2 - sinPhi2;
  const double invRho = 1.0 / point.rho;
  const double invRho2 = invRho * invRho;
  const double angularTerm = u_phi - point.rho * u_rhophi;

  physicalDerivatives->value = xr.value;
  physicalDerivatives->d1 = u_x;
  physicalDerivatives->d2 = u_rho * cosPhi - u_phi * invRho * sinPhi;
  physicalDerivatives->d3 = u_rho * sinPhi + u_phi * invRho * cosPhi;
  physicalDerivatives->d11 = u_xx;
  physicalDerivatives->d12 = u_xrho * cosPhi - u_xphi * invRho * sinPhi;
  physicalDerivatives->d13 = u_xrho * sinPhi + u_xphi * invRho * cosPhi;
  physicalDerivatives->d22 =
      u_rhorho * cosPhi2 + invRho2 * sinPhi2 * (u_phiphi + point.rho * u_rho) +
      sin2Phi * invRho2 * angularTerm;
  physicalDerivatives->d23 =
      0.5 * sin2Phi * (u_rhorho - invRho * u_rho - invRho2 * u_phiphi) -
      cos2Phi * invRho2 * angularTerm;
  physicalDerivatives->d33 =
      u_rhorho * sinPhi2 + invRho2 * cosPhi2 * (u_phiphi + point.rho * u_rho) -
      sin2Phi * invRho2 * angularTerm;
}

inline SpectralCoordinateMap makeTwoPunctureCoordinateMap() {
  return SpectralCoordinateMap{"tensorium_spectral_two_puncture_map",
                               &twoPunctureCoordinateMap, nullptr};
}

inline SpectralDerivativeMap makeTwoPunctureDerivativeMap() {
  return SpectralDerivativeMap{"tensorium_spectral_two_puncture_derivative_map",
                               &twoPunctureDerivativeMap, nullptr};
}

} // namespace tensorium_mlir::runtime
