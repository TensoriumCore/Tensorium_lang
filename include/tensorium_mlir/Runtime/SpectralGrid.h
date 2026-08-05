#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tensorium_mlir::runtime {

inline constexpr double kSpectralPi = 3.14159265358979323846264338327950288;
inline constexpr std::size_t kSpectralParallelPointThreshold = 32768;

enum class SpectralBasis {
  ChebyshevZeros,
  FourierPeriodic,
};

class SpectralGrid3D;

struct SpectralAxis {
  SpectralBasis basis = SpectralBasis::ChebyshevZeros;
  std::vector<double> points;
  double period = 0.0;

  static SpectralAxis chebyshevZeros(std::size_t n, double lower = -1.0,
                                     double upper = 1.0) {
    if (n == 0)
      throw std::runtime_error("spectral Chebyshev axis needs at least 1 point");
    if (!(upper > lower))
      throw std::runtime_error("spectral Chebyshev axis bounds are invalid");

    SpectralAxis axis;
    axis.basis = SpectralBasis::ChebyshevZeros;
    axis.points.resize(n);
    const double center = 0.5 * (lower + upper);
    const double scale = 0.5 * (upper - lower);
    for (std::size_t i = 0; i < n; ++i) {
      const double theta =
          kSpectralPi * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
      axis.points[i] = center + scale * std::cos(theta);
    }
    axis.prepareDerivativeOperators();
    return axis;
  }

  static SpectralAxis fourierPeriodic(std::size_t n,
                                      double period = 2.0 * kSpectralPi,
                                      double origin = 0.0) {
    if (n == 0)
      throw std::runtime_error("spectral Fourier axis needs at least 1 point");
    if (!(period > 0.0))
      throw std::runtime_error("spectral Fourier period must be positive");

    SpectralAxis axis;
    axis.basis = SpectralBasis::FourierPeriodic;
    axis.period = period;
    axis.points.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      axis.points[i] =
          origin + period * static_cast<double>(i) / static_cast<double>(n);
    axis.prepareDerivativeOperators();
    return axis;
  }

  std::size_t size() const { return points.size(); }

  std::vector<double> differentiate(const std::vector<double> &values,
                                    unsigned order) const {
    if (values.size() != points.size())
      throw std::runtime_error("spectral axis value count mismatch");
    std::vector<double> out(values.size(), 0.0);
    std::vector<double> scratch(order > 1 ? values.size() : 0, 0.0);
    std::vector<std::complex<double>> fourierScratch(
        basis == SpectralBasis::FourierPeriodic && order > 0 ? values.size()
                                                             : 0);
    differentiateInto(values, order, out, scratch, fourierScratch);
    return out;
  }

  void differentiateInto(std::span<const double> values, unsigned order,
                         std::span<double> out,
                         std::span<double> scratch = {},
                         std::span<std::complex<double>> fourierScratch = {})
      const {
    if (values.size() != points.size() || out.size() != points.size())
      throw std::runtime_error("spectral axis value count mismatch");
    if (order == 0) {
      std::copy(values.begin(), values.end(), out.begin());
      return;
    }

    prepareDerivativeOperators();
    if (basis == SpectralBasis::FourierPeriodic) {
      differentiateFourierInto(values, order, out, fourierScratch);
      return;
    }

    if (order > 1 && scratch.size() != points.size())
      throw std::runtime_error(
          "spectral derivative scratch buffer has the wrong size");
    std::span<const double> current = values;
    for (unsigned applied = 0; applied < order; ++applied) {
      const unsigned remaining = order - applied;
      std::span<double> target = remaining % 2 == 1 ? out : scratch;
      applyMatrix(polynomialDerivativeMatrix_, current, target);
      current = target;
    }
  }

  double interpolate(const std::vector<double> &values, double coordinate) const {
    if (values.size() != points.size())
      throw std::runtime_error("spectral axis value count mismatch");
    if (!std::isfinite(coordinate))
      throw std::runtime_error("spectral interpolation coordinate is not finite");
    if (basis == SpectralBasis::FourierPeriodic)
      return interpolateFourier(values, coordinate);

    const std::size_t n = points.size();
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double scale =
          std::max({1.0, std::abs(coordinate), std::abs(points[i])});
      if (std::abs(coordinate - points[i]) <=
          8.0 * std::numeric_limits<double>::epsilon() * scale) {
        return values[i];
      }
      const double theta =
          kSpectralPi * (static_cast<double>(i) + 0.5) /
          static_cast<double>(n);
      const double weight = (i % 2 == 0 ? 1.0 : -1.0) * std::sin(theta);
      const double term = weight / (coordinate - points[i]);
      numerator += term * values[i];
      denominator += term;
    }
    if (denominator == 0.0 || !std::isfinite(denominator))
      throw std::runtime_error("spectral interpolation denominator is invalid");
    return numerator / denominator;
  }

private:
  friend class SpectralGrid3D;

  // Axes are immutable after construction in the runtime. Keep the expensive
  // collocation operators with the axis and reuse them for every residual/JVP.
  mutable std::vector<double> polynomialDerivativeMatrix_;
  mutable std::vector<std::complex<double>> fourierForwardPhases_;
  mutable std::vector<std::complex<double>> fourierInversePhases_;
  mutable std::array<std::vector<std::complex<double>>, 2>
      fourierDerivativeFactors_;

  std::vector<double> polynomialDifferentiationMatrix() const {
    const std::size_t n = points.size();
    std::vector<double> weights(n, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (i != j)
          weights[i] /= points[i] - points[j];
      }
    }

    std::vector<double> matrix(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      double rowSum = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        if (i == j)
          continue;
        const double value = weights[j] / (weights[i] * (points[i] - points[j]));
        matrix[i * n + j] = value;
        rowSum += value;
      }
      matrix[i * n + i] = -rowSum;
    }
    return matrix;
  }

  static void applyMatrix(std::span<const double> matrix,
                          std::span<const double> values,
                          std::span<double> out) {
    const std::size_t n = values.size();
    if (matrix.size() != n * n || out.size() != n)
      throw std::runtime_error("spectral derivative matrix size mismatch");
    for (std::size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (std::size_t j = 0; j < n; ++j)
        sum += matrix[i * n + j] * values[j];
      out[i] = sum;
    }
  }

  void prepareDerivativeOperators() const {
    const std::size_t matrixSize = points.size() * points.size();
    if (basis == SpectralBasis::FourierPeriodic) {
      const std::size_t n = points.size();
      const std::complex<double> imaginary(0.0, 1.0);
      if (fourierForwardPhases_.size() != matrixSize ||
          fourierInversePhases_.size() != matrixSize) {
        fourierForwardPhases_.resize(matrixSize);
        fourierInversePhases_.resize(matrixSize);
        for (std::size_t m = 0; m < n; ++m) {
          for (std::size_t j = 0; j < n; ++j) {
            const double forwardPhase =
                -2.0 * kSpectralPi * static_cast<double>(m * j) /
                static_cast<double>(n);
            const double inversePhase =
                2.0 * kSpectralPi * static_cast<double>(m * j) /
                static_cast<double>(n);
            fourierForwardPhases_[m * n + j] =
                std::exp(imaginary * forwardPhase);
            fourierInversePhases_[m * n + j] =
                std::exp(imaginary * inversePhase);
          }
        }
      }
      for (unsigned order = 1; order <= fourierDerivativeFactors_.size();
           ++order) {
        auto &factors = fourierDerivativeFactors_[order - 1];
        if (factors.size() == n)
          continue;
        factors.resize(n);
        for (std::size_t m = 0; m < n; ++m) {
          const int waveNumber =
              m <= n / 2 ? static_cast<int>(m)
                         : static_cast<int>(m) - static_cast<int>(n);
          const double angularWave =
              2.0 * kSpectralPi * static_cast<double>(waveNumber) / period;
          std::complex<double> factor(1.0, 0.0);
          for (unsigned p = 0; p < order; ++p)
            factor *= imaginary * angularWave;
          factors[m] = factor;
        }
      }
      return;
    }
    if (polynomialDerivativeMatrix_.size() != matrixSize)
      polynomialDerivativeMatrix_ = polynomialDifferentiationMatrix();
  }

  void differentiateFourierInto(
      std::span<const double> values, unsigned order, std::span<double> out,
      std::span<std::complex<double>> coeffs) const {
    const std::size_t n = values.size();
    if (coeffs.size() != n)
      throw std::runtime_error(
          "spectral Fourier scratch buffer has the wrong size");
    const std::complex<double> imaginary(0.0, 1.0);

    for (std::size_t m = 0; m < n; ++m) {
      std::complex<double> coeff(0.0, 0.0);
      for (std::size_t j = 0; j < n; ++j)
        coeff += values[j] * fourierForwardPhases_[m * n + j];
      coeffs[m] = coeff / static_cast<double>(n);
    }

    for (std::size_t j = 0; j < n; ++j) {
      std::complex<double> value(0.0, 0.0);
      for (std::size_t m = 0; m < n; ++m) {
        std::complex<double> factor(1.0, 0.0);
        if (order <= fourierDerivativeFactors_.size()) {
          factor = fourierDerivativeFactors_[order - 1][m];
        } else {
          const int waveNumber =
              m <= n / 2 ? static_cast<int>(m)
                         : static_cast<int>(m) - static_cast<int>(n);
          const double angularWave =
              2.0 * kSpectralPi * static_cast<double>(waveNumber) / period;
          for (unsigned p = 0; p < order; ++p)
            factor *= imaginary * angularWave;
        }
        value += factor * coeffs[m] * fourierInversePhases_[m * n + j];
      }
      out[j] = value.real();
    }
  }

  double interpolateFourier(const std::vector<double> &values,
                            double coordinate) const {
    const std::size_t n = values.size();
    const std::complex<double> imaginary(0.0, 1.0);
    const double angle =
        2.0 * kSpectralPi * (coordinate - points.front()) / period;
    std::complex<double> result(0.0, 0.0);
    for (std::size_t m = 0; m < n; ++m) {
      std::complex<double> coefficient(0.0, 0.0);
      for (std::size_t j = 0; j < n; ++j) {
        const double phase =
            -2.0 * kSpectralPi * static_cast<double>(m * j) /
            static_cast<double>(n);
        coefficient += values[j] * std::exp(imaginary * phase);
      }
      coefficient /= static_cast<double>(n);
      const int waveNumber =
          m <= n / 2 ? static_cast<int>(m)
                     : static_cast<int>(m) - static_cast<int>(n);
      result += coefficient *
                std::exp(imaginary * static_cast<double>(waveNumber) * angle);
    }
    return result.real();
  }
};

struct SpectralDerivatives3D {
  std::vector<double> value;
  std::vector<double> d1;
  std::vector<double> d2;
  std::vector<double> d3;
  std::vector<double> d11;
  std::vector<double> d12;
  std::vector<double> d13;
  std::vector<double> d22;
  std::vector<double> d23;
  std::vector<double> d33;
};

struct SpectralPoint3D {
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t k = 0;
  std::size_t index = 0;
  double x1 = 0.0;
  double x2 = 0.0;
  double x3 = 0.0;
};

struct SpectralPointDerivatives3D {
  double value = 0.0;
  double d1 = 0.0;
  double d2 = 0.0;
  double d3 = 0.0;
  double d11 = 0.0;
  double d12 = 0.0;
  double d13 = 0.0;
  double d22 = 0.0;
  double d23 = 0.0;
  double d33 = 0.0;

  double laplacian() const { return d11 + d22 + d33; }
};

class SpectralGrid3D {
public:
  SpectralGrid3D(SpectralAxis a, SpectralAxis b, SpectralAxis phi)
      : axes_{std::move(a), std::move(b), std::move(phi)} {
    if (axes_[0].size() == 0 || axes_[1].size() == 0 || axes_[2].size() == 0)
      throw std::runtime_error("spectral grid axes must be non-empty");
    for (SpectralAxis &axis : axes_)
      axis.prepareDerivativeOperators();
  }

  const SpectralAxis &axis(std::size_t dim) const {
    if (dim >= 3)
      throw std::out_of_range("spectral grid axis index out of range");
    return axes_[dim];
  }

  std::size_t n1() const { return axes_[0].size(); }
  std::size_t n2() const { return axes_[1].size(); }
  std::size_t n3() const { return axes_[2].size(); }
  std::size_t size() const { return n1() * n2() * n3(); }

  std::size_t index(std::size_t i, std::size_t j, std::size_t k) const {
    return i + n1() * (j + n2() * k);
  }

  std::vector<double> derivative(const std::vector<double> &values,
                                 std::size_t dim, unsigned order) const {
    if (values.size() != size())
      throw std::runtime_error("spectral grid value count mismatch");
    if (dim >= 3)
      throw std::out_of_range("spectral grid derivative axis out of range");
    if (order == 0)
      return values;

    std::vector<double> out(values.size(), 0.0);
    if (dim == 0) {
#if defined(_OPENMP)
#pragma omp parallel if(values.size() >= kSpectralParallelPointThreshold)
#endif
      {
        std::vector<double> scratch(order > 1 ? n1() : 0, 0.0);
        std::vector<std::complex<double>> fourierScratch(
            axes_[0].basis == SpectralBasis::FourierPeriodic ? n1() : 0);
#if defined(_OPENMP)
#pragma omp for collapse(2) schedule(static)
#endif
        for (std::size_t k = 0; k < n3(); ++k) {
          for (std::size_t j = 0; j < n2(); ++j) {
            const std::size_t offset = index(0, j, k);
            axes_[0].differentiateInto(
                std::span<const double>(values.data() + offset, n1()), order,
                std::span<double>(out.data() + offset, n1()), scratch,
                fourierScratch);
          }
        }
      }
      return out;
    }

    if (dim == 1) {
#if defined(_OPENMP)
#pragma omp parallel if(values.size() >= kSpectralParallelPointThreshold)
#endif
      {
        std::vector<double> line(n2(), 0.0);
        std::vector<double> diff(n2(), 0.0);
        std::vector<double> scratch(order > 1 ? n2() : 0, 0.0);
        std::vector<std::complex<double>> fourierScratch(
            axes_[1].basis == SpectralBasis::FourierPeriodic ? n2() : 0);
#if defined(_OPENMP)
#pragma omp for collapse(2) schedule(static)
#endif
        for (std::size_t k = 0; k < n3(); ++k) {
          for (std::size_t i = 0; i < n1(); ++i) {
            for (std::size_t j = 0; j < n2(); ++j)
              line[j] = values[index(i, j, k)];
            axes_[1].differentiateInto(line, order, diff, scratch,
                                       fourierScratch);
            for (std::size_t j = 0; j < n2(); ++j)
              out[index(i, j, k)] = diff[j];
          }
        }
      }
      return out;
    }

#if defined(_OPENMP)
#pragma omp parallel if(values.size() >= kSpectralParallelPointThreshold)
#endif
    {
      std::vector<double> line(n3(), 0.0);
      std::vector<double> diff(n3(), 0.0);
      std::vector<double> scratch(order > 1 ? n3() : 0, 0.0);
      std::vector<std::complex<double>> fourierScratch(
          axes_[2].basis == SpectralBasis::FourierPeriodic ? n3() : 0);
#if defined(_OPENMP)
#pragma omp for collapse(2) schedule(static)
#endif
      for (std::size_t j = 0; j < n2(); ++j) {
        for (std::size_t i = 0; i < n1(); ++i) {
          for (std::size_t k = 0; k < n3(); ++k)
            line[k] = values[index(i, j, k)];
          axes_[2].differentiateInto(line, order, diff, scratch,
                                     fourierScratch);
          for (std::size_t k = 0; k < n3(); ++k)
            out[index(i, j, k)] = diff[k];
        }
      }
    }
    return out;
  }

  std::vector<double> laplacian(const std::vector<double> &values) const {
    const auto d11 = derivative(values, 0, 2);
    const auto d22 = derivative(values, 1, 2);
    const auto d33 = derivative(values, 2, 2);
    std::vector<double> out(values.size(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i)
      out[i] = d11[i] + d22[i] + d33[i];
    return out;
  }

  SpectralDerivatives3D derivatives(const std::vector<double> &values) const {
    SpectralDerivatives3D out;
    out.value = values;
    out.d1 = derivative(values, 0, 1);
    out.d2 = derivative(values, 1, 1);
    out.d3 = derivative(values, 2, 1);
    out.d11 = derivative(values, 0, 2);
    out.d12 = derivative(out.d1, 1, 1);
    out.d13 = derivative(out.d1, 2, 1);
    out.d22 = derivative(values, 1, 2);
    out.d23 = derivative(out.d2, 2, 1);
    out.d33 = derivative(values, 2, 2);
    return out;
  }

  double interpolate(const std::vector<double> &values, double x1, double x2,
                     double x3) const {
    if (values.size() != size())
      throw std::runtime_error("spectral grid value count mismatch");

    std::vector<double> axis1Values(n2() * n3(), 0.0);
    std::vector<double> line1(n1(), 0.0);
    for (std::size_t k = 0; k < n3(); ++k) {
      for (std::size_t j = 0; j < n2(); ++j) {
        for (std::size_t i = 0; i < n1(); ++i)
          line1[i] = values[index(i, j, k)];
        axis1Values[j + n2() * k] = axes_[0].interpolate(line1, x1);
      }
    }

    std::vector<double> axis2Values(n3(), 0.0);
    std::vector<double> line2(n2(), 0.0);
    for (std::size_t k = 0; k < n3(); ++k) {
      for (std::size_t j = 0; j < n2(); ++j)
        line2[j] = axis1Values[j + n2() * k];
      axis2Values[k] = axes_[1].interpolate(line2, x2);
    }
    return axes_[2].interpolate(axis2Values, x3);
  }

  SpectralPoint3D point(std::size_t i, std::size_t j, std::size_t k) const {
    return SpectralPoint3D{i, j, k, index(i, j, k), axes_[0].points[i],
                           axes_[1].points[j], axes_[2].points[k]};
  }

  SpectralPointDerivatives3D pointDerivatives(
      const SpectralDerivatives3D &derivs, std::size_t p) const {
    if (p >= size())
      throw std::out_of_range("spectral derivative point index out of range");
    return SpectralPointDerivatives3D{
        derivs.value[p], derivs.d1[p],  derivs.d2[p],  derivs.d3[p],
        derivs.d11[p],  derivs.d12[p], derivs.d13[p], derivs.d22[p],
        derivs.d23[p],  derivs.d33[p]};
  }

  template <typename Fn>
  std::vector<double> evaluateScalarResidual(
      const SpectralDerivatives3D &derivs, Fn &&fn) const {
    if (derivs.value.size() != size() || derivs.d1.size() != size() ||
        derivs.d2.size() != size() || derivs.d3.size() != size() ||
        derivs.d11.size() != size() || derivs.d12.size() != size() ||
        derivs.d13.size() != size() || derivs.d22.size() != size() ||
        derivs.d23.size() != size() || derivs.d33.size() != size()) {
      throw std::runtime_error("spectral derivative bundle size mismatch");
    }

    std::vector<double> out(size(), 0.0);
    for (std::size_t k = 0; k < n3(); ++k) {
      for (std::size_t j = 0; j < n2(); ++j) {
        for (std::size_t i = 0; i < n1(); ++i) {
          const SpectralPoint3D p = point(i, j, k);
          out[p.index] = fn(p, pointDerivatives(derivs, p.index));
        }
      }
    }
    return out;
  }

private:
  SpectralAxis axes_[3];
};

} // namespace tensorium_mlir::runtime
