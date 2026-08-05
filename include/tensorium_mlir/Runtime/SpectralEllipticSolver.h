#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualJVP.h"

#include <complex>

namespace tensorium_mlir::runtime {

inline bool solveDenseLinearSystem(std::vector<double> matrix,
                                   std::vector<double> rhs,
                                   std::vector<double> &solution,
                                   double pivotTolerance) {
  const std::size_t n = rhs.size();
  if (matrix.size() != n * n)
    return false;
  solution.assign(n, 0.0);

  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivotRow = col;
    double pivotAbs = std::fabs(matrix[col * n + col]);
    for (std::size_t row = col + 1; row < n; ++row) {
      const double candidate = std::fabs(matrix[row * n + col]);
      if (candidate > pivotAbs) {
        pivotAbs = candidate;
        pivotRow = row;
      }
    }
    if (!(pivotAbs > pivotTolerance) || !std::isfinite(pivotAbs))
      return false;
    if (pivotRow != col) {
      for (std::size_t j = col; j < n; ++j)
        std::swap(matrix[col * n + j], matrix[pivotRow * n + j]);
      std::swap(rhs[col], rhs[pivotRow]);
    }

    const double pivot = matrix[col * n + col];
    for (std::size_t row = col + 1; row < n; ++row) {
      const double factor = matrix[row * n + col] / pivot;
      matrix[row * n + col] = 0.0;
      for (std::size_t j = col + 1; j < n; ++j)
        matrix[row * n + j] -= factor * matrix[col * n + j];
      rhs[row] -= factor * rhs[col];
    }
  }

  for (std::size_t rev = 0; rev < n; ++rev) {
    const std::size_t row = n - 1 - rev;
    double sum = rhs[row];
    for (std::size_t j = row + 1; j < n; ++j)
      sum -= matrix[row * n + j] * solution[j];
    const double diagonal = matrix[row * n + row];
    if (!(std::fabs(diagonal) > pivotTolerance) || !std::isfinite(diagonal))
      return false;
    solution[row] = sum / diagonal;
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

inline double spectralVectorDot(std::span<const double> lhs,
                                std::span<const double> rhs) {
  if (lhs.size() != rhs.size())
    throw std::runtime_error("spectral vector dot size mismatch");
  double out = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i)
    out += lhs[i] * rhs[i];
  return out;
}

inline double spectralVectorEuclideanNorm(std::span<const double> values) {
  return std::sqrt(std::max(0.0, spectralVectorDot(values, values)));
}

struct SpectralGMRESResult {
  bool converged = false;
  int iterations = 0;
  double residualL2 = std::numeric_limits<double>::infinity();
  bool usedPreconditioner = false;
  std::vector<double> solution;
};

struct SpectralSparseMatrix {
  std::size_t size = 0;
  std::vector<std::size_t> rowOffsets;
  std::vector<std::size_t> columns;
  std::vector<double> values;
  std::vector<double> diagonal;
};

inline bool spectralPreconditionerRequested(
    const SpectralEllipticSolveOptions &options) {
  return options.gmresPreconditioner != SpectralPreconditionerKind::None;
}

struct SpectralLinearPreconditioner {
  SpectralPreconditionerKind kind = SpectralPreconditionerKind::None;
  std::vector<double> inverseDiagonal;
  std::vector<std::vector<double>> denseBlocks;
  std::vector<std::vector<double>> modalBlocks;
  std::vector<SpectralSparseMatrix> sparseBlocks;
  std::size_t blockSize = 0;
  std::size_t modalBlockSize = 0;
  std::array<std::size_t, 3> modalExtents{0, 0, 0};
  std::array<SpectralBasis, 3> modalBases{
      SpectralBasis::ChebyshevZeros, SpectralBasis::ChebyshevZeros,
      SpectralBasis::ChebyshevZeros};
  int relaxationSweeps = 0;
  double relaxationOmega = 1.0;
};

inline bool spectralFieldProjectorEnabled(
    const SpectralResidualProblem &problem) {
  return problem.fieldProjector.project != nullptr;
}

inline void projectSpectralField(const SpectralResidualProblem &problem,
                                 std::span<double> values) {
  if (!spectralFieldProjectorEnabled(problem))
    return;
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  if (values.size() != grid.size())
    throw std::runtime_error("spectral field projector size mismatch");
  problem.fieldProjector.project(
      &grid, values.data(), static_cast<std::int64_t>(values.size()),
      problem.fieldProjector.userData);
}

inline bool spectralSystemUsesFieldProjector(
    const SpectralResidualSystemProblem &system) {
  return std::any_of(system.equations.begin(), system.equations.end(),
                     [](const SpectralResidualSystemEquation &equation) {
                       return spectralFieldProjectorEnabled(equation.problem);
                     });
}

inline void projectSpectralSystemUnknownVector(
    const SpectralResidualSystemProblem &system, std::span<double> values,
    std::size_t fieldCount, std::size_t pointsPerField) {
  if (values.size() != fieldCount * pointsPerField)
    throw std::runtime_error("spectral system projector size mismatch");
  for (const auto &equation : system.equations) {
    if (equation.unknownIndex >= fieldCount)
      throw std::runtime_error("spectral system projector unknown mismatch");
    projectSpectralField(
        equation.problem,
        values.subspan(equation.unknownIndex * pointsPerField, pointsPerField));
  }
}

inline void projectSpectralSystemUnknownFields(
    const SpectralResidualSystemProblem &system,
    std::span<std::vector<double>> fields) {
  for (const auto &equation : system.equations) {
    if (equation.unknownIndex >= fields.size())
      throw std::runtime_error("spectral system projector unknown mismatch");
    projectSpectralField(equation.problem, fields[equation.unknownIndex]);
  }
}

inline double spectralChebyshevAxisLength(const SpectralAxis &axis) {
  const std::size_t n = axis.size();
  if (n <= 1)
    return 1.0;
  double minPoint = axis.points.front();
  double maxPoint = axis.points.front();
  for (double point : axis.points) {
    minPoint = std::min(minPoint, point);
    maxPoint = std::max(maxPoint, point);
  }
  const double edgeCos =
      std::cos(0.5 * kSpectralPi / static_cast<double>(n));
  if (!(edgeCos > 0.0))
    return maxPoint - minPoint;
  return (maxPoint - minPoint) / edgeCos;
}

inline double spectralAxisModalLaplacianEigenvalue(const SpectralAxis &axis,
                                                   std::size_t mode) {
  if (mode == 0)
    return 0.0;
  if (axis.basis == SpectralBasis::FourierPeriodic) {
    const std::size_t n = axis.size();
    const int waveNumber =
        mode <= n / 2 ? static_cast<int>(mode)
                      : static_cast<int>(mode) - static_cast<int>(n);
    const double angularWave =
        2.0 * kSpectralPi * static_cast<double>(waveNumber) / axis.period;
    return -(angularWave * angularWave);
  }

  const double length = spectralChebyshevAxisLength(axis);
  const double angularWave =
      kSpectralPi * static_cast<double>(mode) / std::max(length, 1.0e-15);
  return -(angularWave * angularWave);
}

inline std::vector<double>
buildSpectralModalLaplacianShiftInverseDiagonal(const SpectralGrid3D &grid,
                                                double shift,
                                                double pivotTolerance) {
  const std::size_t n = grid.size();
  std::vector<double> inverse(n, 0.0);
  std::vector<double> lambda1(grid.n1(), 0.0);
  std::vector<double> lambda2(grid.n2(), 0.0);
  std::vector<double> lambda3(grid.n3(), 0.0);
  for (std::size_t i = 0; i < grid.n1(); ++i)
    lambda1[i] = spectralAxisModalLaplacianEigenvalue(grid.axis(0), i);
  for (std::size_t j = 0; j < grid.n2(); ++j)
    lambda2[j] = spectralAxisModalLaplacianEigenvalue(grid.axis(1), j);
  for (std::size_t k = 0; k < grid.n3(); ++k)
    lambda3[k] = spectralAxisModalLaplacianEigenvalue(grid.axis(2), k);

  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const double diagonal = lambda1[i] + lambda2[j] + lambda3[k] + shift;
        if (!(std::fabs(diagonal) > pivotTolerance) ||
            !std::isfinite(diagonal)) {
          throw std::runtime_error(
              "spectral modal Laplacian preconditioner diagonal is singular");
        }
        inverse[grid.index(i, j, k)] = 1.0 / diagonal;
      }
    }
  }
  return inverse;
}

inline void transformSpectralModalLine(
    std::vector<std::complex<double>> &line, SpectralBasis basis,
    bool inverse) {
  const std::size_t n = line.size();
  std::vector<std::complex<double>> out(n);
  const std::complex<double> imaginary(0.0, 1.0);

  if (basis == SpectralBasis::ChebyshevZeros) {
    if (!inverse) {
      for (std::size_t mode = 0; mode < n; ++mode) {
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t point = 0; point < n; ++point) {
          const double theta =
              kSpectralPi * (static_cast<double>(point) + 0.5) /
              static_cast<double>(n);
          sum += line[point] *
                 std::cos(static_cast<double>(mode) * theta);
        }
        out[mode] =
            sum * (mode == 0 ? 1.0 / static_cast<double>(n)
                             : 2.0 / static_cast<double>(n));
      }
    } else {
      for (std::size_t point = 0; point < n; ++point) {
        const double theta =
            kSpectralPi * (static_cast<double>(point) + 0.5) /
            static_cast<double>(n);
        std::complex<double> sum = line[0];
        for (std::size_t mode = 1; mode < n; ++mode)
          sum += line[mode] * std::cos(static_cast<double>(mode) * theta);
        out[point] = sum;
      }
    }
    line = std::move(out);
    return;
  }

  for (std::size_t dst = 0; dst < n; ++dst) {
    std::complex<double> sum(0.0, 0.0);
    for (std::size_t src = 0; src < n; ++src) {
      const double sign = inverse ? 1.0 : -1.0;
      const double phase =
          sign * 2.0 * kSpectralPi * static_cast<double>(dst * src) /
          static_cast<double>(n);
      sum += line[src] * std::exp(imaginary * phase);
    }
    out[dst] = inverse ? sum : sum / static_cast<double>(n);
  }
  line = std::move(out);
}

inline std::vector<double>
buildSpectralAxisSecondDerivativeModalMatrix(const SpectralAxis &axis) {
  const std::size_t n = axis.size();
  std::vector<double> matrix(n * n, 0.0);
  for (std::size_t col = 0; col < n; ++col) {
    std::vector<std::complex<double>> modalLine(n);
    modalLine[col] = 1.0;
    transformSpectralModalLine(modalLine, axis.basis, true);

    std::vector<double> physical(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      if (std::fabs(modalLine[i].imag()) > 1.0e-10)
        throw std::runtime_error("spectral modal transform produced complex data");
      physical[i] = modalLine[i].real();
    }

    const auto secondDerivative = axis.differentiate(physical, 2);
    for (std::size_t i = 0; i < n; ++i)
      modalLine[i] = secondDerivative[i];
    transformSpectralModalLine(modalLine, axis.basis, false);

    for (std::size_t row = 0; row < n; ++row) {
      if (std::fabs(modalLine[row].imag()) > 1.0e-8)
        throw std::runtime_error("spectral modal D2 projection is complex");
      matrix[row * n + col] = modalLine[row].real();
    }
  }
  return matrix;
}

inline bool supportsSpectralModalChebyshevFourierBlocks(
    const SpectralGrid3D &grid) {
  return grid.axis(0).basis == SpectralBasis::ChebyshevZeros &&
         grid.axis(1).basis == SpectralBasis::ChebyshevZeros &&
         grid.axis(2).basis == SpectralBasis::FourierPeriodic;
}

inline std::vector<std::vector<double>>
buildSpectralModalChebyshevFourierLaplacianShiftBlocks(
    const SpectralGrid3D &grid, double shift) {
  if (!supportsSpectralModalChebyshevFourierBlocks(grid))
    return {};

  const auto dxx = buildSpectralAxisSecondDerivativeModalMatrix(grid.axis(0));
  const auto dyy = buildSpectralAxisSecondDerivativeModalMatrix(grid.axis(1));
  const std::size_t n1 = grid.n1();
  const std::size_t n2 = grid.n2();
  const std::size_t blockSize = n1 * n2;
  std::vector<std::vector<double>> blocks;
  blocks.reserve(grid.n3());

  for (std::size_t k = 0; k < grid.n3(); ++k) {
    const double lambda =
        spectralAxisModalLaplacianEigenvalue(grid.axis(2), k) + shift;
    std::vector<double> block(blockSize * blockSize, 0.0);
    for (std::size_t colJ = 0; colJ < n2; ++colJ) {
      for (std::size_t colI = 0; colI < n1; ++colI) {
        const std::size_t col = colI + n1 * colJ;
        for (std::size_t rowI = 0; rowI < n1; ++rowI) {
          const std::size_t row = rowI + n1 * colJ;
          block[row * blockSize + col] += dxx[rowI * n1 + colI];
        }
        for (std::size_t rowJ = 0; rowJ < n2; ++rowJ) {
          const std::size_t row = colI + n1 * rowJ;
          block[row * blockSize + col] += dyy[rowJ * n2 + colJ];
        }
        block[col * blockSize + col] += lambda;
      }
    }
    blocks.push_back(std::move(block));
  }
  return blocks;
}

inline void transformSpectralModalAxis(
    std::vector<std::complex<double>> &values,
    const SpectralLinearPreconditioner &preconditioner, std::size_t dim,
    bool inverse) {
  const std::size_t n1 = preconditioner.modalExtents[0];
  const std::size_t n2 = preconditioner.modalExtents[1];
  const std::size_t n3 = preconditioner.modalExtents[2];
  const auto index = [n1, n2](std::size_t i, std::size_t j, std::size_t k) {
    return i + n1 * (j + n2 * k);
  };

  if (dim == 0) {
    std::vector<std::complex<double>> line(n1);
    for (std::size_t k = 0; k < n3; ++k) {
      for (std::size_t j = 0; j < n2; ++j) {
        for (std::size_t i = 0; i < n1; ++i)
          line[i] = values[index(i, j, k)];
        transformSpectralModalLine(line, preconditioner.modalBases[0], inverse);
        for (std::size_t i = 0; i < n1; ++i)
          values[index(i, j, k)] = line[i];
      }
    }
    return;
  }

  if (dim == 1) {
    std::vector<std::complex<double>> line(n2);
    for (std::size_t k = 0; k < n3; ++k) {
      for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < n2; ++j)
          line[j] = values[index(i, j, k)];
        transformSpectralModalLine(line, preconditioner.modalBases[1], inverse);
        for (std::size_t j = 0; j < n2; ++j)
          values[index(i, j, k)] = line[j];
      }
    }
    return;
  }

  std::vector<std::complex<double>> line(n3);
  for (std::size_t j = 0; j < n2; ++j) {
    for (std::size_t i = 0; i < n1; ++i) {
      for (std::size_t k = 0; k < n3; ++k)
        line[k] = values[index(i, j, k)];
      transformSpectralModalLine(line, preconditioner.modalBases[2], inverse);
      for (std::size_t k = 0; k < n3; ++k)
        values[index(i, j, k)] = line[k];
    }
  }
}

inline bool applySpectralModalPreconditionerBlock(
    const SpectralLinearPreconditioner &preconditioner,
    std::span<const std::vector<double>> modalBlocks,
    std::span<const double> inverseDiagonal, std::span<double> values,
    double pivotTolerance) {
  if (values.size() != preconditioner.blockSize) {
    return false;
  }

  std::vector<std::complex<double>> modal(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    modal[i] = values[i];
  transformSpectralModalAxis(modal, preconditioner, 0, false);
  transformSpectralModalAxis(modal, preconditioner, 1, false);
  transformSpectralModalAxis(modal, preconditioner, 2, false);

  if (!modalBlocks.empty()) {
    const std::size_t n1 = preconditioner.modalExtents[0];
    const std::size_t n2 = preconditioner.modalExtents[1];
    const std::size_t n3 = preconditioner.modalExtents[2];
    const std::size_t blockSize = n1 * n2;
    if (modalBlocks.size() != n3 ||
        preconditioner.modalBlockSize != blockSize) {
      return false;
    }

    for (std::size_t k = 0; k < n3; ++k) {
      if (modalBlocks[k].size() != blockSize * blockSize)
        return false;
      std::vector<double> rhsReal(blockSize, 0.0);
      std::vector<double> rhsImag(blockSize, 0.0);
      for (std::size_t j = 0; j < n2; ++j) {
        for (std::size_t i = 0; i < n1; ++i) {
          const std::size_t row = i + n1 * j;
          const std::size_t p = i + n1 * (j + n2 * k);
          rhsReal[row] = modal[p].real();
          rhsImag[row] = modal[p].imag();
        }
      }

      std::vector<double> solReal;
      std::vector<double> solImag;
      if (!solveDenseLinearSystem(modalBlocks[k], std::move(rhsReal), solReal,
                                  pivotTolerance) ||
          !solveDenseLinearSystem(modalBlocks[k], std::move(rhsImag), solImag,
                                  pivotTolerance)) {
        return false;
      }

      for (std::size_t j = 0; j < n2; ++j) {
        for (std::size_t i = 0; i < n1; ++i) {
          const std::size_t row = i + n1 * j;
          const std::size_t p = i + n1 * (j + n2 * k);
          modal[p] = std::complex<double>(solReal[row], solImag[row]);
        }
      }
    }
  } else {
    if (inverseDiagonal.size() != values.size())
      return false;
    for (std::size_t i = 0; i < modal.size(); ++i)
      modal[i] *= inverseDiagonal[i];
  }

  transformSpectralModalAxis(modal, preconditioner, 2, true);
  transformSpectralModalAxis(modal, preconditioner, 1, true);
  transformSpectralModalAxis(modal, preconditioner, 0, true);

  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = modal[i].real();
    if (!std::isfinite(values[i]) || std::fabs(modal[i].imag()) > 1.0e-9)
      return false;
  }
  return true;
}

inline bool solveSpectralSparseRelaxation(const SpectralSparseMatrix &matrix,
                                          std::span<const double> rhs,
                                          int sweeps, double omega,
                                          double pivotTolerance,
                                          std::vector<double> &solution) {
  if (matrix.size == 0 || rhs.size() != matrix.size ||
      matrix.rowOffsets.size() != matrix.size + 1 ||
      matrix.diagonal.size() != matrix.size || sweeps <= 0 ||
      !(omega > 0.0 && omega <= 2.0)) {
    return false;
  }
  solution.assign(matrix.size, 0.0);
  for (std::size_t row = 0; row < matrix.size; ++row) {
    const double diagonal = matrix.diagonal[row];
    if (!std::isfinite(diagonal) || std::fabs(diagonal) <= pivotTolerance)
      return false;
    solution[row] = rhs[row] / diagonal;
  }

  const auto relaxRow = [&](std::size_t row) {
    double remainder = rhs[row];
    for (std::size_t entry = matrix.rowOffsets[row];
         entry < matrix.rowOffsets[row + 1]; ++entry) {
      const std::size_t column = matrix.columns[entry];
      if (column >= matrix.size)
        return false;
      if (column != row)
        remainder -= matrix.values[entry] * solution[column];
    }
    const double candidate = remainder / matrix.diagonal[row];
    solution[row] += omega * (candidate - solution[row]);
    return std::isfinite(solution[row]);
  };

  for (int sweep = 0; sweep < sweeps; ++sweep) {
    for (std::size_t row = 0; row < matrix.size; ++row) {
      if (!relaxRow(row))
        return false;
    }
    for (std::size_t reversed = 0; reversed < matrix.size; ++reversed) {
      if (!relaxRow(matrix.size - 1 - reversed))
        return false;
    }
  }
  return spectralVectorIsFinite(solution);
}

inline bool applySpectralPreconditioner(
    const SpectralLinearPreconditioner &preconditioner,
    std::vector<double> &values,
    double pivotTolerance) {
  if (preconditioner.kind == SpectralPreconditionerKind::None)
    return true;

  if (preconditioner.kind == SpectralPreconditionerKind::DiagonalJVP) {
    if (preconditioner.inverseDiagonal.size() != values.size())
      return false;
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] *= preconditioner.inverseDiagonal[i];
      if (!std::isfinite(values[i]))
        return false;
    }
    return true;
  }

  if (preconditioner.kind ==
      SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift) {
    if (preconditioner.blockSize == 0 ||
        values.size() !=
            preconditioner.blockSize * preconditioner.sparseBlocks.size())
      return false;
    std::vector<double> out(values.size(), 0.0);
    for (std::size_t block = 0; block < preconditioner.sparseBlocks.size();
         ++block) {
      const std::size_t offset = block * preconditioner.blockSize;
      std::vector<double> solution;
      if (!solveSpectralSparseRelaxation(
              preconditioner.sparseBlocks[block],
              std::span<const double>(&values[offset],
                                      preconditioner.blockSize),
              preconditioner.relaxationSweeps, preconditioner.relaxationOmega,
              pivotTolerance, solution)) {
        return false;
      }
      std::copy(solution.begin(), solution.end(), out.begin() + offset);
    }
    values = std::move(out);
    return spectralVectorIsFinite(values);
  }

  if (preconditioner.kind == SpectralPreconditionerKind::DenseLaplacianShift) {
    if (preconditioner.blockSize == 0 ||
        values.size() != preconditioner.blockSize *
                             preconditioner.denseBlocks.size())
      return false;
    std::vector<double> out(values.size(), 0.0);
    for (std::size_t block = 0; block < preconditioner.denseBlocks.size();
         ++block) {
      const std::size_t offset = block * preconditioner.blockSize;
      std::vector<double> rhs(preconditioner.blockSize, 0.0);
      for (std::size_t i = 0; i < preconditioner.blockSize; ++i)
        rhs[i] = values[offset + i];
      std::vector<double> blockSolution;
      if (!solveDenseLinearSystem(preconditioner.denseBlocks[block],
                                  std::move(rhs), blockSolution,
                                  pivotTolerance)) {
        return false;
      }
      for (std::size_t i = 0; i < preconditioner.blockSize; ++i)
        out[offset + i] = blockSolution[i];
    }
    values = std::move(out);
    return spectralVectorIsFinite(values);
  }

  if (preconditioner.kind == SpectralPreconditionerKind::ModalLaplacianShift) {
    if (preconditioner.blockSize == 0)
      return false;

    std::size_t blockCount = 0;
    if (!preconditioner.modalBlocks.empty()) {
      if (preconditioner.modalExtents[2] == 0 ||
          preconditioner.modalBlocks.size() % preconditioner.modalExtents[2] !=
              0) {
        return false;
      }
      blockCount =
          preconditioner.modalBlocks.size() / preconditioner.modalExtents[2];
    } else {
      if (preconditioner.inverseDiagonal.size() %
              preconditioner.blockSize !=
          0) {
        return false;
      }
      blockCount =
          preconditioner.inverseDiagonal.size() / preconditioner.blockSize;
    }
    if (values.size() != blockCount * preconditioner.blockSize)
      return false;

    for (std::size_t block = 0; block < blockCount; ++block) {
      const std::size_t offset = block * preconditioner.blockSize;
      const std::size_t modalBlockOffset =
          block * preconditioner.modalExtents[2];
      const bool hasModalBlocks = !preconditioner.modalBlocks.empty();
      if (!applySpectralModalPreconditionerBlock(
              preconditioner,
              hasModalBlocks
                  ? std::span<const std::vector<double>>(
                        &preconditioner.modalBlocks[modalBlockOffset],
                        preconditioner.modalExtents[2])
                  : std::span<const std::vector<double>>(),
              preconditioner.inverseDiagonal.empty()
                  ? std::span<const double>()
                  : std::span<const double>(
                        &preconditioner.inverseDiagonal[offset],
                        preconditioner.blockSize),
              std::span<double>(&values[offset],
                                preconditioner.blockSize),
              pivotTolerance)) {
        return false;
      }
    }
    return spectralVectorIsFinite(values);
  }

  return false;
}

struct SpectralThreePointStencil {
  std::array<std::size_t, 3> indices{};
  std::array<double, 3> first{};
  std::array<double, 3> second{};
};

inline SpectralThreePointStencil
buildSpectralThreePointStencil(const SpectralAxis &axis,
                               std::size_t pointIndex) {
  const std::size_t n = axis.size();
  if (n < 3 || pointIndex >= n)
    throw std::runtime_error(
        "mapped finite-difference preconditioner requires three axis points");

  SpectralThreePointStencil stencil;
  std::array<double, 3> coordinates{};
  const double center = axis.points[pointIndex];
  if (axis.basis == SpectralBasis::FourierPeriodic) {
    stencil.indices = {(pointIndex + n - 1) % n, pointIndex,
                       (pointIndex + 1) % n};
    const double spacing = axis.period / static_cast<double>(n);
    coordinates = {center - spacing, center, center + spacing};
  } else {
    const std::size_t first =
        pointIndex == 0 ? 0 : (pointIndex + 1 == n ? n - 3 : pointIndex - 1);
    stencil.indices = {first, first + 1, first + 2};
    for (std::size_t q = 0; q < 3; ++q)
      coordinates[q] = axis.points[stencil.indices[q]];
  }

  for (std::size_t q = 0; q < 3; ++q) {
    const std::size_t p = (q + 1) % 3;
    const std::size_t r = (q + 2) % 3;
    const double denominator =
        (coordinates[q] - coordinates[p]) * (coordinates[q] - coordinates[r]);
    if (!std::isfinite(denominator) || denominator == 0.0)
      throw std::runtime_error("invalid mapped finite-difference stencil");
    stencil.first[q] =
        (2.0 * center - coordinates[p] - coordinates[r]) / denominator;
    stencil.second[q] = 2.0 / denominator;
  }
  return stencil;
}

inline SpectralPointDerivatives3D transformSpectralPreconditionerBundle(
    const SpectralResidualProblem &problem, const double logical[3],
    const SpectralPointDerivatives3D &logicalBundle) {
  SpectralPointDerivatives3D unknownBundle = logicalBundle;
  if (problem.unknownMap.transform) {
    problem.unknownMap.transform(
        logical, &logicalBundle, &unknownBundle,
        problem.unknownMapParams.data(),
        static_cast<std::int64_t>(problem.unknownMapParams.size()),
        problem.unknownMap.userData);
  }
  SpectralPointDerivatives3D physicalBundle = unknownBundle;
  if (problem.derivativeMap.transform) {
    problem.derivativeMap.transform(
        logical, &unknownBundle, &physicalBundle,
        problem.coordinateParams.data(),
        static_cast<std::int64_t>(problem.coordinateParams.size()),
        problem.derivativeMap.userData);
  }
  return physicalBundle;
}

inline SpectralSparseMatrix buildSpectralMappedFiniteDifferenceLaplacianShift(
    const SpectralResidualProblem &problem, double shift,
    double pivotTolerance) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  if (!std::isfinite(shift))
    throw std::runtime_error("mapped finite-difference shift is not finite");

  SpectralSparseMatrix matrix;
  matrix.size = grid.size();
  matrix.rowOffsets.reserve(matrix.size + 1);
  matrix.diagonal.assign(matrix.size, 0.0);
  matrix.rowOffsets.push_back(0);

  for (std::size_t k = 0; k < grid.n3(); ++k) {
    for (std::size_t j = 0; j < grid.n2(); ++j) {
      for (std::size_t i = 0; i < grid.n1(); ++i) {
        const SpectralPoint3D point = grid.point(i, j, k);
        const double logical[3] = {point.x1, point.x2, point.x3};
        std::vector<std::pair<std::size_t, SpectralPointDerivatives3D>> entries;
        entries.reserve(7);
        const auto findOrAdd =
            [&](std::size_t column) -> SpectralPointDerivatives3D & {
          for (auto &entry : entries) {
            if (entry.first == column)
              return entry.second;
          }
          entries.emplace_back(column, SpectralPointDerivatives3D{});
          return entries.back().second;
        };
        findOrAdd(point.index).value = 1.0;

        const std::array<std::size_t, 3> rowIndices = {i, j, k};
        for (std::size_t dim = 0; dim < 3; ++dim) {
          const auto stencil =
              buildSpectralThreePointStencil(grid.axis(dim), rowIndices[dim]);
          for (std::size_t q = 0; q < 3; ++q) {
            std::array<std::size_t, 3> columnIndices = {i, j, k};
            columnIndices[dim] = stencil.indices[q];
            auto &bundle = findOrAdd(grid.index(
                columnIndices[0], columnIndices[1], columnIndices[2]));
            if (dim == 0) {
              bundle.d1 += stencil.first[q];
              bundle.d11 += stencil.second[q];
            } else if (dim == 1) {
              bundle.d2 += stencil.first[q];
              bundle.d22 += stencil.second[q];
            } else {
              bundle.d3 += stencil.first[q];
              bundle.d33 += stencil.second[q];
            }
          }
        }

        std::sort(entries.begin(), entries.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return lhs.first < rhs.first;
                  });
        for (const auto &[column, logicalBundle] : entries) {
          const auto physicalBundle = transformSpectralPreconditionerBundle(
              problem, logical, logicalBundle);
          const double coefficient =
              physicalBundle.laplacian() + shift * physicalBundle.value;
          if (!std::isfinite(coefficient))
            throw std::runtime_error(
                "mapped finite-difference coefficient is not finite");
          matrix.columns.push_back(column);
          matrix.values.push_back(coefficient);
          if (column == point.index)
            matrix.diagonal[point.index] = coefficient;
        }
        if (!std::isfinite(matrix.diagonal[point.index]) ||
            std::fabs(matrix.diagonal[point.index]) <= pivotTolerance) {
          throw std::runtime_error(
              "mapped finite-difference diagonal is singular");
        }
        matrix.rowOffsets.push_back(matrix.columns.size());
      }
    }
  }
  return matrix;
}

inline std::vector<double> buildSpectralLaplacianShiftMatrix(
    const SpectralGrid3D &grid, double shift) {
  const std::size_t n = grid.size();
  std::vector<double> matrix(n * n, 0.0);
  std::vector<double> basis(n, 0.0);
  for (std::size_t col = 0; col < n; ++col) {
    basis[col] = 1.0;
    const auto laplacian = grid.laplacian(basis);
    basis[col] = 0.0;
    for (std::size_t row = 0; row < n; ++row)
      matrix[row * n + col] = laplacian[row];
    matrix[col * n + col] += shift;
  }
  return matrix;
}

inline void populateSpectralModalPreconditionerMetadata(
    const SpectralGrid3D &grid, SpectralLinearPreconditioner &preconditioner) {
  preconditioner.blockSize = grid.size();
  preconditioner.modalExtents = {grid.n1(), grid.n2(), grid.n3()};
  for (std::size_t dim = 0; dim < 3; ++dim) {
    const auto &axis = grid.axis(dim);
    preconditioner.modalBases[dim] = axis.basis;
  }
}

inline void appendSpectralModalLaplacianShiftBlock(
    const SpectralGrid3D &grid, double shift,
    const SpectralEllipticSolveOptions &options,
    SpectralLinearPreconditioner &preconditioner) {
  if (supportsSpectralModalChebyshevFourierBlocks(grid)) {
    const auto blocks =
        buildSpectralModalChebyshevFourierLaplacianShiftBlocks(grid, shift);
    preconditioner.modalBlockSize = grid.n1() * grid.n2();
    preconditioner.modalBlocks.insert(preconditioner.modalBlocks.end(),
                                      blocks.begin(), blocks.end());
    return;
  }

  const auto inverse = buildSpectralModalLaplacianShiftInverseDiagonal(
      grid, shift, options.preconditionerPivotTolerance);
  preconditioner.inverseDiagonal.insert(preconditioner.inverseDiagonal.end(),
                                        inverse.begin(), inverse.end());
}

inline bool buildSpectralDiagonalPreconditionerByJVP(
    const SpectralResidualProblem &problem, const std::vector<double> &values,
    const SpectralEllipticSolveOptions &options,
    SpectralLinearPreconditioner &preconditioner) {
  preconditioner = {};
  if (!spectralPreconditionerRequested(options))
    return true;
  if (options.gmresPreconditioner != SpectralPreconditionerKind::DiagonalJVP)
    return false;

  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  const std::size_t n = grid.size();
  if (values.size() != n)
    return false;
  preconditioner.kind = SpectralPreconditionerKind::DiagonalJVP;
  preconditioner.inverseDiagonal.assign(n, 1.0);
  std::vector<double> direction(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    direction[i] = 1.0;
    const auto jvp =
        evaluateSpectralJacobianVectorProduct(problem, values, direction,
                                              options.jvpOptions);
    direction[i] = 0.0;
    if (!jvp.finite || jvp.values.size() != n)
      return false;
    const double diagonal = jvp.values[i];
    if (std::isfinite(diagonal) &&
        std::fabs(diagonal) > options.preconditionerPivotTolerance) {
      preconditioner.inverseDiagonal[i] = 1.0 / diagonal;
    }
  }
  return true;
}

inline bool buildSpectralScalarPreconditioner(
    const SpectralResidualProblem &problem, const std::vector<double> &values,
    const SpectralEllipticSolveOptions &options,
    SpectralLinearPreconditioner &preconditioner) {
  preconditioner = {};
  if (!spectralPreconditionerRequested(options))
    return true;
  if (options.gmresPreconditioner == SpectralPreconditionerKind::DiagonalJVP) {
    return buildSpectralDiagonalPreconditionerByJVP(problem, values, options,
                                                   preconditioner);
  }
  if (options.gmresPreconditioner ==
      SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift) {
    const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
    if (values.size() != grid.size())
      return false;
    preconditioner.kind =
        SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift;
    preconditioner.blockSize = grid.size();
    preconditioner.relaxationSweeps = options.preconditionerRelaxationSweeps;
    preconditioner.relaxationOmega = options.preconditionerRelaxationOmega;
    preconditioner.sparseBlocks.push_back(
        buildSpectralMappedFiniteDifferenceLaplacianShift(
            problem, options.preconditionerLaplacianShift,
            options.preconditionerPivotTolerance));
    return true;
  }
  if (options.gmresPreconditioner ==
      SpectralPreconditionerKind::DenseLaplacianShift) {
    const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
    if (values.size() != grid.size())
      return false;
    preconditioner.kind = SpectralPreconditionerKind::DenseLaplacianShift;
    preconditioner.blockSize = grid.size();
    preconditioner.denseBlocks.push_back(buildSpectralLaplacianShiftMatrix(
        grid, options.preconditionerLaplacianShift));
    return true;
  }
  if (options.gmresPreconditioner ==
      SpectralPreconditionerKind::ModalLaplacianShift) {
    const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
    if (values.size() != grid.size())
      return false;
    preconditioner.kind = SpectralPreconditionerKind::ModalLaplacianShift;
    populateSpectralModalPreconditionerMetadata(grid, preconditioner);
    appendSpectralModalLaplacianShiftBlock(
        grid, options.preconditionerLaplacianShift, options, preconditioner);
    return true;
  }
  return false;
}

inline bool updateSpectralGMRESQR(
    std::vector<double> &hessenberg, std::size_t leadingDim,
    std::size_t column, std::vector<double> &cosines,
    std::vector<double> &sines, std::vector<double> &rotatedRhs,
    double pivotTolerance, double &residualL2, std::size_t vectorSize) {
  for (std::size_t row = 0; row < column; ++row) {
    const double upper = hessenberg[row * leadingDim + column];
    const double lower = hessenberg[(row + 1) * leadingDim + column];
    hessenberg[row * leadingDim + column] =
        cosines[row] * upper + sines[row] * lower;
    hessenberg[(row + 1) * leadingDim + column] =
        -sines[row] * upper + cosines[row] * lower;
  }

  const double upper = hessenberg[column * leadingDim + column];
  const double lower = hessenberg[(column + 1) * leadingDim + column];
  const double magnitude = std::hypot(upper, lower);
  if (!std::isfinite(magnitude) || magnitude <= pivotTolerance)
    return false;
  cosines[column] = upper / magnitude;
  sines[column] = lower / magnitude;
  hessenberg[column * leadingDim + column] = magnitude;
  hessenberg[(column + 1) * leadingDim + column] = 0.0;

  const double rhs = rotatedRhs[column];
  rotatedRhs[column] = cosines[column] * rhs;
  rotatedRhs[column + 1] = -sines[column] * rhs;
  residualL2 =
      std::fabs(rotatedRhs[column + 1]) /
      std::sqrt(static_cast<double>(std::max<std::size_t>(1, vectorSize)));
  return std::isfinite(residualL2);
}

inline bool solveSpectralGMRESUpperTriangular(
    const std::vector<double> &hessenberg, std::size_t leadingDim,
    std::size_t columns, std::span<const double> rotatedRhs,
    double pivotTolerance, std::vector<double> &solution) {
  if (columns == 0 || rotatedRhs.size() < columns)
    return false;
  solution.assign(columns, 0.0);
  for (std::size_t reversed = 0; reversed < columns; ++reversed) {
    const std::size_t row = columns - 1 - reversed;
    double rhs = rotatedRhs[row];
    for (std::size_t col = row + 1; col < columns; ++col)
      rhs -= hessenberg[row * leadingDim + col] * solution[col];
    const double diagonal = hessenberg[row * leadingDim + row];
    if (!std::isfinite(diagonal) || std::fabs(diagonal) <= pivotTolerance)
      return false;
    solution[row] = rhs / diagonal;
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

template <typename ApplyOperatorFn, typename ApplyRightPreconditionerFn>
inline SpectralGMRESResult solveSpectralRestartedGMRES(
    std::size_t n, std::span<const double> rhs,
    const SpectralEllipticSolveOptions &options,
    ApplyOperatorFn &&applyOperator,
    ApplyRightPreconditionerFn &&applyRightPreconditioner) {
  SpectralGMRESResult result;
  result.solution.assign(n, 0.0);
  if (rhs.size() != n || n == 0 || options.gmresMaxIterations < 0 ||
      options.gmresRestart <= 0)
    return result;

  const double rhsEuclidean = spectralVectorEuclideanNorm(rhs);
  const double rhsL2 = rhsEuclidean / std::sqrt(static_cast<double>(n));
  const double target =
      std::max(options.gmresTolerance, options.gmresRelativeTolerance * rhsL2);
  result.residualL2 = rhsL2;
  if (rhsL2 <= target) {
    result.converged = true;
    return result;
  }

  const std::size_t maxIterations =
      static_cast<std::size_t>(options.gmresMaxIterations);
  const std::size_t restart =
      std::min<std::size_t>(n, static_cast<std::size_t>(options.gmresRestart));
  if (maxIterations == 0 || restart == 0)
    return result;

  std::vector<double> residual(rhs.begin(), rhs.end());
  std::vector<double> operatorValue(n, 0.0);
  std::size_t totalIterations = 0;
  while (totalIterations < maxIterations) {
    const double beta = spectralVectorEuclideanNorm(residual);
    result.residualL2 = beta / std::sqrt(static_cast<double>(n));
    if (result.residualL2 <= target) {
      result.converged = true;
      break;
    }

    const std::size_t cycleIterations =
        std::min(restart, maxIterations - totalIterations);
    std::vector<double> basis((cycleIterations + 1) * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      basis[i] = residual[i] / beta;
    std::vector<double> hessenberg((cycleIterations + 1) * cycleIterations,
                                   0.0);
    std::vector<double> cosines(cycleIterations, 0.0);
    std::vector<double> sines(cycleIterations, 0.0);
    std::vector<double> rotatedRhs(cycleIterations + 1, 0.0);
    rotatedRhs[0] = beta;
    std::vector<double> arnoldiVector(n, 0.0);
    // FGMRES must retain each preconditioned Arnoldi vector because the
    // preconditioner may vary between iterations.
    std::vector<double> preconditionedBasis(cycleIterations * n, 0.0);
    std::size_t columns = 0;
    bool projectedConverged = false;

    for (std::size_t col = 0; col < cycleIterations; ++col) {
      std::vector<double> direction(&basis[col * n], &basis[(col + 1) * n]);
      if (!applyRightPreconditioner(direction))
        return result;
      std::copy(direction.begin(), direction.end(),
                preconditionedBasis.begin() + col * n);
      if (!applyOperator(direction, arnoldiVector) ||
          arnoldiVector.size() != n || !spectralVectorIsFinite(arnoldiVector)) {
        return result;
      }

      for (std::size_t row = 0; row <= col; ++row) {
        std::span<const double> basisVector(&basis[row * n], n);
        const double h = spectralVectorDot(arnoldiVector, basisVector);
        hessenberg[row * cycleIterations + col] = h;
        for (std::size_t i = 0; i < n; ++i)
          arnoldiVector[i] -= h * basisVector[i];
      }

      const double nextNorm = spectralVectorEuclideanNorm(arnoldiVector);
      hessenberg[(col + 1) * cycleIterations + col] = nextNorm;
      if (nextNorm > options.linearPivotTolerance) {
        for (std::size_t i = 0; i < n; ++i)
          basis[(col + 1) * n + i] = arnoldiVector[i] / nextNorm;
      }

      double projectedResidualL2 = std::numeric_limits<double>::infinity();
      if (!updateSpectralGMRESQR(
              hessenberg, cycleIterations, col, cosines, sines, rotatedRhs,
              options.linearPivotTolerance, projectedResidualL2, n)) {
        return result;
      }
      ++totalIterations;
      columns = col + 1;
      result.iterations = static_cast<int>(totalIterations);
      result.residualL2 = projectedResidualL2;
      projectedConverged = projectedResidualL2 <= target;
      if (projectedConverged || nextNorm <= options.linearPivotTolerance)
        break;
    }

    if (columns == 0)
      return result;
    std::vector<double> coefficients;
    if (!solveSpectralGMRESUpperTriangular(
            hessenberg, cycleIterations, columns, rotatedRhs,
            options.linearPivotTolerance, coefficients)) {
      return result;
    }
    std::vector<double> correction(n, 0.0);
    for (std::size_t col = 0; col < columns; ++col) {
      for (std::size_t i = 0; i < n; ++i)
        correction[i] +=
            coefficients[col] * preconditionedBasis[col * n + i];
    }
    for (std::size_t i = 0; i < n; ++i)
      result.solution[i] += correction[i];
    if (projectedConverged) {
      result.converged = true;
      break;
    }

    if (!applyOperator(result.solution, operatorValue) ||
        operatorValue.size() != n || !spectralVectorIsFinite(operatorValue)) {
      return result;
    }
    for (std::size_t i = 0; i < n; ++i)
      residual[i] = rhs[i] - operatorValue[i];
    result.residualL2 = spectralVectorEuclideanNorm(residual) /
                        std::sqrt(static_cast<double>(n));
    if (result.residualL2 <= target) {
      result.converged = true;
      break;
    }
  }
  return result;
}

inline SpectralGMRESResult solveSpectralGMRESByJVP(
    const SpectralResidualProblem &problem, const std::vector<double> &values,
    std::span<const double> rhs,
    const SpectralEllipticSolveOptions &options) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  const std::size_t n = grid.size();
  SpectralGMRESResult result;
  result.solution.assign(n, 0.0);
  if (rhs.size() != n || values.size() != n || options.gmresMaxIterations < 0)
    return result;

  SpectralLinearPreconditioner preconditioner;
  if (!buildSpectralScalarPreconditioner(problem, values, options,
                                         preconditioner))
    return result;
  result.usedPreconditioner =
      preconditioner.kind != SpectralPreconditionerKind::None;

  if (options.gmresRestart > 0) {
    auto restarted = solveSpectralRestartedGMRES(
        n, rhs, options,
        [&](const std::vector<double> &direction, std::vector<double> &out) {
          const auto jvp = evaluateSpectralJacobianVectorProduct(
              problem, values, direction, options.jvpOptions);
          out = jvp.values;
          return jvp.finite && out.size() == n;
        },
        [&](std::vector<double> &direction) {
          if (!applySpectralPreconditioner(
                  preconditioner, direction,
                  options.preconditionerPivotTolerance))
            return false;
          projectSpectralField(problem, direction);
          return true;
        });
    restarted.usedPreconditioner = result.usedPreconditioner;
    return restarted;
  }

  const double rhsEuclidean = spectralVectorEuclideanNorm(rhs);
  const double rhsL2 =
      rhsEuclidean / std::sqrt(static_cast<double>(std::max<std::size_t>(1, n)));
  const double target = std::max(options.gmresTolerance,
                                 options.gmresRelativeTolerance * rhsL2);
  result.residualL2 = rhsL2;
  if (rhsL2 <= target) {
    result.converged = true;
    return result;
  }
  const std::size_t maxIterations = std::min<std::size_t>(
      n, static_cast<std::size_t>(options.gmresMaxIterations));
  if (maxIterations == 0)
    return result;

  std::vector<double> basis((maxIterations + 1) * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    basis[i] = rhs[i] / rhsEuclidean;

  std::vector<double> hessenberg((maxIterations + 1) * maxIterations, 0.0);
  std::vector<double> arnoldiVector(n, 0.0);
  std::vector<double> cosines(maxIterations, 0.0);
  std::vector<double> sines(maxIterations, 0.0);
  std::vector<double> rotatedRhs(maxIterations + 1, 0.0);
  rotatedRhs[0] = rhsEuclidean;
  std::size_t bestColumns = 0;

  for (std::size_t col = 0; col < maxIterations; ++col) {
    std::span<const double> direction(&basis[col * n], n);
    std::vector<double> directionVector(direction.begin(), direction.end());
    if (!applySpectralPreconditioner(preconditioner, directionVector,
                                     options.preconditionerPivotTolerance))
      return result;
    projectSpectralField(problem, directionVector);
    const auto jvp =
        evaluateSpectralJacobianVectorProduct(problem, values, directionVector,
                                              options.jvpOptions);
    if (!jvp.finite || jvp.values.size() != n)
      return result;
    arnoldiVector = jvp.values;

    for (std::size_t row = 0; row <= col; ++row) {
      std::span<const double> basisVector(&basis[row * n], n);
      const double h = spectralVectorDot(arnoldiVector, basisVector);
      hessenberg[row * maxIterations + col] = h;
      for (std::size_t i = 0; i < n; ++i)
        arnoldiVector[i] -= h * basisVector[i];
    }

    const double nextNorm = spectralVectorEuclideanNorm(arnoldiVector);
    hessenberg[(col + 1) * maxIterations + col] = nextNorm;
    if (nextNorm > options.linearPivotTolerance && col + 1 < maxIterations + 1) {
      for (std::size_t i = 0; i < n; ++i)
        basis[(col + 1) * n + i] = arnoldiVector[i] / nextNorm;
    }

    double projectedResidualL2 = std::numeric_limits<double>::infinity();
    if (!updateSpectralGMRESQR(
            hessenberg, maxIterations, col, cosines, sines, rotatedRhs,
            options.linearPivotTolerance, projectedResidualL2, n)) {
      return result;
    }

    const std::size_t columns = col + 1;
    result.iterations = static_cast<int>(columns);
    result.residualL2 = projectedResidualL2;
    bestColumns = columns;
    if (projectedResidualL2 <= target) {
      result.converged = true;
      break;
    }
    if (nextNorm <= options.linearPivotTolerance)
      break;
  }

  if (bestColumns == 0)
    return result;
  std::vector<double> bestY;
  if (!solveSpectralGMRESUpperTriangular(
          hessenberg, maxIterations, bestColumns, rotatedRhs,
          options.linearPivotTolerance, bestY))
    return result;
  result.solution.assign(n, 0.0);
  for (std::size_t col = 0; col < bestColumns; ++col) {
    for (std::size_t i = 0; i < n; ++i)
      result.solution[i] += bestY[col] * basis[col * n + i];
  }
  if (!applySpectralPreconditioner(preconditioner, result.solution,
                                   options.preconditionerPivotTolerance))
    return SpectralGMRESResult{};
  projectSpectralField(problem, result.solution);
  return result;
}

inline bool buildDenseSpectralJacobianByJVP(
    const SpectralResidualProblem &problem, const std::vector<double> &values,
    const SpectralEllipticSolveOptions &options,
    std::vector<double> &jacobian) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  const std::size_t n = grid.size();
  if (values.size() != n)
    throw std::runtime_error("spectral Newton state size mismatch");
  if (n > options.denseJacobianMaxUnknowns)
    return false;

  jacobian.assign(n * n, 0.0);
  std::vector<double> direction(n, 0.0);
  for (std::size_t col = 0; col < n; ++col) {
    direction[col] = 1.0;
    const auto jvp =
        evaluateSpectralJacobianVectorProduct(problem, values, direction,
                                              options.jvpOptions);
    direction[col] = 0.0;
    if (!jvp.finite || jvp.values.size() != n)
      return false;
    for (std::size_t row = 0; row < n; ++row)
      jacobian[row * n + col] = jvp.values[row];
  }
  return true;
}

inline bool validateSpectralSystemSolveLayout(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> values,
    std::size_t pointsPerField) {
  if (values.empty() || system.equations.size() != values.size())
    return false;
  for (const auto &field : values) {
    if (field.size() != pointsPerField)
      return false;
  }
  std::vector<bool> seenUnknown(values.size(), false);
  for (const auto &equation : system.equations) {
    if (equation.unknownIndex >= values.size() ||
        seenUnknown[equation.unknownIndex]) {
      return false;
    }
    seenUnknown[equation.unknownIndex] = true;
  }
  return true;
}

inline std::vector<std::vector<double>>
unflattenSpectralSystemUnknownVectorToFields(
    std::span<const double> values, std::size_t fieldCount,
    std::size_t pointsPerField) {
  if (values.size() != fieldCount * pointsPerField) {
    throw std::runtime_error("spectral residual system vector size mismatch");
  }
  std::vector<std::vector<double>> out(
      fieldCount, std::vector<double>(pointsPerField, 0.0));
  for (std::size_t field = 0; field < fieldCount; ++field) {
    const std::size_t offset = field * pointsPerField;
    for (std::size_t p = 0; p < pointsPerField; ++p)
      out[field][p] = values[offset + p];
  }
  return out;
}

inline std::vector<double> mapSpectralSystemEquationVectorToUnknownOrder(
    const SpectralResidualSystemProblem &system, std::span<const double> values,
    std::size_t fieldCount, std::size_t pointsPerField) {
  if (system.equations.size() != fieldCount ||
      values.size() != system.equations.size() * pointsPerField) {
    throw std::runtime_error("spectral residual system vector size mismatch");
  }
  std::vector<double> out(fieldCount * pointsPerField, 0.0);
  for (std::size_t equation = 0; equation < system.equations.size();
       ++equation) {
    const std::size_t unknown = system.equations[equation].unknownIndex;
    if (unknown >= fieldCount)
      throw std::runtime_error(
          "spectral residual system equation unknown index out of range");
    const std::size_t srcOffset = equation * pointsPerField;
    const std::size_t dstOffset = unknown * pointsPerField;
    for (std::size_t p = 0; p < pointsPerField; ++p)
      out[dstOffset + p] = values[srcOffset + p];
  }
  return out;
}

inline bool buildSpectralSystemDiagonalPreconditionerByJVP(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> values,
    const SpectralEllipticSolveOptions &options,
    SpectralLinearPreconditioner &preconditioner) {
  preconditioner = {};
  if (!spectralPreconditionerRequested(options))
    return true;
  if (options.gmresPreconditioner != SpectralPreconditionerKind::DiagonalJVP)
    return false;

  const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
  const std::size_t pointsPerField = grid.size();
  if (!validateSpectralSystemSolveLayout(system, values, pointsPerField))
    return false;

  const std::size_t fieldCount = values.size();
  const std::size_t equationCount = system.equations.size();
  const std::size_t n = equationCount * pointsPerField;
  preconditioner.kind = SpectralPreconditionerKind::DiagonalJVP;
  preconditioner.inverseDiagonal.assign(n, 1.0);
  std::vector<std::vector<double>> directionFields(
      fieldCount, std::vector<double>(pointsPerField, 0.0));
  for (std::size_t equation = 0; equation < equationCount; ++equation) {
    const std::size_t unknown = system.equations[equation].unknownIndex;
    if (unknown >= fieldCount)
      return false;
    for (std::size_t p = 0; p < pointsPerField; ++p) {
      directionFields[unknown][p] = 1.0;
      const auto jvp = evaluateSpectralResidualSystemJacobianVectorProduct(
          system, values,
          std::span<const std::vector<double>>(directionFields.data(),
                                               directionFields.size()),
          options.jvpOptions);
      directionFields[unknown][p] = 0.0;
      if (!jvp.finite || jvp.values.size() != n)
        return false;
      const std::size_t equationRow = equation * pointsPerField + p;
      const std::size_t unknownRow = unknown * pointsPerField + p;
      const double diagonal = jvp.values[equationRow];
      if (std::isfinite(diagonal) &&
          std::fabs(diagonal) > options.preconditionerPivotTolerance) {
        preconditioner.inverseDiagonal[unknownRow] = 1.0 / diagonal;
      }
    }
  }
  return true;
}

inline double spectralPreconditionerShiftForBlock(
    const SpectralEllipticSolveOptions &options, std::size_t block) {
  if (block < options.preconditionerLaplacianShifts.size())
    return options.preconditionerLaplacianShifts[block];
  return options.preconditionerLaplacianShift;
}

inline bool buildSpectralSystemPreconditioner(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> values,
    const SpectralEllipticSolveOptions &options,
    SpectralLinearPreconditioner &preconditioner) {
  preconditioner = {};
  if (!spectralPreconditionerRequested(options))
    return true;
  if (options.gmresPreconditioner == SpectralPreconditionerKind::DiagonalJVP) {
    return buildSpectralSystemDiagonalPreconditionerByJVP(
        system, values, options, preconditioner);
  }
  if (options.gmresPreconditioner ==
      SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift) {
    const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
    const std::size_t fieldCount = values.size();
    if (!validateSpectralSystemSolveLayout(system, values, grid.size()))
      return false;
    preconditioner.kind =
        SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift;
    preconditioner.blockSize = grid.size();
    preconditioner.relaxationSweeps = options.preconditionerRelaxationSweeps;
    preconditioner.relaxationOmega = options.preconditionerRelaxationOmega;
    preconditioner.sparseBlocks.resize(fieldCount);
    for (const auto &equation : system.equations) {
      preconditioner.sparseBlocks[equation.unknownIndex] =
          buildSpectralMappedFiniteDifferenceLaplacianShift(
              equation.problem,
              spectralPreconditionerShiftForBlock(options,
                                                  equation.unknownIndex),
              options.preconditionerPivotTolerance);
    }
    return true;
  }
  if (options.gmresPreconditioner ==
      SpectralPreconditionerKind::DenseLaplacianShift) {
    const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
    const std::size_t fieldCount = values.size();
    if (!validateSpectralSystemSolveLayout(system, values, grid.size()))
      return false;
    preconditioner.kind = SpectralPreconditionerKind::DenseLaplacianShift;
    preconditioner.blockSize = grid.size();
    preconditioner.denseBlocks.reserve(fieldCount);
    for (std::size_t block = 0; block < fieldCount; ++block) {
      preconditioner.denseBlocks.push_back(buildSpectralLaplacianShiftMatrix(
          grid, spectralPreconditionerShiftForBlock(options, block)));
    }
    return true;
  }
  if (options.gmresPreconditioner ==
      SpectralPreconditionerKind::ModalLaplacianShift) {
    const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
    const std::size_t fieldCount = values.size();
    if (!validateSpectralSystemSolveLayout(system, values, grid.size()))
      return false;
    preconditioner.kind = SpectralPreconditionerKind::ModalLaplacianShift;
    populateSpectralModalPreconditionerMetadata(grid, preconditioner);
    preconditioner.inverseDiagonal.reserve(fieldCount * grid.size());
    for (std::size_t block = 0; block < fieldCount; ++block) {
      appendSpectralModalLaplacianShiftBlock(
          grid, spectralPreconditionerShiftForBlock(options, block), options,
          preconditioner);
    }
    return true;
  }
  return false;
}

inline SpectralGMRESResult solveSpectralSystemGMRESByJVP(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> values, std::span<const double> rhs,
    const SpectralEllipticSolveOptions &options) {
  const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
  const std::size_t fieldCount = values.size();
  const std::size_t n = fieldCount * grid.size();
  SpectralGMRESResult result;
  result.solution.assign(n, 0.0);
  if (rhs.size() != n || options.gmresMaxIterations < 0 ||
      !validateSpectralSystemSolveLayout(system, values, grid.size())) {
    return result;
  }

  SpectralLinearPreconditioner preconditioner;
  if (!buildSpectralSystemPreconditioner(system, values, options,
                                         preconditioner))
    return result;
  result.usedPreconditioner =
      preconditioner.kind != SpectralPreconditionerKind::None;

  const auto rhsUnknownOrder = mapSpectralSystemEquationVectorToUnknownOrder(
      system, rhs, fieldCount, grid.size());
  if (options.gmresRestart > 0) {
    auto restarted = solveSpectralRestartedGMRES(
        n, rhsUnknownOrder, options,
        [&](const std::vector<double> &direction, std::vector<double> &out) {
          const auto directionFields =
              unflattenSpectralSystemUnknownVectorToFields(
                  direction, fieldCount, grid.size());
          const auto jvp = evaluateSpectralResidualSystemJacobianVectorProduct(
              system, values,
              std::span<const std::vector<double>>(directionFields.data(),
                                                   directionFields.size()),
              options.jvpOptions);
          if (!jvp.finite || jvp.values.size() != n)
            return false;
          out = mapSpectralSystemEquationVectorToUnknownOrder(
              system, jvp.values, fieldCount, grid.size());
          return out.size() == n && spectralVectorIsFinite(out);
        },
        [&](std::vector<double> &direction) {
          if (!applySpectralPreconditioner(
                  preconditioner, direction,
                  options.preconditionerPivotTolerance))
            return false;
          projectSpectralSystemUnknownVector(system, direction, fieldCount,
                                             grid.size());
          return true;
        });
    restarted.usedPreconditioner = result.usedPreconditioner;
    return restarted;
  }
  const double rhsEuclidean = spectralVectorEuclideanNorm(rhsUnknownOrder);
  const double rhsL2 =
      rhsEuclidean / std::sqrt(static_cast<double>(std::max<std::size_t>(1, n)));
  const double target = std::max(options.gmresTolerance,
                                 options.gmresRelativeTolerance * rhsL2);
  result.residualL2 = rhsL2;
  if (rhsL2 <= target) {
    result.converged = true;
    return result;
  }
  const std::size_t maxIterations = std::min<std::size_t>(
      n, static_cast<std::size_t>(options.gmresMaxIterations));
  if (maxIterations == 0)
    return result;

  std::vector<double> basis((maxIterations + 1) * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    basis[i] = rhsUnknownOrder[i] / rhsEuclidean;

  std::vector<double> hessenberg((maxIterations + 1) * maxIterations, 0.0);
  std::vector<double> arnoldiVector(n, 0.0);
  std::vector<double> cosines(maxIterations, 0.0);
  std::vector<double> sines(maxIterations, 0.0);
  std::vector<double> rotatedRhs(maxIterations + 1, 0.0);
  rotatedRhs[0] = rhsEuclidean;
  std::size_t bestColumns = 0;

  for (std::size_t col = 0; col < maxIterations; ++col) {
    std::span<const double> direction(&basis[col * n], n);
    std::vector<double> directionVector(direction.begin(), direction.end());
    if (!applySpectralPreconditioner(preconditioner, directionVector,
                                     options.preconditionerPivotTolerance))
      return result;
    projectSpectralSystemUnknownVector(system, directionVector, fieldCount,
                                       grid.size());
    const auto directionFields = unflattenSpectralSystemUnknownVectorToFields(
        directionVector, fieldCount, grid.size());
    const auto jvp = evaluateSpectralResidualSystemJacobianVectorProduct(
        system, values,
        std::span<const std::vector<double>>(directionFields.data(),
                                             directionFields.size()),
        options.jvpOptions);
    if (!jvp.finite || jvp.values.size() != n)
      return result;
    arnoldiVector = mapSpectralSystemEquationVectorToUnknownOrder(
        system, jvp.values, fieldCount, grid.size());

    for (std::size_t row = 0; row <= col; ++row) {
      std::span<const double> basisVector(&basis[row * n], n);
      const double h = spectralVectorDot(arnoldiVector, basisVector);
      hessenberg[row * maxIterations + col] = h;
      for (std::size_t i = 0; i < n; ++i)
        arnoldiVector[i] -= h * basisVector[i];
    }

    const double nextNorm = spectralVectorEuclideanNorm(arnoldiVector);
    hessenberg[(col + 1) * maxIterations + col] = nextNorm;
    if (nextNorm > options.linearPivotTolerance && col + 1 < maxIterations + 1) {
      for (std::size_t i = 0; i < n; ++i)
        basis[(col + 1) * n + i] = arnoldiVector[i] / nextNorm;
    }

    double projectedResidualL2 = std::numeric_limits<double>::infinity();
    if (!updateSpectralGMRESQR(
            hessenberg, maxIterations, col, cosines, sines, rotatedRhs,
            options.linearPivotTolerance, projectedResidualL2, n)) {
      return result;
    }

    const std::size_t columns = col + 1;
    result.iterations = static_cast<int>(columns);
    result.residualL2 = projectedResidualL2;
    bestColumns = columns;
    if (projectedResidualL2 <= target) {
      result.converged = true;
      break;
    }
    if (nextNorm <= options.linearPivotTolerance)
      break;
  }

  if (bestColumns == 0)
    return result;
  std::vector<double> bestY;
  if (!solveSpectralGMRESUpperTriangular(
          hessenberg, maxIterations, bestColumns, rotatedRhs,
          options.linearPivotTolerance, bestY))
    return result;
  result.solution.assign(n, 0.0);
  for (std::size_t col = 0; col < bestColumns; ++col) {
    for (std::size_t i = 0; i < n; ++i)
      result.solution[i] += bestY[col] * basis[col * n + i];
  }
  if (!applySpectralPreconditioner(preconditioner, result.solution,
                                   options.preconditionerPivotTolerance))
    return SpectralGMRESResult{};
  projectSpectralSystemUnknownVector(system, result.solution, fieldCount,
                                     grid.size());
  return result;
}

inline bool buildDenseSpectralSystemJacobianByJVP(
    const SpectralResidualSystemProblem &system,
    std::span<const std::vector<double>> values,
    const SpectralEllipticSolveOptions &options,
    std::vector<double> &jacobian) {
  const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
  const std::size_t fieldCount = values.size();
  const std::size_t n = fieldCount * grid.size();
  if (fieldCount == 0)
    throw std::runtime_error("spectral Newton system has no unknown fields");
  if (!validateSpectralSystemSolveLayout(system, values, grid.size()))
    throw std::runtime_error("spectral Newton system layout mismatch");
  if (n > options.denseJacobianMaxUnknowns)
    return false;

  jacobian.assign(n * n, 0.0);
  std::vector<std::vector<double>> directionFields(
      fieldCount, std::vector<double>(grid.size(), 0.0));
  for (std::size_t col = 0; col < n; ++col) {
    const std::size_t field = col / grid.size();
    const std::size_t point = col % grid.size();
    directionFields[field][point] = 1.0;
    const auto jvp = evaluateSpectralResidualSystemJacobianVectorProduct(
        system, values,
        std::span<const std::vector<double>>(directionFields.data(),
                                             directionFields.size()),
        options.jvpOptions);
    directionFields[field][point] = 0.0;
    if (!jvp.finite || jvp.values.size() != n)
      return false;
    for (std::size_t row = 0; row < n; ++row)
      jacobian[row * n + col] = jvp.values[row];
  }
  return true;
}

inline void updateSpectralSolveResidualState(
    SpectralEllipticSolveResult &result,
    const SpectralResidualAssemblyResult &residual) {
  result.finalResidualL2 = residual.l2Norm;
  result.finalResidualMaxAbs = residual.maxAbs;
  result.residualRatio =
      spectralResidualRatio(result.initialResidualL2, result.finalResidualL2);
  result.usedGeneratedGridKernel =
      result.usedGeneratedGridKernel || residual.usedGeneratedGridKernel;
}

inline void updateSpectralSolveResidualState(
    SpectralEllipticSolveResult &result,
    const SpectralResidualSystemAssemblyResult &residual) {
  result.finalResidualL2 = residual.l2Norm;
  result.finalResidualMaxAbs = residual.maxAbs;
  result.residualRatio =
      spectralResidualRatio(result.initialResidualL2, result.finalResidualL2);
  result.usedGeneratedGridKernel =
      result.usedGeneratedGridKernel || residual.usedGeneratedGridKernels;
}

inline SpectralEllipticSolveResult solveSpectralNewton(
    const SpectralResidualProblem &problem, std::vector<double> &values,
    const SpectralEllipticSolveOptions &options = {}) {
  const SpectralGrid3D &grid = requireSpectralResidualGrid(problem);
  SpectralEllipticSolveResult result;
  result.maxSteps = options.maxNewtonSteps;
  result.unknowns = grid.size();

  if (values.size() != grid.size() || options.maxNewtonSteps < 0 ||
      options.maxLineSearchSteps < 0 || !(options.initialDamping > 0.0) ||
      !(options.lineSearchReduction > 0.0 &&
        options.lineSearchReduction < 1.0) ||
      !(options.minDamping > 0.0) || !(options.linearPivotTolerance > 0.0) ||
      options.gmresMaxIterations < 0 || options.gmresRestart < 0 ||
      options.gmresTolerance < 0.0 || options.gmresRelativeTolerance < 0.0 ||
      !(options.preconditionerPivotTolerance > 0.0) ||
      !std::isfinite(options.preconditionerPivotTolerance) ||
      (options.gmresPreconditioner ==
           SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift &&
       (options.preconditionerRelaxationSweeps <= 0 ||
        !std::isfinite(options.preconditionerRelaxationOmega) ||
        !(options.preconditionerRelaxationOmega > 0.0 &&
          options.preconditionerRelaxationOmega <= 2.0)))) {
    result.status = SpectralEllipticSolveStatus::InvalidInput;
    return result;
  }

  projectSpectralField(problem, values);
  result.usedFieldProjector = spectralFieldProjectorEnabled(problem);
  auto residual = assembleSpectralResidual(problem, values);
  result.initialResidualL2 = residual.l2Norm;
  updateSpectralSolveResidualState(result, residual);
  if (!residual.finite) {
    result.status = SpectralEllipticSolveStatus::InvalidResidual;
    return result;
  }
  if (reachedSpectralResidualTarget(result, options)) {
    result.status = SpectralEllipticSolveStatus::Converged;
    return result;
  }

  const std::size_t n = grid.size();
  const bool denseAllowed =
      options.denseJacobianMaxUnknowns > 0 &&
      n <= options.denseJacobianMaxUnknowns;
  const bool useDense =
      options.linearSolver == SpectralLinearSolveKind::DenseJacobian ||
      (options.linearSolver == SpectralLinearSolveKind::Auto && denseAllowed);
  if (useDense && !denseAllowed) {
    result.status = SpectralEllipticSolveStatus::InvalidInput;
    return result;
  }

  for (int step = 1; step <= options.maxNewtonSteps; ++step) {
    std::vector<double> rhs(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      rhs[i] = -residual.values[i];

    std::vector<double> correction;
    if (useDense) {
      std::vector<double> jacobian;
      if (!buildDenseSpectralJacobianByJVP(problem, values, options,
                                           jacobian)) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      if (!solveDenseLinearSystem(std::move(jacobian), std::move(rhs),
                                  correction,
                                  options.linearPivotTolerance)) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      result.linearIterations += static_cast<int>(n);
      result.finalLinearResidualL2 = 0.0;
    } else {
      const auto linear =
          solveSpectralGMRESByJVP(problem, values, rhs, options);
      result.linearIterations += linear.iterations;
      result.finalLinearResidualL2 = linear.residualL2;
      result.usedMatrixFreeGMRES = true;
      result.usedPreconditioner =
          result.usedPreconditioner || linear.usedPreconditioner;
      if (!linear.converged || linear.solution.size() != n) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      correction = linear.solution;
    }
    projectSpectralField(problem, correction);

    bool accepted = false;
    double damping = options.initialDamping;
    std::vector<double> candidate(values.size(), 0.0);
    SpectralResidualAssemblyResult candidateResidual;
    for (int attempt = 0; attempt <= options.maxLineSearchSteps; ++attempt) {
      for (std::size_t i = 0; i < n; ++i)
        candidate[i] = values[i] + damping * correction[i];
      projectSpectralField(problem, candidate);
      candidateResidual = assembleSpectralResidual(problem, candidate);
      if (candidateResidual.finite &&
          (candidateResidual.l2Norm < residual.l2Norm ||
           candidateResidual.l2Norm <= options.residualTolerance ||
           residual.l2Norm == 0.0)) {
        accepted = true;
        break;
      }
      damping *= options.lineSearchReduction;
      if (damping < options.minDamping)
        break;
    }

    if (!accepted) {
      result.status = SpectralEllipticSolveStatus::LineSearchFailed;
      return result;
    }

    values = std::move(candidate);
    residual = std::move(candidateResidual);
    result.steps = step;
    result.lastDamping = damping;
    updateSpectralSolveResidualState(result, residual);
    if (!residual.finite) {
      result.status = SpectralEllipticSolveStatus::InvalidResidual;
      return result;
    }
    if (reachedSpectralResidualTarget(result, options)) {
      result.status = SpectralEllipticSolveStatus::Converged;
      return result;
    }
  }

  result.status = SpectralEllipticSolveStatus::MaxSteps;
  return result;
}

inline SpectralEllipticSolveResult solveSpectralNewton(
    const SpectralResidualSystemProblem &system,
    std::span<std::vector<double>> unknownFields,
    const SpectralEllipticSolveOptions &options = {}) {
  const SpectralGrid3D &grid = requireSpectralResidualSystemGrid(system);
  SpectralEllipticSolveResult result;
  result.maxSteps = options.maxNewtonSteps;
  result.unknowns = unknownFields.size() * grid.size();

  if (unknownFields.empty() || system.equations.empty() ||
      unknownFields.size() != system.equations.size() ||
      options.maxNewtonSteps < 0 || options.maxLineSearchSteps < 0 ||
      !(options.initialDamping > 0.0) ||
      !(options.lineSearchReduction > 0.0 &&
        options.lineSearchReduction < 1.0) ||
      !(options.minDamping > 0.0) || !(options.linearPivotTolerance > 0.0) ||
      options.gmresMaxIterations < 0 || options.gmresRestart < 0 ||
      options.gmresTolerance < 0.0 || options.gmresRelativeTolerance < 0.0 ||
      !(options.preconditionerPivotTolerance > 0.0) ||
      !std::isfinite(options.preconditionerPivotTolerance) ||
      (options.gmresPreconditioner ==
           SpectralPreconditionerKind::MappedFiniteDifferenceLaplacianShift &&
       (options.preconditionerRelaxationSweeps <= 0 ||
        !std::isfinite(options.preconditionerRelaxationOmega) ||
        !(options.preconditionerRelaxationOmega > 0.0 &&
          options.preconditionerRelaxationOmega <= 2.0)))) {
    result.status = SpectralEllipticSolveStatus::InvalidInput;
    return result;
  }
  for (const auto &field : unknownFields) {
    if (field.size() != grid.size()) {
      result.status = SpectralEllipticSolveStatus::InvalidInput;
      return result;
    }
  }
  if (!validateSpectralSystemSolveLayout(
          system,
          std::span<const std::vector<double>>(unknownFields.data(),
                                               unknownFields.size()),
          grid.size())) {
    result.status = SpectralEllipticSolveStatus::InvalidInput;
    return result;
  }

  projectSpectralSystemUnknownFields(system, unknownFields);
  result.usedFieldProjector = spectralSystemUsesFieldProjector(system);
  auto residual = assembleSpectralResidualSystem(system, unknownFields);
  result.initialResidualL2 = residual.l2Norm;
  updateSpectralSolveResidualState(result, residual);
  if (!residual.finite) {
    result.status = SpectralEllipticSolveStatus::InvalidResidual;
    return result;
  }
  if (reachedSpectralResidualTarget(result, options)) {
    result.status = SpectralEllipticSolveStatus::Converged;
    return result;
  }

  const std::size_t fieldCount = unknownFields.size();
  const std::size_t pointsPerField = grid.size();
  const std::size_t n = fieldCount * pointsPerField;
  const bool denseAllowed =
      options.denseJacobianMaxUnknowns > 0 &&
      n <= options.denseJacobianMaxUnknowns;
  const bool useDense =
      options.linearSolver == SpectralLinearSolveKind::DenseJacobian ||
      (options.linearSolver == SpectralLinearSolveKind::Auto && denseAllowed);
  if (useDense && !denseAllowed) {
    result.status = SpectralEllipticSolveStatus::InvalidInput;
    return result;
  }

  for (int step = 1; step <= options.maxNewtonSteps; ++step) {
    std::vector<double> rhs(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      rhs[i] = -residual.values[i];

    std::vector<double> correction;
    if (useDense) {
      std::vector<double> jacobian;
      if (!buildDenseSpectralSystemJacobianByJVP(
              system, std::span<const std::vector<double>>(
                          unknownFields.data(), unknownFields.size()),
              options, jacobian)) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      if (!solveDenseLinearSystem(std::move(jacobian), std::move(rhs),
                                  correction,
                                  options.linearPivotTolerance)) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      result.linearIterations += static_cast<int>(n);
      result.finalLinearResidualL2 = 0.0;
    } else {
      const auto linear = solveSpectralSystemGMRESByJVP(
          system,
          std::span<const std::vector<double>>(unknownFields.data(),
                                               unknownFields.size()),
          rhs, options);
      result.linearIterations += linear.iterations;
      result.finalLinearResidualL2 = linear.residualL2;
      result.usedMatrixFreeGMRES = true;
      result.usedPreconditioner =
          result.usedPreconditioner || linear.usedPreconditioner;
      if (!linear.converged || linear.solution.size() != n) {
        result.status = SpectralEllipticSolveStatus::LinearSolveFailed;
        return result;
      }
      correction = linear.solution;
    }
    projectSpectralSystemUnknownVector(system, correction, fieldCount,
                                       pointsPerField);

    bool accepted = false;
    double damping = options.initialDamping;
    std::vector<std::vector<double>> candidate(
        fieldCount, std::vector<double>(pointsPerField, 0.0));
    SpectralResidualSystemAssemblyResult candidateResidual;
    for (int attempt = 0; attempt <= options.maxLineSearchSteps; ++attempt) {
      for (std::size_t field = 0; field < fieldCount; ++field) {
        const std::size_t offset = field * pointsPerField;
        for (std::size_t p = 0; p < pointsPerField; ++p) {
          candidate[field][p] =
              unknownFields[field][p] + damping * correction[offset + p];
        }
      }
      projectSpectralSystemUnknownFields(
          system,
          std::span<std::vector<double>>(candidate.data(), candidate.size()));
      candidateResidual = assembleSpectralResidualSystem(
          system, std::span<const std::vector<double>>(candidate.data(),
                                                       candidate.size()));
      if (candidateResidual.finite &&
          (candidateResidual.l2Norm < residual.l2Norm ||
           candidateResidual.l2Norm <= options.residualTolerance ||
           residual.l2Norm == 0.0)) {
        accepted = true;
        break;
      }
      damping *= options.lineSearchReduction;
      if (damping < options.minDamping)
        break;
    }

    if (!accepted) {
      result.status = SpectralEllipticSolveStatus::LineSearchFailed;
      return result;
    }

    for (std::size_t field = 0; field < fieldCount; ++field)
      unknownFields[field] = std::move(candidate[field]);
    residual = std::move(candidateResidual);
    result.steps = step;
    result.lastDamping = damping;
    updateSpectralSolveResidualState(result, residual);
    if (!residual.finite) {
      result.status = SpectralEllipticSolveStatus::InvalidResidual;
      return result;
    }
    if (reachedSpectralResidualTarget(result, options)) {
      result.status = SpectralEllipticSolveStatus::Converged;
      return result;
    }
  }

  result.status = SpectralEllipticSolveStatus::MaxSteps;
  return result;
}


} // namespace tensorium_mlir::runtime
