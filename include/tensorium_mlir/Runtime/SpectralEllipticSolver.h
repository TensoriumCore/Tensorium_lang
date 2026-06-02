#pragma once

#include "tensorium_mlir/Runtime/SpectralResidualJVP.h"

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

inline bool spectralPreconditionerRequested(
    const SpectralEllipticSolveOptions &options) {
  return options.gmresPreconditioner != SpectralPreconditionerKind::None;
}

struct SpectralLinearPreconditioner {
  SpectralPreconditionerKind kind = SpectralPreconditionerKind::None;
  std::vector<double> inverseDiagonal;
  std::vector<std::vector<double>> denseBlocks;
  std::size_t blockSize = 0;
};

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

  return false;
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
  return false;
}

inline bool solveSpectralLeastSquaresNormalEquations(
    const std::vector<double> &hessenberg, std::size_t rows, std::size_t cols,
    std::size_t leadingDim, double beta, double pivotTolerance,
    std::vector<double> &solution, double &residualL2,
    std::size_t vectorSize) {
  std::vector<double> normal(cols * cols, 0.0);
  std::vector<double> rhs(cols, 0.0);
  for (std::size_t col = 0; col < cols; ++col) {
    rhs[col] = beta * hessenberg[col];
    for (std::size_t other = 0; other < cols; ++other) {
      double sum = 0.0;
      for (std::size_t row = 0; row < rows; ++row)
        sum += hessenberg[row * leadingDim + col] *
               hessenberg[row * leadingDim + other];
      normal[col * cols + other] = sum;
    }
  }

  if (!solveDenseLinearSystem(std::move(normal), std::move(rhs), solution,
                              pivotTolerance))
    return false;

  double residualSquared = 0.0;
  for (std::size_t row = 0; row < rows; ++row) {
    double value = row == 0 ? beta : 0.0;
    for (std::size_t col = 0; col < cols; ++col)
      value -= hessenberg[row * leadingDim + col] * solution[col];
    residualSquared += value * value;
  }
  residualL2 =
      std::sqrt(std::max(0.0, residualSquared) /
                static_cast<double>(std::max<std::size_t>(1, vectorSize)));
  return std::isfinite(residualL2);
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
  std::vector<double> y;
  std::vector<double> bestY;
  std::size_t bestColumns = 0;

  for (std::size_t col = 0; col < maxIterations; ++col) {
    std::span<const double> direction(&basis[col * n], n);
    std::vector<double> directionVector(direction.begin(), direction.end());
    if (!applySpectralPreconditioner(preconditioner, directionVector,
                                     options.preconditionerPivotTolerance))
      return result;
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

    const std::size_t columns = col + 1;
    const std::size_t rows = col + 2;
    double projectedResidualL2 = std::numeric_limits<double>::infinity();
    if (!solveSpectralLeastSquaresNormalEquations(
            hessenberg, rows, columns, maxIterations, rhsEuclidean,
            options.linearPivotTolerance, y, projectedResidualL2, n)) {
      return result;
    }

    result.iterations = static_cast<int>(columns);
    result.residualL2 = projectedResidualL2;
    bestY = y;
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
  result.solution.assign(n, 0.0);
  for (std::size_t col = 0; col < bestColumns; ++col) {
    for (std::size_t i = 0; i < n; ++i)
      result.solution[i] += bestY[col] * basis[col * n + i];
  }
  if (!applySpectralPreconditioner(preconditioner, result.solution,
                                   options.preconditionerPivotTolerance))
    return SpectralGMRESResult{};
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
  std::vector<double> y;
  std::vector<double> bestY;
  std::size_t bestColumns = 0;

  for (std::size_t col = 0; col < maxIterations; ++col) {
    std::span<const double> direction(&basis[col * n], n);
    std::vector<double> directionVector(direction.begin(), direction.end());
    if (!applySpectralPreconditioner(preconditioner, directionVector,
                                     options.preconditionerPivotTolerance))
      return result;
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

    const std::size_t columns = col + 1;
    const std::size_t rows = col + 2;
    double projectedResidualL2 = std::numeric_limits<double>::infinity();
    if (!solveSpectralLeastSquaresNormalEquations(
            hessenberg, rows, columns, maxIterations, rhsEuclidean,
            options.linearPivotTolerance, y, projectedResidualL2, n)) {
      return result;
    }

    result.iterations = static_cast<int>(columns);
    result.residualL2 = projectedResidualL2;
    bestY = y;
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
  result.solution.assign(n, 0.0);
  for (std::size_t col = 0; col < bestColumns; ++col) {
    for (std::size_t i = 0; i < n; ++i)
      result.solution[i] += bestY[col] * basis[col * n + i];
  }
  if (!applySpectralPreconditioner(preconditioner, result.solution,
                                   options.preconditionerPivotTolerance))
    return SpectralGMRESResult{};
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
      options.maxLineSearchSteps < 0 ||
      !(options.initialDamping > 0.0) ||
      !(options.lineSearchReduction > 0.0 &&
        options.lineSearchReduction < 1.0) ||
      !(options.minDamping > 0.0) ||
      !(options.linearPivotTolerance > 0.0) ||
      options.gmresMaxIterations < 0 || options.gmresTolerance < 0.0 ||
      options.gmresRelativeTolerance < 0.0 ||
      !(options.preconditionerPivotTolerance > 0.0) ||
      !std::isfinite(options.preconditionerPivotTolerance)) {
    result.status = SpectralEllipticSolveStatus::InvalidInput;
    return result;
  }

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

    bool accepted = false;
    double damping = options.initialDamping;
    std::vector<double> candidate(values.size(), 0.0);
    SpectralResidualAssemblyResult candidateResidual;
    for (int attempt = 0; attempt <= options.maxLineSearchSteps; ++attempt) {
      for (std::size_t i = 0; i < n; ++i)
        candidate[i] = values[i] + damping * correction[i];
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
      !(options.minDamping > 0.0) ||
      !(options.linearPivotTolerance > 0.0) ||
      options.gmresMaxIterations < 0 || options.gmresTolerance < 0.0 ||
      options.gmresRelativeTolerance < 0.0 ||
      !(options.preconditionerPivotTolerance > 0.0) ||
      !std::isfinite(options.preconditionerPivotTolerance)) {
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
