#include "tensorium_mlir/Runtime/EllipticRelaxation.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

using GeneratedHostStorage = tensorium_mlir::runtime::GeneratedHostStorage;
using GeneratedHostGridShape = tensorium_mlir::runtime::GeneratedHostGridShape;

struct RuntimeViews {
  double *u = nullptr;
  double *v = nullptr;
  double *H = nullptr;
};

std::int64_t flatIndex(std::int64_t i, std::int64_t j, std::int64_t k,
                       std::int64_t ny, std::int64_t nz) {
  return (i * ny + j) * nz + k;
}

RuntimeViews bindViews(GeneratedHostStorage &storage) {
  RuntimeViews views;
  views.u = storage.data("field:u");
  views.v = storage.data("field:v");
  views.H = storage.data("field:H");
  return views;
}

void initializeSineMode(RuntimeViews views, std::int64_t nx, std::int64_t ny,
                        std::int64_t nz) {
  constexpr double pi = 3.14159265358979323846264338327950288;

  for (std::int64_t i = 0; i < nx; ++i) {
    const double sx = std::sin(pi * static_cast<double>(i) /
                               static_cast<double>(nx - 1));
    for (std::int64_t j = 0; j < ny; ++j) {
      const double sy = std::sin(pi * static_cast<double>(j) /
                                 static_cast<double>(ny - 1));
      for (std::int64_t k = 0; k < nz; ++k) {
        const double sz = std::sin(pi * static_cast<double>(k) /
                                   static_cast<double>(nz - 1));
        const std::int64_t p = flatIndex(i, j, k, ny, nz);
        views.u[p] = sx * sy * sz;
        views.v[p] = 0.0;
        views.H[p] = 0.0;
      }
    }
  }
}

double l2Interior(const double *values, std::int64_t nx, std::int64_t ny,
                  std::int64_t nz, std::int64_t radius) {
  double sum = 0.0;
  std::int64_t count = 0;
  for (std::int64_t i = radius; i < nx - radius; ++i) {
    for (std::int64_t j = radius; j < ny - radius; ++j) {
      for (std::int64_t k = radius; k < nz - radius; ++k) {
        const double value = values[flatIndex(i, j, k, ny, nz)];
        sum += value * value;
        ++count;
      }
    }
  }

  if (count == 0)
    return 0.0;
  return std::sqrt(sum / static_cast<double>(count));
}

} // namespace

int main() {
  const std::int64_t nx = 16;
  const std::int64_t ny = 16;
  const std::int64_t nz = 16;
  const double eta = 2.0;
  const double c = 0.05;
  const int steps = 400;
  const GeneratedHostGridShape shape{nx, ny, nz};

  try {
    GeneratedHostStorage storage(
        std::span<const tensorium_host_kernel_desc>(
            tensorium_host_kernels, TENSORIUM_HOST_KERNEL_COUNT),
        std::span<const tensorium_host_buffer_desc>(
            tensorium_host_buffers, TENSORIUM_HOST_BUFFER_COUNT),
        shape);
    RuntimeViews views = bindViews(storage);
    initializeSineMode(views, nx, ny, nz);

    const std::span<const tensorium_host_kernel_adapter_desc> adapters(
        tensorium_host_kernel_adapters, TENSORIUM_HOST_KERNEL_ADAPTER_COUNT);
    // RhsGrid ABI exposes parameters in sorted name order.
    const double rhsParams[] = {c, eta};
    tensorium_mlir::runtime::EllipticSolveOptions solveOptions;
    solveOptions.maxSteps = steps;
    solveOptions.residualRatioTarget = 0.7;
    solveOptions.jacobiWeight = 2.0 / 3.0;

    const auto solveResult =
        tensorium_mlir::runtime::solveWeightedJacobiRelaxation(
            storage, adapters, std::span<const double>(rhsParams, 2), views.u,
            views.H, solveOptions);
    const double initialL2 = solveResult.initialResidualL2;
    const double finalL2 = solveResult.finalResidualL2;
    std::printf("[poisson-relax-l2] initial ||H||2 = %.17g\n", initialL2);
    std::printf("[poisson-relax-l2] final   ||H||2 = %.17g\n", finalL2);
    std::printf("[poisson-relax-l2] ratio         = %.17g\n",
                solveResult.residualRatio);
    std::printf("[poisson-relax-l2] steps         = %d\n", solveResult.steps);

    if (!(initialL2 > 0.0)) {
      std::fprintf(stderr, "initial residual norm is not positive\n");
      return 3;
    }
    if (!(solveResult.residualRatio < 0.7)) {
      std::fprintf(stderr, "residual norm did not decrease enough\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "poisson relaxation runner failed: %s\n", ex.what());
    return 2;
  }

  return 0;
}
