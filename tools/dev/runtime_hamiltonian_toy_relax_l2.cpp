#include "tensorium_mlir/Runtime/GeneratedHostStorage.h"

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

struct RuntimeViews {
  double *u = nullptr;
  double *v = nullptr;
  double *psi = nullptr;
  double *A2 = nullptr;
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
  views.psi = storage.data("field:psi");
  views.A2 = storage.data("field:A2");
  views.H = storage.data("field:H");
  return views;
}

double sineMode(std::int64_t i, std::int64_t j, std::int64_t k,
                std::int64_t nx, std::int64_t ny, std::int64_t nz) {
  constexpr double pi = 3.14159265358979323846264338327950288;
  const double sx =
      std::sin(pi * static_cast<double>(i) / static_cast<double>(nx - 1));
  const double sy =
      std::sin(pi * static_cast<double>(j) / static_cast<double>(ny - 1));
  const double sz =
      std::sin(pi * static_cast<double>(k) / static_cast<double>(nz - 1));
  return sx * sy * sz;
}

void initializeManufacturedHamiltonian(RuntimeViews views, std::int64_t nx,
                                       std::int64_t ny, std::int64_t nz,
                                       double amplitude) {
  constexpr double pi = 3.14159265358979323846264338327950288;
  const double theta = pi / static_cast<double>(nx - 1);
  const double lambdaDiscrete = 6.0 * (1.0 - std::cos(theta));

  for (std::int64_t i = 0; i < nx; ++i) {
    for (std::int64_t j = 0; j < ny; ++j) {
      for (std::int64_t k = 0; k < nz; ++k) {
        const std::int64_t p = flatIndex(i, j, k, ny, nz);
        const double exact = amplitude * sineMode(i, j, k, nx, ny, nz);
        const double psiTotal = 1.0 + exact;
        views.u[p] = 0.0;
        views.v[p] = 0.0;
        views.psi[p] = 1.0;
        views.A2[p] = 8.0 * lambdaDiscrete * exact * std::pow(psiTotal, 7.0);
        views.H[p] = 0.0;
      }
    }
  }
}

double l2InteriorField(const double *values, std::int64_t nx, std::int64_t ny,
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

double l2InteriorError(const double *u, std::int64_t nx, std::int64_t ny,
                       std::int64_t nz, std::int64_t radius,
                       double amplitude) {
  double sum = 0.0;
  std::int64_t count = 0;
  for (std::int64_t i = radius; i < nx - radius; ++i) {
    for (std::int64_t j = radius; j < ny - radius; ++j) {
      for (std::int64_t k = radius; k < nz - radius; ++k) {
        const double exact =
            amplitude * sineMode(i, j, k, nx, ny, nz);
        const double error = u[flatIndex(i, j, k, ny, nz)] - exact;
        sum += error * error;
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
  const double amplitude = 0.1;
  const double eta = 2.0;
  const double c = 1.0;
  const double dt = 0.01;
  const int steps = 1200;

  try {
    GeneratedHostStorage storage(
        std::span<const tensorium_host_kernel_desc>(
            tensorium_host_kernels, TENSORIUM_HOST_KERNEL_COUNT),
        std::span<const tensorium_host_buffer_desc>(
            tensorium_host_buffers, TENSORIUM_HOST_BUFFER_COUNT),
        {nx, ny, nz});
    RuntimeViews views = bindViews(storage);
    initializeManufacturedHamiltonian(views, nx, ny, nz, amplitude);

    const auto eulerUpdates = storage.eulerUpdatePairsFromDerivativePrefix();
    if (eulerUpdates.size() != 2) {
      std::fprintf(stderr, "Euler update plan mismatch: updates=%zu\n",
                   eulerUpdates.size());
      return 2;
    }

    const char *kernelSymbol = nullptr;
    const char *candidateKernels[] = {
        "tensorium_residual_grid_parallel", "tensorium_residual_grid_affine",
        "tensorium_rhs_grid_parallel", "tensorium_rhs_grid_affine"};
    for (const char *candidate : candidateKernels) {
      if (storage.findKernelPlan(candidate)) {
        kernelSymbol = candidate;
        break;
      }
    }
    const auto *plan = kernelSymbol ? storage.findKernelPlan(kernelSymbol)
                                    : nullptr;
    if (!plan) {
      std::fprintf(stderr, "missing residual/rhs grid plan\n");
      return 2;
    }
    const std::int64_t radius =
        plan->stencilRadius > 0 ? plan->stencilRadius : 1;

    const std::span<const tensorium_host_kernel_adapter_desc> adapters(
        tensorium_host_kernel_adapters, TENSORIUM_HOST_KERNEL_ADAPTER_COUNT);
    // RhsGrid ABI exposes parameters in sorted name order.
    const double rhsParams[] = {c, eta};
    const tensorium_mlir::runtime::GeneratedHostGridSpacing spacing{
        1.0, 1.0, 1.0};

    storage.invoke(adapters, kernelSymbol,
                   std::span<const double>(rhsParams, 2), spacing);
    const double initialResidualL2 =
        l2InteriorField(views.H, nx, ny, nz, radius);
    const double initialErrorL2 =
        l2InteriorError(views.u, nx, ny, nz, radius, amplitude);

    for (int step = 0; step < steps; ++step) {
      storage.applyEulerUpdate(eulerUpdates, dt);
      storage.invoke(adapters, kernelSymbol,
                     std::span<const double>(rhsParams, 2), spacing);
    }

    const double finalResidualL2 =
        l2InteriorField(views.H, nx, ny, nz, radius);
    const double finalErrorL2 =
        l2InteriorError(views.u, nx, ny, nz, radius, amplitude);

    std::printf("[hamiltonian-toy-relax-l2] initial ||H||2   = %.17g\n",
                initialResidualL2);
    std::printf("[hamiltonian-toy-relax-l2] final   ||H||2   = %.17g\n",
                finalResidualL2);
    std::printf("[hamiltonian-toy-relax-l2] residual ratio  = %.17g\n",
                finalResidualL2 / initialResidualL2);
    std::printf("[hamiltonian-toy-relax-l2] initial ||err||2 = %.17g\n",
                initialErrorL2);
    std::printf("[hamiltonian-toy-relax-l2] final   ||err||2 = %.17g\n",
                finalErrorL2);

    if (!(initialResidualL2 > 0.0)) {
      std::fprintf(stderr, "initial residual norm is not positive\n");
      return 3;
    }
    if (!(finalResidualL2 < 0.6 * initialResidualL2)) {
      std::fprintf(stderr, "residual norm did not decrease enough\n");
      return 3;
    }
    if (!(finalErrorL2 < 0.6 * initialErrorL2)) {
      std::fprintf(stderr, "solution error did not decrease enough\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "Hamiltonian toy relaxation runner failed: %s\n",
                 ex.what());
    return 2;
  }

  return 0;
}
