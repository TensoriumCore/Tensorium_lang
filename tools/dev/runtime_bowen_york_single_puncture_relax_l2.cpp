#include "tensorium_mlir/Runtime/GeneratedHostStorage.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

using GeneratedHostStorage = tensorium_mlir::runtime::GeneratedHostStorage;

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

void initializeState(RuntimeViews views, std::int64_t n) {
  for (std::int64_t p = 0; p < n; ++p) {
    views.u[p] = 0.0;
    views.v[p] = 0.0;
    views.H[p] = 0.0;
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
        if (!std::isfinite(value))
          return value;
        sum += value * value;
        ++count;
      }
    }
  }

  if (count == 0)
    return 0.0;
  return std::sqrt(sum / static_cast<double>(count));
}

double maxAbsInterior(const double *values, std::int64_t nx, std::int64_t ny,
                      std::int64_t nz, std::int64_t radius) {
  double maxValue = 0.0;
  for (std::int64_t i = radius; i < nx - radius; ++i) {
    for (std::int64_t j = radius; j < ny - radius; ++j) {
      for (std::int64_t k = radius; k < nz - radius; ++k) {
        const double value = values[flatIndex(i, j, k, ny, nz)];
        if (!std::isfinite(value))
          return value;
        maxValue = std::fmax(maxValue, std::fabs(value));
      }
    }
  }
  return maxValue;
}

double readDoubleEnv(const char *name, double fallback) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
    return fallback;

  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(raw, &end);
  if (end == raw || *end != '\0' || errno == ERANGE ||
      !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid numeric environment value ") +
                             name + "=" + raw);
  }
  return value;
}

int readIntEnv(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
    return fallback;

  char *end = nullptr;
  errno = 0;
  const long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || errno == ERANGE || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid integer environment value ") +
                             name + "=" + raw);
  }
  return static_cast<int>(value);
}

bool readBoolEnv(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
    return fallback;
  if (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
      std::strcmp(raw, "yes") == 0 || std::strcmp(raw, "on") == 0)
    return true;
  if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 ||
      std::strcmp(raw, "no") == 0 || std::strcmp(raw, "off") == 0)
    return false;
  throw std::runtime_error(std::string("invalid boolean environment value ") +
                           name + "=" + raw);
}

bool isCsvOutput() {
  const char *raw = std::getenv("BY_OUTPUT");
  return raw && std::strcmp(raw, "csv") == 0;
}

bool shouldPrintCheckpoint(int step, int finalStep) {
  if (step == finalStep)
    return true;
  return step == 0 || step == 10 || step == 50 || step == 100 ||
         step == 250 || step == 500;
}

void printCheckpoint(int step, int finalStep, double residualL2, double maxU) {
  std::printf("[bowen-york-single-puncture-l2] checkpoint step=%d "
              "||H||2=%.17g max|u|=%.17g%s\n",
              step, residualL2, maxU, step == finalStep ? " final" : "");
}

} // namespace

int main() {
  const std::int64_t nx = 16;
  const std::int64_t ny = 16;
  const std::int64_t nz = 16;

  try {
    const bool csvOutput = isCsvOutput();
    const bool checkpoints = readBoolEnv("BY_CHECKPOINTS", !csvOutput);
    const bool expectZero = readBoolEnv("BY_EXPECT_ZERO", false);
    const bool failOnWeak = readBoolEnv("BY_FAIL_ON_WEAK", !csvOutput);

    const double c = readDoubleEnv("BY_C", 1.0);
    const double eps2 = readDoubleEnv("BY_EPS2", 1.0);
    const double eta = readDoubleEnv("BY_ETA", 2.0);
    const double mass = readDoubleEnv("BY_MASS", 1.0);
    const double px = readDoubleEnv("BY_PX", 0.2);
    const double x0 = readDoubleEnv("BY_X0", 7.5);
    const double y0 = readDoubleEnv("BY_Y0", 7.5);
    const double z0 = readDoubleEnv("BY_Z0", 7.5);
    const double dt = readDoubleEnv("BY_DT", 0.005);
    const int steps = readIntEnv("BY_STEPS", 1600);
    const double decreaseFactor = readDoubleEnv("BY_DECREASE_FACTOR", 0.7);
    const double growthFactor = readDoubleEnv("BY_GROWTH_FACTOR", 1.05);
    const double zeroTolerance = readDoubleEnv("BY_ZERO_TOL", 1.0e-12);

    GeneratedHostStorage storage(
        std::span<const tensorium_host_kernel_desc>(
            tensorium_host_kernels, TENSORIUM_HOST_KERNEL_COUNT),
        std::span<const tensorium_host_buffer_desc>(
            tensorium_host_buffers, TENSORIUM_HOST_BUFFER_COUNT),
        {nx, ny, nz});
    RuntimeViews views = bindViews(storage);
    initializeState(views, nx * ny * nz);

    const auto eulerUpdates = storage.eulerUpdatePairsFromDerivativePrefix();
    if (eulerUpdates.size() != 2) {
      std::fprintf(stderr, "Euler update plan mismatch: updates=%zu\n",
                   eulerUpdates.size());
      return 2;
    }

    const char *kernelSymbol =
        storage.findKernelPlan("tensorium_residual_grid_affine")
            ? "tensorium_residual_grid_affine"
            : "tensorium_rhs_grid_affine";
    const auto *plan = storage.findKernelPlan(kernelSymbol);
    if (!plan) {
      std::fprintf(stderr, "missing residual/rhs grid plan\n");
      return 2;
    }
    const std::int64_t radius =
        plan->stencilRadius > 0 ? plan->stencilRadius : 1;

    const std::span<const tensorium_host_kernel_adapter_desc> adapters(
        tensorium_host_kernel_adapters, TENSORIUM_HOST_KERNEL_ADAPTER_COUNT);
    // RhsGrid ABI exposes parameters in sorted name order.
    const double rhsParams[] = {c, eps2, eta, mass, px, x0, y0, z0};
    const tensorium_mlir::runtime::GeneratedHostGridSpacing spacing{
        1.0, 1.0, 1.0};

    if (!csvOutput) {
      std::printf("[bowen-york-single-puncture-l2] params eta=%.17g c=%.17g "
                  "dt=%.17g steps=%d px=%.17g eps2=%.17g\n",
                  eta, c, dt, steps, px, eps2);
    }

    storage.invoke(adapters, kernelSymbol,
                   std::span<const double>(rhsParams, 8), spacing);
    const double initialResidualL2 =
        l2InteriorField(views.H, nx, ny, nz, radius);
    if (checkpoints) {
      printCheckpoint(0, steps, initialResidualL2,
                      maxAbsInterior(views.u, nx, ny, nz, radius));
    }

    for (int step = 1; step <= steps; ++step) {
      storage.applyEulerUpdate(eulerUpdates, dt);
      storage.invoke(adapters, kernelSymbol,
                     std::span<const double>(rhsParams, 8), spacing);
      if (checkpoints && shouldPrintCheckpoint(step, steps)) {
        printCheckpoint(step, steps,
                        l2InteriorField(views.H, nx, ny, nz, radius),
                        maxAbsInterior(views.u, nx, ny, nz, radius));
      }
    }

    const double finalResidualL2 =
        l2InteriorField(views.H, nx, ny, nz, radius);
    const double maxU = maxAbsInterior(views.u, nx, ny, nz, radius);
    const double ratio =
        initialResidualL2 > 0.0
            ? finalResidualL2 / initialResidualL2
            : (finalResidualL2 == 0.0 ? 0.0
                                      : std::numeric_limits<double>::infinity());

    const char *status = "ok";
    bool hardFailure = false;
    if (!std::isfinite(initialResidualL2) || !std::isfinite(finalResidualL2) ||
        !std::isfinite(maxU)) {
      status = "invalid";
      hardFailure = true;
    } else if (expectZero) {
      if (initialResidualL2 > zeroTolerance ||
          finalResidualL2 > zeroTolerance || maxU > zeroTolerance) {
        status = "zero_fail";
        hardFailure = true;
      }
    } else if (!(initialResidualL2 > 0.0)) {
      status = "invalid_initial";
      hardFailure = true;
    } else if (!(maxU > 0.0)) {
      status = "static";
    } else if (!(ratio < decreaseFactor)) {
      status = ratio <= growthFactor ? "weak" : "grew";
    }

    if (csvOutput) {
      std::printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%s\n", eta, c,
                  initialResidualL2, finalResidualL2, ratio, maxU, status);
    } else {
      std::printf("[bowen-york-single-puncture-l2] initial ||H||2 = %.17g\n",
                  initialResidualL2);
      std::printf("[bowen-york-single-puncture-l2] final   ||H||2 = %.17g\n",
                  finalResidualL2);
      std::printf("[bowen-york-single-puncture-l2] residual ratio = %.17g\n",
                  ratio);
      std::printf("[bowen-york-single-puncture-l2] max |u| = %.17g\n",
                  maxU);
      std::printf("[bowen-york-single-puncture-l2] status = %s\n", status);
    }

    if (hardFailure || (failOnWeak && std::strcmp(status, "ok") != 0)) {
      std::fprintf(stderr,
                   "Bowen-York single-puncture validation status: %s\n",
                   status);
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "Bowen-York single-puncture runner failed: %s\n",
                 ex.what());
    return 2;
  }

  return 0;
}
