#include "tensorium_mlir/Runtime/EllipticRelaxation.h"

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

struct CheckpointObserverContext {
  RuntimeViews views;
  GeneratedHostGridShape shape;
  bool checkpoints = false;
};

void observeCheckpoint(const tensorium_mlir::runtime::EllipticSolveResult &result,
                       GeneratedHostStorage &, void *rawContext) {
  auto *context = static_cast<CheckpointObserverContext *>(rawContext);
  if (!context || !context->checkpoints ||
      !shouldPrintCheckpoint(result.steps, result.maxSteps))
    return;
  printCheckpoint(result.steps, result.maxSteps, result.finalResidualL2,
                  tensorium_mlir::runtime::maxAbsInteriorField(
                      context->views.u, context->shape,
                      result.stencilRadius));
}

} // namespace

int main() {
  const std::int64_t nx = 16;
  const std::int64_t ny = 16;
  const std::int64_t nz = 16;
  const GeneratedHostGridShape shape{nx, ny, nz};

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
        shape);
    RuntimeViews views = bindViews(storage);
    initializeState(views, nx * ny * nz);

    const std::span<const tensorium_host_kernel_adapter_desc> adapters(
        tensorium_host_kernel_adapters, TENSORIUM_HOST_KERNEL_ADAPTER_COUNT);
    // RhsGrid ABI exposes parameters in sorted name order.
    const double rhsParams[] = {c, eps2, eta, mass, px, x0, y0, z0};

    if (!csvOutput) {
      std::printf("[bowen-york-single-puncture-l2] params eta=%.17g c=%.17g "
                  "dt=%.17g steps=%d px=%.17g eps2=%.17g\n",
                  eta, c, dt, steps, px, eps2);
    }

    CheckpointObserverContext observerContext{views, shape, checkpoints};
    tensorium_mlir::runtime::EllipticSolveOptions solveOptions;
    solveOptions.dt = dt;
    solveOptions.maxSteps = steps;
    solveOptions.residualRatioTarget = expectZero ? 0.0 : decreaseFactor;
    solveOptions.residualTolerance = expectZero ? zeroTolerance : 0.0;
    solveOptions.expectedEulerUpdateCount = 2;
    solveOptions.observer = observeCheckpoint;
    solveOptions.observerUserData = &observerContext;

    const auto solveResult = tensorium_mlir::runtime::solveExplicitEulerRelaxation(
        storage, adapters, std::span<const double>(rhsParams, 8), views.H,
        solveOptions);
    const double initialResidualL2 = solveResult.initialResidualL2;
    const double finalResidualL2 = solveResult.finalResidualL2;
    const double maxU =
        tensorium_mlir::runtime::maxAbsInteriorField(
            views.u, shape, solveResult.stencilRadius);
    const double ratio = solveResult.residualRatio;

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
      std::printf("[bowen-york-single-puncture-l2] steps = %d\n",
                  solveResult.steps);
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
