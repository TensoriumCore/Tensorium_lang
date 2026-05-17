#include "tensorium_mlir/Runtime/GeneratedHostStorage.h"

#include <cfloat>
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
  double *chi = nullptr;
  double *alpha = nullptr;
  double *beta = nullptr;
  double *B = nullptr;
  double *K = nullptr;
  double *gammatilde = nullptr;
  double *gammatildeU = nullptr;
  double *Atilde = nullptr;
  double *Gammahat = nullptr;
  double *dchi = nullptr;
  double *dalpha = nullptr;
  double *dbeta = nullptr;
  double *dB = nullptr;
  double *dK = nullptr;
  double *dgammatilde = nullptr;
  double *dAtilde = nullptr;
  double *dGammahat = nullptr;
};

std::int64_t flatIndex(std::int64_t i, std::int64_t j, std::int64_t k,
                       std::int64_t ny, std::int64_t nz) {
  return (i * ny + j) * nz + k;
}

int comp2(int i, int j) { return i * 3 + j; }

RuntimeViews bindViews(GeneratedHostStorage &storage) {
  RuntimeViews v;
  v.chi = storage.data("field:chi");
  v.alpha = storage.data("field:alpha");
  v.beta = storage.data("field:beta");
  v.B = storage.data("field:B");
  v.K = storage.data("field:K");
  v.gammatilde = storage.data("field:gammatilde");
  v.gammatildeU = storage.data("field:gammatildeU");
  v.Atilde = storage.data("field:Atilde");
  v.Gammahat = storage.data("field:Gammahat");
  v.dchi = storage.data("field:dchi");
  v.dalpha = storage.data("field:dalpha");
  v.dbeta = storage.data("field:dbeta");
  v.dB = storage.data("field:dB");
  v.dK = storage.data("field:dK");
  v.dgammatilde = storage.data("field:dgammatilde");
  v.dAtilde = storage.data("field:dAtilde");
  v.dGammahat = storage.data("field:dGammahat");
  return v;
}

double tolerance(double expected) {
  return 512.0 * DBL_EPSILON * std::fmax(1.0, std::fabs(expected));
}

bool checkValue(const char *name, double got, double expected) {
  const double diff = std::fabs(got - expected);
  const double tol = tolerance(expected);
  if (diff <= tol)
    return true;
  std::fprintf(stderr,
               "%s mismatch: got %.17g expected %.17g diff %.3e tol %.3e\n",
               name, got, expected, diff, tol);
  return false;
}

void printNonzero(const char *name, double value) {
  if (std::fabs(value) > 1.0e-14)
    std::printf("  %s = %.17g\n", name, value);
}

void printNonzeroVector(const char *name, const double *values,
                        std::int64_t n, std::int64_t p) {
  for (int i = 0; i < 3; ++i) {
    char label[96];
    std::snprintf(label, sizeof(label), "%s[%d]", name, i);
    printNonzero(label, values[(std::int64_t)i * n + p]);
  }
}

void printNonzeroMatrix(const char *name, const double *values,
                        std::int64_t n, std::int64_t p) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      char label[96];
      std::snprintf(label, sizeof(label), "%s[%d,%d]", name, i, j);
      printNonzero(label, values[(std::int64_t)comp2(i, j) * n + p]);
    }
  }
}

bool checkVectorZero(const char *name, const double *values, std::int64_t n,
                     std::int64_t p) {
  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    char label[96];
    std::snprintf(label, sizeof(label), "%s[%d]", name, i);
    ok &= checkValue(label, values[(std::int64_t)i * n + p], 0.0);
  }
  return ok;
}

bool checkMatrix(const char *name, const double *values,
                 const double expected[9], std::int64_t n, std::int64_t p) {
  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      char label[96];
      std::snprintf(label, sizeof(label), "%s[%d,%d]", name, i, j);
      ok &= checkValue(label, values[(std::int64_t)comp2(i, j) * n + p],
                       expected[comp2(i, j)]);
    }
  }
  return ok;
}

} // namespace

int main() {
  const std::int64_t nx = 3;
  const std::int64_t ny = 3;
  const std::int64_t nz = 3;
  const std::int64_t n = nx * ny * nz;
  const std::int64_t center = flatIndex(1, 1, 1, ny, nz);
  const double eta = 2.0;
  const double dt = 0.01;

  try {
    GeneratedHostStorage storage(
        std::span<const tensorium_host_kernel_desc>(
            tensorium_host_kernels, TENSORIUM_HOST_KERNEL_COUNT),
        std::span<const tensorium_host_buffer_desc>(
            tensorium_host_buffers, TENSORIUM_HOST_BUFFER_COUNT),
        {nx, ny, nz});
    RuntimeViews v = bindViews(storage);

    for (std::int64_t p = 0; p < n; ++p) {
      v.chi[p] = 1.0;
      v.alpha[p] = 1.0;
      v.K[p] = -1.0;

      v.gammatilde[(std::int64_t)comp2(0, 0) * n + p] = 1.0;
      v.gammatilde[(std::int64_t)comp2(1, 1) * n + p] = 1.0;
      v.gammatilde[(std::int64_t)comp2(2, 2) * n + p] = 1.0;
      v.gammatildeU[(std::int64_t)comp2(0, 0) * n + p] = 1.0;
      v.gammatildeU[(std::int64_t)comp2(1, 1) * n + p] = 1.0;
      v.gammatildeU[(std::int64_t)comp2(2, 2) * n + p] = 1.0;

      v.Atilde[(std::int64_t)comp2(0, 0) * n + p] = -1.0 / 3.0;
      v.Atilde[(std::int64_t)comp2(1, 1) * n + p] = -1.0 / 3.0;
      v.Atilde[(std::int64_t)comp2(2, 2) * n + p] = 2.0 / 3.0;
    }

    const auto eulerUpdates = storage.eulerUpdatePairsFromDerivativePrefix();
    if (eulerUpdates.size() != 8) {
      std::fprintf(stderr, "Euler update plan mismatch: updates=%zu\n",
                   eulerUpdates.size());
      return 2;
    }

    const std::span<const tensorium_host_kernel_adapter_desc> adapters(
        tensorium_host_kernel_adapters, TENSORIUM_HOST_KERNEL_ADAPTER_COUNT);
    const double rhsParams[] = {eta};
    storage.invoke(adapters, "tensorium_rhs_grid_affine",
                   std::span<const double>(rhsParams, 1), {1.0, 1.0, 1.0});

    std::printf("[runtime-kasner-euler] nonzero RHS components at center\n");
    printNonzero("dchi", v.dchi[center]);
    printNonzero("dalpha", v.dalpha[center]);
    printNonzeroVector("dbeta", v.dbeta, n, center);
    printNonzeroVector("dB", v.dB, n, center);
    printNonzero("dK", v.dK[center]);
    printNonzeroMatrix("dgammatilde", v.dgammatilde, n, center);
    printNonzeroMatrix("dAtilde", v.dAtilde, n, center);
    printNonzeroVector("dGammahat", v.dGammahat, n, center);

    storage.applyEulerUpdate(eulerUpdates, dt);

    std::printf("[runtime-kasner-euler] nonzero state components at center "
                "after dt=%.17g\n",
                dt);
    printNonzero("chi", v.chi[center]);
    printNonzero("alpha", v.alpha[center]);
    printNonzeroVector("beta", v.beta, n, center);
    printNonzeroVector("B", v.B, n, center);
    printNonzero("K", v.K[center]);
    printNonzeroMatrix("gammatilde", v.gammatilde, n, center);
    printNonzeroMatrix("gammatildeU", v.gammatildeU, n, center);
    printNonzeroMatrix("Atilde", v.Atilde, n, center);
    printNonzeroVector("Gammahat", v.Gammahat, n, center);

    const double expectedDgammatilde[9] = {
        2.0 / 3.0, 0.0, 0.0, 0.0, 2.0 / 3.0,
        0.0,       0.0, 0.0, -4.0 / 3.0,
    };
    const double expectedDAtilde[9] = {
        1.0 / 9.0, 0.0, 0.0, 0.0, 1.0 / 9.0,
        0.0,       0.0, 0.0, -14.0 / 9.0,
    };
    const double expectedGammatilde[9] = {
        1.0 + dt * 2.0 / 3.0, 0.0, 0.0,
        0.0, 1.0 + dt * 2.0 / 3.0, 0.0,
        0.0, 0.0, 1.0 - dt * 4.0 / 3.0,
    };
    const double expectedGammatildeU[9] = {
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    };
    const double expectedAtilde[9] = {
        -1.0 / 3.0 + dt / 9.0, 0.0, 0.0,
        0.0, -1.0 / 3.0 + dt / 9.0, 0.0,
        0.0, 0.0, 2.0 / 3.0 - dt * 14.0 / 9.0,
    };

    bool ok = true;
    ok &= checkValue("dchi", v.dchi[center], -2.0 / 3.0);
    ok &= checkValue("dalpha", v.dalpha[center], 2.0);
    ok &= checkVectorZero("dbeta", v.dbeta, n, center);
    ok &= checkVectorZero("dB", v.dB, n, center);
    ok &= checkValue("dK", v.dK[center], 1.0);
    ok &= checkMatrix("dgammatilde", v.dgammatilde, expectedDgammatilde, n,
                      center);
    ok &= checkMatrix("dAtilde", v.dAtilde, expectedDAtilde, n, center);
    ok &= checkVectorZero("dGammahat", v.dGammahat, n, center);

    ok &= checkValue("chi", v.chi[center], 1.0 - dt * 2.0 / 3.0);
    ok &= checkValue("alpha", v.alpha[center], 1.0 + 2.0 * dt);
    ok &= checkVectorZero("beta", v.beta, n, center);
    ok &= checkVectorZero("B", v.B, n, center);
    ok &= checkValue("K", v.K[center], -1.0 + dt);
    ok &= checkMatrix("gammatilde", v.gammatilde, expectedGammatilde, n,
                      center);
    ok &= checkMatrix("gammatildeU", v.gammatildeU, expectedGammatildeU, n,
                      center);
    ok &= checkMatrix("Atilde", v.Atilde, expectedAtilde, n, center);
    ok &= checkVectorZero("Gammahat", v.Gammahat, n, center);

    if (!ok) {
      std::fprintf(stderr, "runtime BSSN Kasner Euler iteration mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "runtime BSSN Kasner Euler runner failed: %s\n",
                 ex.what());
    return 2;
  }

  return 0;
}
