#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef TENSORIUM_GENERATED_HOST_H
#error "compile this runner with -include <generated Tensorium host header>"
#endif

namespace {

struct StorageBuffer {
  std::string key;
  std::int64_t componentCount = 1;
  std::int64_t scalarCount = 0;
  std::int64_t scalarOffset = 0;
};

class GeneratedUniformStorage {
public:
  explicit GeneratedUniformStorage(std::int64_t nPoints) : nPoints_(nPoints) {
    if (nPoints_ <= 0)
      throw std::invalid_argument("nPoints must be positive");

    std::int64_t total = 0;
    for (std::int64_t i = 0; i < TENSORIUM_HOST_BUFFER_COUNT; ++i) {
      const tensorium_host_buffer_desc &desc = tensorium_host_buffers[i];
      const std::string key = storageKey(desc);
      auto found = indexByKey_.find(key);
      if (found != indexByKey_.end()) {
        StorageBuffer &existing = buffers_[found->second];
        if (existing.componentCount != desc.component_count)
          throw std::runtime_error("component count mismatch for " + key);
        continue;
      }

      StorageBuffer buffer;
      buffer.key = key;
      buffer.componentCount = desc.component_count;
      buffer.scalarCount = desc.component_count * nPoints_;
      buffer.scalarOffset = total;
      total += buffer.scalarCount;

      const std::size_t index = buffers_.size();
      indexByKey_.emplace(buffer.key, index);
      buffers_.push_back(std::move(buffer));
    }

    arena_.assign(static_cast<std::size_t>(total), 0.0);
  }

  std::size_t dataAllocationCount() const { return arena_.empty() ? 0u : 1u; }
  std::size_t bufferCount() const { return buffers_.size(); }
  std::int64_t totalScalars() const {
    return static_cast<std::int64_t>(arena_.size());
  }

  double *data(const char *key) {
    auto found = indexByKey_.find(key);
    if (found == indexByKey_.end())
      throw std::runtime_error(std::string("missing runtime buffer: ") + key);
    const StorageBuffer &buffer = buffers_[found->second];
    return arena_.data() + buffer.scalarOffset;
  }

private:
  static std::string storageKey(const tensorium_host_buffer_desc &desc) {
    const char *prefix =
        desc.role == TENSORIUM_HOST_BUFFER_ROLE_COORDINATE ? "coord:" : "field:";
    return std::string(prefix) + desc.name;
  }

  std::int64_t nPoints_ = 0;
  std::vector<double> arena_;
  std::vector<StorageBuffer> buffers_;
  std::unordered_map<std::string, std::size_t> indexByKey_;
};

struct RuntimeViews {
  double *r = nullptr;
  double *theta = nullptr;
  double *phi = nullptr;
  double *g = nullptr;
  double *gU = nullptr;
  double *alpha = nullptr;
  double *gamma = nullptr;
  double *gammaU = nullptr;
  double *chi = nullptr;
  double *beta = nullptr;
  double *B = nullptr;
  double *K = nullptr;
  double *gammatilde = nullptr;
  double *gammatildeU = nullptr;
  double *Atilde = nullptr;
  double *Gammahat = nullptr;
  double *Rcoord = nullptr;
  double *radialBasis = nullptr;
  double *dchi = nullptr;
  double *dalpha = nullptr;
  double *dbeta = nullptr;
  double *dB = nullptr;
  double *dK = nullptr;
  double *dgammatilde = nullptr;
  double *dAtilde = nullptr;
  double *dGammahat = nullptr;
  double *RicciAnalytic = nullptr;
  double *HessianAlpha = nullptr;
  double *DAtilde = nullptr;
  double *Hamiltonian = nullptr;
  double *Momentum = nullptr;
};

RuntimeViews bindViews(GeneratedUniformStorage &storage) {
  RuntimeViews v;
  v.r = storage.data("coord:r");
  v.theta = storage.data("coord:theta");
  v.phi = storage.data("coord:phi");
  v.g = storage.data("field:g");
  v.gU = storage.data("field:gU");
  v.alpha = storage.data("field:alpha");
  v.gamma = storage.data("field:gamma");
  v.gammaU = storage.data("field:gammaU");
  v.chi = storage.data("field:chi");
  v.beta = storage.data("field:beta");
  v.B = storage.data("field:B");
  v.K = storage.data("field:K");
  v.gammatilde = storage.data("field:gammatilde");
  v.gammatildeU = storage.data("field:gammatildeU");
  v.Atilde = storage.data("field:Atilde");
  v.Gammahat = storage.data("field:Gammahat");
  v.Rcoord = storage.data("field:Rcoord");
  v.radialBasis = storage.data("field:radialBasis");
  v.dchi = storage.data("field:dchi");
  v.dalpha = storage.data("field:dalpha");
  v.dbeta = storage.data("field:dbeta");
  v.dB = storage.data("field:dB");
  v.dK = storage.data("field:dK");
  v.dgammatilde = storage.data("field:dgammatilde");
  v.dAtilde = storage.data("field:dAtilde");
  v.dGammahat = storage.data("field:dGammahat");
  v.RicciAnalytic = storage.data("field:RicciAnalytic");
  v.HessianAlpha = storage.data("field:HessianAlpha");
  v.DAtilde = storage.data("field:DAtilde");
  v.Hamiltonian = storage.data("field:Hamiltonian");
  v.Momentum = storage.data("field:Momentum");
  return v;
}

std::int64_t flatIndex(std::int64_t i, std::int64_t j, std::int64_t k,
                       std::int64_t ny, std::int64_t nz) {
  return (i * ny + j) * nz + k;
}

int comp2(int i, int j) { return i * 3 + j; }
int comp3(int i, int j, int k) { return (i * 3 + j) * 3 + k; }

double exactTol(double expected) {
  return 4096.0 * DBL_EPSILON * std::max(1.0, std::fabs(expected));
}

bool checkValue(const char *name, double got, double expected) {
  const double diff = std::fabs(got - expected);
  const double tol = exactTol(expected);
  if (diff <= tol)
    return true;
  std::fprintf(stderr,
               "%s mismatch: got %.17g expected %.17g diff %.3e tol %.3e\n",
               name, got, expected, diff, tol);
  return false;
}

void expectedSchwarzschild(double m, double r, double theta, double *alpha,
                           double gamma[9], double gammaU[9],
                           double ricci[9], double hessianAlpha[9]) {
  const double s = std::sin(theta);
  const double f = 1.0 - 2.0 * m / r;
  for (int c = 0; c < 9; ++c) {
    gamma[c] = 0.0;
    gammaU[c] = 0.0;
    ricci[c] = 0.0;
    hessianAlpha[c] = 0.0;
  }
  *alpha = std::sqrt(f);
  gamma[comp2(0, 0)] = 1.0 / f;
  gamma[comp2(1, 1)] = r * r;
  gamma[comp2(2, 2)] = r * r * s * s;
  gammaU[comp2(0, 0)] = f;
  gammaU[comp2(1, 1)] = 1.0 / (r * r);
  gammaU[comp2(2, 2)] = 1.0 / (r * r * s * s);
  ricci[comp2(0, 0)] = -2.0 * m / (r * r * r * f);
  ricci[comp2(1, 1)] = m / r;
  ricci[comp2(2, 2)] = (m / r) * s * s;
  for (int c = 0; c < 9; ++c)
    hessianAlpha[c] = (*alpha) * ricci[c];
}

bool checkVectorZero(const char *name, const double *got, std::int64_t n,
                     std::int64_t p) {
  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    char label[96];
    std::snprintf(label, sizeof(label), "%s[%d]", name, i);
    ok &= checkValue(label, got[(std::int64_t)i * n + p], 0.0);
  }
  return ok;
}

bool checkMatrix(const char *name, const double *got, const double expected[9],
                 std::int64_t n, std::int64_t p) {
  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      char label[96];
      std::snprintf(label, sizeof(label), "%s[%d,%d]", name, i, j);
      ok &= checkValue(label, got[(std::int64_t)comp2(i, j) * n + p],
                       expected[comp2(i, j)]);
    }
  }
  return ok;
}

bool checkMatrixZero(const char *name, const double *got, std::int64_t n,
                     std::int64_t p) {
  const double zero[9] = {0.0};
  return checkMatrix(name, got, zero, n, p);
}

bool checkTensor3Zero(const char *name, const double *got, std::int64_t n,
                      std::int64_t p) {
  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        char label[96];
        std::snprintf(label, sizeof(label), "%s[%d,%d,%d]", name, i, j, k);
        ok &= checkValue(label, got[(std::int64_t)comp3(i, j, k) * n + p],
                         0.0);
      }
    }
  }
  return ok;
}

bool checkPoint(const RuntimeViews &v, std::int64_t p, std::int64_t n,
                double m) {
  double expAlpha = 0.0;
  double expGamma[9];
  double expGammaU[9];
  double expRicci[9];
  double expHessianAlpha[9];
  expectedSchwarzschild(m, v.r[p], v.theta[p], &expAlpha, expGamma,
                        expGammaU, expRicci, expHessianAlpha);

  bool ok = true;
  ok &= checkValue("alpha", v.alpha[p], expAlpha);
  ok &= checkMatrix("gamma", v.gamma, expGamma, n, p);
  ok &= checkMatrix("gammaU", v.gammaU, expGammaU, n, p);
  ok &= checkValue("chi", v.chi[p], 1.0);
  ok &= checkVectorZero("beta", v.beta, n, p);
  ok &= checkVectorZero("B", v.B, n, p);
  ok &= checkValue("K", v.K[p], 0.0);
  ok &= checkMatrix("gammatilde", v.gammatilde, expGamma, n, p);
  ok &= checkMatrix("gammatildeU", v.gammatildeU, expGammaU, n, p);
  ok &= checkMatrixZero("Atilde", v.Atilde, n, p);
  ok &= checkVectorZero("Gammahat", v.Gammahat, n, p);

  ok &= checkValue("dchi", v.dchi[p], 0.0);
  ok &= checkValue("dalpha", v.dalpha[p], 0.0);
  ok &= checkVectorZero("dbeta", v.dbeta, n, p);
  ok &= checkVectorZero("dB", v.dB, n, p);
  ok &= checkValue("dK", v.dK[p], 0.0);
  ok &= checkMatrixZero("dgammatilde", v.dgammatilde, n, p);
  ok &= checkMatrixZero("dAtilde", v.dAtilde, n, p);
  ok &= checkVectorZero("dGammahat", v.dGammahat, n, p);

  ok &= checkMatrix("RicciAnalytic", v.RicciAnalytic, expRicci, n, p);
  ok &= checkMatrix("HessianAlpha", v.HessianAlpha, expHessianAlpha, n, p);
  ok &= checkTensor3Zero("DAtilde", v.DAtilde, n, p);
  ok &= checkValue("Hamiltonian", v.Hamiltonian[p], 0.0);
  ok &= checkVectorZero("Momentum", v.Momentum, n, p);
  return ok;
}

} // namespace

int main() {
  const std::int64_t nx = 5;
  const std::int64_t ny = 5;
  const std::int64_t nz = 5;
  const std::int64_t n = nx * ny * nz;
  const std::int64_t ci = 2;
  const std::int64_t cj = 2;
  const std::int64_t ck = 2;
  const std::int64_t center = flatIndex(ci, cj, ck, ny, nz);

  const double m = 1.0;
  const double eta = 2.0;
  const double r0 = 8.0;
  const double theta0 = 0.7;
  const double phi0 = 0.2;
  const double dr = 0.25;
  const double dtheta = 0.15;
  const double dphi = 0.2;

  try {
    GeneratedUniformStorage storage(n);
    RuntimeViews v = bindViews(storage);

    if (storage.dataAllocationCount() != 1 || storage.bufferCount() != 31) {
      std::fprintf(stderr,
                   "runtime storage layout mismatch: allocations=%zu "
                   "unique_buffers=%zu\n",
                   storage.dataAllocationCount(), storage.bufferCount());
      return 2;
    }

    for (std::int64_t i = 0; i < nx; ++i) {
      for (std::int64_t j = 0; j < ny; ++j) {
        for (std::int64_t k = 0; k < nz; ++k) {
          const std::int64_t p = flatIndex(i, j, k, ny, nz);
          v.r[p] = r0 + static_cast<double>(i - ci) * dr;
          v.theta[p] = theta0 + static_cast<double>(j - cj) * dtheta;
          v.phi[p] = phi0 + static_cast<double>(k - ck) * dphi;
          v.Rcoord[p] = v.r[p];
          v.chi[p] = 1.0;
          v.radialBasis[(std::int64_t)comp2(0, 0) * n + p] = 1.0;
        }
      }
    }

    tensorium_call_init_grid_affine(m, v.r, v.theta, v.phi, v.alpha, v.gamma,
                                    v.gammaU, n);
    for (std::int64_t p = 0; p < n; ++p) {
      for (int c = 0; c < 9; ++c) {
        const std::int64_t idx = (std::int64_t)c * n + p;
        v.g[idx] = v.gamma[idx];
        v.gU[idx] = v.gammaU[idx];
        v.gammatilde[idx] = v.gamma[idx];
        v.gammatildeU[idx] = v.gammaU[idx];
      }
    }

    tensorium_call_rhs_grid_affine(
        nx, ny, nz, dr, dtheta, dphi, m, eta, v.g, v.gU, v.alpha, v.gamma,
        v.gammaU, v.chi, v.beta, v.B, v.K, v.gammatilde, v.gammatildeU,
        v.Atilde, v.Gammahat, v.Rcoord, v.radialBasis, v.dchi, v.dalpha,
        v.dbeta, v.dB, v.dK, v.dgammatilde, v.dAtilde, v.dGammahat,
        v.RicciAnalytic, v.HessianAlpha, v.DAtilde, v.Hamiltonian,
        v.Momentum);

    std::printf("[runtime-uniform] generated descriptor buffers=%lld "
                "unique_buffers=%zu total_scalars=%lld arena_allocations=%zu\n",
                (long long)TENSORIUM_HOST_BUFFER_COUNT, storage.bufferCount(),
                (long long)storage.totalScalars(),
                storage.dataAllocationCount());
    std::printf("[runtime-uniform] center r=%.17g theta=%.17g "
                "Hamiltonian=%.17g Momentum=[%.17g, %.17g, %.17g]\n",
                v.r[center], v.theta[center], v.Hamiltonian[center],
                v.Momentum[center], v.Momentum[n + center],
                v.Momentum[2 * n + center]);

    bool ok = true;
    for (std::int64_t i = 1; i < nx - 1; ++i) {
      for (std::int64_t j = 1; j < ny - 1; ++j) {
        for (std::int64_t k = 1; k < nz - 1; ++k) {
          const std::int64_t p = flatIndex(i, j, k, ny, nz);
          ok &= checkPoint(v, p, n, m);
        }
      }
    }

    if (!ok) {
      std::fprintf(stderr, "runtime uniform Schwarzschild BSSN mismatch\n");
      return 3;
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "runtime uniform runner failed: %s\n", ex.what());
    return 2;
  }

  return 0;
}
