#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void tensorium_rhs_grid_affine(
    int64_t nx, int64_t ny, int64_t nz, double dr, double dtheta, double dphi,
    double *k_alloc, double *k_aligned, int64_t k_offset, int64_t k_size,
    int64_t k_stride, double *atilde_alloc, double *atilde_aligned,
    int64_t atilde_offset, int64_t atilde_size, int64_t atilde_stride,
    double *gamma_alloc, double *gamma_aligned, int64_t gamma_offset,
    int64_t gamma_size, int64_t gamma_stride, double *ricci_alloc,
    double *ricci_aligned, int64_t ricci_offset, int64_t ricci_size,
    int64_t ricci_stride, double *riemann_alloc, double *riemann_aligned,
    int64_t riemann_offset, int64_t riemann_size, int64_t riemann_stride,
    double *datilde_alloc, double *datilde_aligned, int64_t datilde_offset,
    int64_t datilde_size, int64_t datilde_stride, double *alpha_alloc,
    double *alpha_aligned, int64_t alpha_offset, int64_t alpha_size,
    int64_t alpha_stride);

static int almost_equal(double got, double expected, double rel_tol,
                        double abs_tol) {
  const double diff = fabs(got - expected);
  const double scale = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
  const double tol = fmax(abs_tol, rel_tol * scale);
  return diff <= tol;
}

static int64_t flat_index(int64_t i, int64_t j, int64_t k, int64_t ny,
                          int64_t nz) {
  return (i * ny + j) * nz + k;
}

static int comp2(int i, int j) { return i * 3 + j; }
static int comp3(int i, int j, int k) { return (i * 3 + j) * 3 + k; }
static int comp4(int i, int j, int k, int l) { return ((i * 3 + j) * 3 + k) * 3 + l; }

static double init_riemann_component(int i, int j, int k, int l) {
  return (double)(1000 * i + 100 * j + 10 * k + l + 1);
}

int main(void) {
  const int64_t nx = 10;
  const int64_t ny = 10;
  const int64_t nz = 10;
  const int64_t n = nx * ny * nz;
  const int64_t ci = nx / 2;
  const int64_t cj = ny / 2;
  const int64_t ck = nz / 2;
  const int64_t cidx = flat_index(ci, cj, ck, ny, nz);

  const double k0 = 3.0;
  const double alpha0 = 2.0;
  const double dr = 0.1;
  const double dtheta = 0.1;
  const double dphi = 0.1;

  double *k = (double *)calloc((size_t)n, sizeof(double));
  double *atilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *gamma = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *ricci = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *riemann = (double *)calloc((size_t)(81 * n), sizeof(double));
  double *datilde = (double *)calloc((size_t)(27 * n), sizeof(double));
  double *alpha = (double *)calloc((size_t)n, sizeof(double));
  if (!k || !atilde || !gamma || !ricci || !riemann || !datilde || !alpha) {
    fprintf(stderr, "allocation failure\n");
    free(k);
    free(atilde);
    free(gamma);
    free(ricci);
    free(riemann);
    free(datilde);
    free(alpha);
    return 2;
  }

  for (int64_t p = 0; p < n; ++p) {
    k[p] = k0;
    alpha[p] = alpha0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        atilde[(int64_t)comp2(i, j) * n + p] = (double)(comp2(i, j) + 1);
        gamma[(int64_t)comp2(i, j) * n + p] = (i == j) ? 1.0 : 0.0;
        ricci[(int64_t)comp2(i, j) * n + p] = -777.0;
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k2 = 0; k2 < 3; ++k2) {
          for (int l = 0; l < 3; ++l) {
            riemann[(int64_t)comp4(i, j, k2, l) * n + p] =
                init_riemann_component(i, j, k2, l);
          }
        }
      }
    }
    for (int c = 0; c < 27; ++c) {
      datilde[(int64_t)c * n + p] = -555.0;
    }
  }

  tensorium_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, k, k, 0, n, 1,
                            atilde, atilde, 0, 9 * n, 1, gamma, gamma, 0,
                            9 * n, 1, ricci, ricci, 0, 9 * n, 1, riemann,
                            riemann, 0, 81 * n, 1, datilde, datilde, 0, 27 * n,
                            1, alpha, alpha, 0, n, 1);

  int ok = 1;

  const double expectedK = -alpha0 * k0 * k0 + alpha0 * (1.0 + 5.0 + 9.0);
  const double gotK = k[cidx];
  printf("[ll-smoke] BSSN minimal center dt(K) got=%.17g expected=%.17g\n",
         gotK, expectedK);
  ok &= almost_equal(gotK, expectedK, 1e-12, 1e-12);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const int c2 = comp2(i, j);
      const double expectedAtilde = alpha0 * (double)(c2 + 1);
      const double gotAtilde = atilde[(int64_t)c2 * n + cidx];
      if (i == 0 && j < 3) {
        printf("[ll-smoke] BSSN minimal center dt(Atilde)[0,%d] got=%.17g "
               "expected=%.17g\n",
               j, gotAtilde, expectedAtilde);
      }
      ok &= almost_equal(gotAtilde, expectedAtilde, 1e-12, 1e-12);

      double expectedRicci = 0.0;
      for (int s = 0; s < 3; ++s) {
        expectedRicci += init_riemann_component(s, i, s, j);
      }
      const double gotRicci = ricci[(int64_t)c2 * n + cidx];
      if (i == 0 && j < 3) {
        printf("[ll-smoke] BSSN minimal center dt(Ricci)[0,%d] got=%.17g "
               "expected=%.17g\n",
               j, gotRicci, expectedRicci);
      }
      ok &= almost_equal(gotRicci, expectedRicci, 1e-12, 1e-12);
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k2 = 0; k2 < 3; ++k2) {
        const double got = datilde[(int64_t)comp3(i, j, k2) * n + cidx];
        ok &= almost_equal(got, 0.0, 1e-12, 1e-12);
      }
    }
  }
  printf("[ll-smoke] BSSN minimal center dt(DAtilde) all components ~ 0\n");

  free(k);
  free(atilde);
  free(gamma);
  free(ricci);
  free(riemann);
  free(datilde);
  free(alpha);

  if (!ok) {
    fprintf(stderr, "BSSN minimal RHS mismatch\n");
    return 3;
  }
  return 0;
}
