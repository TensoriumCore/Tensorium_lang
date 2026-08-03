#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void tensorium_rhs_grid_affine(
    int64_t nx, int64_t ny, int64_t nz, double dr, double dtheta, double dphi,
    double *chi_alloc, double *chi_aligned, int64_t chi_offset,
    int64_t chi_size, int64_t chi_stride, double *gamma_alloc,
    double *gamma_aligned, int64_t gamma_offset, int64_t gamma_size,
    int64_t gamma_stride, double *atilde_alloc, double *atilde_aligned,
    int64_t atilde_offset, int64_t atilde_size, int64_t atilde_stride,
    double *alpha_alloc, double *alpha_aligned, int64_t alpha_offset,
    int64_t alpha_size, int64_t alpha_stride, double *chi_rhs_alloc,
    double *chi_rhs_aligned, int64_t chi_rhs_offset, int64_t chi_rhs_size,
    int64_t chi_rhs_stride, double *gamma_rhs_alloc, double *gamma_rhs_aligned,
    int64_t gamma_rhs_offset, int64_t gamma_rhs_size, int64_t gamma_rhs_stride);

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

int main(void) {
  const int64_t nx = 10;
  const int64_t ny = 10;
  const int64_t nz = 10;
  const int64_t n = nx * ny * nz;
  const int64_t ci = nx / 2;
  const int64_t cj = ny / 2;
  const int64_t ck = nz / 2;
  const int64_t cidx = flat_index(ci, cj, ck, ny, nz);

  const double chi0 = 1.5;
  const double alpha0 = 2.0;
  const double dr = 0.1;
  const double dtheta = 0.1;
  const double dphi = 0.1;

  double *chi = (double *)calloc((size_t)n, sizeof(double));
  double *gamma = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *atilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *alpha = (double *)calloc((size_t)n, sizeof(double));
  double *chiRhs = (double *)calloc((size_t)n, sizeof(double));
  double *gammaRhs = (double *)calloc((size_t)(9 * n), sizeof(double));
  if (!chi || !gamma || !atilde || !alpha || !chiRhs || !gammaRhs) {
    fprintf(stderr, "allocation failure\n");
    free(chi);
    free(gamma);
    free(atilde);
    free(alpha);
    free(chiRhs);
    free(gammaRhs);
    return 2;
  }

  for (int64_t p = 0; p < n; ++p) {
    chi[p] = chi0;
    alpha[p] = alpha0;
    for (int c = 0; c < 9; ++c) {
      atilde[(int64_t)c * n + p] = (double)(c + 1);
      gamma[(int64_t)c * n + p] = 0.0;
    }
  }

  tensorium_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, chi, chi, 0, n, 1,
                            gamma, gamma, 0, 9 * n, 1, atilde, atilde, 0, 9 * n,
                            1, alpha, alpha, 0, n, 1, chiRhs, chiRhs, 0, n, 1,
                            gammaRhs, gammaRhs, 0, 9 * n, 1);

  const double expectedChi = -2.0 * alpha0 * chi0;
  const double gotChi = chiRhs[cidx];
  printf("[ll-smoke] BSSN reduced center dt(chi) got=%.17g expected=%.17g\n",
         gotChi, expectedChi);

  int ok = almost_equal(gotChi, expectedChi, 1e-12, 1e-12);

  for (int c = 0; c < 9; ++c) {
    const double expected = -2.0 * alpha0 * (double)(c + 1);
    const double got = gammaRhs[(int64_t)c * n + cidx];
    if (c < 3) {
      printf("[ll-smoke] BSSN reduced center dt(gamma)[%d] got=%.17g "
             "expected=%.17g\n",
             c, got, expected);
    }
    ok &= almost_equal(got, expected, 1e-12, 1e-12);
  }

  free(chi);
  free(gamma);
  free(atilde);
  free(alpha);
  free(chiRhs);
  free(gammaRhs);

  if (!ok) {
    fprintf(stderr, "BSSN reduced RHS mismatch\n");
    return 3;
  }
  return 0;
}
