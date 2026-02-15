#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void tensorium_rhs_grid_affine(
    int64_t nx, int64_t ny, int64_t nz, double dr, double dtheta, double dphi,
    double *chi_alloc, double *chi_aligned, int64_t chi_offset,
    int64_t chi_size, int64_t chi_stride, double *gamma_alloc,
    double *gamma_aligned, int64_t gamma_offset, int64_t gamma_size,
    int64_t gamma_stride, double *atilde_alloc, double *atilde_aligned,
    int64_t atilde_offset, int64_t atilde_size, int64_t atilde_stride,
    double *alpha_alloc, double *alpha_aligned, int64_t alpha_offset,
    int64_t alpha_size, int64_t alpha_stride);

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

static void eval_rhs(int64_t nx, int64_t ny, int64_t nz, double dr,
                     double dtheta, double dphi, const double *chi_state,
                     const double *gamma_state, const double *atilde,
                     const double *alpha, double *chi_rhs, double *gamma_rhs,
                     double *chi_work, double *gamma_work) {
  const int64_t n = nx * ny * nz;
  memcpy(chi_work, chi_state, (size_t)n * sizeof(double));
  memcpy(gamma_work, gamma_state, (size_t)(9 * n) * sizeof(double));

  tensorium_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, chi_work, chi_work, 0,
                            n, 1, gamma_work, gamma_work, 0, 9 * n, 1,
                            (double *)atilde, (double *)atilde, 0, 9 * n, 1,
                            (double *)alpha, (double *)alpha, 0, n, 1);

  memcpy(chi_rhs, chi_work, (size_t)n * sizeof(double));
  memcpy(gamma_rhs, gamma_work, (size_t)(9 * n) * sizeof(double));
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

  const double dr = 0.1;
  const double dtheta = 0.1;
  const double dphi = 0.1;
  const double dt = 0.01;
  const int steps = 20;

  const double chi0 = 1.5;
  const double alpha0 = 2.0;

  double *chi = (double *)calloc((size_t)n, sizeof(double));
  double *gamma = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *atilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *alpha = (double *)calloc((size_t)n, sizeof(double));

  double *k1_chi = (double *)calloc((size_t)n, sizeof(double));
  double *k1_gamma = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *k2_chi = (double *)calloc((size_t)n, sizeof(double));
  double *k2_gamma = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *mid_chi = (double *)calloc((size_t)n, sizeof(double));
  double *mid_gamma = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *chi_work = (double *)calloc((size_t)n, sizeof(double));
  double *gamma_work = (double *)calloc((size_t)(9 * n), sizeof(double));

  if (!chi || !gamma || !atilde || !alpha || !k1_chi || !k1_gamma || !k2_chi ||
      !k2_gamma || !mid_chi || !mid_gamma || !chi_work || !gamma_work) {
    fprintf(stderr, "allocation failure\n");
    free(chi);
    free(gamma);
    free(atilde);
    free(alpha);
    free(k1_chi);
    free(k1_gamma);
    free(k2_chi);
    free(k2_gamma);
    free(mid_chi);
    free(mid_gamma);
    free(chi_work);
    free(gamma_work);
    return 2;
  }

  for (int64_t p = 0; p < n; ++p) {
    chi[p] = chi0;
    alpha[p] = alpha0;
    for (int c = 0; c < 9; ++c) {
      gamma[(int64_t)c * n + p] = 100.0 + (double)c;
      atilde[(int64_t)c * n + p] = (double)(c + 1);
    }
  }

  for (int step = 0; step < steps; ++step) {
    eval_rhs(nx, ny, nz, dr, dtheta, dphi, chi, gamma, atilde, alpha, k1_chi,
             k1_gamma, chi_work, gamma_work);

    for (int64_t p = 0; p < n; ++p) {
      mid_chi[p] = chi[p] + 0.5 * dt * k1_chi[p];
    }
    for (int c = 0; c < 9; ++c) {
      for (int64_t p = 0; p < n; ++p) {
        const int64_t off = (int64_t)c * n + p;
        mid_gamma[off] = gamma[off] + 0.5 * dt * k1_gamma[off];
      }
    }

    eval_rhs(nx, ny, nz, dr, dtheta, dphi, mid_chi, mid_gamma, atilde, alpha,
             k2_chi, k2_gamma, chi_work, gamma_work);

    for (int64_t p = 0; p < n; ++p) {
      chi[p] += dt * k2_chi[p];
    }
    for (int c = 0; c < 9; ++c) {
      for (int64_t p = 0; p < n; ++p) {
        const int64_t off = (int64_t)c * n + p;
        gamma[off] += dt * k2_gamma[off];
      }
    }
  }

  const double lambda = -2.0 * alpha0;
  const double factor = 1.0 + lambda * dt + 0.5 * lambda * lambda * dt * dt;
  const double expectedChi = chi0 * pow(factor, (double)steps);
  const double gotChi = chi[cidx];

  printf("[ll-smoke] BSSN reduced RK2 center chi(%d) got=%.17g expected=%.17g\n",
         steps, gotChi, expectedChi);

  int ok = almost_equal(gotChi, expectedChi, 1e-12, 1e-12);

  for (int c = 0; c < 9; ++c) {
    const double gamma0 = 100.0 + (double)c;
    const double rhs = -2.0 * alpha0 * (double)(c + 1);
    const double expectedGamma = gamma0 + (double)steps * dt * rhs;
    const double gotGamma = gamma[(int64_t)c * n + cidx];
    if (c < 3) {
      printf("[ll-smoke] BSSN reduced RK2 center gamma[%d] got=%.17g "
             "expected=%.17g\n",
             c, gotGamma, expectedGamma);
    }
    ok &= almost_equal(gotGamma, expectedGamma, 1e-12, 1e-12);
  }

  free(chi);
  free(gamma);
  free(atilde);
  free(alpha);
  free(k1_chi);
  free(k1_gamma);
  free(k2_chi);
  free(k2_gamma);
  free(mid_chi);
  free(mid_gamma);
  free(chi_work);
  free(gamma_work);

  if (!ok) {
    fprintf(stderr, "BSSN reduced RK2 mismatch\n");
    return 3;
  }
  return 0;
}
