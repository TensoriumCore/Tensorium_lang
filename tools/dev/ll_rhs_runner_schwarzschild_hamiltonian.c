#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

static void free_all(double *x, double *y, double *z, double *alpha,
                     double *gamma, double *gamma_u, double *ricci,
                     double *hamiltonian) {
  free(x);
  free(y);
  free(z);
  free(alpha);
  free(gamma);
  free(gamma_u);
  free(ricci);
  free(hamiltonian);
}

static int check_point(int64_t idx, int64_t n, double m, const double *r,
                       const double *theta, const double *alpha,
                       const double *gamma, const double *gamma_u,
                       const double *ricci, const double *hamiltonian,
                       double *max_abs_hamiltonian,
                       double *max_abs_scalar_mismatch) {
  const double rr = r[idx];
  const double th = theta[idx];
  const double sin_th = sin(th);
  const double sin2 = sin_th * sin_th;
  const double f = 1.0 - 2.0 * m / rr;

  const double exp_alpha = sqrt(f);
  const double exp_gamma[9] = {
      1.0 / f, 0.0, 0.0,
      0.0, rr * rr, 0.0,
      0.0, 0.0, rr * rr * sin2,
  };
  const double exp_gamma_u[9] = {
      f, 0.0, 0.0,
      0.0, 1.0 / (rr * rr), 0.0,
      0.0, 0.0, 1.0 / (rr * rr * sin2),
  };
  const double exp_ricci[9] = {
      -2.0 * m / (rr * rr * rr * f), 0.0, 0.0,
      0.0, m / rr, 0.0,
      0.0, 0.0, (m / rr) * sin2,
  };

  int ok = 1;
  ok &= almost_equal(alpha[idx], exp_alpha, 1e-12, 1e-12);

  for (int c = 0; c < 9; ++c) {
    const double got_gamma = gamma[(int64_t)c * n + idx];
    const double got_gamma_u = gamma_u[(int64_t)c * n + idx];
    const double got_ricci = ricci[(int64_t)c * n + idx];

    ok &= almost_equal(got_gamma, exp_gamma[c], 1e-12, 1e-12);
    ok &= almost_equal(got_gamma_u, exp_gamma_u[c], 1e-12, 1e-12);
    ok &= almost_equal(got_ricci, exp_ricci[c], 5e-2, 5e-3);
    ok &= isfinite(got_gamma);
    ok &= isfinite(got_gamma_u);
    ok &= isfinite(got_ricci);
  }

  double ricci_scalar = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const int c = comp2(i, j);
      ricci_scalar += gamma_u[(int64_t)c * n + idx] *
                      ricci[(int64_t)c * n + idx];
    }
  }

  const double scalar_mismatch = fabs(hamiltonian[idx] - ricci_scalar);
  const double abs_hamiltonian = fabs(hamiltonian[idx]);
  if (abs_hamiltonian > *max_abs_hamiltonian)
    *max_abs_hamiltonian = abs_hamiltonian;
  if (scalar_mismatch > *max_abs_scalar_mismatch)
    *max_abs_scalar_mismatch = scalar_mismatch;

  ok &= fabs(hamiltonian[idx]) <= 2e-3;
  ok &= scalar_mismatch <= 1e-12;
  ok &= isfinite(hamiltonian[idx]);

  return ok;
}

int main(void) {
  const int64_t nx = 20;
  const int64_t ny = 20;
  const int64_t nz = 20;
  const int64_t n = nx * ny * nz;
  const int64_t ci = nx / 2;
  const int64_t cj = ny / 2;
  const int64_t ck = nz / 2;

  const double m = 1.0;
  const double r0 = 10.0;
  const double theta0 = 1.0;
  const double phi0 = 0.7;
  const double dr = 0.01;
  const double dtheta = 0.01;
  const double dphi = 0.01;

  double *r = (double *)malloc((size_t)n * sizeof(double));
  double *theta = (double *)malloc((size_t)n * sizeof(double));
  double *phi = (double *)malloc((size_t)n * sizeof(double));
  double *alpha = (double *)malloc((size_t)n * sizeof(double));
  double *gamma = (double *)malloc((size_t)(9 * n) * sizeof(double));
  double *gamma_u = (double *)malloc((size_t)(9 * n) * sizeof(double));
  double *ricci = (double *)malloc((size_t)(9 * n) * sizeof(double));
  double *hamiltonian = (double *)malloc((size_t)n * sizeof(double));

  if (!r || !theta || !phi || !alpha || !gamma || !gamma_u || !ricci ||
      !hamiltonian) {
    fprintf(stderr, "allocation failure\n");
    free_all(r, theta, phi, alpha, gamma, gamma_u, ricci, hamiltonian);
    return 2;
  }

  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      for (int64_t k = 0; k < nz; ++k) {
        const int64_t idx = flat_index(i, j, k, ny, nz);
        r[idx] = r0 + (double)(i - ci) * dr;
        theta[idx] = theta0 + (double)(j - cj) * dtheta;
        phi[idx] = phi0 + (double)(k - ck) * dphi;
        hamiltonian[idx] = NAN;
        for (int c = 0; c < 9; ++c)
          ricci[(int64_t)c * n + idx] = NAN;
      }
    }
  }

  tensorium_call_init_grid_affine(m, r, theta, phi, alpha, gamma, gamma_u,
                                  n);
  tensorium_call_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, gamma, gamma_u,
                                 ricci, hamiltonian);

  const int64_t samples[][3] = {
      {ci, cj, ck},
      {ci - 3, cj + 2, ck - 2},
      {ci + 3, cj - 2, ck + 2},
  };

  int ok = 1;
  double max_abs_hamiltonian = 0.0;
  double max_abs_scalar_mismatch = 0.0;
  for (size_t s = 0; s < sizeof(samples) / sizeof(samples[0]); ++s) {
    const int64_t idx = flat_index(samples[s][0], samples[s][1], samples[s][2],
                                   ny, nz);
    ok &= check_point(idx, n, m, r, theta, alpha, gamma, gamma_u, ricci,
                      hamiltonian, &max_abs_hamiltonian,
                      &max_abs_scalar_mismatch);
  }

  const int64_t cidx = flat_index(ci, cj, ck, ny, nz);
  printf("[ll-smoke] Schwarzschild Hamiltonian center M=%.17g r=%.17g "
         "theta=%.17g\n",
         m, r[cidx], theta[cidx]);
  printf("Hamiltonian center got=%.17g expected=0\n", hamiltonian[cidx]);
  printf("Ricci_rr center got=%.17g expected=%.17g\n",
         ricci[(int64_t)comp2(0, 0) * n + cidx],
         -2.0 * m / (r[cidx] * r[cidx] * r[cidx] *
                     (1.0 - 2.0 * m / r[cidx])));
  printf("Ricci_thetatheta center got=%.17g expected=%.17g\n",
         ricci[(int64_t)comp2(1, 1) * n + cidx], m / r[cidx]);
  printf("max |Hamiltonian| over samples = %.17g\n", max_abs_hamiltonian);
  printf("max |Hamiltonian - gammaU*Ricci| over samples = %.17g\n",
         max_abs_scalar_mismatch);

  free_all(r, theta, phi, alpha, gamma, gamma_u, ricci, hamiltonian);

  if (!ok) {
    fprintf(stderr, "Schwarzschild Hamiltonian constraint mismatch\n");
    return 3;
  }
  return 0;
}
