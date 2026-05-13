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

static void free_all(double *chi, double *alpha, double *theta, double *khat,
                     double *gammatilde_u, double *atilde, double *gammahat,
                     double *dchi, double *dkhat, double *dgammatilde,
                     double *datilde, double *dgammahat) {
  free(chi);
  free(alpha);
  free(theta);
  free(khat);
  free(gammatilde_u);
  free(atilde);
  free(gammahat);
  free(dchi);
  free(dkhat);
  free(dgammatilde);
  free(datilde);
  free(dgammahat);
}

int main(void) {
  const int64_t nx = 3;
  const int64_t ny = 3;
  const int64_t nz = 3;
  const int64_t n = nx * ny * nz;
  const int64_t cidx = flat_index(1, 1, 1, ny, nz);

  double *chi = (double *)calloc((size_t)n, sizeof(double));
  double *alpha = (double *)calloc((size_t)n, sizeof(double));
  double *theta = (double *)calloc((size_t)n, sizeof(double));
  double *khat = (double *)calloc((size_t)n, sizeof(double));
  double *gammatilde_u = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *atilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *gammahat = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *dchi = (double *)calloc((size_t)n, sizeof(double));
  double *dkhat = (double *)calloc((size_t)n, sizeof(double));
  double *dgammatilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *datilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *dgammahat = (double *)calloc((size_t)(3 * n), sizeof(double));

  if (!chi || !alpha || !theta || !khat || !gammatilde_u || !atilde ||
      !gammahat || !dchi || !dkhat || !dgammatilde || !datilde ||
      !dgammahat) {
    fprintf(stderr, "allocation failure\n");
    free_all(chi, alpha, theta, khat, gammatilde_u, atilde, gammahat, dchi,
             dkhat, dgammatilde, datilde, dgammahat);
    return 2;
  }

  for (int64_t p = 0; p < n; ++p) {
    chi[p] = 1.0;
    alpha[p] = 1.0;
    theta[p] = 0.0;
    khat[p] = -1.0;

    gammatilde_u[(int64_t)comp2(0, 0) * n + p] = 1.0;
    gammatilde_u[(int64_t)comp2(1, 1) * n + p] = 1.0;
    gammatilde_u[(int64_t)comp2(2, 2) * n + p] = 1.0;

    atilde[(int64_t)comp2(0, 0) * n + p] = -1.0 / 3.0;
    atilde[(int64_t)comp2(1, 1) * n + p] = -1.0 / 3.0;
    atilde[(int64_t)comp2(2, 2) * n + p] = 2.0 / 3.0;

    dchi[p] = NAN;
    dkhat[p] = NAN;
    for (int c = 0; c < 9; ++c) {
      dgammatilde[(int64_t)c * n + p] = NAN;
      datilde[(int64_t)c * n + p] = NAN;
    }
    for (int c = 0; c < 3; ++c)
      dgammahat[(int64_t)c * n + p] = NAN;
  }

  tensorium_call_rhs_grid_affine(nx, ny, nz, 1.0, 1.0, 1.0, chi, alpha, theta,
                                 khat, gammatilde_u, atilde, gammahat, dchi,
                                 dkhat, dgammatilde, datilde, dgammahat);

  const double expected_dgammatilde[9] = {
      2.0 / 3.0, 0.0, 0.0,
      0.0, 2.0 / 3.0, 0.0,
      0.0, 0.0, -4.0 / 3.0,
  };
  const double expected_datilde[9] = {
      1.0 / 9.0, 0.0, 0.0,
      0.0, 1.0 / 9.0, 0.0,
      0.0, 0.0, -14.0 / 9.0,
  };

  int ok = 1;
  ok &= almost_equal(dchi[cidx], -2.0 / 3.0, 1e-12, 1e-12);
  ok &= almost_equal(dkhat[cidx], 1.0, 1e-12, 1e-12);
  for (int c = 0; c < 9; ++c) {
    ok &= almost_equal(dgammatilde[(int64_t)c * n + cidx],
                       expected_dgammatilde[c], 1e-12, 1e-12);
    ok &= almost_equal(datilde[(int64_t)c * n + cidx], expected_datilde[c],
                       1e-12, 1e-12);
  }
  ok &= almost_equal(dgammahat[cidx], 0.0, 1e-12, 1e-12);
  ok &= almost_equal(dgammahat[n + cidx], 0.0, 1e-12, 1e-12);
  ok &= almost_equal(dgammahat[2 * n + cidx], 0.0, 1e-12, 1e-12);

  printf("[ll-smoke] Z4c Kasner center dchi got=%.17g expected=%.17g\n",
         dchi[cidx], -2.0 / 3.0);
  printf("[ll-smoke] Z4c Kasner center dKhat got=%.17g expected=1\n",
         dkhat[cidx]);
  printf("[ll-smoke] Z4c Kasner center dgammatilde[0,0] got=%.17g "
         "expected=%.17g\n",
         dgammatilde[(int64_t)comp2(0, 0) * n + cidx], 2.0 / 3.0);
  printf("[ll-smoke] Z4c Kasner center dAtilde[2,2] got=%.17g "
         "expected=%.17g\n",
         datilde[(int64_t)comp2(2, 2) * n + cidx], -14.0 / 9.0);

  free_all(chi, alpha, theta, khat, gammatilde_u, atilde, gammahat, dchi,
           dkhat, dgammatilde, datilde, dgammahat);

  if (!ok) {
    fprintf(stderr, "Z4c Kasner RHS mismatch\n");
    return 3;
  }
  return 0;
}
