#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define TENSORIUM_STANDARD_METRIC_MINKOWSKI 1
#define TENSORIUM_STANDARD_METRIC_SCHWARZSCHILD 2
#define TENSORIUM_STANDARD_METRIC_REISSNER_NORDSTROM 3
#define TENSORIUM_STANDARD_METRIC_KERR_LIKE 4
#define TENSORIUM_STANDARD_METRIC_SPATIAL_OFFDIAG 5
#define TENSORIUM_STANDARD_METRIC_SCHWARZSCHILD_ISOTROPIC 6

#ifndef TENSORIUM_STANDARD_METRIC_CASE
#error "TENSORIUM_STANDARD_METRIC_CASE must be defined"
#endif

static int cidx(int i, int j) { return i * 3 + j; }

static double exact_tol(double expected) {
  const double scale = fmax(1.0, fabs(expected));
  return 256.0 * DBL_EPSILON * scale;
}

static int check_scalar(const char *name, double got, double expected) {
  const double diff = fabs(got - expected);
  const double tol = exact_tol(expected);
  printf("  %-18s got=% .17g expected=% .17g diff=%.3e tol=%.3e\n",
         name, got, expected, diff, tol);
  if (diff <= tol)
    return 1;
  fprintf(stderr, "%s mismatch: got %.17g expected %.17g\n", name, got,
          expected);
  return 0;
}

static int check_scalar_quiet(const char *name, double got, double expected) {
  const double diff = fabs(got - expected);
  const double tol = exact_tol(expected);
  if (diff <= tol)
    return 1;
  fprintf(stderr, "%s mismatch: got %.17g expected %.17g diff %.3e tol %.3e\n",
          name, got, expected, diff, tol);
  return 0;
}

static int check_array(const char *name, const double *got,
                       const double *expected, int count) {
  int ok = 1;
  for (int i = 0; i < count; ++i) {
    char label[64];
    (void)snprintf(label, sizeof(label), "%s[%d]", name, i);
    ok &= check_scalar(label, got[i], expected[i]);
  }
  return ok;
}

static int check_array_quiet(const char *name, const double *got,
                             const double *expected, int count) {
  int ok = 1;
  for (int i = 0; i < count; ++i) {
    char label[64];
    (void)snprintf(label, sizeof(label), "%s[%d]", name, i);
    ok &= check_scalar_quiet(label, got[i], expected[i]);
  }
  return ok;
}

static int check_inverse_identity(const double *gamma, const double *gammaU) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double value = 0.0;
      for (int k = 0; k < 3; ++k)
        value += gammaU[cidx(i, k)] * gamma[cidx(k, j)];
      const double expected = i == j ? 1.0 : 0.0;
      char label[64];
      (void)snprintf(label, sizeof(label), "gammaU*gamma[%d,%d]", i, j);
      ok &= check_scalar(label, value, expected);
    }
  }
  return ok;
}

static int check_inverse_identity_quiet(const double *gamma,
                                        const double *gammaU) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double value = 0.0;
      for (int k = 0; k < 3; ++k)
        value += gammaU[cidx(i, k)] * gamma[cidx(k, j)];
      const double expected = i == j ? 1.0 : 0.0;
      char label[64];
      (void)snprintf(label, sizeof(label), "gammaU*gamma[%d,%d]", i, j);
      ok &= check_scalar_quiet(label, value, expected);
    }
  }
  return ok;
}

static void zero3(double *m) {
  for (int i = 0; i < 9; ++i)
    m[i] = 0.0;
}

static void set_diag3(double *m, double a, double b, double c) {
  zero3(m);
  m[cidx(0, 0)] = a;
  m[cidx(1, 1)] = b;
  m[cidx(2, 2)] = c;
}

static void expected_minkowski(double *alpha, double *gamma, double *gammaU) {
  *alpha = 1.0;
  set_diag3(gamma, 1.0, 1.0, 1.0);
  set_diag3(gammaU, 1.0, 1.0, 1.0);
}

static void expected_schwarzschild(double M, double r, double theta,
                                   double *alpha, double *gamma,
                                   double *gammaU) {
  const double s = sin(theta);
  const double f = 1.0 - 2.0 * M / r;
  *alpha = sqrt(f);
  set_diag3(gamma, 1.0 / f, r * r, r * r * s * s);
  set_diag3(gammaU, f, 1.0 / (r * r), 1.0 / (r * r * s * s));
}

static void expected_reissner_nordstrom(double M, double Q, double r,
                                        double theta, double *alpha,
                                        double *gamma, double *gammaU) {
  const double s = sin(theta);
  const double f = 1.0 - 2.0 * M / r + (Q * Q) / (r * r);
  *alpha = sqrt(f);
  set_diag3(gamma, 1.0 / f, r * r, r * r * s * s);
  set_diag3(gammaU, f, 1.0 / (r * r), 1.0 / (r * r * s * s));
}

static void expected_kerr_like(double M, double a, double r, double theta,
                               double *alpha, double *gamma, double *gammaU) {
  const double s = sin(theta);
  const double f = 1.0 - 2.0 * M / r;
  const double betaPhi = -(2.0 * a * M / r * s * s);
  const double betaSq = betaPhi * betaPhi / (r * r * s * s);
  *alpha = sqrt(f + betaSq);
  set_diag3(gamma, 1.0 / f, r * r, r * r * s * s);
  set_diag3(gammaU, f, 1.0 / (r * r), 1.0 / (r * r * s * s));
}

static void expected_spatial_offdiag(double *alpha, double *gamma,
                                     double *gammaU) {
  *alpha = 1.0;
  zero3(gamma);
  zero3(gammaU);
  gamma[cidx(0, 0)] = 2.0;
  gamma[cidx(0, 1)] = 1.0;
  gamma[cidx(1, 0)] = 1.0;
  gamma[cidx(1, 1)] = 3.0;
  gamma[cidx(2, 2)] = 4.0;
  gammaU[cidx(0, 0)] = 0.6;
  gammaU[cidx(0, 1)] = -0.2;
  gammaU[cidx(1, 0)] = -0.2;
  gammaU[cidx(1, 1)] = 0.4;
  gammaU[cidx(2, 2)] = 0.25;
}

static void expected_schwarzschild_isotropic(double M, double x, double y,
                                             double z, double *alpha,
                                             double *gamma, double *gammaU) {
  const double rho = sqrt(x * x + y * y + z * z + 0.25);
  const double psi = 1.0 + M / (2.0 * rho);
  const double eta = 1.0 - M / (2.0 * rho);
  const double psi2 = psi * psi;
  const double psi4 = psi2 * psi2;
  *alpha = eta / psi;
  set_diag3(gamma, psi4, psi4, psi4);
  set_diag3(gammaU, 1.0 / psi4, 1.0 / psi4, 1.0 / psi4);
}

static void load_soa_point(const double *soa, int64_t n, int64_t point,
                           double *out) {
  for (int comp = 0; comp < 9; ++comp)
    out[comp] = soa[(int64_t)comp * n + point];
}

static int check_grid_point(const char *label, double gotAlpha,
                            const double *gotGamma, const double *gotGammaU,
                            double expectedAlpha,
                            const double *expectedGamma,
                            const double *expectedGammaU) {
  int ok = 1;
  char scalarLabel[96];
  (void)snprintf(scalarLabel, sizeof(scalarLabel), "%s.alpha", label);
  ok &= check_scalar_quiet(scalarLabel, gotAlpha, expectedAlpha);
  ok &= check_array_quiet("grid.gamma", gotGamma, expectedGamma, 9);
  ok &= check_array_quiet("grid.gammaU", gotGammaU, expectedGammaU, 9);
  ok &= check_inverse_identity_quiet(gotGamma, gotGammaU);
  if (ok)
    printf("  %s exact grid comparison OK\n", label);
  return ok;
}

int main(void) {
  double alpha[1] = {0.0};
  double gamma[9] = {0.0};
  double gammaU[9] = {0.0};
  double expectedAlpha = 0.0;
  double expectedGamma[9];
  double expectedGammaU[9];
  zero3(expectedGamma);
  zero3(expectedGammaU);

#if TENSORIUM_STANDARD_METRIC_CASE == TENSORIUM_STANDARD_METRIC_MINKOWSKI
  const char *caseName = "Minkowski Cartesian";
  const double x = 1.25;
  const double y = -0.5;
  const double z = 2.0;
  tensorium_call_init_point(x, y, z, alpha, gamma, gammaU);
  expectedAlpha = 1.0;
  set_diag3(expectedGamma, 1.0, 1.0, 1.0);
  set_diag3(expectedGammaU, 1.0, 1.0, 1.0);

#elif TENSORIUM_STANDARD_METRIC_CASE == TENSORIUM_STANDARD_METRIC_SCHWARZSCHILD
  const char *caseName = "Schwarzschild spherical";
  const double M = 1.0;
  const double r = 10.0;
  const double theta = 1.0;
  const double phi = 0.5;
  const double s = sin(theta);
  const double f = 1.0 - 2.0 * M / r;
  tensorium_call_init_point(M, r, theta, phi, alpha, gamma, gammaU);
  expectedAlpha = sqrt(f);
  set_diag3(expectedGamma, 1.0 / f, r * r, r * r * s * s);
  set_diag3(expectedGammaU, f, 1.0 / (r * r), 1.0 / (r * r * s * s));

#elif TENSORIUM_STANDARD_METRIC_CASE == \
    TENSORIUM_STANDARD_METRIC_REISSNER_NORDSTROM
  const char *caseName = "Reissner-Nordstrom spherical";
  const double M = 1.0;
  const double Q = 0.5;
  const double r = 10.0;
  const double theta = 1.0;
  const double phi = 0.5;
  const double s = sin(theta);
  const double f = 1.0 - 2.0 * M / r + (Q * Q) / (r * r);
  tensorium_call_init_point(M, Q, r, theta, phi, alpha, gamma, gammaU);
  expectedAlpha = sqrt(f);
  set_diag3(expectedGamma, 1.0 / f, r * r, r * r * s * s);
  set_diag3(expectedGammaU, f, 1.0 / (r * r), 1.0 / (r * r * s * s));

#elif TENSORIUM_STANDARD_METRIC_CASE == TENSORIUM_STANDARD_METRIC_KERR_LIKE
  const char *caseName = "Kerr-like shift spherical";
  const double M = 1.0;
  const double a = 0.3;
  const double r = 10.0;
  const double theta = 1.0;
  const double phi = 0.5;
  const double s = sin(theta);
  const double f = 1.0 - 2.0 * M / r;
  const double betaPhi = -(2.0 * a * M / r * s * s);
  const double betaSq = betaPhi * betaPhi / (r * r * s * s);
  tensorium_call_init_point(M, a, r, theta, phi, alpha, gamma, gammaU);
  expectedAlpha = sqrt(f + betaSq);
  set_diag3(expectedGamma, 1.0 / f, r * r, r * r * s * s);
  set_diag3(expectedGammaU, f, 1.0 / (r * r), 1.0 / (r * r * s * s));

#elif TENSORIUM_STANDARD_METRIC_CASE == TENSORIUM_STANDARD_METRIC_SPATIAL_OFFDIAG
  const char *caseName = "Spatial off-diagonal Cartesian";
  const double x = 1.25;
  const double y = -0.5;
  const double z = 2.0;
  tensorium_call_init_point(x, y, z, alpha, gamma, gammaU);
  expectedAlpha = 1.0;
  expectedGamma[cidx(0, 0)] = 2.0;
  expectedGamma[cidx(0, 1)] = 1.0;
  expectedGamma[cidx(1, 0)] = 1.0;
  expectedGamma[cidx(1, 1)] = 3.0;
  expectedGamma[cidx(2, 2)] = 4.0;
  expectedGammaU[cidx(0, 0)] = 0.6;
  expectedGammaU[cidx(0, 1)] = -0.2;
  expectedGammaU[cidx(1, 0)] = -0.2;
  expectedGammaU[cidx(1, 1)] = 0.4;
  expectedGammaU[cidx(2, 2)] = 0.25;

#elif TENSORIUM_STANDARD_METRIC_CASE == \
    TENSORIUM_STANDARD_METRIC_SCHWARZSCHILD_ISOTROPIC
  const char *caseName = "Schwarzschild isotropic Cartesian";
  const double M = 1.0;
  const double x = 4.0;
  const double y = 3.0;
  const double z = 2.0;
  const double rho = sqrt(x * x + y * y + z * z + 0.25);
  const double psi = 1.0 + M / (2.0 * rho);
  const double eta = 1.0 - M / (2.0 * rho);
  const double psi2 = psi * psi;
  const double psi4 = psi2 * psi2;
  tensorium_call_init_point(M, x, y, z, alpha, gamma, gammaU);
  expectedAlpha = eta / psi;
  set_diag3(expectedGamma, psi4, psi4, psi4);
  set_diag3(expectedGammaU, 1.0 / psi4, 1.0 / psi4, 1.0 / psi4);

#else
#error "Unsupported TENSORIUM_STANDARD_METRIC_CASE"
#endif

  printf("[analytic-init] %s\n", caseName);
  int ok = 1;
  ok &= check_scalar("alpha", alpha[0], expectedAlpha);
  ok &= check_array("gamma", gamma, expectedGamma, 9);
  ok &= check_array("gammaU", gammaU, expectedGammaU, 9);
  ok &= check_inverse_identity(gamma, gammaU);

  {
    const int64_t n = 3;
    double gridAlpha[3] = {0.0, 0.0, 0.0};
    double gridGamma[27] = {0.0};
    double gridGammaU[27] = {0.0};
    double gotGamma[9];
    double gotGammaU[9];
    double expAlpha = 0.0;
    double expGamma[9];
    double expGammaU[9];
    zero3(expGamma);
    zero3(expGammaU);

#if TENSORIUM_STANDARD_METRIC_CASE == TENSORIUM_STANDARD_METRIC_MINKOWSKI
    double xg[3] = {x, x + 0.5, x - 0.75};
    double yg[3] = {y, y + 0.25, y - 0.5};
    double zg[3] = {z, z - 0.125, z + 0.625};
    tensorium_call_init_grid_affine(xg, yg, zg, gridAlpha, gridGamma,
                                    gridGammaU, n);
    for (int64_t p = 0; p < n; ++p) {
      (void)xg[p];
      (void)yg[p];
      (void)zg[p];
      expected_minkowski(&expAlpha, expGamma, expGammaU);
      load_soa_point(gridGamma, n, p, gotGamma);
      load_soa_point(gridGammaU, n, p, gotGammaU);
      char label[64];
      (void)snprintf(label, sizeof(label), "grid[%lld]",
                     (long long)p);
      ok &= check_grid_point(label, gridAlpha[p], gotGamma, gotGammaU,
                             expAlpha, expGamma, expGammaU);
    }

#elif TENSORIUM_STANDARD_METRIC_CASE == TENSORIUM_STANDARD_METRIC_SCHWARZSCHILD
    double rg[3] = {r, r + 0.5, r - 0.75};
    double thetag[3] = {theta, theta + 0.1, theta - 0.2};
    double phig[3] = {phi, phi + 0.25, phi - 0.4};
    tensorium_call_init_grid_affine(M, rg, thetag, phig, gridAlpha, gridGamma,
                                    gridGammaU, n);
    for (int64_t p = 0; p < n; ++p) {
      expected_schwarzschild(M, rg[p], thetag[p], &expAlpha, expGamma,
                             expGammaU);
      load_soa_point(gridGamma, n, p, gotGamma);
      load_soa_point(gridGammaU, n, p, gotGammaU);
      char label[64];
      (void)snprintf(label, sizeof(label), "grid[%lld]",
                     (long long)p);
      ok &= check_grid_point(label, gridAlpha[p], gotGamma, gotGammaU,
                             expAlpha, expGamma, expGammaU);
    }

#elif TENSORIUM_STANDARD_METRIC_CASE == \
    TENSORIUM_STANDARD_METRIC_REISSNER_NORDSTROM
    double rg[3] = {r, r + 0.5, r - 0.75};
    double thetag[3] = {theta, theta + 0.1, theta - 0.2};
    double phig[3] = {phi, phi + 0.25, phi - 0.4};
    tensorium_call_init_grid_affine(M, Q, rg, thetag, phig, gridAlpha,
                                    gridGamma, gridGammaU, n);
    for (int64_t p = 0; p < n; ++p) {
      expected_reissner_nordstrom(M, Q, rg[p], thetag[p], &expAlpha,
                                  expGamma, expGammaU);
      load_soa_point(gridGamma, n, p, gotGamma);
      load_soa_point(gridGammaU, n, p, gotGammaU);
      char label[64];
      (void)snprintf(label, sizeof(label), "grid[%lld]",
                     (long long)p);
      ok &= check_grid_point(label, gridAlpha[p], gotGamma, gotGammaU,
                             expAlpha, expGamma, expGammaU);
    }

#elif TENSORIUM_STANDARD_METRIC_CASE == TENSORIUM_STANDARD_METRIC_KERR_LIKE
    double rg[3] = {r, r + 0.5, r - 0.75};
    double thetag[3] = {theta, theta + 0.1, theta - 0.2};
    double phig[3] = {phi, phi + 0.25, phi - 0.4};
    tensorium_call_init_grid_affine(M, a, rg, thetag, phig, gridAlpha,
                                    gridGamma, gridGammaU, n);
    for (int64_t p = 0; p < n; ++p) {
      expected_kerr_like(M, a, rg[p], thetag[p], &expAlpha, expGamma,
                         expGammaU);
      load_soa_point(gridGamma, n, p, gotGamma);
      load_soa_point(gridGammaU, n, p, gotGammaU);
      char label[64];
      (void)snprintf(label, sizeof(label), "grid[%lld]",
                     (long long)p);
      ok &= check_grid_point(label, gridAlpha[p], gotGamma, gotGammaU,
                             expAlpha, expGamma, expGammaU);
    }

#elif TENSORIUM_STANDARD_METRIC_CASE == TENSORIUM_STANDARD_METRIC_SPATIAL_OFFDIAG
    double xg[3] = {x, x + 0.5, x - 0.75};
    double yg[3] = {y, y + 0.25, y - 0.5};
    double zg[3] = {z, z - 0.125, z + 0.625};
    tensorium_call_init_grid_affine(xg, yg, zg, gridAlpha, gridGamma,
                                    gridGammaU, n);
    for (int64_t p = 0; p < n; ++p) {
      (void)xg[p];
      (void)yg[p];
      (void)zg[p];
      expected_spatial_offdiag(&expAlpha, expGamma, expGammaU);
      load_soa_point(gridGamma, n, p, gotGamma);
      load_soa_point(gridGammaU, n, p, gotGammaU);
      char label[64];
      (void)snprintf(label, sizeof(label), "grid[%lld]",
                     (long long)p);
      ok &= check_grid_point(label, gridAlpha[p], gotGamma, gotGammaU,
                             expAlpha, expGamma, expGammaU);
    }

#elif TENSORIUM_STANDARD_METRIC_CASE == \
    TENSORIUM_STANDARD_METRIC_SCHWARZSCHILD_ISOTROPIC
    double xg[3] = {x, x + 0.5, x - 0.75};
    double yg[3] = {y, y + 0.25, y - 0.5};
    double zg[3] = {z, z - 0.125, z + 0.625};
    tensorium_call_init_grid_affine(M, xg, yg, zg, gridAlpha, gridGamma,
                                    gridGammaU, n);
    for (int64_t p = 0; p < n; ++p) {
      expected_schwarzschild_isotropic(M, xg[p], yg[p], zg[p], &expAlpha,
                                       expGamma, expGammaU);
      load_soa_point(gridGamma, n, p, gotGamma);
      load_soa_point(gridGammaU, n, p, gotGammaU);
      char label[64];
      (void)snprintf(label, sizeof(label), "grid[%lld]",
                     (long long)p);
      ok &= check_grid_point(label, gridAlpha[p], gotGamma, gotGammaU,
                             expAlpha, expGamma, expGammaU);
    }
#endif
  }

  if (!ok)
    return 3;
  return 0;
}
