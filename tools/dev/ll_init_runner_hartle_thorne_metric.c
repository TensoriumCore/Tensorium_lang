#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int cidx3(int i, int j) { return i * 3 + j; }
static int cidx4(int i, int j) { return i * 4 + j; }

static double exact_tol(double expected) {
  const double scale = fmax(1.0, fabs(expected));
  return 512.0 * DBL_EPSILON * scale;
}

static int check_scalar(const char *name, double got, double expected) {
  const double diff = fabs(got - expected);
  const double tol = exact_tol(expected);
  printf("  %-20s got=% .17g expected=% .17g diff=%.3e tol=%.3e\n",
         name, got, expected, diff, tol);
  if (diff <= tol)
    return 1;
  fprintf(stderr, "%s mismatch: got %.17g expected %.17g\n", name, got,
          expected);
  return 0;
}

static int check_scalar_tol(const char *name, double got, double expected,
                            double relTol, double absTol) {
  const double diff = fabs(got - expected);
  const double scale = fmax(1.0, fabs(expected));
  const double tol = fmax(absTol, relTol * scale);
  printf("  %-20s got=% .17g expected=% .17g diff=%.3e tol=%.3e\n",
         name, got, expected, diff, tol);
  if (diff <= tol)
    return 1;
  fprintf(stderr, "%s mismatch: got %.17g expected %.17g\n", name, got,
          expected);
  return 0;
}

static void zero3(double *m) {
  for (int i = 0; i < 9; ++i)
    m[i] = 0.0;
}

static void fill_hartle_thorne_metric4(double h, double f, double omega,
                                       double r, double theta, double *g) {
  const double s = sin(theta);
  for (int i = 0; i < 16; ++i)
    g[i] = 0.0;
  g[cidx4(0, 0)] = -h;
  g[cidx4(1, 1)] = 1.0 / f;
  g[cidx4(2, 2)] = r * r;
  g[cidx4(3, 3)] = r * r * s * s;
  g[cidx4(0, 3)] = -(omega * r * r * s * s);
  g[cidx4(3, 0)] = g[cidx4(0, 3)];
}

static void expected_split3p1(double h, double f, double omega, double r,
                              double theta, double *alpha, double *gamma,
                              double *gammaU) {
  const double s = sin(theta);
  zero3(gamma);
  zero3(gammaU);
  gamma[cidx3(0, 0)] = 1.0 / f;
  gamma[cidx3(1, 1)] = r * r;
  gamma[cidx3(2, 2)] = r * r * s * s;
  gammaU[cidx3(0, 0)] = f;
  gammaU[cidx3(1, 1)] = 1.0 / (r * r);
  gammaU[cidx3(2, 2)] = 1.0 / (r * r * s * s);
  *alpha = sqrt(h + omega * omega * r * r * s * s);
}

static int check_linearized_3p1(double h, double omega, double r,
                                double theta, double alphaFull) {
  const double s = sin(theta);
  const double r2s2 = r * r * s * s;
  const double alphaLinearized = sqrt(h);
  const double betaPhi = -(omega * r2s2);
  const double betaPhiContravariant = betaPhi / r2s2;
  const double gtphiContravariant =
      betaPhiContravariant / (alphaLinearized * alphaLinearized);
  const double alphaFullDelta = alphaFull - alphaLinearized;
  const double alphaQuadraticCoeff =
      r2s2 / (alphaFull + alphaLinearized);

  printf("[hartle-thorne-linearized] first-order 3+1 check\n");
  printf("  alpha_full - sqrt(h) = %.17g\n", alphaFullDelta);
  printf("  (alpha_full - sqrt(h)) / omega^2 = %.17g\n",
         alphaFullDelta / (omega * omega));

  int ok = 1;
  ok &= check_scalar("alpha_linearized", alphaLinearized, sqrt(h));
  ok &= check_scalar("beta_phi", betaPhi, -(omega * r2s2));
  ok &= check_scalar("beta^phi", betaPhiContravariant, -omega);
  ok &= check_scalar("g^tphi", gtphiContravariant, -omega / h);
  ok &= check_scalar("alpha_full delta", alphaFullDelta,
                     omega * omega * alphaQuadraticCoeff);
  ok &= check_scalar("alpha delta/omega", alphaFullDelta / omega,
                     omega * alphaQuadraticCoeff);
  return ok;
}

static double hartle_thorne_omega_gr(double r, double J) {
  return 2.0 * J / (r * r * r);
}

static double hartle_thorne_omega_prime_gr(double r, double J) {
  return -6.0 * J / (r * r * r * r);
}

static double hartle_thorne_radial_flux_gr(double r, double J) {
  return r * r * r * r * hartle_thorne_omega_prime_gr(r, J);
}

static int check_hartle_thorne_radial_equation_gr(void) {
  const double J = 0.1;
  const double radii[4] = {4.0, 6.0, 10.0, 20.0};
  const double expectedFlux = -6.0 * J;

  printf("[hartle-thorne-radial-gr] d/dr(r^4 omega') = 0 check\n");
  printf("  J=%.17g\n", J);

  int ok = 1;
  for (int i = 0; i < 4; ++i) {
    const double r = radii[i];
    const double omega = hartle_thorne_omega_gr(r, J);
    const double omegaPrime = hartle_thorne_omega_prime_gr(r, J);
    const double radialFlux = hartle_thorne_radial_flux_gr(r, J);
    const double fdStep = 1.0e-5 * r;
    const double omegaPrimeFD =
        (hartle_thorne_omega_gr(r + fdStep, J) -
         hartle_thorne_omega_gr(r - fdStep, J)) /
        (2.0 * fdStep);

    printf("  r=% .17g omega=% .17g omega_prime=% .17g r^4*omega'=% .17g\n",
           r, omega, omegaPrime, radialFlux);
    ok &= check_scalar("r^4 omega'", radialFlux, expectedFlux);
    ok &= check_scalar_tol("omega' finite diff", omegaPrime, omegaPrimeFD,
                           1.0e-8, 1.0e-12);
  }

  for (int i = 0; i + 1 < 4; ++i) {
    const double leftFlux = hartle_thorne_radial_flux_gr(radii[i], J);
    const double rightFlux = hartle_thorne_radial_flux_gr(radii[i + 1], J);
    const double radialEqFD =
        (rightFlux - leftFlux) / (radii[i + 1] - radii[i]);
    char label[96];
    (void)snprintf(label, sizeof(label), "d_flux/dr [%g,%g]", radii[i],
                   radii[i + 1]);
    ok &= check_scalar(label, radialEqFD, 0.0);
  }

  return ok;
}

static void print_matrix4(const char *name, const double *m) {
  printf("%s = [\n", name);
  for (int i = 0; i < 4; ++i) {
    printf("  [");
    for (int j = 0; j < 4; ++j) {
      printf("% .17g%s", m[cidx4(i, j)], j == 3 ? "" : ", ");
    }
    printf("]%s\n", i == 3 ? "" : ",");
  }
  printf("]\n");
}

static void print_matrix3(const char *name, const double *m) {
  printf("%s = [\n", name);
  for (int i = 0; i < 3; ++i) {
    printf("  [");
    for (int j = 0; j < 3; ++j) {
      printf("% .17g%s", m[cidx3(i, j)], j == 2 ? "" : ", ");
    }
    printf("]%s\n", i == 2 ? "" : ",");
  }
  printf("]\n");
}

static void load_soa_point(const double *soa, int64_t n, int64_t point,
                           double *out) {
  for (int comp = 0; comp < 9; ++comp)
    out[comp] = soa[(int64_t)comp * n + point];
}

static int check_matrix3(const char *name, const double *got,
                         const double *expected) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      char label[96];
      (void)snprintf(label, sizeof(label), "%s[%d,%d]", name, i, j);
      ok &= check_scalar(label, got[cidx3(i, j)], expected[cidx3(i, j)]);
    }
  }
  return ok;
}

static int check_point_outputs(const char *label, double h, double f,
                               double omega, double r, double theta,
                               double gotAlpha, const double *gotGamma,
                               const double *gotGammaU) {
  double metric4[16];
  double expAlpha = 0.0;
  double expGamma[9];
  double expGammaU[9];

  fill_hartle_thorne_metric4(h, f, omega, r, theta, metric4);
  expected_split3p1(h, f, omega, r, theta, &expAlpha, expGamma, expGammaU);

  printf("[hartle-thorne-init] %s\n", label);
  printf("  r=%.17g theta=%.17g h=%.17g f=%.17g omega=%.17g\n", r, theta,
         h, f, omega);
  print_matrix4("g_mu_nu", metric4);
  print_matrix3("Tensorium gamma_ij", gotGamma);
  print_matrix3("Tensorium gammaU^ij", gotGammaU);

  int ok = 1;
  ok &= check_scalar("g_tphi == g_phit", metric4[cidx4(0, 3)],
                     metric4[cidx4(3, 0)]);
  ok &= check_scalar("alpha", gotAlpha, expAlpha);
  ok &= check_matrix3("gamma", gotGamma, expGamma);
  ok &= check_matrix3("gammaU", gotGammaU, expGammaU);
  return ok;
}

int main(void) {
  const double M = 1.0;
  const double J = 0.25;
  const double r = 10.0;
  const double theta = 1.0;
  const double phi = 0.5;
  const double h = 1.0 - 2.0 * M / r;
  const double f = h;
  const double omega = 2.0 * J / (r * r * r);

  double alpha[1] = {0.0};
  double gamma[9] = {0.0};
  double gammaU[9] = {0.0};
  tensorium_call_init_point(h, omega, f, r, theta, phi, alpha, gamma, gammaU);

  int ok = check_point_outputs("single point", h, f, omega, r, theta, alpha[0],
                               gamma, gammaU);
  ok &= check_linearized_3p1(h, omega, r, theta, alpha[0]);
  ok &= check_hartle_thorne_radial_equation_gr();

  {
    const int64_t n = 3;
    double rg[3] = {r, r, r};
    double thetag[3] = {theta, theta + 0.2, theta - 0.25};
    double phig[3] = {phi, phi + 0.1, phi - 0.3};
    double gridAlpha[3] = {0.0, 0.0, 0.0};
    double gridGamma[27] = {0.0};
    double gridGammaU[27] = {0.0};
    tensorium_call_init_grid_affine(h, omega, f, rg, thetag, phig, gridAlpha,
                                    gridGamma, gridGammaU, n);

    for (int64_t p = 0; p < n; ++p) {
      double gotGamma[9];
      double gotGammaU[9];
      char label[64];
      load_soa_point(gridGamma, n, p, gotGamma);
      load_soa_point(gridGammaU, n, p, gotGammaU);
      (void)snprintf(label, sizeof(label), "grid[%lld]", (long long)p);
      ok &= check_point_outputs(label, h, f, omega, rg[p], thetag[p],
                                gridAlpha[p], gotGamma, gotGammaU);
    }
  }

  if (!ok) {
    fprintf(stderr, "Hartle-Thorne metric init test failed\n");
    return 3;
  }
  return 0;
}
