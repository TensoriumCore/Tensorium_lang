#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t flat_index(int64_t i, int64_t j, int64_t k, int64_t ny,
                          int64_t nz) {
  return (i * ny + j) * nz + k;
}

static int comp2(int i, int j) { return i * 3 + j; }

static int comp3(int i, int j, int k) { return (i * 3 + j) * 3 + k; }

static double exact_tol(double expected) {
  const double scale = fmax(1.0, fabs(expected));
  return 4096.0 * DBL_EPSILON * scale;
}

static int check_value(const char *name, double got, double expected,
                       int verbose) {
  const double diff = fabs(got - expected);
  const double tol = exact_tol(expected);
  if (verbose) {
    printf("  %-28s got=% .17g expected=% .17g diff=%.3e tol=%.3e\n",
           name, got, expected, diff, tol);
  }
  if (diff <= tol)
    return 1;
  fprintf(stderr, "%s mismatch: got %.17g expected %.17g diff %.3e tol %.3e\n",
          name, got, expected, diff, tol);
  return 0;
}

static void print_vector_at(const char *name, const double *v, int64_t n,
                            int64_t p) {
  printf("%s = [%.17g, %.17g, %.17g]\n", name, v[p], v[n + p],
         v[2 * n + p]);
}

static void print_matrix_at(const char *name, const double *m, int64_t n,
                            int64_t p) {
  printf("%s = [\n", name);
  for (int i = 0; i < 3; ++i) {
    printf("  [");
    for (int j = 0; j < 3; ++j) {
      if (j != 0)
        printf(", ");
      printf("%.17g", m[(int64_t)comp2(i, j) * n + p]);
    }
    printf(i == 2 ? "]\n" : "],\n");
  }
  printf("]\n");
}

static void print_tensor3_at(const char *name, const double *t, int64_t n,
                             int64_t p) {
  printf("%s = [\n", name);
  for (int i = 0; i < 3; ++i) {
    printf("  [\n");
    for (int j = 0; j < 3; ++j) {
      printf("    [");
      for (int k = 0; k < 3; ++k) {
        if (k != 0)
          printf(", ");
        printf("%.17g", t[(int64_t)comp3(i, j, k) * n + p]);
      }
      printf(j == 2 ? "]\n" : "],\n");
    }
    printf(i == 2 ? "  ]\n" : "  ],\n");
  }
  printf("]\n");
}

static void expected_schwarzschild(double m, double r, double theta,
                                   double *alpha, double gamma[9],
                                   double gamma_u[9], double ricci[9],
                                   double hessian_alpha[9]) {
  const double s = sin(theta);
  const double s2 = s * s;
  const double f = 1.0 - 2.0 * m / r;
  for (int c = 0; c < 9; ++c) {
    gamma[c] = 0.0;
    gamma_u[c] = 0.0;
    ricci[c] = 0.0;
    hessian_alpha[c] = 0.0;
  }

  *alpha = sqrt(f);

  gamma[comp2(0, 0)] = 1.0 / f;
  gamma[comp2(1, 1)] = r * r;
  gamma[comp2(2, 2)] = r * r * s2;

  gamma_u[comp2(0, 0)] = f;
  gamma_u[comp2(1, 1)] = 1.0 / (r * r);
  gamma_u[comp2(2, 2)] = 1.0 / (r * r * s2);

  ricci[comp2(0, 0)] = -2.0 * m / (r * r * r * f);
  ricci[comp2(1, 1)] = m / r;
  ricci[comp2(2, 2)] = (m / r) * s2;

  for (int c = 0; c < 9; ++c)
    hessian_alpha[c] = (*alpha) * ricci[c];
}

static int check_vector_at(const char *name, const double *got,
                           const double expected[3], int64_t n, int64_t p,
                           int verbose) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    char label[96];
    (void)snprintf(label, sizeof(label), "%s[%d]", name, i);
    ok &= check_value(label, got[(int64_t)i * n + p], expected[i], verbose);
  }
  return ok;
}

static int check_vector_zero_at(const char *name, const double *got, int64_t n,
                                int64_t p, int verbose) {
  const double zero[3] = {0.0, 0.0, 0.0};
  return check_vector_at(name, got, zero, n, p, verbose);
}

static int check_matrix_at(const char *name, const double *got,
                           const double expected[9], int64_t n, int64_t p,
                           int verbose) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      char label[96];
      (void)snprintf(label, sizeof(label), "%s[%d,%d]", name, i, j);
      ok &= check_value(label, got[(int64_t)comp2(i, j) * n + p],
                        expected[comp2(i, j)], verbose);
    }
  }
  return ok;
}

static int check_matrix_zero_at(const char *name, const double *got, int64_t n,
                                int64_t p, int verbose) {
  double zero[9] = {0.0};
  return check_matrix_at(name, got, zero, n, p, verbose);
}

static int check_tensor3_zero_at(const char *name, const double *got, int64_t n,
                                 int64_t p, int verbose) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        char label[96];
        (void)snprintf(label, sizeof(label), "%s[%d,%d,%d]", name, i, j, k);
        ok &= check_value(label, got[(int64_t)comp3(i, j, k) * n + p], 0.0,
                          verbose);
      }
    }
  }
  return ok;
}

static int check_inverse_identity(const double *gamma, const double *gamma_u,
                                  int64_t n, int64_t p, int verbose) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double value = 0.0;
      for (int k = 0; k < 3; ++k) {
        value += gamma_u[(int64_t)comp2(i, k) * n + p] *
                 gamma[(int64_t)comp2(k, j) * n + p];
      }
      char label[96];
      (void)snprintf(label, sizeof(label), "gammaU*gamma[%d,%d]", i, j);
      ok &= check_value(label, value, i == j ? 1.0 : 0.0, verbose);
    }
  }
  return ok;
}

static int check_point(int64_t p, int64_t n, double m, const double *r,
                       const double *theta, const double *alpha,
                       const double *gamma, const double *gamma_u,
                       const double *chi, const double *beta, const double *B,
                       const double *K, const double *gammatilde,
                       const double *gammatilde_u, const double *Atilde,
                       const double *Gammahat, const double *dchi,
                       const double *dalpha, const double *dbeta,
                       const double *dB, const double *dK,
                       const double *dgammatilde, const double *dAtilde,
                       const double *dGammahat, const double *ricci,
                       const double *hessian_alpha, const double *datilde,
                       const double *hamiltonian, const double *momentum,
                       int verbose) {
  double exp_alpha = 0.0;
  double exp_gamma[9];
  double exp_gamma_u[9];
  double exp_ricci[9];
  double exp_hessian_alpha[9];
  expected_schwarzschild(m, r[p], theta[p], &exp_alpha, exp_gamma,
                         exp_gamma_u, exp_ricci, exp_hessian_alpha);

  int ok = 1;
  ok &= check_value("alpha", alpha[p], exp_alpha, verbose);
  ok &= check_matrix_at("gamma", gamma, exp_gamma, n, p, verbose);
  ok &= check_matrix_at("gammaU", gamma_u, exp_gamma_u, n, p, verbose);
  ok &= check_inverse_identity(gamma, gamma_u, n, p, verbose);

  ok &= check_value("chi", chi[p], 1.0, verbose);
  ok &= check_vector_zero_at("beta", beta, n, p, verbose);
  ok &= check_vector_zero_at("B", B, n, p, verbose);
  ok &= check_value("K", K[p], 0.0, verbose);
  ok &= check_matrix_at("gammatilde", gammatilde, exp_gamma, n, p, verbose);
  ok &= check_matrix_at("gammatildeU", gammatilde_u, exp_gamma_u, n, p,
                        verbose);
  ok &= check_matrix_zero_at("Atilde", Atilde, n, p, verbose);
  ok &= check_vector_zero_at("Gammahat", Gammahat, n, p, verbose);

  ok &= check_value("dchi", dchi[p], 0.0, verbose);
  ok &= check_value("dalpha", dalpha[p], 0.0, verbose);
  ok &= check_vector_zero_at("dbeta", dbeta, n, p, verbose);
  ok &= check_vector_zero_at("dB", dB, n, p, verbose);
  ok &= check_value("dK", dK[p], 0.0, verbose);
  ok &= check_matrix_zero_at("dgammatilde", dgammatilde, n, p, verbose);
  ok &= check_matrix_zero_at("dAtilde", dAtilde, n, p, verbose);
  ok &= check_vector_zero_at("dGammahat", dGammahat, n, p, verbose);

  ok &= check_matrix_at("RicciAnalytic", ricci, exp_ricci, n, p, verbose);
  ok &= check_matrix_at("HessianAlpha", hessian_alpha, exp_hessian_alpha, n,
                        p, verbose);
  ok &= check_tensor3_zero_at("DAtilde", datilde, n, p, verbose);
  ok &= check_value("Hamiltonian", hamiltonian[p], 0.0, verbose);
  ok &= check_vector_zero_at("Momentum", momentum, n, p, verbose);
  return ok;
}

static void free_all(double **ptrs, size_t count) {
  for (size_t i = 0; i < count; ++i)
    free(ptrs[i]);
}

int main(void) {
  const int64_t nx = 5;
  const int64_t ny = 5;
  const int64_t nz = 5;
  const int64_t n = nx * ny * nz;
  const int64_t ci = 2;
  const int64_t cj = 2;
  const int64_t ck = 2;
  const int64_t cidx = flat_index(ci, cj, ck, ny, nz);

  const double m = 1.0;
  const double eta = 2.0;
  const double r0 = 8.0;
  const double theta0 = 0.7;
  const double phi0 = 0.2;
  const double dr = 0.25;
  const double dtheta = 0.15;
  const double dphi = 0.2;

  double *r = (double *)calloc((size_t)n, sizeof(double));
  double *theta = (double *)calloc((size_t)n, sizeof(double));
  double *phi = (double *)calloc((size_t)n, sizeof(double));
  double *g = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *g_u = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *alpha = (double *)calloc((size_t)n, sizeof(double));
  double *gamma = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *gamma_u = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *chi = (double *)calloc((size_t)n, sizeof(double));
  double *beta = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *B = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *K = (double *)calloc((size_t)n, sizeof(double));
  double *gammatilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *gammatilde_u = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *Atilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *Gammahat = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *rcoord = (double *)calloc((size_t)n, sizeof(double));
  double *radial_basis = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *dchi = (double *)calloc((size_t)n, sizeof(double));
  double *dalpha = (double *)calloc((size_t)n, sizeof(double));
  double *dbeta = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *dB = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *dK = (double *)calloc((size_t)n, sizeof(double));
  double *dgammatilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *dAtilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *dGammahat = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *ricci = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *hessian_alpha = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *datilde = (double *)calloc((size_t)(27 * n), sizeof(double));
  double *hamiltonian = (double *)calloc((size_t)n, sizeof(double));
  double *momentum = (double *)calloc((size_t)(3 * n), sizeof(double));

  double *all_ptrs[] = {
      r,          theta,        phi,        g,            g_u,
      alpha,      gamma,        gamma_u,    chi,          beta,
      B,          K,            gammatilde, gammatilde_u, Atilde,
      Gammahat,   rcoord,       radial_basis,
      dchi,       dalpha,       dbeta,      dB,           dK,
      dgammatilde, dAtilde,     dGammahat,  ricci,        hessian_alpha,
      datilde,    hamiltonian,  momentum,
  };
  const size_t all_count = sizeof(all_ptrs) / sizeof(all_ptrs[0]);
  for (size_t i = 0; i < all_count; ++i) {
    if (!all_ptrs[i]) {
      fprintf(stderr, "allocation failure\n");
      free_all(all_ptrs, all_count);
      return 2;
    }
  }

  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      for (int64_t k = 0; k < nz; ++k) {
        const int64_t p = flat_index(i, j, k, ny, nz);
        r[p] = r0 + (double)(i - ci) * dr;
        theta[p] = theta0 + (double)(j - cj) * dtheta;
        phi[p] = phi0 + (double)(k - ck) * dphi;
        rcoord[p] = r[p];
        chi[p] = 1.0;
        radial_basis[(int64_t)comp2(0, 0) * n + p] = 1.0;

        dchi[p] = NAN;
        dalpha[p] = NAN;
        dK[p] = NAN;
        hamiltonian[p] = NAN;
        for (int c = 0; c < 3; ++c) {
          dbeta[(int64_t)c * n + p] = NAN;
          dB[(int64_t)c * n + p] = NAN;
          dGammahat[(int64_t)c * n + p] = NAN;
          momentum[(int64_t)c * n + p] = NAN;
        }
        for (int c = 0; c < 9; ++c) {
          dgammatilde[(int64_t)c * n + p] = NAN;
          dAtilde[(int64_t)c * n + p] = NAN;
          ricci[(int64_t)c * n + p] = NAN;
          hessian_alpha[(int64_t)c * n + p] = NAN;
        }
        for (int c = 0; c < 27; ++c)
          datilde[(int64_t)c * n + p] = NAN;
      }
    }
  }

  tensorium_call_init_grid_affine(m, r, theta, phi, alpha, gamma, gamma_u, n);
  for (int64_t p = 0; p < n; ++p) {
    for (int c = 0; c < 9; ++c) {
      const double gij = gamma[(int64_t)c * n + p];
      const double guij = gamma_u[(int64_t)c * n + p];
      g[(int64_t)c * n + p] = gij;
      g_u[(int64_t)c * n + p] = guij;
      gammatilde[(int64_t)c * n + p] = gij;
      gammatilde_u[(int64_t)c * n + p] = guij;
    }
  }

  tensorium_call_rhs_grid_affine(
      nx, ny, nz, dr, dtheta, dphi, m, eta, g, g_u, alpha, gamma, gamma_u, chi,
      beta, B, K, gammatilde, gammatilde_u, Atilde, Gammahat, rcoord,
      radial_basis, dchi, dalpha, dbeta, dB, dK, dgammatilde, dAtilde,
      dGammahat, ricci, hessian_alpha, datilde, hamiltonian, momentum);

  printf("[analytic-schwarzschild-bssn] center coordinates M=%.17g eta=%.17g "
         "r=%.17g theta=%.17g phi=%.17g\n",
         m, eta, r[cidx], theta[cidx], phi[cidx]);
  printf("[analytic-schwarzschild-bssn] generated complete BSSN state\n");
  printf("chi = %.17g\n", chi[cidx]);
  printf("alpha = %.17g\n", alpha[cidx]);
  print_vector_at("beta^i", beta, n, cidx);
  print_vector_at("B^i", B, n, cidx);
  printf("K = %.17g\n", K[cidx]);
  print_matrix_at("gammatilde_ij", gammatilde, n, cidx);
  print_matrix_at("gammatilde^ij", gammatilde_u, n, cidx);
  print_matrix_at("Atilde_ij", Atilde, n, cidx);
  print_vector_at("Gammahat^i", Gammahat, n, cidx);
  printf("Rcoord = %.17g\n", rcoord[cidx]);
  print_matrix_at("radialBasis_ij", radial_basis, n, cidx);

  printf("[analytic-schwarzschild-bssn] generated complete BSSN RHS\n");
  printf("dchi = %.17g\n", dchi[cidx]);
  printf("dalpha = %.17g\n", dalpha[cidx]);
  print_vector_at("dbeta^i", dbeta, n, cidx);
  print_vector_at("dB^i", dB, n, cidx);
  printf("dK = %.17g\n", dK[cidx]);
  print_matrix_at("dgammatilde_ij", dgammatilde, n, cidx);
  print_matrix_at("dAtilde_ij", dAtilde, n, cidx);
  print_vector_at("dGammahat^i", dGammahat, n, cidx);

  printf("[analytic-schwarzschild-bssn] generated constraints\n");
  print_matrix_at("RicciAnalytic_ij", ricci, n, cidx);
  print_matrix_at("HessianAlpha_ij", hessian_alpha, n, cidx);
  print_tensor3_at("DAtilde_ijk", datilde, n, cidx);
  printf("Hamiltonian = %.17g\n", hamiltonian[cidx]);
  print_vector_at("Momentum_i", momentum, n, cidx);

  printf("[analytic-schwarzschild-bssn] exact center comparisons\n");
  int ok = check_point(cidx, n, m, r, theta, alpha, gamma, gamma_u, chi, beta,
                       B, K, gammatilde, gammatilde_u, Atilde, Gammahat,
                       dchi, dalpha, dbeta, dB, dK, dgammatilde, dAtilde,
                       dGammahat, ricci, hessian_alpha, datilde, hamiltonian,
                       momentum, 1);

  for (int64_t i = 1; i < nx - 1; ++i) {
    for (int64_t j = 1; j < ny - 1; ++j) {
      for (int64_t k = 1; k < nz - 1; ++k) {
        const int64_t p = flat_index(i, j, k, ny, nz);
        ok &= check_point(p, n, m, r, theta, alpha, gamma, gamma_u, chi, beta,
                          B, K, gammatilde, gammatilde_u, Atilde, Gammahat,
                          dchi, dalpha, dbeta, dB, dK, dgammatilde, dAtilde,
                          dGammahat, ricci, hessian_alpha, datilde,
                          hamiltonian, momentum, 0);
      }
    }
  }

  free_all(all_ptrs, all_count);

  if (!ok) {
    fprintf(stderr,
            "Schwarzschild complete BSSN analytic constraints mismatch\n");
    return 3;
  }
  return 0;
}
