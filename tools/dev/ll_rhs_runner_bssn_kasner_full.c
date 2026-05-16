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

static double exact_tol(double expected) {
  const double scale = fmax(1.0, fabs(expected));
  return 512.0 * DBL_EPSILON * scale;
}

static int check_value(const char *name, double got, double expected) {
  const double diff = fabs(got - expected);
  const double tol = exact_tol(expected);
  printf("  %-24s got=% .17g expected=% .17g diff=%.3e tol=%.3e\n",
         name, got, expected, diff, tol);
  if (diff <= tol)
    return 1;
  fprintf(stderr, "%s mismatch: got %.17g expected %.17g\n", name, got,
          expected);
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

static int check_vector_at(const char *name, const double *got,
                           const double expected[3], int64_t n, int64_t p) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    char label[80];
    (void)snprintf(label, sizeof(label), "%s[%d]", name, i);
    ok &= check_value(label, got[(int64_t)i * n + p], expected[i]);
  }
  return ok;
}

static int check_matrix_at(const char *name, const double *got,
                           const double expected[9], int64_t n, int64_t p) {
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      char label[80];
      (void)snprintf(label, sizeof(label), "%s[%d,%d]", name, i, j);
      ok &= check_value(label, got[(int64_t)comp2(i, j) * n + p],
                        expected[comp2(i, j)]);
    }
  }
  return ok;
}

static void free_all(double *chi, double *alpha, double *beta, double *B,
                     double *K, double *gammatilde, double *gammatildeU,
                     double *Atilde, double *Gammahat, double *dchi,
                     double *dalpha, double *dbeta, double *dB, double *dK,
                     double *dgammatilde, double *dAtilde,
                     double *dGammahat) {
  free(chi);
  free(alpha);
  free(beta);
  free(B);
  free(K);
  free(gammatilde);
  free(gammatildeU);
  free(Atilde);
  free(Gammahat);
  free(dchi);
  free(dalpha);
  free(dbeta);
  free(dB);
  free(dK);
  free(dgammatilde);
  free(dAtilde);
  free(dGammahat);
}

int main(void) {
  const int64_t nx = 3;
  const int64_t ny = 3;
  const int64_t nz = 3;
  const int64_t n = nx * ny * nz;
  const int64_t cidx = flat_index(1, 1, 1, ny, nz);
  const double eta = 2.0;

  double *chi = (double *)calloc((size_t)n, sizeof(double));
  double *alpha = (double *)calloc((size_t)n, sizeof(double));
  double *beta = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *B = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *K = (double *)calloc((size_t)n, sizeof(double));
  double *gammatilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *gammatildeU = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *Atilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *Gammahat = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *dchi = (double *)calloc((size_t)n, sizeof(double));
  double *dalpha = (double *)calloc((size_t)n, sizeof(double));
  double *dbeta = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *dB = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *dK = (double *)calloc((size_t)n, sizeof(double));
  double *dgammatilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *dAtilde = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *dGammahat = (double *)calloc((size_t)(3 * n), sizeof(double));

  if (!chi || !alpha || !beta || !B || !K || !gammatilde || !gammatildeU ||
      !Atilde || !Gammahat || !dchi || !dalpha || !dbeta || !dB || !dK ||
      !dgammatilde || !dAtilde || !dGammahat) {
    fprintf(stderr, "allocation failure\n");
    free_all(chi, alpha, beta, B, K, gammatilde, gammatildeU, Atilde, Gammahat,
             dchi, dalpha, dbeta, dB, dK, dgammatilde, dAtilde, dGammahat);
    return 2;
  }

  for (int64_t p = 0; p < n; ++p) {
    chi[p] = 1.0;
    alpha[p] = 1.0;
    K[p] = -1.0;

    gammatilde[(int64_t)comp2(0, 0) * n + p] = 1.0;
    gammatilde[(int64_t)comp2(1, 1) * n + p] = 1.0;
    gammatilde[(int64_t)comp2(2, 2) * n + p] = 1.0;
    gammatildeU[(int64_t)comp2(0, 0) * n + p] = 1.0;
    gammatildeU[(int64_t)comp2(1, 1) * n + p] = 1.0;
    gammatildeU[(int64_t)comp2(2, 2) * n + p] = 1.0;

    Atilde[(int64_t)comp2(0, 0) * n + p] = -1.0 / 3.0;
    Atilde[(int64_t)comp2(1, 1) * n + p] = -1.0 / 3.0;
    Atilde[(int64_t)comp2(2, 2) * n + p] = 2.0 / 3.0;

    dchi[p] = NAN;
    dalpha[p] = NAN;
    dK[p] = NAN;
    for (int c = 0; c < 3; ++c) {
      dbeta[(int64_t)c * n + p] = NAN;
      dB[(int64_t)c * n + p] = NAN;
      dGammahat[(int64_t)c * n + p] = NAN;
    }
    for (int c = 0; c < 9; ++c) {
      dgammatilde[(int64_t)c * n + p] = NAN;
      dAtilde[(int64_t)c * n + p] = NAN;
    }
  }

  tensorium_call_rhs_grid_affine(nx, ny, nz, 1.0, 1.0, 1.0, eta, chi, alpha,
                                 beta, B, K, gammatilde, gammatildeU, Atilde,
                                 Gammahat, dchi, dalpha, dbeta, dB, dK,
                                 dgammatilde, dAtilde, dGammahat);

  const double zero3[3] = {0.0, 0.0, 0.0};
  const double expected_dgammatilde[9] = {
      2.0 / 3.0, 0.0, 0.0,
      0.0, 2.0 / 3.0, 0.0,
      0.0, 0.0, -4.0 / 3.0,
  };
  const double expected_dAtilde[9] = {
      1.0 / 9.0, 0.0, 0.0,
      0.0, 1.0 / 9.0, 0.0,
      0.0, 0.0, -14.0 / 9.0,
  };

  printf("[analytic-bssn] Complete BSSN Kasner center state\n");
  printf("chi = %.17g\n", chi[cidx]);
  printf("alpha = %.17g\n", alpha[cidx]);
  print_vector_at("beta^i", beta, n, cidx);
  print_vector_at("B^i", B, n, cidx);
  printf("K = %.17g\n", K[cidx]);
  print_matrix_at("gammatilde_ij", gammatilde, n, cidx);
  print_matrix_at("gammatilde^ij", gammatildeU, n, cidx);
  print_matrix_at("Atilde_ij", Atilde, n, cidx);
  print_vector_at("Gammahat^i", Gammahat, n, cidx);

  printf("[analytic-bssn] Complete BSSN Kasner generated RHS\n");
  printf("dchi = %.17g\n", dchi[cidx]);
  printf("dalpha = %.17g\n", dalpha[cidx]);
  print_vector_at("dbeta^i", dbeta, n, cidx);
  print_vector_at("dB^i", dB, n, cidx);
  printf("dK = %.17g\n", dK[cidx]);
  print_matrix_at("dgammatilde_ij", dgammatilde, n, cidx);
  print_matrix_at("dAtilde_ij", dAtilde, n, cidx);
  print_vector_at("dGammahat^i", dGammahat, n, cidx);

  printf("[analytic-bssn] Exact component comparisons\n");
  int ok = 1;
  ok &= check_value("dchi", dchi[cidx], -2.0 / 3.0);
  ok &= check_value("dalpha", dalpha[cidx], 2.0);
  ok &= check_vector_at("dbeta", dbeta, zero3, n, cidx);
  ok &= check_vector_at("dB", dB, zero3, n, cidx);
  ok &= check_value("dK", dK[cidx], 1.0);
  ok &= check_matrix_at("dgammatilde", dgammatilde, expected_dgammatilde, n,
                        cidx);
  ok &= check_matrix_at("dAtilde", dAtilde, expected_dAtilde, n, cidx);
  ok &= check_vector_at("dGammahat", dGammahat, zero3, n, cidx);

  free_all(chi, alpha, beta, B, K, gammatilde, gammatildeU, Atilde, Gammahat,
           dchi, dalpha, dbeta, dB, dK, dgammatilde, dAtilde, dGammahat);

  if (!ok) {
    fprintf(stderr, "Complete BSSN Kasner analytic RHS mismatch\n");
    return 3;
  }
  return 0;
}
