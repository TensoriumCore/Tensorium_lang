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
static int comp3(int i, int j, int k) { return (i * 3 + j) * 3 + k; }

static void print_matrix3(const char *name, const double m[3][3]) {
  printf("%s = [[%.17g, %.17g, %.17g],\n", name, m[0][0], m[0][1], m[0][2]);
  printf("      [%.17g, %.17g, %.17g],\n", m[1][0], m[1][1], m[1][2]);
  printf("      [%.17g, %.17g, %.17g]]\n", m[2][0], m[2][1], m[2][2]);
}

int main(void) {
  const int64_t nx = 16;
  const int64_t ny = 16;
  const int64_t nz = 16;
  const int64_t n = nx * ny * nz;
  const int64_t ci = nx / 2;
  const int64_t cj = ny / 2;
  const int64_t ck = nz / 2;
  const int64_t cidx = flat_index(ci, cj, ck, ny, nz);

  const double M = 1.0;
  const double x0 = 10.0;
  const double y0 = 1.0;
  const double z0 = 0.7;
  const double dr = 0.01;
  const double dtheta = 0.01;
  const double dphi = 0.01;

  double *x = (double *)malloc((size_t)n * sizeof(double));
  double *y = (double *)malloc((size_t)n * sizeof(double));
  double *z = (double *)malloc((size_t)n * sizeof(double));
  double *alpha = (double *)malloc((size_t)n * sizeof(double));
  double *gamma = (double *)malloc((size_t)(16 * n) * sizeof(double));
  double *gammaU = (double *)malloc((size_t)(16 * n) * sizeof(double));
  double *chr = (double *)malloc((size_t)(27 * n) * sizeof(double));
  double *ricci = (double *)malloc((size_t)(16 * n) * sizeof(double));
  if (!x || !y || !z || !alpha || !gamma || !gammaU || !chr || !ricci) {
    fprintf(stderr, "allocation failure\n");
    return 2;
  }

  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      for (int64_t k = 0; k < nz; ++k) {
        const int64_t idx = flat_index(i, j, k, ny, nz);
        x[idx] = x0 + (double)(i - ci) * dr;
        y[idx] = y0 + (double)(j - cj) * dtheta;
        z[idx] = z0 + (double)(k - ck) * dphi;
      }
    }
  }

  tensorium_call_init_grid_affine(M, x, y, z, alpha, gamma, gammaU, n);

  // First pass computes Christoffel from gamma/gammaU into chr.
  tensorium_call_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, gamma, gammaU,
                                 chr, ricci);
  // Second pass evaluates Ricci with the updated Christoffel state.
  tensorium_call_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, gamma, gammaU,
                                 chr, ricci);

  double gamma_cov[3][3] = {{0.0}};
  double gamma_con[3][3] = {{0.0}};
  double ricci_cov[3][3] = {{0.0}};
  double chr_r[3][3] = {{0.0}};
  double chr_th[3][3] = {{0.0}};
  double chr_ph[3][3] = {{0.0}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      gamma_cov[i][j] = gamma[(int64_t)comp2(i, j) * n + cidx];
      gamma_con[i][j] = gammaU[(int64_t)comp2(i, j) * n + cidx];
      ricci_cov[i][j] = ricci[(int64_t)comp2(i, j) * n + cidx];
    }
  }
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      chr_r[j][k] = chr[(int64_t)comp3(0, j, k) * n + cidx];
      chr_th[j][k] = chr[(int64_t)comp3(1, j, k) * n + cidx];
      chr_ph[j][k] = chr[(int64_t)comp3(2, j, k) * n + cidx];
    }
  }

  const double r = x[cidx];
  const double theta = y[cidx];
  const double f = 1.0 - 2.0 * M / r;
  const double exp_rr = -2.0 * M / (r * r * r * f);
  const double exp_tt = M / r;
  const double exp_pp = exp_tt * sin(theta) * sin(theta);
  const double exp_scalar = 0.0;

  double ricci_scalar = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ricci_scalar += gamma_con[i][j] * ricci_cov[i][j];

  printf("[ll-smoke] Schwarzschild spatial Ricci center point M=%.17g r=%.17g theta=%.17g\n",
         M, r, theta);
  printf("  note: this is the 3D Ricci(gamma_ij), not the 4D vacuum Ricci_mu_nu\n");
  print_matrix3("gamma_ij", gamma_cov);
  print_matrix3("gammaU^ij", gamma_con);
  print_matrix3("Christoffel^r_jk", chr_r);
  print_matrix3("Christoffel^theta_jk", chr_th);
  print_matrix3("Christoffel^phi_jk", chr_ph);
  print_matrix3("spatial Ricci_ij", ricci_cov);
  printf("spatial Ricci_rr expected=%.17g got=%.17g\n", exp_rr,
         ricci_cov[0][0]);
  printf("spatial Ricci_thetatheta expected=%.17g got=%.17g\n", exp_tt,
         ricci_cov[1][1]);
  printf("spatial Ricci_phiphi expected=%.17g got=%.17g\n", exp_pp,
         ricci_cov[2][2]);
  printf("spatial Ricci scalar gammaU^ij*Ricci_ij expected=%.17g got=%.17g\n",
         exp_scalar, ricci_scalar);

  int ok = 1;
  const double diag_rel_tol = 5e-2;
  const double diag_abs_tol = 5e-3;
  const double off_abs_tol = 5e-3;
  const double scalar_abs_tol = 5e-4;

  ok &= almost_equal(ricci_cov[0][0], exp_rr, diag_rel_tol, diag_abs_tol);
  ok &= almost_equal(ricci_cov[1][1], exp_tt, diag_rel_tol, diag_abs_tol);
  ok &= almost_equal(ricci_cov[2][2], exp_pp, diag_rel_tol, diag_abs_tol);
  ok &= fabs(ricci_scalar - exp_scalar) <= scalar_abs_tol;
  ok &= fabs(ricci_cov[0][1]) <= off_abs_tol;
  ok &= fabs(ricci_cov[1][0]) <= off_abs_tol;
  ok &= fabs(ricci_cov[0][2]) <= off_abs_tol;
  ok &= fabs(ricci_cov[2][0]) <= off_abs_tol;
  ok &= fabs(ricci_cov[1][2]) <= off_abs_tol;
  ok &= fabs(ricci_cov[2][1]) <= off_abs_tol;

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ok &= isfinite(ricci_cov[i][j]);

  free(x);
  free(y);
  free(z);
  free(alpha);
  free(gamma);
  free(gammaU);
  free(chr);
  free(ricci);

  if (!ok) {
    fprintf(stderr, "Schwarzschild spatial Ricci mismatch beyond tolerance\n");
    return 3;
  }
  return 0;
}
