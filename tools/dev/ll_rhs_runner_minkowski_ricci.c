#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void tensorium_init_grid_affine(
    double *x_alloc, double *x_aligned, int64_t x_offset, int64_t x_size,
    int64_t x_stride, double *y_alloc, double *y_aligned, int64_t y_offset,
    int64_t y_size, int64_t y_stride, double *z_alloc, double *z_aligned,
    int64_t z_offset, int64_t z_size, int64_t z_stride, double *alpha_alloc,
    double *alpha_aligned, int64_t alpha_offset, int64_t alpha_size,
    int64_t alpha_stride, double *gamma_alloc, double *gamma_aligned,
    int64_t gamma_offset, int64_t gamma_size, int64_t gamma_stride,
    double *gammaU_alloc, double *gammaU_aligned, int64_t gammaU_offset,
    int64_t gammaU_size, int64_t gammaU_stride);

extern void tensorium_rhs_grid_affine(
    int64_t nx, int64_t ny, int64_t nz, double dr, double dtheta, double dphi,
    double *gamma_alloc, double *gamma_aligned, int64_t gamma_offset,
    int64_t gamma_size, int64_t gamma_stride, double *gammaU_alloc,
    double *gammaU_aligned, int64_t gammaU_offset, int64_t gammaU_size,
    int64_t gammaU_stride, double *chr_alloc, double *chr_aligned,
    int64_t chr_offset, int64_t chr_size, int64_t chr_stride,
    double *ricci_alloc, double *ricci_aligned, int64_t ricci_offset,
    int64_t ricci_size, int64_t ricci_stride);

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

static void print_matrix4(const char *name, const double m[4][4]) {
  printf("%s = [[%.17g, %.17g, %.17g, %.17g],\n", name, m[0][0], m[0][1],
         m[0][2], m[0][3]);
  printf("      [%.17g, %.17g, %.17g, %.17g],\n", m[1][0], m[1][1], m[1][2],
         m[1][3]);
  printf("      [%.17g, %.17g, %.17g, %.17g],\n", m[2][0], m[2][1], m[2][2],
         m[2][3]);
  printf("      [%.17g, %.17g, %.17g, %.17g]]\n", m[3][0], m[3][1], m[3][2],
         m[3][3]);
}

int main(void) {
  const int64_t nx = 9;
  const int64_t ny = 9;
  const int64_t nz = 9;
  const int64_t n = nx * ny * nz;
  const int64_t ci = nx / 2;
  const int64_t cj = ny / 2;
  const int64_t ck = nz / 2;
  const int64_t cidx = flat_index(ci, cj, ck, ny, nz);

  // Minkowski metric in Cartesian coordinates:
  // gamma_ij = delta_ij, Christoffel^i_jk = 0, Ricci_ij = 0.
  const double x0 = 1.0;
  const double y0 = -0.25;
  const double z0 = 0.5;
  const double dr = 0.1;
  const double dtheta = 0.05;
  const double dphi = 0.1;

  double *x = (double *)malloc((size_t)n * sizeof(double));
  double *y = (double *)malloc((size_t)n * sizeof(double));
  double *z = (double *)malloc((size_t)n * sizeof(double));
  double *alpha = (double *)malloc((size_t)n * sizeof(double));
  double *gamma = (double *)malloc((size_t)(9 * n) * sizeof(double));
  double *gammaU = (double *)malloc((size_t)(9 * n) * sizeof(double));
  double *chr = (double *)malloc((size_t)(27 * n) * sizeof(double));
  double *ricci = (double *)malloc((size_t)(9 * n) * sizeof(double));
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

  tensorium_init_grid_affine(x, x, 0, n, 1, y, y, 0, n, 1, z, z, 0, n, 1,
                             alpha, alpha, 0, n, 1, gamma, gamma, 0, 9 * n,
                             1, gammaU, gammaU, 0, 9 * n, 1);

  tensorium_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, gamma, gamma, 0,
                            9 * n, 1, gammaU, gammaU, 0, 9 * n, 1, chr, chr, 0,
                            27 * n, 1, ricci, ricci, 0, 9 * n, 1);

  double g_cov[4][4] = {{0.0}};
  double gamma_cov[3][3] = {{0.0}};
  double gamma_con[3][3] = {{0.0}};
  double ricci_cov[3][3] = {{0.0}};
  double chr_r[3][3] = {{0.0}};
  double chr_th[3][3] = {{0.0}};
  double chr_ph[3][3] = {{0.0}};

  g_cov[0][0] = -alpha[cidx] * alpha[cidx];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      gamma_cov[i][j] = gamma[(int64_t)comp2(i, j) * n + cidx];
      gamma_con[i][j] = gammaU[(int64_t)comp2(i, j) * n + cidx];
      ricci_cov[i][j] = ricci[(int64_t)comp2(i, j) * n + cidx];
      g_cov[i + 1][j + 1] = gamma_cov[i][j];
    }
  }
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      chr_r[j][k] = chr[(int64_t)comp3(0, j, k) * n + cidx];
      chr_th[j][k] = chr[(int64_t)comp3(1, j, k) * n + cidx];
      chr_ph[j][k] = chr[(int64_t)comp3(2, j, k) * n + cidx];
    }
  }

  printf("[ll-smoke] Minkowski Ricci center point (Cartesian)\n");
  printf("  center coords: x=%.17g y=%.17g z=%.17g\n", x[cidx], y[cidx], z[cidx]);
  print_matrix4("g_uv (reconstructed)", g_cov);
  printf("alpha = %.17g\n", alpha[cidx]);
  print_matrix3("Gamma_ij (cov)", gamma_cov);
  print_matrix3("GammaU^ij (con)", gamma_con);
  print_matrix3("Christoffel^r_jk", chr_r);
  print_matrix3("Christoffel^theta_jk", chr_th);
  print_matrix3("Christoffel^phi_jk", chr_ph);
  print_matrix3("Ricci_ij", ricci_cov);

  const double expected_chr = 0.0;

  const double got_r_rr = chr[(int64_t)comp3(0, 0, 0) * n + cidx];
  const double got_r_thth = chr[(int64_t)comp3(0, 1, 1) * n + cidx];
  const double got_r_phph = chr[(int64_t)comp3(0, 2, 2) * n + cidx];
  const double got_th_rth = chr[(int64_t)comp3(1, 0, 1) * n + cidx];
  const double got_ph_rph = chr[(int64_t)comp3(2, 0, 2) * n + cidx];
  const double got_ph_thph = chr[(int64_t)comp3(2, 1, 2) * n + cidx];

  printf("Christoffel^r_rr      got=%.17g expected=0\n", got_r_rr);
  printf("Christoffel^r_yy      got=%.17g expected=0\n", got_r_thth);
  printf("Christoffel^r_zz      got=%.17g expected=0\n", got_r_phph);
  printf("Christoffel^y_xy      got=%.17g expected=0\n", got_th_rth);
  printf("Christoffel^z_xz      got=%.17g expected=0\n", got_ph_rph);
  printf("Christoffel^z_yz      got=%.17g expected=0\n", got_ph_thph);

  const double chr_rel_tol = 1e-12;
  const double chr_abs_tol = 1e-12;
  int ok = 1;
  ok &= almost_equal(got_r_rr, expected_chr, chr_rel_tol, chr_abs_tol);
  ok &= almost_equal(got_r_thth, expected_chr, chr_rel_tol, chr_abs_tol);
  ok &= almost_equal(got_r_phph, expected_chr, chr_rel_tol, chr_abs_tol);
  ok &= almost_equal(got_th_rth, expected_chr, chr_rel_tol, chr_abs_tol);
  ok &= almost_equal(got_ph_rph, expected_chr, chr_rel_tol, chr_abs_tol);
  ok &= almost_equal(got_ph_thph, expected_chr, chr_rel_tol, chr_abs_tol);

  const double ricci_abs_tol = 1e-12;
  double ricci_max_abs = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double v = fabs(ricci_cov[i][j]);
      if (v > ricci_max_abs)
        ricci_max_abs = v;
      ok &= isfinite(ricci_cov[i][j]);
      ok &= (v <= ricci_abs_tol);
    }
  }
  printf("Ricci max|component| = %.17g (expected ~0)\n", ricci_max_abs);

  free(x);
  free(y);
  free(z);
  free(alpha);
  free(gamma);
  free(gammaU);
  free(chr);
  free(ricci);

  if (!ok) {
    fprintf(stderr, "Ricci/Christoffel LLVM smoke mismatch beyond tolerance\n");
    return 3;
  }
  return 0;
}
