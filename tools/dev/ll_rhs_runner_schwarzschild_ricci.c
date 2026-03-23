#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void tensorium_init_grid_affine(
    double unused_param, double *x_alloc, double *x_aligned, int64_t x_offset,
    int64_t x_size, int64_t x_stride, double *y_alloc, double *y_aligned,
    int64_t y_offset, int64_t y_size, int64_t y_stride, double *z_alloc,
    double *z_aligned, int64_t z_offset, int64_t z_size, int64_t z_stride,
    double *alpha_alloc, double *alpha_aligned,
    int64_t alpha_offset, int64_t alpha_size, int64_t alpha_stride,
    double *gamma_alloc, double *gamma_aligned, int64_t gamma_offset,
    int64_t gamma_size, int64_t gamma_stride, double *gammaU_alloc,
    double *gammaU_aligned, int64_t gammaU_offset, int64_t gammaU_size,
    int64_t gammaU_stride);

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

static void print_matrix3(const char *name, const double m[3][3]) {
  printf("%s = [[%.17g, %.17g, %.17g],\n", name, m[0][0], m[0][1], m[0][2]);
  printf("      [%.17g, %.17g, %.17g],\n", m[1][0], m[1][1], m[1][2]);
  printf("      [%.17g, %.17g, %.17g]]\n", m[2][0], m[2][1], m[2][2]);
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

  const double M = 1.0;
  const double x0 = 10.0;
  const double y0 = 1.0;
  const double z0 = 0.7;
  const double dr = 0.1;
  const double dtheta = 0.05;
  const double dphi = 0.05;

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

  tensorium_init_grid_affine(M, x, x, 0, n, 1, y, y, 0, n, 1, z, z, 0, n, 1,
                             alpha, alpha, 0, n, 1, gamma, gamma, 0, 9 * n, 1,
                             gammaU, gammaU, 0, 9 * n, 1);

  // First pass computes Christoffel from gamma/gammaU into chr.
  tensorium_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, gamma, gamma, 0,
                            9 * n, 1, gammaU, gammaU, 0, 9 * n, 1, chr, chr, 0,
                            27 * n, 1, ricci, ricci, 0, 9 * n, 1);
  // Second pass evaluates Ricci with the updated Christoffel state.
  tensorium_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, gamma, gamma, 0,
                            9 * n, 1, gammaU, gammaU, 0, 9 * n, 1, chr, chr, 0,
                            27 * n, 1, ricci, ricci, 0, 9 * n, 1);

  double ricci_cov[3][3] = {{0.0}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ricci_cov[i][j] = ricci[(int64_t)comp2(i, j) * n + cidx];

  const double r = x[cidx];
  const double theta = y[cidx];
  const double f = 1.0 - 2.0 * M / r;
  const double exp_rr = -2.0 * M / (r * r * r * f);
  const double exp_tt = M / r;
  const double exp_pp = exp_tt * sin(theta) * sin(theta);

  printf("[ll-smoke] Schwarzschild Ricci center point M=%.17g r=%.17g theta=%.17g\n",
         M, r, theta);
  print_matrix3("Ricci_ij", ricci_cov);
  printf("Ricci_rr expected=%.17g got=%.17g\n", exp_rr, ricci_cov[0][0]);
  printf("Ricci_thetatheta expected=%.17g got=%.17g\n", exp_tt, ricci_cov[1][1]);
  printf("Ricci_phiphi expected=%.17g got=%.17g\n", exp_pp, ricci_cov[2][2]);

  int ok = 1;
  const double diag_rel_tol = 5e-2;
  const double diag_abs_tol = 5e-3;
  const double off_abs_tol = 5e-3;

  ok &= almost_equal(ricci_cov[0][0], exp_rr, diag_rel_tol, diag_abs_tol);
  ok &= almost_equal(ricci_cov[1][1], exp_tt, diag_rel_tol, diag_abs_tol);
  ok &= almost_equal(ricci_cov[2][2], exp_pp, diag_rel_tol, diag_abs_tol);
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
    fprintf(stderr, "Schwarzschild Ricci mismatch beyond tolerance\n");
    return 3;
  }
  return 0;
}
