#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void tensorium_init_grid_affine(
    double p0, double *x_alloc, double *x_aligned, int64_t x_offset,
    int64_t x_size, int64_t x_stride, double *y_alloc, double *y_aligned,
    int64_t y_offset, int64_t y_size, int64_t y_stride, double *z_alloc,
    double *z_aligned, int64_t z_offset, int64_t z_size, int64_t z_stride,
    double *alpha_alloc, double *alpha_aligned, int64_t alpha_offset,
    int64_t alpha_size, int64_t alpha_stride, double *gamma_alloc,
    double *gamma_aligned, int64_t gamma_offset, int64_t gamma_size,
    int64_t gamma_stride, double *gammaU_alloc, double *gammaU_aligned,
    int64_t gammaU_offset, int64_t gammaU_size, int64_t gammaU_stride);

extern void tensorium_rhs_grid_affine(
    int64_t nx, int64_t ny, int64_t nz, double dx, double dy, double dz,
    double *gamma_alloc, double *gamma_aligned, int64_t gamma_offset,
    int64_t gamma_size, int64_t gamma_stride, double *gammaU_alloc,
    double *gammaU_aligned, int64_t gammaU_offset, int64_t gammaU_size,
    int64_t gammaU_stride, double *chr_alloc, double *chr_aligned,
    int64_t chr_offset, int64_t chr_size, int64_t chr_stride,
    double *ricci_alloc, double *ricci_aligned, int64_t ricci_offset,
    int64_t ricci_size, int64_t ricci_stride);

static int64_t flat_index(int64_t i, int64_t j, int64_t k, int64_t ny,
                          int64_t nz) {
  return (i * ny + j) * nz + k;
}

static int comp2(int i, int j) { return i * 3 + j; }

int main(int argc, char **argv) {
  const char *csv_path =
      (argc > 1) ? argv[1] : "/tmp/bh_cartesian_slice64.csv";

  const int64_t nx = 64;
  const int64_t ny = 64;
  const int64_t nz = 64;
  const int64_t n = nx * ny * nz;
  const int64_t ck = nz / 2;

  const double M = 1.0;
  const double dx = 0.25;
  const double dy = 0.25;
  const double dz = 0.25;

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

  const int64_t ci = nx / 2;
  const int64_t cj = ny / 2;
  const int64_t ck0 = nz / 2;
  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      for (int64_t k = 0; k < nz; ++k) {
        const int64_t idx = flat_index(i, j, k, ny, nz);
        x[idx] = (double)(i - ci) * dx;
        y[idx] = (double)(j - cj) * dy;
        z[idx] = (double)(k - ck0) * dz;
      }
    }
  }

  tensorium_init_grid_affine(M, x, x, 0, n, 1, y, y, 0, n, 1, z, z, 0, n, 1,
                             alpha, alpha, 0, n, 1, gamma, gamma, 0, 9 * n, 1,
                             gammaU, gammaU, 0, 9 * n, 1);

  // Pass 1 computes Christoffel; pass 2 evaluates Ricci with updated Christoffel.
  tensorium_rhs_grid_affine(nx, ny, nz, dx, dy, dz, gamma, gamma, 0, 9 * n, 1,
                            gammaU, gammaU, 0, 9 * n, 1, chr, chr, 0,
                            27 * n, 1, ricci, ricci, 0, 9 * n, 1);
  tensorium_rhs_grid_affine(nx, ny, nz, dx, dy, dz, gamma, gamma, 0, 9 * n, 1,
                            gammaU, gammaU, 0, 9 * n, 1, chr, chr, 0,
                            27 * n, 1, ricci, ricci, 0, 9 * n, 1);

  FILE *f = fopen(csv_path, "w");
  if (!f) {
    perror("fopen");
    return 3;
  }

  // Here grr is exported as gamma_xx on the Cartesian slice.
  fprintf(f, "i,j,k,x,y,z,alpha,grr,ricci_xx,ricci_xy,ricci_yy,ricci_trace\n");

  int finite_fail = 0;
  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      const int64_t idx = flat_index(i, j, ck, ny, nz);

      const double gu_xx = gammaU[(int64_t)comp2(0, 0) * n + idx];
      const double gu_xy = gammaU[(int64_t)comp2(0, 1) * n + idx];
      const double gu_yx = gammaU[(int64_t)comp2(1, 0) * n + idx];
      const double gu_yy = gammaU[(int64_t)comp2(1, 1) * n + idx];

      const double g_rr = gamma[(int64_t)comp2(0, 0) * n + idx];
      const double r_xx = ricci[(int64_t)comp2(0, 0) * n + idx];
      const double r_xy = ricci[(int64_t)comp2(0, 1) * n + idx];
      const double r_yx = ricci[(int64_t)comp2(1, 0) * n + idx];
      const double r_yy = ricci[(int64_t)comp2(1, 1) * n + idx];

      const double r_trace =
          gu_xx * r_xx + gu_xy * r_yx + gu_yx * r_xy + gu_yy * r_yy;

      if (!isfinite(alpha[idx]) || !isfinite(g_rr) || !isfinite(r_xx) ||
          !isfinite(r_xy) || !isfinite(r_yy) || !isfinite(r_trace)) {
        finite_fail = 1;
      }

      fprintf(
          f,
          "%lld,%lld,%lld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
          (long long)i, (long long)j, (long long)ck, x[idx], y[idx], z[idx],
          alpha[idx], g_rr, r_xx, r_xy, r_yy, r_trace);
    }
  }

  fclose(f);
  free(x);
  free(y);
  free(z);
  free(alpha);
  free(gamma);
  free(gammaU);
  free(chr);
  free(ricci);

  if (finite_fail) {
    fprintf(stderr, "non-finite values found in exported slice\n");
    return 4;
  }

  printf("CSV exported: %s\n", csv_path);
  return 0;
}
