#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void tensorium_rhs_grid_affine(
    int64_t nx, int64_t ny, int64_t nz, double dr, double dtheta, double dphi,
    double *chr_alloc, double *chr_aligned, int64_t chr_offset,
    int64_t chr_size, int64_t chr_stride, double *gamma_u_alloc,
    double *gamma_u_aligned, int64_t gamma_u_offset, int64_t gamma_u_size,
    int64_t gamma_u_stride, double *v_alloc, double *v_aligned,
    int64_t v_offset, int64_t v_size, int64_t v_stride, double *w_alloc,
    double *w_aligned, int64_t w_offset, int64_t w_size, int64_t w_stride,
    double *a_alloc, double *a_aligned, int64_t a_offset, int64_t a_size,
    int64_t a_stride, double *nabla_up_v_alloc,
    double *nabla_up_v_aligned, int64_t nabla_up_v_offset,
    int64_t nabla_up_v_size, int64_t nabla_up_v_stride,
    double *nabla_up_w_alloc, double *nabla_up_w_aligned,
    int64_t nabla_up_w_offset, int64_t nabla_up_w_size,
    int64_t nabla_up_w_stride, double *nabla_up_a_alloc,
    double *nabla_up_a_aligned, int64_t nabla_up_a_offset,
    int64_t nabla_up_a_size, int64_t nabla_up_a_stride);

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
  printf("%s = [[%.17g, %.17g, %.17g],\n", name, m[0][0], m[0][1],
         m[0][2]);
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

  const double dr = 1.0;
  const double dtheta = 1.0;
  const double dphi = 1.0;

  double *chr = (double *)calloc((size_t)(27 * n), sizeof(double));
  double *gammaU = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *v = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *w = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *a = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *nablaUpV = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *nablaUpW = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *nablaUpA = (double *)calloc((size_t)(27 * n), sizeof(double));
  if (!chr || !gammaU || !v || !w || !a || !nablaUpV || !nablaUpW ||
      !nablaUpA) {
    fprintf(stderr, "allocation failure\n");
    return 2;
  }

  for (int64_t p = 0; p < n; ++p) {
    chr[(int64_t)comp3(0, 0, 1) * n + p] = 2.0;
    chr[(int64_t)comp3(0, 1, 2) * n + p] = 3.0;
    chr[(int64_t)comp3(2, 2, 1) * n + p] = 5.0;

    gammaU[(int64_t)comp2(0, 0) * n + p] = 1.0;
    gammaU[(int64_t)comp2(1, 1) * n + p] = 1.0;
    gammaU[(int64_t)comp2(2, 2) * n + p] = 1.0;

    v[(int64_t)0 * n + p] = 1.0;
    w[(int64_t)2 * n + p] = 6.0;

    a[(int64_t)comp2(2, 2) * n + p] = 7.0;
    a[(int64_t)comp2(0, 2) * n + p] = 11.0;
  }

  tensorium_rhs_grid_affine(
      nx, ny, nz, dr, dtheta, dphi, chr, chr, 0, 27 * n, 1, gammaU, gammaU, 0,
      9 * n, 1, v, v, 0, 3 * n, 1, w, w, 0, 3 * n, 1, a, a, 0, 9 * n, 1,
      nablaUpV, nablaUpV, 0, 9 * n, 1, nablaUpW, nablaUpW, 0, 9 * n, 1,
      nablaUpA, nablaUpA, 0, 27 * n, 1);

  double matV[3][3];
  double matW[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      matV[i][j] = nablaUpV[(int64_t)comp2(i, j) * n + cidx];
      matW[i][j] = nablaUpW[(int64_t)comp2(i, j) * n + cidx];
    }
  }

  const double got_v_10 = matV[1][0];
  const double got_w_01 = matW[0][1];
  const double got_a_012 = nablaUpA[(int64_t)comp3(0, 1, 2) * n + cidx];

  const double exp_v_10 = -2.0;
  const double exp_w_01 = 18.0;
  const double exp_a_012 = -34.0;

  printf("[ll-smoke] Contravariant nabla all-cases center point\n");
  print_matrix3("nabla^i(V_j)", matV);
  print_matrix3("nabla^j(W^i)", matW);
  printf("nabla^k(A^i_j)[0,1,2] got=%.17g expected=%.17g\n", got_a_012,
         exp_a_012);
  printf("nabla^i(V_j)[1,0] got=%.17g expected=%.17g\n", got_v_10,
         exp_v_10);
  printf("nabla^j(W^i)[0,1] got=%.17g expected=%.17g\n", got_w_01,
         exp_w_01);

  int ok = 1;
  ok &= almost_equal(got_v_10, exp_v_10, 1e-12, 1e-12);
  ok &= almost_equal(got_w_01, exp_w_01, 1e-12, 1e-12);
  ok &= almost_equal(got_a_012, exp_a_012, 1e-12, 1e-12);

  free(chr);
  free(gammaU);
  free(v);
  free(w);
  free(a);
  free(nablaUpV);
  free(nablaUpW);
  free(nablaUpA);

  if (!ok) {
    fprintf(stderr, "Contravariant nabla LLVM smoke mismatch\n");
    return 3;
  }
  return 0;
}
