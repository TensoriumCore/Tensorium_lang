#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void tensorium_rhs_grid_affine(
    int64_t nx, int64_t ny, int64_t nz, double dr, double dtheta, double dphi,
    double *chr_alloc, double *chr_aligned, int64_t chr_offset,
    int64_t chr_size, int64_t chr_stride, double *v_alloc, double *v_aligned,
    int64_t v_offset, int64_t v_size, int64_t v_stride, double *w_alloc,
    double *w_aligned, int64_t w_offset, int64_t w_size, int64_t w_stride,
    double *nabla_v_alloc, double *nabla_v_aligned, int64_t nabla_v_offset,
    int64_t nabla_v_size, int64_t nabla_v_stride, double *nabla_w_alloc,
    double *nabla_w_aligned, int64_t nabla_w_offset, int64_t nabla_w_size,
    int64_t nabla_w_stride, double *nabla_v_rhs_alloc,
    double *nabla_v_rhs_aligned, int64_t nabla_v_rhs_offset,
    int64_t nabla_v_rhs_size, int64_t nabla_v_rhs_stride,
    double *nabla_w_rhs_alloc, double *nabla_w_rhs_aligned,
    int64_t nabla_w_rhs_offset, int64_t nabla_w_rhs_size,
    int64_t nabla_w_rhs_stride);

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
  double *v = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *w = (double *)calloc((size_t)(3 * n), sizeof(double));
  double *nablaV = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *nablaW = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *nablaVRhs = (double *)calloc((size_t)(9 * n), sizeof(double));
  double *nablaWRhs = (double *)calloc((size_t)(9 * n), sizeof(double));
  if (!chr || !v || !w || !nablaV || !nablaW || !nablaVRhs || !nablaWRhs) {
    fprintf(stderr, "allocation failure\n");
    return 2;
  }

  for (int64_t p = 0; p < n; ++p) {
    // Gamma^0_{0 1} = 2, Gamma^0_{1 2} = 3.
    chr[(int64_t)comp3(0, 0, 1) * n + p] = 2.0;
    chr[(int64_t)comp3(0, 1, 2) * n + p] = 3.0;

    // V_i and W^i constants -> partial terms are exactly zero.
    v[(int64_t)0 * n + p] = 1.0;
    v[(int64_t)1 * n + p] = 2.0;
    v[(int64_t)2 * n + p] = 3.0;

    w[(int64_t)0 * n + p] = 4.0;
    w[(int64_t)1 * n + p] = 5.0;
    w[(int64_t)2 * n + p] = 6.0;
  }

  tensorium_rhs_grid_affine(
      nx, ny, nz, dr, dtheta, dphi, chr, chr, 0, 27 * n, 1, v, v, 0, 3 * n, 1,
      w, w, 0, 3 * n, 1, nablaV, nablaV, 0, 9 * n, 1, nablaW, nablaW, 0, 9 * n,
      1, nablaVRhs, nablaVRhs, 0, 9 * n, 1, nablaWRhs, nablaWRhs, 0, 9 * n, 1);

  double matV[3][3];
  double matW[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      matV[i][j] = nablaVRhs[(int64_t)comp2(i, j) * n + cidx];
      matW[i][j] = nablaWRhs[(int64_t)comp2(i, j) * n + cidx];
    }
  }

  printf("[ll-smoke] Covariant rank-1 center point\n");
  print_matrix3("nabla_j(V_i)", matV);
  print_matrix3("nabla_j(W^i)", matW);

  const double got_v_01 = matV[0][1];
  const double got_v_12 = matV[1][2];
  const double got_w_00 = matW[0][0];
  const double got_w_01 = matW[0][1];

  const double exp_v_01 = -2.0;
  const double exp_v_12 = -3.0;
  const double exp_w_00 = 10.0;
  const double exp_w_01 = 18.0;

  printf("nabla_j(V_i)[0,1] got=%.17g expected=%.17g\n", got_v_01, exp_v_01);
  printf("nabla_j(V_i)[1,2] got=%.17g expected=%.17g\n", got_v_12, exp_v_12);
  printf("nabla_j(W^i)[0,0] got=%.17g expected=%.17g\n", got_w_00, exp_w_00);
  printf("nabla_j(W^i)[0,1] got=%.17g expected=%.17g\n", got_w_01, exp_w_01);

  int ok = 1;
  ok &= almost_equal(got_v_01, exp_v_01, 1e-12, 1e-12);
  ok &= almost_equal(got_v_12, exp_v_12, 1e-12, 1e-12);
  ok &= almost_equal(got_w_00, exp_w_00, 1e-12, 1e-12);
  ok &= almost_equal(got_w_01, exp_w_01, 1e-12, 1e-12);

  free(chr);
  free(v);
  free(w);
  free(nablaV);
  free(nablaW);
  free(nablaVRhs);
  free(nablaWRhs);

  if (!ok) {
    fprintf(stderr, "Covariant rank-1 LLVM smoke mismatch\n");
    return 3;
  }
  return 0;
}
