#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

double heat_source(double value) { return 2.0 * value + 3.0; }

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

int main(void) {
  const int64_t nx = 7;
  const int64_t ny = 7;
  const int64_t nz = 7;
  const int64_t n = nx * ny * nz;
  const int64_t ci = nx / 2;
  const int64_t cj = ny / 2;
  const int64_t ck = nz / 2;
  const int64_t cidx = flat_index(ci, cj, ck, ny, nz);

  double *phi = (double *)calloc((size_t)n, sizeof(double));
  double *rhs = (double *)calloc((size_t)n, sizeof(double));
  if (!phi || !rhs) {
    fprintf(stderr, "allocation failure\n");
    return 2;
  }

  for (int64_t p = 0; p < n; ++p)
    phi[p] = 0.25 * (double)p;

  tensorium_call_rhs_grid_affine(nx, ny, nz, 1.0, 1.0, 1.0, phi, rhs);

  const double got = rhs[cidx];
  const double expected = heat_source(phi[cidx]);
  printf("[ll-smoke] extern scalar RHS center got=%.17g expected=%.17g\n",
         got, expected);

  free(phi);
  free(rhs);

  if (!almost_equal(got, expected, 1e-12, 1e-12)) {
    fprintf(stderr, "extern scalar RHS LLVM smoke mismatch\n");
    return 3;
  }
  return 0;
}
