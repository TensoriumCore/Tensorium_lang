#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Lowered from memref<1xf64> + memref<9xf64> descriptors.
extern void tensorium_init_point(double M, double r, double theta, double phi,
                                 double *alpha_alloc, double *alpha_aligned,
                                 int64_t alpha_offset, int64_t alpha_size,
                                 int64_t alpha_stride, double *gamma_alloc,
                                 double *gamma_aligned, int64_t gamma_offset,
                                 int64_t gamma_size, int64_t gamma_stride,
                                 double *gammaU_alloc, double *gammaU_aligned,
                                 int64_t gammaU_offset, int64_t gammaU_size,
                                 int64_t gammaU_stride);

static int almost_equal(double got, double expected, double rel_tol,
                        double abs_tol) {
  const double diff = fabs(got - expected);
  const double scale = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
  const double tol = fmax(abs_tol, rel_tol * scale);
  return diff <= tol;
}

static void print_mat3(const char *name, const double *m) {
  printf("%s = [[%.17g, %.17g, %.17g],\n", name, m[0], m[1], m[2]);
  printf("      [%.17g, %.17g, %.17g],\n", m[3], m[4], m[5]);
  printf("      [%.17g, %.17g, %.17g]]\n", m[6], m[7], m[8]);
}

int main(void) {
  const double M = 1.0;
  const double r = 10.0;
  const double theta = 1.5707963267948966; // pi/2
  const double phi = 0.0;
  const double f = 1.0 - (2.0 * M / r);

  double alpha[1] = {0.0};
  double gamma[9] = {0.0};
  double gammaU[9] = {0.0};

  tensorium_init_point(M, r, theta, phi, alpha, alpha, 0, 1, 1, gamma, gamma,
                       0, 9, 1, gammaU, gammaU, 0, 9, 1);

  const double alpha_expected = sqrt(f);
  const double gamma_expected[9] = {1.0 / f, 0.0, 0.0, 0.0, r * r, 0.0,
                                    0.0,     0.0, r * r};
  const double gammaU_expected[9] = {f, 0.0, 0.0, 0.0, 1.0 / (r * r), 0.0,
                                     0.0, 0.0, 1.0 / (r * r)};

  printf("[ll-smoke] Schwarzschild init point M=%.17g r=%.17g theta=%.17g\n", M,
         r, theta);
  printf("alpha = %.17g (expected %.17g)\n", alpha[0], alpha_expected);
  print_mat3("gamma_ij", gamma);
  print_mat3("gammaU^ij", gammaU);

  if (!almost_equal(alpha[0], alpha_expected, 1e-12, 1e-12)) {
    fprintf(stderr, "alpha mismatch: got %.17g expected %.17g\n", alpha[0],
            alpha_expected);
    return 2;
  }

  for (int i = 0; i < 9; ++i) {
    if (!almost_equal(gamma[i], gamma_expected[i], 1e-12, 1e-12)) {
      fprintf(stderr, "gamma mismatch at %d: got %.17g expected %.17g\n", i,
              gamma[i], gamma_expected[i]);
      return 3;
    }
    if (!almost_equal(gammaU[i], gammaU_expected[i], 1e-12, 1e-12)) {
      fprintf(stderr, "gammaU mismatch at %d: got %.17g expected %.17g\n", i,
              gammaU[i], gammaU_expected[i]);
      return 4;
    }
  }

  return 0;
}
