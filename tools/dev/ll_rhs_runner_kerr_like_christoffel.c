#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void tensorium_init_grid_affine(
    double M, double *r_alloc, double *r_aligned, int64_t r_offset,
    int64_t r_size, int64_t r_stride, double *theta_alloc,
    double *theta_aligned, int64_t theta_offset, int64_t theta_size,
    int64_t theta_stride, double *phi_alloc, double *phi_aligned,
    int64_t phi_offset, int64_t phi_size, int64_t phi_stride,
    double *alpha_alloc, double *alpha_aligned, int64_t alpha_offset,
    int64_t alpha_size, int64_t alpha_stride, double *gamma_alloc,
    double *gamma_aligned, int64_t gamma_offset, int64_t gamma_size,
    int64_t gamma_stride, double *gammaU_alloc, double *gammaU_aligned,
    int64_t gammaU_offset, int64_t gammaU_size, int64_t gammaU_stride);

extern void tensorium_rhs_grid_affine(
    int64_t nx, int64_t ny, int64_t nz, double dr, double dtheta, double dphi,
    double *gamma_alloc, double *gamma_aligned, int64_t gamma_offset,
    int64_t gamma_size, int64_t gamma_stride, double *gammaU_alloc,
    double *gammaU_aligned, int64_t gammaU_offset, int64_t gammaU_size,
    int64_t gammaU_stride, double *chr_alloc, double *chr_aligned,
    int64_t chr_offset, int64_t chr_size, int64_t chr_stride,
    double *chr_rhs_alloc, double *chr_rhs_aligned, int64_t chr_rhs_offset,
    int64_t chr_rhs_size, int64_t chr_rhs_stride);

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

static int comp_index(int iu, int j, int k) { return (iu * 3 + j) * 3 + k; }

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
  const int64_t center_i = nx / 2;
  const int64_t center_j = ny / 2;
  const int64_t center_k = nz / 2;
  const int64_t cidx = flat_index(center_i, center_j, center_k, ny, nz);

  const double M = 1.0;
  const double a = 0.3;
  const double r0 = 10.0;
  const double theta0 = 1.0;
  const double phi0 = 0.5;
  const double dr = 0.1;
  const double dtheta = 0.05;
  const double dphi = 0.1;

  double *r = (double *)malloc((size_t)n * sizeof(double));
  double *theta = (double *)malloc((size_t)n * sizeof(double));
  double *phi = (double *)malloc((size_t)n * sizeof(double));
  double *alpha = (double *)malloc((size_t)n * sizeof(double));
  double *gamma = (double *)malloc((size_t)(9 * n) * sizeof(double));
  double *gammaU = (double *)malloc((size_t)(9 * n) * sizeof(double));
  double *chr = (double *)calloc((size_t)(27 * n), sizeof(double));
  double *chrRhs = (double *)malloc((size_t)(27 * n) * sizeof(double));
  if (!r || !theta || !phi || !alpha || !gamma || !gammaU || !chr || !chrRhs) {
    fprintf(stderr, "allocation failure\n");
    return 2;
  }

  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      for (int64_t k = 0; k < nz; ++k) {
        const int64_t idx = flat_index(i, j, k, ny, nz);
        r[idx] = r0 + (double)(i - center_i) * dr;
        theta[idx] = theta0 + (double)(j - center_j) * dtheta;
        phi[idx] = phi0 + (double)(k - center_k) * dphi;
      }
    }
  }

  tensorium_init_grid_affine(M, r, r, 0, n, 1, theta, theta, 0, n, 1, phi, phi,
                             0, n, 1, alpha, alpha, 0, n, 1, gamma, gamma, 0,
                             9 * n, 1, gammaU, gammaU, 0, 9 * n, 1);

  tensorium_rhs_grid_affine(nx, ny, nz, dr, dtheta, dphi, gamma, gamma, 0,
                            9 * n, 1, gammaU, gammaU, 0, 9 * n, 1, chr, chr, 0,
                            27 * n, 1, chrRhs, chrRhs, 0, 27 * n, 1);

  const char *upper_name[3] = {"r", "theta", "phi"};
  printf("[ll-smoke] Kerr-like Christoffel center point M=%.17g a=%.17g "
         "r=%.17g theta=%.17g\n",
         M, a, r[cidx], theta[cidx]);

  for (int iu = 0; iu < 3; ++iu) {
    double mat[3][3];
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const int comp = comp_index(iu, j, k);
        mat[j][k] = chrRhs[(int64_t)comp * n + cidx];
      }
    }
    char label[64];
    (void)snprintf(label, sizeof(label), "Gamma^%s_{jk}", upper_name[iu]);
    print_matrix3(label, mat);
  }

  // Spatial Christoffels are expected to match Schwarzschild for this
  // simplified Kerr-like metric because only g_tphi carries the a-dependent
  // term.
  const double f = 1.0 - 2.0 * M / r0;
  const double expected_r_rr = -M / (r0 * (r0 - 2.0 * M));
  const double expected_r_thth = -r0 * f;
  const double expected_r_phph = -r0 * f * sin(theta0) * sin(theta0);
  const double expected_th_rth = 1.0 / r0;
  const double expected_ph_rph = 1.0 / r0;
  const double expected_ph_thph = cos(theta0) / sin(theta0);

  const double got_r_rr = chrRhs[(int64_t)comp_index(0, 0, 0) * n + cidx];
  const double got_r_thth = chrRhs[(int64_t)comp_index(0, 1, 1) * n + cidx];
  const double got_r_phph = chrRhs[(int64_t)comp_index(0, 2, 2) * n + cidx];
  const double got_th_rth = chrRhs[(int64_t)comp_index(1, 0, 1) * n + cidx];
  const double got_ph_rph = chrRhs[(int64_t)comp_index(2, 0, 2) * n + cidx];
  const double got_ph_thph = chrRhs[(int64_t)comp_index(2, 1, 2) * n + cidx];

  printf("Gamma^r_rr          got=%.17g expected=%.17g\n", got_r_rr,
         expected_r_rr);
  printf("Gamma^r_thetatheta  got=%.17g expected=%.17g\n", got_r_thth,
         expected_r_thth);
  printf("Gamma^r_phiphi      got=%.17g expected=%.17g\n", got_r_phph,
         expected_r_phph);
  printf("Gamma^theta_rtheta  got=%.17g expected=%.17g\n", got_th_rth,
         expected_th_rth);
  printf("Gamma^phi_rphi      got=%.17g expected=%.17g\n", got_ph_rph,
         expected_ph_rph);
  printf("Gamma^phi_thetaphi  got=%.17g expected=%.17g\n", got_ph_thph,
         expected_ph_thph);

  const double rel_tol = 3e-3;
  const double abs_tol = 3e-3;
  int ok = 1;
  ok &= almost_equal(got_r_rr, expected_r_rr, rel_tol, abs_tol);
  ok &= almost_equal(got_r_thth, expected_r_thth, rel_tol, abs_tol);
  ok &= almost_equal(got_r_phph, expected_r_phph, rel_tol, abs_tol);
  ok &= almost_equal(got_th_rth, expected_th_rth, rel_tol, abs_tol);
  ok &= almost_equal(got_ph_rph, expected_ph_rph, rel_tol, abs_tol);
  ok &= almost_equal(got_ph_thph, expected_ph_thph, rel_tol, abs_tol);

  free(r);
  free(theta);
  free(phi);
  free(alpha);
  free(gamma);
  free(gammaU);
  free(chr);
  free(chrRhs);

  if (!ok) {
    fprintf(stderr, "Kerr-like Christoffel mismatch beyond tolerance\n");
    return 3;
  }
  return 0;
}
