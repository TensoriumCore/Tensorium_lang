#include "tensorium/Solver/ConstraintHandoff.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { POINT_COUNT = 3, ERROR_CAPACITY = 1024 };

static int fail(tensorium_constraint_solution_v1 *solution,
                const char *stage, tensorium_constraint_status_v1 status,
                const char *error) {
  fprintf(stderr, "%s failed (status=%d): %s\n", stage, (int)status,
          error && error[0] ? error : "no diagnostic");
  tensorium_constraint_solution_destroy_v1(solution);
  return 1;
}

static int nearly_equal(double lhs, double rhs, double tolerance) {
  return fabs(lhs - rhs) <=
         tolerance * fmax(1.0, fmax(fabs(lhs), fabs(rhs)));
}

static double determinant(const double matrix[9][POINT_COUNT], int point) {
  const double a = matrix[0][point];
  const double b = matrix[1][point];
  const double c = matrix[2][point];
  const double d = matrix[3][point];
  const double e = matrix[4][point];
  const double f = matrix[5][point];
  const double g = matrix[6][point];
  const double h = matrix[7][point];
  const double i = matrix[8][point];
  return a * (e * i - f * h) - b * (d * i - f * g) +
         c * (d * h - e * g);
}

static int test_electromagnetic_handoff(const char *path, char *error) {
  tensorium_constraint_parameter_v1 parameter = {0};
  parameter.struct_size = sizeof(parameter);
  parameter.name = "charge";
  parameter.value = 0.6;

  tensorium_constraint_solution_v1 *solution = NULL;
  tensorium_constraint_status_v1 status =
      tensorium_solve_radial_constraints_file_v1(path, &parameter, 1, &solution,
                                                 error, ERROR_CAPACITY);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return fail(solution, "Einstein-Maxwell constraint solve", status, error);

  const double x[POINT_COUNT] = {0.5, 0.0, 0.0};
  const double y[POINT_COUNT] = {0.0, 1.0, 0.0};
  const double z[POINT_COUNT] = {0.0, 0.0, 4.0};
  tensorium_ctt_target_grid_v1 target = {0};
  target.struct_size = sizeof(target);
  target.coordinates = TENSORIUM_CTT_COORDINATES_CARTESIAN;
  target.point_count = POINT_COUNT;
  target.coordinate_components[0] = x;
  target.coordinate_components[1] = y;
  target.coordinate_components[2] = z;

  double electric[3][POINT_COUNT] = {{0}};
  double magnetic[3][POINT_COUNT] = {{0}};
  tensorium_electromagnetic_buffers_v1 outputs = {0};
  outputs.struct_size = sizeof(outputs);
  outputs.point_count = POINT_COUNT;
  for (int component = 0; component < 3; ++component) {
    outputs.electric_field[component] = electric[component];
    outputs.magnetic_field[component] = magnetic[component];
  }

  status = tensorium_interpolate_radial_electromagnetic_v1(
      solution, &target, &outputs, error, ERROR_CAPACITY);
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return fail(solution, "electromagnetic interpolation", status, error);

  for (int point = 0; point < POINT_COUNT; ++point) {
    const double radius = x[point] + y[point] + z[point];
    const double psi = sqrt(pow(1.0 + 1.0 / (2.0 * radius), 2.0) -
                            0.6 * 0.6 / (4.0 * radius * radius));
    const double expected = 0.6 / (radius * radius * pow(psi, 6.0));
    for (int component = 0; component < 3; ++component) {
      const double wanted = component == point ? expected : 0.0;
      if (!nearly_equal(electric[component][point], wanted, 1.0e-8) ||
          !nearly_equal(magnetic[component][point], 0.0, 1.0e-15)) {
        return fail(solution, "electromagnetic field validation",
                    TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                    "unexpected electric or magnetic field component");
      }
    }
  }

  tensorium_constraint_solution_destroy_v1(solution);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <ctt-dsl-file> <einstein-maxwell-dsl-file>\n",
            argv[0]);
    return 2;
  }
  if (tensorium_constraint_handoff_abi_version() !=
      TENSORIUM_CONSTRAINT_HANDOFF_ABI_VERSION) {
    fprintf(stderr, "constraint handoff ABI version mismatch\n");
    return 1;
  }

  char error[ERROR_CAPACITY] = {0};
  tensorium_constraint_parameter_v1 parameter = {0};
  parameter.struct_size = sizeof(parameter);
  parameter.name = "amplitude";
  parameter.value = 0.2;

  tensorium_constraint_solution_v1 *solution = NULL;
  tensorium_constraint_status_v1 status =
      tensorium_solve_radial_constraints_file_v1(
          argv[1], &parameter, 1, &solution, error, sizeof(error));
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return fail(solution, "constraint solve", status, error);

  tensorium_constraint_solution_info_v1 info = {0};
  info.struct_size = sizeof(info);
  status = tensorium_constraint_solution_info_get_v1(
      solution, &info, error, sizeof(error));
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return fail(solution, "solution metadata", status, error);
  if (!info.converged || info.iterations <= 0 || info.iterations > 15 ||
      !(info.residual_norm < 1.0e-10) || info.source_point_count != 50 ||
      info.domain_count != 2 || !info.has_physical_ctt) {
    snprintf(error, sizeof(error),
             "unexpected metadata: converged=%lld iterations=%lld "
             "residual=%.17g points=%lld domains=%lld physical=%lld",
             (long long)info.converged, (long long)info.iterations,
             info.residual_norm, (long long)info.source_point_count,
             (long long)info.domain_count, (long long)info.has_physical_ctt);
    return fail(solution, "solution metadata validation",
                TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR, error);
  }

  const double x[POINT_COUNT] = {1.25, 2.5, 3.75};
  const double y[POINT_COUNT] = {0.0, 0.0, 0.0};
  const double z[POINT_COUNT] = {0.0, 0.0, 0.0};
  tensorium_ctt_target_grid_v1 target = {0};
  target.struct_size = sizeof(target);
  target.coordinates = TENSORIUM_CTT_COORDINATES_CARTESIAN;
  target.point_count = POINT_COUNT;
  target.coordinate_components[0] = x;
  target.coordinate_components[1] = y;
  target.coordinate_components[2] = z;

  double spatial_metric[9][POINT_COUNT] = {{0}};
  double inverse_spatial_metric[9][POINT_COUNT] = {{0}};
  double extrinsic_curvature[9][POINT_COUNT] = {{0}};
  double physical_mean_curvature[POINT_COUNT] = {0};
  tensorium_ctt_physical_buffers_v1 physical = {0};
  physical.struct_size = sizeof(physical);
  physical.point_count = POINT_COUNT;
  physical.mean_curvature = physical_mean_curvature;
  for (int component = 0; component < 9; ++component) {
    physical.spatial_metric[component] = spatial_metric[component];
    physical.inverse_spatial_metric[component] =
        inverse_spatial_metric[component];
    physical.extrinsic_curvature[component] =
        extrinsic_curvature[component];
  }
  status = tensorium_interpolate_radial_ctt_v1(
      solution, &target, &physical, error, sizeof(error));
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return fail(solution, "physical CTT interpolation", status, error);

  for (int point = 0; point < POINT_COUNT; ++point) {
    if (!(determinant(spatial_metric, point) > 0.0) ||
        !isfinite(physical_mean_curvature[point])) {
      return fail(solution, "physical CTT validation",
                  TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "non-positive metric determinant or non-finite mean curvature");
    }
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        double product = 0.0;
        for (int inner = 0; inner < 3; ++inner)
          product += spatial_metric[3 * row + inner][point] *
                     inverse_spatial_metric[3 * inner + column][point];
        if (!nearly_equal(product, row == column ? 1.0 : 0.0, 1.0e-10))
          return fail(solution, "physical inverse-metric validation",
                      TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                      "gamma_ik gamma^kj is not the identity");
      }
    }
  }

  double chi[POINT_COUNT] = {0};
  double conformal_metric[9][POINT_COUNT] = {{0}};
  double inverse_conformal_metric[9][POINT_COUNT] = {{0}};
  double trace_free_extrinsic_curvature[9][POINT_COUNT] = {{0}};
  double bssn_mean_curvature[POINT_COUNT] = {0};
  double lapse[POINT_COUNT] = {0};
  double shift[3][POINT_COUNT] = {{0}};
  tensorium_ctt_bssn_buffers_v1 bssn = {0};
  bssn.struct_size = sizeof(bssn);
  bssn.point_count = POINT_COUNT;
  bssn.chi = chi;
  bssn.mean_curvature = bssn_mean_curvature;
  bssn.lapse = lapse;
  for (int component = 0; component < 9; ++component) {
    bssn.conformal_metric[component] = conformal_metric[component];
    bssn.inverse_conformal_metric[component] =
        inverse_conformal_metric[component];
    bssn.trace_free_extrinsic_curvature[component] =
        trace_free_extrinsic_curvature[component];
  }
  for (int component = 0; component < 3; ++component)
    bssn.shift[component] = shift[component];

  tensorium_bssn_gauge_seed_v1 gauge = {0};
  gauge.struct_size = sizeof(gauge);
  gauge.lapse = 0.9;
  gauge.shift[0] = 0.01;
  gauge.shift[1] = -0.02;
  gauge.shift[2] = 0.03;
  status = tensorium_initialize_bssn_from_radial_ctt_v1(
      solution, &target, &bssn, &gauge, error, sizeof(error));
  if (status != TENSORIUM_CONSTRAINT_STATUS_OK)
    return fail(solution, "BSSN initialization", status, error);

  for (int point = 0; point < POINT_COUNT; ++point) {
    if (!(chi[point] > 0.0) ||
        !nearly_equal(determinant(conformal_metric, point), 1.0, 1.0e-10) ||
        !nearly_equal(bssn_mean_curvature[point],
                      physical_mean_curvature[point], 1.0e-10) ||
        !nearly_equal(lapse[point], gauge.lapse, 1.0e-15)) {
      return fail(solution, "BSSN scalar validation",
                  TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "invalid chi, conformal determinant, K, or lapse");
    }
    double trace = 0.0;
    for (int component = 0; component < 9; ++component)
      trace += inverse_conformal_metric[component][point] *
               trace_free_extrinsic_curvature[component][point];
    if (!nearly_equal(trace, 0.0, 1.0e-10))
      return fail(solution, "BSSN trace-free validation",
                  TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                  "conformal trace of A_tilde is non-zero");
    for (int component = 0; component < 3; ++component) {
      if (!nearly_equal(shift[component][point], gauge.shift[component],
                        1.0e-15))
        return fail(solution, "BSSN shift validation",
                    TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR,
                    "gauge shift was not copied");
    }
  }

  tensorium_ctt_target_grid_v1 incompatible_target = target;
  incompatible_target.struct_size = 0;
  error[0] = '\0';
  status = tensorium_interpolate_radial_ctt_v1(
      solution, &incompatible_target, &physical, error, sizeof(error));
  if (status != TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH || !error[0])
    return fail(solution, "ABI mismatch rejection", status, error);

  printf("constraint handoff ABI v%lld: converged in %lld iterations, "
         "residual %.3e, %lld source points -> %d BSSN points\n",
         (long long)tensorium_constraint_handoff_abi_version(),
         (long long)info.iterations, info.residual_norm,
         (long long)info.source_point_count, POINT_COUNT);
  tensorium_constraint_solution_destroy_v1(solution);
  return test_electromagnetic_handoff(argv[2], error);
}
