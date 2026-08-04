#ifndef TENSORIUM_SOLVER_CONSTRAINT_HANDOFF_H
#define TENSORIUM_SOLVER_CONSTRAINT_HANDOFF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TENSORIUM_CONSTRAINT_HANDOFF_ABI_VERSION 1

typedef struct tensorium_constraint_solution_v1
    tensorium_constraint_solution_v1;

typedef int32_t tensorium_constraint_status_v1;
enum tensorium_constraint_status_v1_value {
  TENSORIUM_CONSTRAINT_STATUS_OK = 0,
  TENSORIUM_CONSTRAINT_STATUS_INVALID_ARGUMENT = 1,
  TENSORIUM_CONSTRAINT_STATUS_IO_ERROR = 2,
  TENSORIUM_CONSTRAINT_STATUS_FRONTEND_ERROR = 3,
  TENSORIUM_CONSTRAINT_STATUS_SOLVER_ERROR = 4,
  TENSORIUM_CONSTRAINT_STATUS_ABI_MISMATCH = 5,
  TENSORIUM_CONSTRAINT_STATUS_INTERNAL_ERROR = 6
};

typedef int32_t tensorium_ctt_coordinates_v1;
enum tensorium_ctt_coordinates_v1_value {
  TENSORIUM_CTT_COORDINATES_SPHERICAL = 1,
  TENSORIUM_CTT_COORDINATES_CARTESIAN = 2
};

typedef struct tensorium_constraint_parameter_v1 {
  uint64_t struct_size;
  const char *name;
  double value;
} tensorium_constraint_parameter_v1;

typedef struct tensorium_constraint_solution_info_v1 {
  uint64_t struct_size;
  int64_t converged;
  int64_t iterations;
  double residual_norm;
  int64_t source_point_count;
  int64_t domain_count;
  int64_t has_physical_ctt;
} tensorium_constraint_solution_info_v1;

typedef struct tensorium_ctt_target_grid_v1 {
  uint64_t struct_size;
  tensorium_ctt_coordinates_v1 coordinates;
  int64_t point_count;
  /* Spherical: r, theta, phi. Cartesian: x, y, z. */
  const double *coordinate_components[3];
} tensorium_ctt_target_grid_v1;

typedef struct tensorium_ctt_physical_buffers_v1 {
  uint64_t struct_size;
  int64_t point_count;
  /* SoA, row-major component 3*i+j, each array has point_count entries. */
  double *spatial_metric[9];
  double *inverse_spatial_metric[9];
  double *extrinsic_curvature[9];
  /* Optional. */
  double *mean_curvature;
} tensorium_ctt_physical_buffers_v1;

typedef struct tensorium_bssn_gauge_seed_v1 {
  uint64_t struct_size;
  double lapse;
  double shift[3];
} tensorium_bssn_gauge_seed_v1;

typedef struct tensorium_ctt_bssn_buffers_v1 {
  uint64_t struct_size;
  int64_t point_count;
  double *chi;
  /* SoA, row-major component 3*i+j, each array has point_count entries. */
  double *conformal_metric[9];
  double *inverse_conformal_metric[9];
  double *trace_free_extrinsic_curvature[9];
  double *mean_curvature;
  /* Gauge outputs are optional. Shift must provide all components or none. */
  double *lapse;
  double *shift[3];
} tensorium_ctt_bssn_buffers_v1;

/* Returns the runtime ABI version implemented by the linked library. */
int64_t tensorium_constraint_handoff_abi_version(void);

/*
 * Parses, validates, and solves a radial constraint problem from DSL source.
 * On success, *solution_out owns an opaque handle that must be destroyed with
 * tensorium_constraint_solution_destroy_v1.
 */
tensorium_constraint_status_v1 tensorium_solve_radial_constraints_source_v1(
    const char *source, const tensorium_constraint_parameter_v1 *parameters,
    int64_t parameter_count, tensorium_constraint_solution_v1 **solution_out,
    char *error_message, int64_t error_capacity);

/* Same operation using a UTF-8 filesystem path. */
tensorium_constraint_status_v1 tensorium_solve_radial_constraints_file_v1(
    const char *path, const tensorium_constraint_parameter_v1 *parameters,
    int64_t parameter_count, tensorium_constraint_solution_v1 **solution_out,
    char *error_message, int64_t error_capacity);

tensorium_constraint_status_v1 tensorium_constraint_solution_info_get_v1(
    const tensorium_constraint_solution_v1 *solution,
    tensorium_constraint_solution_info_v1 *info, char *error_message,
    int64_t error_capacity);

tensorium_constraint_status_v1 tensorium_interpolate_radial_ctt_v1(
    const tensorium_constraint_solution_v1 *solution,
    const tensorium_ctt_target_grid_v1 *target,
    const tensorium_ctt_physical_buffers_v1 *outputs, char *error_message,
    int64_t error_capacity);

tensorium_constraint_status_v1 tensorium_initialize_bssn_from_radial_ctt_v1(
    const tensorium_constraint_solution_v1 *solution,
    const tensorium_ctt_target_grid_v1 *target,
    const tensorium_ctt_bssn_buffers_v1 *outputs,
    const tensorium_bssn_gauge_seed_v1 *gauge, char *error_message,
    int64_t error_capacity);

void tensorium_constraint_solution_destroy_v1(
    tensorium_constraint_solution_v1 *solution);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TENSORIUM_SOLVER_CONSTRAINT_HANDOFF_H */
