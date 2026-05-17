#pragma once

#include <stdint.h>

#ifndef TENSORIUM_GENERATED_HOST_MEMREF_TYPES_H
#define TENSORIUM_GENERATED_HOST_MEMREF_TYPES_H

typedef struct tensorium_memref1d_f64 {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} tensorium_memref1d_f64;

static inline tensorium_memref1d_f64
tensorium_make_memref1d_f64(double *data, int64_t size) {
  tensorium_memref1d_f64 ref = {data, data, 0, size, 1};
  return ref;
}

#endif /* TENSORIUM_GENERATED_HOST_MEMREF_TYPES_H */

#if !defined(TENSORIUM_GENERATED_HOST_DESCRIPTOR_TYPES_H) &&                 \
    !defined(TENSORIUM_GENERATED_HOST_H)
#define TENSORIUM_GENERATED_HOST_DESCRIPTOR_TYPES_H

typedef enum tensorium_host_buffer_role {
  TENSORIUM_HOST_BUFFER_ROLE_COORDINATE = 1,
  TENSORIUM_HOST_BUFFER_ROLE_FIELD = 2,
  TENSORIUM_HOST_BUFFER_ROLE_OUTPUT = 3
} tensorium_host_buffer_role;

typedef enum tensorium_host_arg_access {
  TENSORIUM_HOST_ARG_ACCESS_NONE = 0,
  TENSORIUM_HOST_ARG_ACCESS_READ = 1,
  TENSORIUM_HOST_ARG_ACCESS_WRITE = 2,
  TENSORIUM_HOST_ARG_ACCESS_READWRITE = 3
} tensorium_host_arg_access;

typedef struct tensorium_host_kernel_desc {
  const char *symbol_name;
  const char *wrapper_name;
  const char *kind;
  int64_t buffer_begin;
  int64_t buffer_count;
  int64_t stencil_radius;
} tensorium_host_kernel_desc;

typedef struct tensorium_host_buffer_desc {
  const char *kernel_symbol;
  const char *name;
  const char *c_name;
  int64_t kernel_index;
  int64_t arg_index;
  int64_t role;
  int64_t access;
  int64_t up;
  int64_t down;
  int64_t rank;
  int64_t component_count;
} tensorium_host_buffer_desc;

#endif /* TENSORIUM_GENERATED_HOST_DESCRIPTOR_TYPES_H */

#ifndef TENSORIUM_GENERATED_HOST_DESCRIPTOR_TYPES_H
#define TENSORIUM_GENERATED_HOST_DESCRIPTOR_TYPES_H
#endif

#ifndef TENSORIUM_GENERATED_HOST_INVOKE_TYPES_H
#define TENSORIUM_GENERATED_HOST_INVOKE_TYPES_H

typedef struct tensorium_host_grid_desc {
  int64_t nx;
  int64_t ny;
  int64_t nz;
  double dx;
  double dy;
  double dz;
  int64_t n_points;
} tensorium_host_grid_desc;

typedef int (*tensorium_host_kernel_invoke_fn)(
    const double *params, int64_t param_count,
    const tensorium_memref1d_f64 *buffers, int64_t buffer_count,
    const tensorium_host_grid_desc *grid);

typedef struct tensorium_host_kernel_adapter_desc {
  const char *symbol_name;
  tensorium_host_kernel_invoke_fn invoke;
} tensorium_host_kernel_adapter_desc;

#endif /* TENSORIUM_GENERATED_HOST_INVOKE_TYPES_H */
