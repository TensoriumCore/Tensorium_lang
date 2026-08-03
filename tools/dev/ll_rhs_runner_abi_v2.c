#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MEMREF_ARGS(name)                                                      \
  double *name##_allocated, double *name##_aligned, int64_t name##_offset,     \
      int64_t name##_size, int64_t name##_stride

extern void tensorium_rhs_grid_affine(int64_t nx, int64_t ny, int64_t nz,
                                      double dx, double dy, double dz,
                                      MEMREF_ARGS(chi), MEMREF_ARGS(gamma),
                                      MEMREF_ARGS(atilde), MEMREF_ARGS(alpha),
                                      MEMREF_ARGS(chi_rhs),
                                      MEMREF_ARGS(gamma_rhs));

typedef struct {
  double *storage;
  int64_t capacity;
  int64_t offset;
  int64_t size;
  int64_t stride;
} MemRefView;

#define MEMREF_PASS(view)                                                      \
  (view).storage, (view).storage, (view).offset, (view).size, (view).stride

static const double sentinel = -1234567.0;

static MemRefView make_view(int64_t size, int64_t offset, int64_t stride) {
  MemRefView view;
  view.capacity = offset + (size - 1) * stride + 1;
  view.storage = (double *)malloc((size_t)view.capacity * sizeof(double));
  view.offset = offset;
  view.size = size;
  view.stride = stride;
  if (view.storage) {
    for (int64_t i = 0; i < view.capacity; ++i)
      view.storage[i] = sentinel;
  }
  return view;
}

static double *logical_element(MemRefView view, int64_t logical_index) {
  return &view.storage[view.offset + logical_index * view.stride];
}

static int almost_equal(double got, double expected) {
  return fabs(got - expected) <= 1.0e-12 * fmax(1.0, fabs(expected));
}

static int padding_is_untouched(MemRefView view) {
  for (int64_t raw = 0; raw < view.capacity; ++raw) {
    if (raw >= view.offset && (raw - view.offset) % view.stride == 0)
      continue;
    if (view.storage[raw] != sentinel)
      return 0;
  }
  return 1;
}

static int storage_is_untouched(MemRefView view) {
  for (int64_t raw = 0; raw < view.capacity; ++raw) {
    if (view.storage[raw] != sentinel)
      return 0;
  }
  return 1;
}

int main(void) {
  const int64_t nx = 3;
  const int64_t ny = 3;
  const int64_t nz = 3;
  const int64_t points = nx * ny * nz;
  const int64_t tensor_values = 9 * points;
  const double chi_value = 1.5;
  const double alpha_value = 2.0;

  MemRefView chi = make_view(points, 3, 2);
  MemRefView gamma = make_view(tensor_values, 4, 3);
  MemRefView atilde = make_view(tensor_values, 2, 2);
  MemRefView alpha = make_view(points, 5, 4);
  MemRefView chi_rhs = make_view(points, 7, 3);
  MemRefView gamma_rhs = make_view(tensor_values, 6, 2);
  if (!chi.storage || !gamma.storage || !atilde.storage || !alpha.storage ||
      !chi_rhs.storage || !gamma_rhs.storage) {
    fprintf(stderr, "ABI v2 runner allocation failure\n");
    return 2;
  }

  for (int64_t point = 0; point < points; ++point) {
    *logical_element(chi, point) = chi_value;
    *logical_element(alpha, point) = alpha_value;
    for (int64_t component = 0; component < 9; ++component) {
      const int64_t logical = component * points + point;
      *logical_element(gamma, logical) = 100.0 + (double)component;
      *logical_element(atilde, logical) = (double)(component + 1);
    }
  }

  tensorium_rhs_grid_affine(nx, ny, nz, 0.25, 0.5, 0.75, MEMREF_PASS(chi),
                            MEMREF_PASS(gamma), MEMREF_PASS(atilde),
                            MEMREF_PASS(alpha), MEMREF_PASS(chi_rhs),
                            MEMREF_PASS(gamma_rhs));

  int ok = 1;
  for (int64_t point = 0; point < points; ++point) {
    const double got_chi_rhs = *logical_element(chi_rhs, point);
    if (!almost_equal(got_chi_rhs, -2.0 * alpha_value * chi_value)) {
      fprintf(stderr, "chi_rhs[%lld] = %.17g\n", (long long)point, got_chi_rhs);
      ok = 0;
    }
    if (!almost_equal(*logical_element(chi, point), chi_value)) {
      fprintf(stderr, "input chi[%lld] was modified\n", (long long)point);
      ok = 0;
    }
    for (int64_t component = 0; component < 9; ++component) {
      const int64_t logical = component * points + point;
      const double got_gamma_rhs = *logical_element(gamma_rhs, logical);
      if (!almost_equal(got_gamma_rhs,
                        -2.0 * alpha_value * (double)(component + 1))) {
        fprintf(stderr, "gamma_rhs[%lld,%lld] = %.17g\n", (long long)component,
                (long long)point, got_gamma_rhs);
        ok = 0;
      }
      if (!almost_equal(*logical_element(gamma, logical),
                        100.0 + (double)component)) {
        fprintf(stderr, "input gamma[%lld,%lld] was modified\n",
                (long long)component, (long long)point);
        ok = 0;
      }
    }
  }
  if (!padding_is_untouched(chi_rhs) || !padding_is_untouched(gamma_rhs)) {
    fprintf(stderr, "output memref padding was modified\n");
    ok = 0;
  }

  MemRefView short_chi_rhs = make_view(points - 1, 1, 2);
  MemRefView guarded_gamma_rhs = make_view(tensor_values, 2, 2);
  if (!short_chi_rhs.storage || !guarded_gamma_rhs.storage) {
    fprintf(stderr, "ABI v2 guard allocation failure\n");
    ok = 0;
  } else {
    tensorium_rhs_grid_affine(
        nx, ny, nz, 0.25, 0.5, 0.75, MEMREF_PASS(chi), MEMREF_PASS(gamma),
        MEMREF_PASS(atilde), MEMREF_PASS(alpha), MEMREF_PASS(short_chi_rhs),
        MEMREF_PASS(guarded_gamma_rhs));
    if (!storage_is_untouched(short_chi_rhs) ||
        !storage_is_untouched(guarded_gamma_rhs)) {
      fprintf(stderr, "undersized descriptors were not rejected\n");
      ok = 0;
    }
  }

  printf("[ll-smoke] ABI v2 offsets/strides, separate outputs, size guard, and "
         "all %lld grid points: %s\n",
         (long long)points, ok ? "PASS" : "FAIL");

  free(chi.storage);
  free(gamma.storage);
  free(atilde.storage);
  free(alpha.storage);
  free(chi_rhs.storage);
  free(gamma_rhs.storage);
  free(short_chi_rhs.storage);
  free(guarded_gamma_rhs.storage);
  return ok ? 0 : 3;
}
