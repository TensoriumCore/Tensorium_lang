#pragma once

struct TensorType {
  int up = 0;
  int down = 0;

  bool isScalar() const { return up == 0 && down == 0; }
  int rank() const { return up + down; }
  bool sameVariance(const TensorType &o) const { return up == o.up && down == o.down; }
};
