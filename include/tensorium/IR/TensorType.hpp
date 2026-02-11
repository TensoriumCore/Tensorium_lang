#pragma once

namespace tensorium::ir {

struct TensorType {
  int up = 0;
  int down = 0;

  int rank() const { return up + down; }
  bool isScalar() const { return up == 0 && down == 0; }
};

} // namespace tensorium::ir
