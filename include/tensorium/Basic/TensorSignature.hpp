#pragma once
#include <string>
#include <vector>

namespace tensorium {

enum class VarianceKind { Scalar, Contravariant, Covariant, Mixed };

struct TensorSignature {
  int up = 0;
  int down = 0;
  std::vector<std::string> indices;

  int getRank() const { return up + down; }
  bool isScalar() const { return up == 0 && down == 0; }

  VarianceKind getVariance() const {
    if (isScalar())
      return VarianceKind::Scalar;
    if (up > 0 && down == 0)
      return VarianceKind::Contravariant;
    if (up == 0 && down > 0)
      return VarianceKind::Covariant;
    return VarianceKind::Mixed;
  }
};

} // namespace tensorium
