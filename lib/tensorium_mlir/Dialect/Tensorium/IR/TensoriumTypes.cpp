#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

using namespace tensorium::mlir;

Variance FieldType::getVariance() const {
  if (getUp() == 0 && getDown() == 0)
    return Variance::Scalar;
  if (getUp() > 0 && getDown() == 0)
    return Variance::Contravariant;
  if (getUp() == 0 && getDown() > 0)
    return Variance::Covariant;
  return Variance::Mixed;
}
