#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

using namespace tensorium::mlir;

FieldType FieldType::get(::mlir::MLIRContext *ctx, ::mlir::Type elementType,
                         unsigned up, unsigned down) {
  return Base::get(ctx, elementType, up, down);
}

::mlir::Type FieldType::getElementType() const {
  return getImpl()->elementType;
}

unsigned FieldType::getRank() const { return getImpl()->up + getImpl()->down; }

unsigned FieldType::getUp() const { return getImpl()->up; }

unsigned FieldType::getDown() const { return getImpl()->down; }

Variance FieldType::getVariance() const {
  if (getUp() == 0 && getDown() == 0)
    return Variance::Scalar;
  if (getUp() > 0 && getDown() == 0)
    return Variance::Contravariant;
  if (getUp() == 0 && getDown() > 0)
    return Variance::Covariant;
  return Variance::Mixed;
}
