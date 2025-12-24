#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

using namespace tensorium::mlir;

FieldType FieldType::get(::mlir::MLIRContext *ctx, ::mlir::Type elementType,
                         unsigned rank, Variance variance) {
  return Base::get(ctx, elementType, rank, variance);
}

::mlir::Type FieldType::getElementType() const {
  return getImpl()->elementType;
}

unsigned FieldType::getRank() const { return getImpl()->rank; }

Variance FieldType::getVariance() const { return getImpl()->variance; }
