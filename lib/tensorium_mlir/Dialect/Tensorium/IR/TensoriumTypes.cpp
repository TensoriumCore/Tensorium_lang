
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

using namespace tensorium::mlir;
using namespace mlir;

FieldType FieldType::get(MLIRContext *ctx, Type elementType, unsigned rank) {
  return Base::get(ctx, elementType, rank);
}

Type FieldType::getElementType() const { return getImpl()->elementType; }

unsigned FieldType::getRank() const { return getImpl()->rank; }
