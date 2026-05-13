#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "TensoriumTypes.cpp.inc"

namespace tensorium {
namespace mlir {

TensoriumDialect::TensoriumDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<TensoriumDialect>()) {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TensoriumTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "TensoriumOps.cpp.inc"
      >();
}

} // namespace mlir
} // namespace tensorium
