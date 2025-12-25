#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "mlir/IR/Builders.h" // Often required by generated code

using namespace mlir;
namespace tensorium {
namespace mlir {

LogicalResult IndexOp::verify() {
  auto fieldTy =
      llvm::dyn_cast<tensorium::mlir::FieldType>(getField().getType());
  if (!fieldTy)
    return emitOpError("operand must be a tensorium.field");

  auto idx = getIndices();
  if (!idx)
    return emitOpError("missing indices attribute");

  unsigned rank = fieldTy.getRank();
  unsigned nidx = idx.size();

  if (rank != nidx)
    return emitOpError() << "wrong number of indices: expected " << rank
                         << ", got " << nidx;

  llvm::SmallDenseSet<llvm::StringRef, 16> seen;
  for (Attribute a : idx) {
    auto s = llvm::dyn_cast<StringAttr>(a);
    if (!s)
      return emitOpError("indices must be an array of string attributes");
    auto v = s.getValue();
    if (!seen.insert(v).second)
      return emitOpError() << "duplicate index '" << v << "'";
  }

  return success();
}

} // namespace mlir
} // namespace tensorium

#define GET_OP_CLASSES
#include "TensoriumOps.cpp.inc"
