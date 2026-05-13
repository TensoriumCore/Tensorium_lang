#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "mlir/IR/Builders.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

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

ParseResult EinsumOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  Type resultType;

  if (parser.parseOperandList(operands))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseType(resultType))
    return failure();

  SmallVector<Type, 8> operandTypes(operands.size(), resultType);
  if (parser.resolveOperands(operands, operandTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  result.addTypes(resultType);

  if (succeeded(parser.parseOptionalAttrDict(result.attributes)))
    return success();

  return success();
}

void EinsumOp::print(OpAsmPrinter &p) {
  p << " " << getOperands();
  p << " {";
  p.printNewline();
  p.increaseIndent();

  auto printOne = [&](StringRef name) {
    if (auto a = (*this)->getAttr(name)) {
      p.printNewline();
      p << name << " = ";
      p.printAttribute(a);
    }
  };

  printOne("spec");
  printOne("tin.idx.ins");
  printOne("tin.idx.out");
  printOne("tin.idx.all");
  printOne("tin.idx.counts");
  printOne("tin.idx.roles");
  printOne("tin.idx.valid");

  for (auto na : (*this)->getAttrs()) {
    auto n = na.getName().strref();
    if (n == "spec" || n.starts_with("tin.idx."))
      continue;
    p.printNewline();
    p << n << " = ";
    p.printAttribute(na.getValue());
  }

  p.decreaseIndent();
  p.printNewline();
  p << "}";
  p << " : " << getResult().getType();
}
} // namespace mlir
} // namespace tensorium

#define GET_OP_CLASSES
#include "TensoriumOps.cpp.inc"
