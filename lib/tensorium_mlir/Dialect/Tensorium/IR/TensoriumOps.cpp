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

using tensorium::mlir::FieldType;

static LogicalResult requireFieldType(Value v, Operation *op,
                                      StringRef what, FieldType &out) {
  if (auto ty = mlir::dyn_cast<FieldType>(v.getType())) {
    out = ty;
    return success();
  }
  return op->emitOpError() << what << " must be tensorium.field";
}

static LogicalResult requireScalarFieldType(Value v, Operation *op,
                                            StringRef what, FieldType &out) {
  if (failed(requireFieldType(v, op, what, out)))
    return failure();
  if (out.getRank() != 0)
    return op->emitOpError() << what << " must be scalar tensorium.field";
  return success();
}

LogicalResult tensorium::mlir::ConstOp::verify() {
  FieldType type;
  if (failed(requireFieldType(getResult(), *this, "result", type)))
    return failure();
  return success();
}

LogicalResult tensorium::mlir::ParamOp::verify() {
  FieldType resultTy;
  if (failed(requireScalarFieldType(getResult(), *this, "result", resultTy)))
    return failure();
  if (getName().empty())
    return emitOpError("param name must not be empty");
  return success();
}

LogicalResult tensorium::mlir::CoordOp::verify() {
  FieldType resultTy;
  if (failed(requireScalarFieldType(getResult(), *this, "result", resultTy)))
    return failure();
  if (getName().empty())
    return emitOpError("coord name must not be empty");
  return success();
}

LogicalResult tensorium::mlir::MulOp::verify() {
  FieldType lhsTy, rhsTy, resTy;
  if (failed(requireFieldType(getLhs(), *this, "lhs", lhsTy)) ||
      failed(requireFieldType(getRhs(), *this, "rhs", rhsTy)) ||
      failed(requireFieldType(getRes(), *this, "result", resTy)))
    return failure();
  if (resTy.getRank() != lhsTy.getRank() + rhsTy.getRank())
    return emitOpError("result rank must equal operand rank sum");
  return success();
}

LogicalResult tensorium::mlir::DerivOp::verify() {
  FieldType inTy, outTy;
  if (failed(requireFieldType(getIn(), *this, "input", inTy)) ||
      failed(requireFieldType(getOut(), *this, "result", outTy)))
    return failure();
  if (outTy.getRank() != inTy.getRank() + 1)
    return emitOpError("derivative must add one covariant index");
  return success();
}

LogicalResult tensorium::mlir::ContractOp::verify() {
  FieldType inTy, outTy;
  if (failed(requireFieldType(getIn(), *this, "input", inTy)) ||
      failed(requireFieldType(getOut(), *this, "result", outTy)))
    return failure();
  if (inTy.getRank() < outTy.getRank())
    return emitOpError("result cannot have more indices than input");
  return success();
}

LogicalResult tensorium::mlir::ExternCallOp::verify() {
  if (getCallee().empty())
    return emitOpError("callee must not be empty");
  return success();
}

LogicalResult tensorium::mlir::Metric4Op::verify() {
  if (getIndices().size() != 2)
    return emitOpError("indices attribute must contain exactly 2 symbols");

  if (getComponents().size() != 16)
    return emitOpError("metric4 requires exactly 16 scalar components");

  auto coords = getCoordSystem();
  if (!(coords == "cartesian" || coords == "spherical" ||
        coords == "cylindrical")) {
    return emitOpError("coord_system must be cartesian/spherical/cylindrical");
  }

  return success();
}

LogicalResult tensorium::mlir::BuildCovectorOp::verify() {
  if (getComponents().size() != 3)
    return emitOpError("build_covector expects 3 scalar components");
  return success();
}

LogicalResult tensorium::mlir::BuildCovTensor2Op::verify() {
  if (getComponents().size() != 9)
    return emitOpError("build_cov_tensor2 expects 9 scalar components");
  return success();
}

LogicalResult tensorium::mlir::BuildConTensor2Op::verify() {
  if (getComponents().size() != 9)
    return emitOpError("build_con_tensor2 expects 9 scalar components");
  return success();
}

#define GET_OP_CLASSES
#include "TensoriumOps.cpp.inc"
