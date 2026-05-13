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

LogicalResult tensorium::mlir::RefOp::verify() {
  FieldType srcTy, resTy;
  if (failed(requireFieldType(getSource(), *this, "source", srcTy)) ||
      failed(requireFieldType(getResult(), *this, "result", resTy)))
    return failure();
  if (srcTy != resTy)
    return emitOpError("source/result types must match");
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

LogicalResult tensorium::mlir::AddOp::verify() {
  FieldType lhsTy, rhsTy, resTy;
  if (failed(requireFieldType(getLhs(), *this, "lhs", lhsTy)) ||
      failed(requireFieldType(getRhs(), *this, "rhs", rhsTy)) ||
      failed(requireFieldType(getRes(), *this, "result", resTy)))
    return failure();
  if (lhsTy != rhsTy || lhsTy != resTy)
    return emitOpError("lhs, rhs, result must share identical tensor type");
  return success();
}

LogicalResult tensorium::mlir::SubOp::verify() {
  FieldType lhsTy, rhsTy, resTy;
  if (failed(requireFieldType(getLhs(), *this, "lhs", lhsTy)) ||
      failed(requireFieldType(getRhs(), *this, "rhs", rhsTy)) ||
      failed(requireFieldType(getRes(), *this, "result", resTy)))
    return failure();
  if (lhsTy != rhsTy || lhsTy != resTy)
    return emitOpError("lhs, rhs, result must share identical tensor type");
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

LogicalResult tensorium::mlir::DivOp::verify() {
  FieldType lhsTy, rhsTy, resTy;
  if (failed(requireFieldType(getLhs(), *this, "lhs", lhsTy)) ||
      failed(requireFieldType(getRhs(), *this, "rhs", rhsTy)) ||
      failed(requireFieldType(getRes(), *this, "result", resTy)))
    return failure();
  if (rhsTy.getRank() != 0)
    return emitOpError("rhs must be scalar");
  if (resTy != lhsTy)
    return emitOpError("result must match lhs tensor type");
  return success();
}

LogicalResult tensorium::mlir::SinOp::verify() {
  FieldType inTy, outTy;
  if (failed(requireScalarFieldType(getIn(), *this, "input", inTy)) ||
      failed(requireScalarFieldType(getOut(), *this, "result", outTy)))
    return failure();
  return success();
}

LogicalResult tensorium::mlir::SqrtOp::verify() {
  FieldType inTy, outTy;
  if (failed(requireScalarFieldType(getIn(), *this, "input", inTy)) ||
      failed(requireScalarFieldType(getOut(), *this, "result", outTy)))
    return failure();
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

LogicalResult tensorium::mlir::PromoteOp::verify() {
  FieldType inTy, outTy;
  if (failed(requireFieldType(getIn(), *this, "input", inTy)) ||
      failed(requireFieldType(getOut(), *this, "result", outTy)))
    return failure();
  if (inTy.getRank() != 0)
    return emitOpError("input must be scalar");
  return success();
}

LogicalResult tensorium::mlir::ExternCallOp::verify() {
  FieldType resultTy;
  if (failed(requireScalarFieldType(getResult(), *this, "result", resultTy)))
    return failure();
  if (getCallee().empty())
    return emitOpError("callee must not be empty");
  for (Value arg : getArgs()) {
    FieldType argTy;
    if (failed(requireScalarFieldType(arg, *this, "argument", argTy)))
      return failure();
  }
  return success();
}

LogicalResult tensorium::mlir::DtAssignOp::verify() {
  FieldType fieldTy, rhsTy;
  if (failed(requireFieldType(getField(), *this, "field", fieldTy)) ||
      failed(requireFieldType(getRhs(), *this, "rhs", rhsTy)))
    return failure();
  if (fieldTy != rhsTy)
    return emitOpError("rhs tensor type must match field type");
  return success();
}

LogicalResult tensorium::mlir::AssignOp::verify() {
  FieldType fieldTy, rhsTy;
  if (failed(requireFieldType(getField(), *this, "field", fieldTy)) ||
      failed(requireFieldType(getRhs(), *this, "rhs", rhsTy)))
    return failure();
  if (fieldTy != rhsTy)
    return emitOpError("rhs tensor type must match field type");
  return success();
}

LogicalResult tensorium::mlir::Metric4Op::verify() {
  FieldType metricTy;
  if (failed(requireFieldType(getMetric(), *this, "metric result", metricTy)))
    return failure();
  if (metricTy.getUp() != 0 || metricTy.getDown() != 2)
    return emitOpError("metric result must be covariant rank-2");

  if (getIndices().size() != 2)
    return emitOpError("indices attribute must contain exactly 2 symbols");

  if (getComponents().size() != 16)
    return emitOpError("metric4 requires exactly 16 scalar components");
  for (Value v : getComponents()) {
    FieldType compTy;
    if (failed(requireScalarFieldType(v, *this, "metric component", compTy)))
      return failure();
  }

  auto coords = getCoordSystem();
  if (!(coords == "cartesian" || coords == "spherical" ||
        coords == "cylindrical")) {
    return emitOpError("coord_system must be cartesian/spherical/cylindrical");
  }

  return success();
}

LogicalResult tensorium::mlir::Decompose3P1FromMetricOp::verify() {
  FieldType metricTy, alphaTy, betaTy, gammaTy, gammaUTy;
  if (failed(requireFieldType(getMetric4(), *this, "metric operand", metricTy)) ||
      failed(requireFieldType(getAlpha(), *this, "alpha result", alphaTy)) ||
      failed(requireFieldType(getBeta(), *this, "beta result", betaTy)) ||
      failed(requireFieldType(getGamma(), *this, "gamma result", gammaTy)) ||
      failed(requireFieldType(getGammaU(), *this, "gammaU result", gammaUTy)))
    return failure();

  if (metricTy.getUp() != 0 || metricTy.getDown() != 2)
    return emitOpError("metric operand must be covariant rank-2");
  if (alphaTy.getRank() != 0)
    return emitOpError("alpha result must be scalar");
  if (betaTy.getUp() != 0 || betaTy.getDown() != 1)
    return emitOpError("beta result must be covector rank-1");
  if (gammaTy.getUp() != 0 || gammaTy.getDown() != 2)
    return emitOpError("gamma result must be covariant rank-2");
  if (gammaUTy.getUp() != 2 || gammaUTy.getDown() != 0)
    return emitOpError("gammaU result must be contravariant rank-2");

  return success();
}

LogicalResult tensorium::mlir::Init3P1Op::verify() {
  FieldType alphaTy, betaTy, gammaTy, gammaUTy;
  FieldType alphaInTy, betaInTy, gammaInTy, gammaUInTy;
  if (failed(requireFieldType(getAlpha(), *this, "alpha result", alphaTy)) ||
      failed(requireFieldType(getBeta(), *this, "beta result", betaTy)) ||
      failed(requireFieldType(getGamma(), *this, "gamma result", gammaTy)) ||
      failed(requireFieldType(getGammaU(), *this, "gammaU result", gammaUTy)) ||
      failed(requireFieldType(getAlphaIn(), *this, "alpha input", alphaInTy)) ||
      failed(requireFieldType(getBetaIn(), *this, "beta input", betaInTy)) ||
      failed(requireFieldType(getGammaIn(), *this, "gamma input", gammaInTy)) ||
      failed(requireFieldType(getGammaUIn(), *this, "gammaU input", gammaUInTy)))
    return failure();

  if (alphaTy.getRank() != 0)
    return emitOpError("alpha result must be scalar");
  if (betaTy.getUp() != 0 || betaTy.getDown() != 1)
    return emitOpError("beta result must be covector rank-1");
  if (gammaTy.getUp() != 0 || gammaTy.getDown() != 2)
    return emitOpError("gamma result must be covariant rank-2");
  if (gammaUTy.getUp() != 2 || gammaUTy.getDown() != 0)
    return emitOpError("gammaU result must be contravariant rank-2");

  if (alphaInTy != alphaTy || betaInTy != betaTy || gammaInTy != gammaTy ||
      gammaUInTy != gammaUTy)
    return emitOpError("init3p1 inputs and outputs must have matching types");

  return success();
}

LogicalResult tensorium::mlir::BuildCovectorOp::verify() {
  FieldType outTy;
  if (failed(requireFieldType(getOut(), *this, "result", outTy)))
    return failure();
  if (outTy.getUp() != 0 || outTy.getDown() != 1)
    return emitOpError("result must be covector rank-1");
  if (getComponents().size() != 3)
    return emitOpError("build_covector expects 3 scalar components");
  for (Value v : getComponents()) {
    FieldType compTy;
    if (failed(requireScalarFieldType(v, *this, "component", compTy)))
      return failure();
  }
  return success();
}

LogicalResult tensorium::mlir::BuildCovTensor2Op::verify() {
  FieldType outTy;
  if (failed(requireFieldType(getOut(), *this, "result", outTy)))
    return failure();
  if (outTy.getUp() != 0 || outTy.getDown() != 2)
    return emitOpError("result must be covariant rank-2");
  if (getComponents().size() != 9)
    return emitOpError("build_cov_tensor2 expects 9 scalar components");
  for (Value v : getComponents()) {
    FieldType compTy;
    if (failed(requireScalarFieldType(v, *this, "component", compTy)))
      return failure();
  }
  return success();
}

LogicalResult tensorium::mlir::BuildConTensor2Op::verify() {
  FieldType outTy;
  if (failed(requireFieldType(getOut(), *this, "result", outTy)))
    return failure();
  if (outTy.getUp() != 2 || outTy.getDown() != 0)
    return emitOpError("result must be contravariant rank-2");
  if (getComponents().size() != 9)
    return emitOpError("build_con_tensor2 expects 9 scalar components");
  for (Value v : getComponents()) {
    FieldType compTy;
    if (failed(requireScalarFieldType(v, *this, "component", compTy)))
      return failure();
  }
  return success();
}

#define GET_OP_CLASSES
#include "TensoriumOps.cpp.inc"
