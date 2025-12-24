
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace tensorium {
namespace mlir {

TensoriumDialect::TensoriumDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<TensoriumDialect>()) {
  addTypes<FieldType>();
}

Type TensoriumDialect::parseType(DialectAsmParser &parser) const {
  StringRef tag;
  if (failed(parser.parseKeyword(&tag)))
    return Type();

  if (tag != "field") {
    parser.emitError(parser.getNameLoc(), "unknown tensorium type: ") << tag;
    return Type();
  }

  if (failed(parser.parseLess()))
    return Type();

  Type elementType;
  if (failed(parser.parseType(elementType)))
    return Type();

  if (failed(parser.parseComma()))
    return Type();

  unsigned rank = 0;
  if (failed(parser.parseInteger(rank)))
    return Type();

  if (failed(parser.parseComma()))
    return Type();

  StringRef varTag;
  if (failed(parser.parseKeyword(&varTag)))
    return Type();

  Variance variance;
  if (varTag == "scalar")
    variance = Variance::Scalar;
  else if (varTag == "cov")
    variance = Variance::Covariant;
  else if (varTag == "con")
    variance = Variance::Contravariant;
  else if (varTag == "mixed")
    variance = Variance::Mixed;
  else {
    parser.emitError(parser.getNameLoc(), "unknown variance: ") << varTag;
    return Type();
  }

  if (failed(parser.parseGreater()))
    return Type();

  return FieldType::get(getContext(), elementType, rank, variance);
}

void TensoriumDialect::printType(Type type, DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
      .Case<FieldType>([&](FieldType t) {
        printer << "field<";
        printer.printType(t.getElementType());
        printer << ", " << t.getRank() << ", ";

        switch (t.getVariance()) {
        case Variance::Scalar:
          printer << "scalar";
          break;
        case Variance::Covariant:
          printer << "cov";
          break;
        case Variance::Contravariant:
          printer << "con";
          break;
        case Variance::Mixed:
          printer << "mixed";
          break;
        }

        printer << ">";
      })
      .Default([&](Type) { llvm_unreachable("unexpected 'tensorium' type"); });
}

} // namespace mlir
} // namespace tensorium
