
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

  if (tag != "field")
    return (parser.emitError(parser.getNameLoc(), "unknown tensorium type: ")
                << tag,
            Type());

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

  if (failed(parser.parseGreater()))
    return Type();

  return FieldType::get(getContext(), elementType, rank);
}

void TensoriumDialect::printType(Type type, DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
      .Case<FieldType>([&](FieldType t) {
        printer << "field<";
        printer.printType(t.getElementType());
        printer << ", " << t.getRank() << ">";
      })
      .Default([&](Type) { llvm_unreachable("unexpected 'tensorium' type"); });
}

} // namespace mlir
} // namespace tensorium
