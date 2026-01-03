#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace tensorium {
namespace mlir {

TensoriumDialect::TensoriumDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<TensoriumDialect>()) {
  addTypes<FieldType>();
  addOperations<
#define GET_OP_LIST
#include "TensoriumOps.cpp.inc"
      >();
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

  unsigned up = 0;
  if (failed(parser.parseInteger(up)))
    return Type();

  if (failed(parser.parseComma()))
    return Type();

  unsigned down = 0;
  if (failed(parser.parseInteger(down)))
    return Type();

  if (failed(parser.parseGreater()))
    return Type();

  return FieldType::get(getContext(), elementType, up, down);
}

void TensoriumDialect::printType(Type type, DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
      .Case<FieldType>([&](FieldType t) {
        printer << "field<";
        printer.printType(t.getElementType());
        printer << ", " << t.getUp() << ", " << t.getDown() << ">";
      })
      .Default([&](Type) { llvm_unreachable("unexpected 'tensorium' type"); });
}

} // namespace mlir
} // namespace tensorium
