
#pragma once

#include "mlir/IR/Dialect.h"

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir

namespace tensorium {
namespace mlir {

class TensoriumDialect : public ::mlir::Dialect {
public:
  explicit TensoriumDialect(::mlir::MLIRContext *ctx);

  static ::llvm::StringRef getDialectNamespace() { return "tensorium"; }

  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &printer) const override;
};

} // namespace mlir
} // namespace tensorium
