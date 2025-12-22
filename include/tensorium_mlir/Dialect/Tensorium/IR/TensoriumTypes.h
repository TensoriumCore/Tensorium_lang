
#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

namespace tensorium {
namespace mlir {

class TensoriumDialect;

struct FieldTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::pair<::mlir::Type, unsigned>;

  FieldTypeStorage(::mlir::Type elementType, unsigned rank)
      : elementType(elementType), rank(rank) {}

  bool operator==(const KeyTy &key) const {
    return key.first == elementType && key.second == rank;
  }

  static FieldTypeStorage *construct(::mlir::TypeStorageAllocator &alloc,
                                     const KeyTy &key) {
    return new (alloc.allocate<FieldTypeStorage>())
        FieldTypeStorage(key.first, key.second);
  }

  ::mlir::Type elementType;
  unsigned rank;
};

class FieldType
    : public ::mlir::Type::TypeBase<FieldType, ::mlir::Type, FieldTypeStorage> {
public:
  using Base::Base;

  static constexpr ::llvm::StringLiteral name = "tensorium.field";

  static FieldType get(::mlir::MLIRContext *ctx, ::mlir::Type elementType,
                       unsigned rank);

  ::mlir::Type getElementType() const;
  unsigned getRank() const;
};

} // namespace mlir
} // namespace tensorium
