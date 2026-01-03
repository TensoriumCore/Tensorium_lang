#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <tuple>

namespace tensorium {
namespace mlir {

enum class Variance : uint8_t { Scalar, Contravariant, Covariant, Mixed };

struct FieldTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::Type, unsigned, unsigned>;

  FieldTypeStorage(::mlir::Type elementType, unsigned up, unsigned down)
      : elementType(elementType), up(up), down(down) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, up, down);
  }

  static FieldTypeStorage *construct(::mlir::TypeStorageAllocator &alloc,
                                     const KeyTy &key) {
    return new (alloc.allocate<FieldTypeStorage>())
        FieldTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  ::mlir::Type elementType;
  unsigned up;
  unsigned down;
};

class FieldType
    : public ::mlir::Type::TypeBase<FieldType, ::mlir::Type, FieldTypeStorage> {
public:
  using Base::Base;

  static constexpr ::llvm::StringLiteral name = "tensorium.field";

  static FieldType get(::mlir::MLIRContext *ctx, ::mlir::Type elementType,
                       unsigned up, unsigned down);

  ::mlir::Type getElementType() const;
  unsigned getRank() const;
  unsigned getUp() const;
  unsigned getDown() const;
  Variance getVariance() const;
};

} // namespace mlir
} // namespace tensorium
