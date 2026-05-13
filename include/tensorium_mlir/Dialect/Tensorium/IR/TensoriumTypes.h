#pragma once

#include "mlir/IR/Types.h"

#include <cstdint>

namespace tensorium {
namespace mlir {

enum class Variance : uint8_t { Scalar, Contravariant, Covariant, Mixed };

} // namespace mlir
} // namespace tensorium

#define GET_TYPEDEF_CLASSES
#include "TensoriumTypes.h.inc"
