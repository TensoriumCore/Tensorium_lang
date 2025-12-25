#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h" 
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace tensorium::mlir {
  using Attribute = ::mlir::Attribute; 
}
namespace tensorium {
namespace mlir {
class TensoriumDialect;
} // namespace mlir
} // namespace tensorium

#define GET_OP_CLASSES
#include "TensoriumOps.h.inc"
