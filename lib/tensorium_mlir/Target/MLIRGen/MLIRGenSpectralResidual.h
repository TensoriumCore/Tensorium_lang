#pragma once

#include "tensorium/IR/DomainIR.hpp"

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
} // namespace mlir

namespace tensorium_mlir {

void emitSpectralResidualKernels(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::ModuleOp moduleOp,
                                 const tensorium::backend::ModuleIR &module);

} // namespace tensorium_mlir
