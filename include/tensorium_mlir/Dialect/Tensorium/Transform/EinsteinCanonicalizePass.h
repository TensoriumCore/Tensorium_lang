#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace tensorium::mlir {
std::unique_ptr<::mlir::Pass> createTensoriumEinsteinCanonicalizePass();
}
