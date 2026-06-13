#pragma once

#include "tensorium/IR/DomainIR.hpp"

namespace tensorium::validation {

void canonicalizeDifferentialIR(backend::ModuleIR &module);
void canonicalizeEinsteinIR(backend::ModuleIR &module);

} // namespace tensorium::validation
