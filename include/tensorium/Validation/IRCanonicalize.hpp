#pragma once

#include "tensorium/Backend/DomainIR.hpp"

namespace tensorium::validation {

void canonicalizeDifferentialIR(backend::ModuleIR &module);
void canonicalizeEinsteinIR(backend::ModuleIR &module);

} // namespace tensorium::validation
