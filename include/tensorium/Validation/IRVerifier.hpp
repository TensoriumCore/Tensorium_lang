#pragma once

#include "tensorium/Validation/ProgramValidator.hpp"

namespace tensorium::validation {

ValidationResult verifyIR(const backend::ModuleIR &module);

} // namespace tensorium::validation
