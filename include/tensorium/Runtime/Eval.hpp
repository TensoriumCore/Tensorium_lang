
#pragma once
#include "tensorium/Backend/IR.hpp"
#include <unordered_map>
#include <stdexcept>

namespace tensorium::runtime {

struct ScalarEnv {
  std::unordered_map<std::string, const double*> fieldPtr;
  std::unordered_map<std::string, double> params;
};

double evalScalar(const backend::ExprIR* e, const ScalarEnv& env);

} // namespace tensorium::runtime
