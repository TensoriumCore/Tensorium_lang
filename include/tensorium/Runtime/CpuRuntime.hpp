
#pragma once
#include "tensorium/Backend/IR.hpp"
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace tensorium::runtime {

struct CpuState1D {
  std::size_t n = 0;
  std::unordered_map<std::string, std::vector<double>> fields;
  std::unordered_map<std::string, double> params;
};

struct RunOptions {
  std::size_t steps = 10;
};

CpuState1D initState1D(const backend::ModuleIR &mod, double initScalar,
                       double initAlpha);

void runEuler1D(const backend::ModuleIR &mod, CpuState1D &st,
                const RunOptions &opt);

} // namespace tensorium::runtime
