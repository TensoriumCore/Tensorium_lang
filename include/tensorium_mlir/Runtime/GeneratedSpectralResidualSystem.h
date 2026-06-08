#pragma once

#include "tensorium_mlir/Runtime/SpectralEllipticSolver.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tensorium_mlir::runtime {

inline std::span<const tensorium_spectral_residual_system_desc>
generatedSpectralResidualSystemDescs(
    const tensorium_spectral_residual_system_desc *descs, std::size_t count) {
  if (!descs || count == 0)
    return {};
  return std::span<const tensorium_spectral_residual_system_desc>(descs, count);
}

inline const tensorium_spectral_residual_system_desc &
requireGeneratedSpectralResidualSystemDesc(
    std::span<const tensorium_spectral_residual_system_desc> systems,
    std::string_view symbolName = {}) {
  if (systems.empty())
    throw std::runtime_error("generated spectral residual system table is empty");

  if (symbolName.empty()) {
    const auto &desc = systems.front();
    if (!desc.symbol_name || desc.symbol_name[0] == '\0')
      throw std::runtime_error("generated spectral residual system name is empty");
    return desc;
  }

  for (const auto &desc : systems) {
    if (desc.symbol_name && symbolName == desc.symbol_name)
      return desc;
  }
  throw std::runtime_error("missing generated spectral residual system: " +
                           std::string(symbolName));
}

inline void requireGeneratedSpectralResidualSystemShape(
    const tensorium_spectral_residual_system_desc &desc,
    std::int64_t expectedUnknownCount, std::int64_t expectedEquationCount) {
  if (!desc.symbol_name || desc.symbol_name[0] == '\0')
    throw std::runtime_error("generated spectral residual system name is empty");
  if (expectedUnknownCount >= 0 && desc.unknown_count != expectedUnknownCount) {
    throw std::runtime_error("generated spectral residual system '" +
                             std::string(desc.symbol_name) +
                             "' has unexpected unknown count");
  }
  if (expectedEquationCount >= 0 &&
      desc.equation_count != expectedEquationCount) {
    throw std::runtime_error("generated spectral residual system '" +
                             std::string(desc.symbol_name) +
                             "' has unexpected equation count");
  }
}

inline const tensorium_spectral_residual_system_desc &
requireGeneratedSpectralResidualSystemDesc(
    const tensorium_spectral_residual_system_desc *descs, std::size_t count,
    std::string_view symbolName = {}, std::int64_t expectedUnknownCount = -1,
    std::int64_t expectedEquationCount = -1) {
  const auto &desc = requireGeneratedSpectralResidualSystemDesc(
      generatedSpectralResidualSystemDescs(descs, count), symbolName);
  requireGeneratedSpectralResidualSystemShape(
      desc, expectedUnknownCount, expectedEquationCount);
  return desc;
}

inline SpectralGeneratedResidualSystem makeGeneratedSpectralResidualSystem(
    const tensorium_spectral_residual_system_desc *systemDescs,
    std::size_t systemCount, std::string_view symbolName,
    const SpectralGrid3D &grid,
    const tensorium_spectral_residual_kernel_desc *pointKernelDescs,
    std::size_t pointKernelCount,
    const tensorium_spectral_residual_grid_kernel_desc *gridKernelDescs,
    std::size_t gridKernelCount,
    std::span<const SpectralGeneratedResidualSystemEquationInputs> inputs,
    std::int64_t expectedUnknownCount = -1,
    std::int64_t expectedEquationCount = -1) {
  const auto &desc = requireGeneratedSpectralResidualSystemDesc(
      systemDescs, systemCount, symbolName, expectedUnknownCount,
      expectedEquationCount);
  return makeSpectralResidualSystemFromDesc(
      desc, grid, pointKernelDescs, pointKernelCount, gridKernelDescs,
      gridKernelCount, inputs);
}

inline SpectralResidualSystemAssemblyResult
assembleGeneratedSpectralResidualSystem(
    const SpectralGeneratedResidualSystem &generatedSystem,
    std::span<const std::vector<double>> unknownFields) {
  return assembleSpectralResidualSystem(generatedSystem.view(), unknownFields);
}

inline SpectralResidualSystemAssemblyResult
assembleGeneratedSpectralResidualSystem(
    const SpectralGeneratedResidualSystem &generatedSystem,
    std::span<std::vector<double>> unknownFields) {
  return assembleGeneratedSpectralResidualSystem(
      generatedSystem,
      std::span<const std::vector<double>>(unknownFields.data(),
                                           unknownFields.size()));
}

struct GeneratedSpectralEllipticSolveRun {
  SpectralResidualSystemAssemblyResult initialResidual;
  SpectralEllipticSolveResult solveResult;
  SpectralResidualSystemAssemblyResult finalResidual;
};

inline GeneratedSpectralEllipticSolveRun solveGeneratedSpectralEllipticSystem(
    const SpectralResidualSystemProblem &system,
    std::span<std::vector<double>> unknownFields,
    const SpectralEllipticSolveOptions &options = {}) {
  GeneratedSpectralEllipticSolveRun run;
  run.initialResidual = assembleSpectralResidualSystem(
      system, std::span<const std::vector<double>>(unknownFields.data(),
                                                   unknownFields.size()));
  run.solveResult = solveSpectralNewton(system, unknownFields, options);
  run.finalResidual = assembleSpectralResidualSystem(
      system, std::span<const std::vector<double>>(unknownFields.data(),
                                                   unknownFields.size()));
  return run;
}

inline GeneratedSpectralEllipticSolveRun solveGeneratedSpectralEllipticSystem(
    const SpectralGeneratedResidualSystem &generatedSystem,
    std::span<std::vector<double>> unknownFields,
    const SpectralEllipticSolveOptions &options = {}) {
  return solveGeneratedSpectralEllipticSystem(generatedSystem.view(),
                                              unknownFields, options);
}

} // namespace tensorium_mlir::runtime
