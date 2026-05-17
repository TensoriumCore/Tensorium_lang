#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Validation/IRCanonicalize.hpp"
#include "tensorium/Validation/IRVerifier.hpp"
#include "tensorium_mlir/Runtime/HostBuffers.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"
#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string inputPath;
  std::array<std::int64_t, 3> dims = {0, 0, 0};
};

std::string readFile(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open input file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " [--nx N --ny N --nz N] <file.tn>\n"
      << "\n"
      << "Builds the lowered host ABI descriptor and prints a uniform-grid\n"
      << "runtime allocation/binding plan for generated kernels.\n";
}

std::int64_t parsePositiveI64(const std::string &value,
                              const std::string &flag) {
  std::size_t consumed = 0;
  std::int64_t parsed = 0;
  try {
    parsed = std::stoll(value, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error(flag + " expects an integer");
  }
  if (consumed != value.size() || parsed <= 0)
    throw std::runtime_error(flag + " expects a positive integer");
  return parsed;
}

Options parseOptions(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto parseDim = [&](std::int64_t &slot, const char *flag) {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string(flag) + " expects a value");
      slot = parsePositiveI64(argv[++i], flag);
    };

    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (arg == "--nx") {
      parseDim(opts.dims[0], "--nx");
    } else if (arg == "--ny") {
      parseDim(opts.dims[1], "--ny");
    } else if (arg == "--nz") {
      parseDim(opts.dims[2], "--nz");
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else if (opts.inputPath.empty()) {
      opts.inputPath = arg;
    } else {
      throw std::runtime_error("multiple input files provided");
    }
  }

  if (opts.inputPath.empty())
    throw std::runtime_error("missing input .tn file");
  return opts;
}

tensorium::backend::ModuleIR buildModuleFromFile(const std::string &path) {
  const std::string source = readFile(path);
  tensorium::Lexer lex(source.c_str());
  tensorium::Parser parser(lex);
  tensorium::Program program = parser.parseProgram();
  tensorium::SemanticAnalyzer sem(program, tensorium::CompilationMode::Executable);
  auto module = tensorium::backend::BackendBuilder::build(program, sem);

  tensorium::validation::canonicalizeDifferentialIR(module);
  tensorium::validation::canonicalizeEinsteinIR(module);
  const auto verify = tensorium::validation::verifyIR(module);
  if (!verify.ok()) {
    std::ostringstream oss;
    oss << "IR verification failed";
    for (const auto &diag : verify.diags)
      oss << "\n  - " << diag.message;
    throw std::runtime_error(oss.str());
  }
  return module;
}

tensorium_mlir::MLIRGenOptions makeRuntimeProbeOptions() {
  auto opts = tensorium_mlir::makeMLIRGenOptions(
      tensorium_mlir::OptimizationLevel::O2);
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridAffinePass = true;
  opts.enableRhsGridAffinePass = true;
  opts.enableStripSourceFuncsPass = true;
  opts.enableStencilLoweringPass = true;
  opts.enableEinsteinLoweringPass = true;
  opts.enableEinsteinAnalyzeEinsumPass = true;
  opts.enableEinsteinCanonicalizePass = true;
  opts.enableEinsteinValidityPass = true;
  return opts;
}

const char *roleName(tensorium_mlir::HostBufferRole role) {
  switch (role) {
  case tensorium_mlir::HostBufferRole::Coordinate:
    return "coordinate";
  case tensorium_mlir::HostBufferRole::Field:
    return "field";
  case tensorium_mlir::HostBufferRole::Output:
    return "output";
  }
  return "unknown";
}

const char *accessName(tensorium_mlir::HostArgAccess access) {
  switch (access) {
  case tensorium_mlir::HostArgAccess::None:
    return "none";
  case tensorium_mlir::HostArgAccess::Read:
    return "read";
  case tensorium_mlir::HostArgAccess::Write:
    return "write";
  case tensorium_mlir::HostArgAccess::ReadWrite:
    return "readwrite";
  }
  return "unknown";
}

const char *argKindName(tensorium_mlir::HostArgKind kind) {
  switch (kind) {
  case tensorium_mlir::HostArgKind::F64:
    return "f64";
  case tensorium_mlir::HostArgKind::Index:
    return "index";
  case tensorium_mlir::HostArgKind::Memref1DF64:
    return "memref1d_f64";
  }
  return "unknown";
}

std::string joinStrings(const std::vector<std::string> &values) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << values[i];
  }
  os << "]";
  return os.str();
}

std::string joinI64(const std::vector<std::int64_t> &values) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << values[i];
  }
  os << "]";
  return os.str();
}

std::array<std::int64_t, 3>
uniformDims(const tensorium_mlir::HostModuleABI &abi,
            const std::array<std::int64_t, 3> &overrides) {
  std::array<std::int64_t, 3> dims = {1, 1, 1};
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (overrides[i] > 0) {
      dims[i] = overrides[i];
    } else if (i < abi.resolution.size() && abi.resolution[i] > 0) {
      dims[i] = abi.resolution[i];
    }
  }
  return dims;
}

void printFieldPlan(const tensorium_mlir::HostModuleABI &abi,
                    std::int64_t nPoints) {
  std::cout << "fields:\n";
  for (const auto &field : abi.fields) {
    const std::int64_t scalars = field.componentCount * nPoints;
    std::cout << "  " << field.name << " rank=" << field.rank
              << " variance=(" << field.up << "," << field.down << ")"
              << " components=" << field.componentCount
              << " scalars=" << scalars << "\n";
  }
}

void printStoragePlan(const tensorium_mlir::runtime::HostFieldStorage &storage) {
  std::cout << "storage:\n";
  std::cout << "  data_arena_allocations=" << storage.dataAllocationCount()
            << " unique_buffers=" << storage.buffers().size()
            << " total_scalars=" << storage.totalScalars() << "\n";
  for (const auto &buffer : storage.buffers()) {
    std::cout << "  buffer key=" << buffer.key
              << " c_name=" << buffer.cName
              << " role=" << roleName(buffer.role)
              << " access=" << accessName(buffer.access)
              << " offset=" << buffer.scalarOffset
              << " scalars=" << buffer.scalarCount
              << " components=" << buffer.componentCount << "\n";
  }
  std::cout << "  kernel_bindings:\n";
  for (const auto &plan : storage.kernelPlans()) {
    std::cout << "    kernel " << plan.symbolName << "\n";
    for (const auto &binding : plan.buffers) {
      const auto &buffer = storage.buffers()[binding.storageIndex];
      std::cout << "      arg=" << binding.argIndex
                << " key=" << buffer.key
                << " access=" << accessName(binding.access)
                << " offset=" << buffer.scalarOffset
                << " scalars=" << buffer.scalarCount << "\n";
    }
  }
}

bool printKernelPlan(const tensorium_mlir::HostKernelABI &kernel,
                     const std::array<std::int64_t, 3> &dims,
                     std::int64_t nPoints) {
  std::cout << "kernel " << kernel.symbolName << " kind=" << kernel.kind
            << " wrapper=" << kernel.wrapperName << "\n";
  std::cout << "  params=" << joinStrings(kernel.params)
            << " coords=" << joinStrings(kernel.coords)
            << " fields=" << joinStrings(kernel.fields)
            << " outputs=" << joinStrings(kernel.outputs) << "\n";
  std::cout << "  read_args=" << joinI64(kernel.readArgIndices)
            << " write_args=" << joinI64(kernel.writeArgIndices)
            << " stencil_radius=" << kernel.stencilRadius << "\n";

  bool ok = true;
  if (kernel.kind == tensorium_mlir::abi::kKindRhsGridAffine ||
      kernel.kind == tensorium_mlir::abi::kKindRhsGridScf) {
    const std::int64_t radius = kernel.stencilRadius;
    const std::int64_t ix = dims[0] - 2 * radius;
    const std::int64_t iy = dims[1] - 2 * radius;
    const std::int64_t iz = dims[2] - 2 * radius;
    std::cout << "  uniform_grid nx=" << dims[0] << " ny=" << dims[1]
              << " nz=" << dims[2] << " n_points=" << nPoints
              << " ghost_required=" << radius << "\n";
    if (ix <= 0 || iy <= 0 || iz <= 0) {
      std::cout << "  ERROR interior is empty for this stencil radius\n";
      ok = false;
    } else {
      std::cout << "  interior nx=" << ix << " ny=" << iy << " nz=" << iz
                << "\n";
    }
  }

  std::cout << "  raw_args:\n";
  for (std::size_t i = 0; i < kernel.rawArgs.size(); ++i) {
    const auto &arg = kernel.rawArgs[i];
    std::cout << "    [" << i << "] " << arg.cName
              << " kind=" << argKindName(arg.kind) << "\n";
  }

  std::cout << "  buffers:\n";
  for (const auto &buffer : kernel.buffers) {
    std::cout << "    " << buffer.name << " c_name=" << buffer.cName
              << " arg=" << buffer.argIndex
              << " role=" << roleName(buffer.role)
              << " access=" << accessName(buffer.access)
              << " rank=" << buffer.rank
              << " variance=(" << buffer.up << "," << buffer.down << ")"
              << " components=" << buffer.componentCount
              << " scalars="
              << tensorium_mlir::requiredBufferScalars(buffer, nPoints)
              << "\n";
  }
  return ok;
}

int run(const Options &opts) {
  auto moduleIR = buildModuleFromFile(opts.inputPath);

  mlir::MLIRContext ctx;
  bool pipelineOk = true;
  auto module = tensorium_mlir::buildMLIRModule(
      moduleIR, ctx, makeRuntimeProbeOptions(), &pipelineOk);
  if (!pipelineOk || !module)
    throw std::runtime_error("MLIR lowering pipeline failed");

  const auto abi = tensorium_mlir::buildHostModuleABI(moduleIR, *module);
  const auto errors = tensorium_mlir::validateHostModuleABI(abi);
  if (!errors.empty()) {
    std::cerr << "ABI descriptor validation failed:\n";
    for (const auto &error : errors)
      std::cerr << "  - " << error << "\n";
    return 2;
  }

  const auto dims = uniformDims(abi, opts.dims);
  const std::int64_t nPoints = dims[0] * dims[1] * dims[2];
  tensorium_mlir::runtime::HostFieldStorage storage(
      abi, tensorium_mlir::runtime::HostGridShape{dims[0], dims[1], dims[2]});

  std::cout << "Tensorium ABI contract probe\n";
  std::cout << "file: " << opts.inputPath << "\n";
  std::cout << "abi_version="
            << tensorium_mlir::abi::kGeneratedKernelABIVersion
            << " mem_layout="
            << tensorium_mlir::abi::kMemLayoutSoAComponentMajor
            << " memref_abi="
            << tensorium_mlir::abi::kMemrefABI1DStridedF64 << "\n";
  std::cout << "simulation dim=" << abi.dimension
            << " coords=" << abi.coordSystem
            << " resolution=" << joinI64(std::vector<std::int64_t>(
                   abi.resolution.begin(), abi.resolution.end()))
            << " spatial=" << abi.spatialScheme << "/"
            << abi.derivativeScheme << " order=" << abi.spatialOrder << "\n";
  std::cout << "uniform_allocation nx=" << dims[0] << " ny=" << dims[1]
            << " nz=" << dims[2] << " n_points=" << nPoints << "\n";

  printFieldPlan(abi, nPoints);
  printStoragePlan(storage);

  bool ok = true;
  std::cout << "kernels:\n";
  for (const auto &kernel : abi.kernels)
    ok &= printKernelPlan(kernel, dims, nPoints);

  if (!ok)
    return 3;

  std::cout << "ABI contract OK\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(parseOptions(argc, argv));
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
