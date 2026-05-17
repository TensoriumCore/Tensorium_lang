#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"
#include "tensorium_mlir/Target/MLIRGen/MLIRGenHostABI.h"
#include "MLIRGenExpr.h"
#include "MLIRGenInitialData.h"
#include "MLIRGenPipeline.h"
#include "MLIRGenShared.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace tensorium_mlir {
namespace {

llvm::StringRef coordSystemToAttr(tensorium::backend::CoordSystem coords) {
  using tensorium::backend::CoordSystem;
  switch (coords) {
  case CoordSystem::Cartesian:
    return "cartesian";
  case CoordSystem::Spherical:
    return "spherical";
  case CoordSystem::Cylindrical:
    return "cylindrical";
  }
  return "cartesian";
}

std::string cStringLiteral(const std::string &input) {
  std::string out = "\"";
  for (char ch : input) {
    switch (ch) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out.push_back(ch);
      break;
    }
  }
  out += "\"";
  return out;
}

std::int64_t componentCountFor(
    llvm::StringRef name,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  auto it = componentCounts.find(name.str());
  if (it == componentCounts.end() || it->second < 1)
    return 1;
  return it->second;
}

std::string sizeExprFor(
    llvm::StringRef name, llvm::StringRef nExpr,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  const std::int64_t components = componentCountFor(name, componentCounts);
  if (components == 1)
    return nExpr.str();
  return std::to_string(components) + " * " + nExpr.str();
}

void appendComma(std::ostringstream &os, bool &first) {
  if (!first)
    os << ", ";
  first = false;
}

void emitRawFormal(std::ostringstream &os, const HostArgABI &arg,
                   bool &first) {
  const std::string &base = arg.cName;
  if (arg.kind == HostArgKind::F64) {
    appendComma(os, first);
    os << "double " << base;
    return;
  }
  if (arg.kind == HostArgKind::Index) {
    appendComma(os, first);
    os << "int64_t " << base;
    return;
  }
  if (arg.kind == HostArgKind::Memref1DF64) {
    appendComma(os, first);
    os << "double *" << base << "_alloc";
    appendComma(os, first);
    os << "double *" << base << "_aligned";
    appendComma(os, first);
    os << "int64_t " << base << "_offset";
    appendComma(os, first);
    os << "int64_t " << base << "_size";
    appendComma(os, first);
    os << "int64_t " << base << "_stride";
  }
}

void emitRawPrototype(std::ostringstream &os, const HostKernelABI &kernel) {
  os << "extern void " << kernel.symbolName << "(";
  bool first = true;
  for (const auto &arg : kernel.rawArgs)
    emitRawFormal(os, arg, first);
  if (first)
    os << "void";
  os << ");\n";
}

void emitScalarFormal(std::ostringstream &os, llvm::StringRef type,
                      llvm::StringRef name, bool &first) {
  appendComma(os, first);
  os << type.str() << " " << makeHostCIdentifier(name.str(), "arg");
}

void emitBufferFormal(std::ostringstream &os, llvm::StringRef name,
                      bool &first) {
  appendComma(os, first);
  os << "double *" << makeHostCIdentifier(name.str(), "buffer");
}

void emitDescriptorCallArgs(std::ostringstream &os, llvm::StringRef name,
                            llvm::StringRef sizeExpr, bool &first) {
  const std::string cName = makeHostCIdentifier(name.str(), "buffer");
  appendComma(os, first);
  os << cName;
  appendComma(os, first);
  os << cName;
  appendComma(os, first);
  os << "0";
  appendComma(os, first);
  os << sizeExpr.str();
  appendComma(os, first);
  os << "1";
}

void emitInitPointWrapper(
    std::ostringstream &os, const HostKernelABI &kernel,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  if (kernel.outputs.empty())
    return;

  os << "static inline void " << kernel.wrapperName << "(";
  bool first = true;
  for (const auto &param : kernel.params)
    emitScalarFormal(os, "double", param, first);
  for (const auto &coord : kernel.coords)
    emitScalarFormal(os, "double", coord, first);
  for (const auto &output : kernel.outputs)
    emitBufferFormal(os, output, first);
  os << ") {\n  " << kernel.symbolName << "(";

  first = true;
  for (const auto &param : kernel.params) {
    appendComma(os, first);
    os << makeHostCIdentifier(param, "param");
  }
  for (const auto &coord : kernel.coords) {
    appendComma(os, first);
    os << makeHostCIdentifier(coord, "coord");
  }
  for (const auto &output : kernel.outputs) {
    emitDescriptorCallArgs(os, output,
                           std::to_string(componentCountFor(output,
                                                            componentCounts)),
                           first);
  }
  os << ");\n}\n";
}

void emitInitGridWrapper(
    std::ostringstream &os, const HostKernelABI &kernel,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  if (kernel.outputs.empty())
    return;

  os << "static inline void " << kernel.wrapperName << "(";
  bool first = true;
  for (const auto &param : kernel.params)
    emitScalarFormal(os, "double", param, first);
  for (const auto &coord : kernel.coords)
    emitBufferFormal(os, coord, first);
  for (const auto &output : kernel.outputs)
    emitBufferFormal(os, output, first);
  emitScalarFormal(os, "int64_t", "n_points", first);
  os << ") {\n  " << kernel.symbolName << "(";

  first = true;
  for (const auto &param : kernel.params) {
    appendComma(os, first);
    os << makeHostCIdentifier(param, "param");
  }
  for (const auto &coord : kernel.coords)
    emitDescriptorCallArgs(os, coord, "n_points", first);
  for (const auto &output : kernel.outputs)
    emitDescriptorCallArgs(os, output,
                           sizeExprFor(output, "n_points", componentCounts),
                           first);
  os << ");\n}\n";
}

void emitRhsGridWrapper(
    std::ostringstream &os, const HostKernelABI &kernel,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  if (kernel.fields.empty())
    return;

  os << "static inline void " << kernel.wrapperName << "(";
  bool first = true;
  emitScalarFormal(os, "int64_t", "nx", first);
  emitScalarFormal(os, "int64_t", "ny", first);
  emitScalarFormal(os, "int64_t", "nz", first);
  emitScalarFormal(os, "double", "dx", first);
  emitScalarFormal(os, "double", "dy", first);
  emitScalarFormal(os, "double", "dz", first);
  for (const auto &param : kernel.params)
    emitScalarFormal(os, "double", param, first);
  for (const auto &field : kernel.fields)
    emitBufferFormal(os, field, first);
  os << ") {\n"
     << "  const int64_t n_points = nx * ny * nz;\n"
     << "  " << kernel.symbolName << "(";

  first = true;
  for (llvm::StringRef name : {"nx", "ny", "nz", "dx", "dy", "dz"}) {
    appendComma(os, first);
    os << name.str();
  }
  for (const auto &param : kernel.params) {
    appendComma(os, first);
    os << makeHostCIdentifier(param, "param");
  }
  for (const auto &field : kernel.fields)
    emitDescriptorCallArgs(os, field,
                           sizeExprFor(field, "n_points", componentCounts),
                           first);
  os << ");\n}\n";
}

void emitConvenienceWrapper(
    std::ostringstream &os, const HostKernelABI &kernel,
    const std::unordered_map<std::string, std::int64_t> &componentCounts) {
  if (kernel.kind == tensorium_mlir::abi::kKindInitPoint) {
    emitInitPointWrapper(os, kernel, componentCounts);
  } else if (kernel.kind == tensorium_mlir::abi::kKindInitGridScf ||
             kernel.kind == tensorium_mlir::abi::kKindInitGridAffine) {
    emitInitGridWrapper(os, kernel, componentCounts);
  } else if (kernel.kind == tensorium_mlir::abi::kKindRhsGridScf ||
             kernel.kind == tensorium_mlir::abi::kKindRhsGridAffine) {
    emitRhsGridWrapper(os, kernel, componentCounts);
  }
}

const char *hostHeaderRoleEnum(HostBufferRole role) {
  switch (role) {
  case HostBufferRole::Coordinate:
    return "TENSORIUM_HOST_BUFFER_ROLE_COORDINATE";
  case HostBufferRole::Field:
    return "TENSORIUM_HOST_BUFFER_ROLE_FIELD";
  case HostBufferRole::Output:
    return "TENSORIUM_HOST_BUFFER_ROLE_OUTPUT";
  }
  return "TENSORIUM_HOST_BUFFER_ROLE_FIELD";
}

const char *hostHeaderAccessEnum(HostArgAccess access) {
  switch (access) {
  case HostArgAccess::None:
    return "TENSORIUM_HOST_ARG_ACCESS_NONE";
  case HostArgAccess::Read:
    return "TENSORIUM_HOST_ARG_ACCESS_READ";
  case HostArgAccess::Write:
    return "TENSORIUM_HOST_ARG_ACCESS_WRITE";
  case HostArgAccess::ReadWrite:
    return "TENSORIUM_HOST_ARG_ACCESS_READWRITE";
  }
  return "TENSORIUM_HOST_ARG_ACCESS_NONE";
}

std::size_t hostBufferDescriptorCount(const HostModuleABI &abi) {
  std::size_t count = 0;
  for (const auto &kernel : abi.kernels)
    count += kernel.buffers.size();
  return count;
}

void emitRuntimeDescriptorTypes(std::ostringstream &os) {
  os << "#ifndef TENSORIUM_GENERATED_HOST_DESCRIPTOR_TYPES_H\n"
     << "#define TENSORIUM_GENERATED_HOST_DESCRIPTOR_TYPES_H\n\n"
     << "typedef enum tensorium_host_buffer_role {\n"
     << "  TENSORIUM_HOST_BUFFER_ROLE_COORDINATE = 1,\n"
     << "  TENSORIUM_HOST_BUFFER_ROLE_FIELD = 2,\n"
     << "  TENSORIUM_HOST_BUFFER_ROLE_OUTPUT = 3\n"
     << "} tensorium_host_buffer_role;\n\n"
     << "typedef enum tensorium_host_arg_access {\n"
     << "  TENSORIUM_HOST_ARG_ACCESS_NONE = 0,\n"
     << "  TENSORIUM_HOST_ARG_ACCESS_READ = 1,\n"
     << "  TENSORIUM_HOST_ARG_ACCESS_WRITE = 2,\n"
     << "  TENSORIUM_HOST_ARG_ACCESS_READWRITE = 3\n"
     << "} tensorium_host_arg_access;\n\n"
     << "typedef struct tensorium_host_kernel_desc {\n"
     << "  const char *symbol_name;\n"
     << "  const char *wrapper_name;\n"
     << "  const char *kind;\n"
     << "  int64_t buffer_begin;\n"
     << "  int64_t buffer_count;\n"
     << "  int64_t stencil_radius;\n"
     << "} tensorium_host_kernel_desc;\n\n"
     << "typedef struct tensorium_host_buffer_desc {\n"
     << "  const char *kernel_symbol;\n"
     << "  const char *name;\n"
     << "  const char *c_name;\n"
     << "  int64_t kernel_index;\n"
     << "  int64_t arg_index;\n"
     << "  int64_t role;\n"
     << "  int64_t access;\n"
     << "  int64_t up;\n"
     << "  int64_t down;\n"
     << "  int64_t rank;\n"
     << "  int64_t component_count;\n"
     << "} tensorium_host_buffer_desc;\n\n"
     << "#endif /* TENSORIUM_GENERATED_HOST_DESCRIPTOR_TYPES_H */\n\n";
}

void emitRuntimeInvokeTypes(std::ostringstream &os) {
  os << "#ifndef TENSORIUM_GENERATED_HOST_INVOKE_TYPES_H\n"
     << "#define TENSORIUM_GENERATED_HOST_INVOKE_TYPES_H\n\n"
     << "typedef struct tensorium_host_grid_desc {\n"
     << "  int64_t nx;\n"
     << "  int64_t ny;\n"
     << "  int64_t nz;\n"
     << "  double dx;\n"
     << "  double dy;\n"
     << "  double dz;\n"
     << "  int64_t n_points;\n"
     << "} tensorium_host_grid_desc;\n\n"
     << "typedef int (*tensorium_host_kernel_invoke_fn)(\n"
     << "    const double *params, int64_t param_count,\n"
     << "    const tensorium_memref1d_f64 *buffers, int64_t buffer_count,\n"
     << "    const tensorium_host_grid_desc *grid);\n\n"
     << "typedef struct tensorium_host_kernel_adapter_desc {\n"
     << "  const char *symbol_name;\n"
     << "  tensorium_host_kernel_invoke_fn invoke;\n"
     << "} tensorium_host_kernel_adapter_desc;\n\n"
     << "#endif /* TENSORIUM_GENERATED_HOST_INVOKE_TYPES_H */\n\n";
}

void emitRuntimeDescriptors(std::ostringstream &os, const HostModuleABI &abi) {
  const std::size_t kernelCount = abi.kernels.size();
  const std::size_t bufferCount = hostBufferDescriptorCount(abi);

  os << "#define TENSORIUM_HOST_KERNEL_COUNT " << kernelCount << "\n"
     << "#define TENSORIUM_HOST_BUFFER_COUNT " << bufferCount << "\n\n";

  os << "static const tensorium_host_kernel_desc tensorium_host_kernels["
     << (kernelCount == 0 ? 1 : kernelCount) << "] = {\n";
  std::size_t bufferBegin = 0;
  if (kernelCount == 0) {
    os << "  {0, 0, 0, 0, 0, 0}\n";
  } else {
    for (std::size_t i = 0; i < abi.kernels.size(); ++i) {
      const auto &kernel = abi.kernels[i];
      os << "  {" << cStringLiteral(kernel.symbolName) << ", "
         << cStringLiteral(kernel.wrapperName) << ", "
         << cStringLiteral(kernel.kind) << ", "
         << static_cast<std::int64_t>(bufferBegin) << ", "
         << static_cast<std::int64_t>(kernel.buffers.size()) << ", "
         << kernel.stencilRadius << "}";
      bufferBegin += kernel.buffers.size();
      os << (i + 1 == abi.kernels.size() ? "\n" : ",\n");
    }
  }
  os << "};\n\n";

  os << "static const tensorium_host_buffer_desc tensorium_host_buffers["
     << (bufferCount == 0 ? 1 : bufferCount) << "] = {\n";
  if (bufferCount == 0) {
    os << "  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}\n";
  } else {
    bool first = true;
    for (std::size_t kernelIndex = 0; kernelIndex < abi.kernels.size();
         ++kernelIndex) {
      const auto &kernel = abi.kernels[kernelIndex];
      for (const auto &buffer : kernel.buffers) {
        if (!first)
          os << ",\n";
        first = false;
        os << "  {" << cStringLiteral(kernel.symbolName) << ", "
           << cStringLiteral(buffer.name) << ", "
           << cStringLiteral(buffer.cName) << ", "
           << static_cast<std::int64_t>(kernelIndex) << ", "
           << buffer.argIndex << ", " << hostHeaderRoleEnum(buffer.role)
           << ", " << hostHeaderAccessEnum(buffer.access) << ", "
           << buffer.up << ", " << buffer.down << ", " << buffer.rank << ", "
           << buffer.componentCount << "}";
      }
    }
    os << "\n";
  }
  os << "};\n\n";
}

bool isRuntimeInvokableGridKernel(const HostKernelABI &kernel) {
  return kernel.kind == tensorium_mlir::abi::kKindInitGridScf ||
         kernel.kind == tensorium_mlir::abi::kKindInitGridAffine ||
         kernel.kind == tensorium_mlir::abi::kKindRhsGridScf ||
         kernel.kind == tensorium_mlir::abi::kKindRhsGridAffine;
}

std::string adapterNameFor(const HostKernelABI &kernel) {
  return "tensorium_invoke_" +
         makeHostCIdentifier(kernel.symbolName, "kernel");
}

void emitMemrefRawArgs(std::ostringstream &os, llvm::StringRef arrayName,
                       std::int64_t index, bool &first) {
  for (llvm::StringRef member : {"allocated", "aligned", "offset", "size",
                                 "stride"}) {
    appendComma(os, first);
    os << arrayName.str() << "[" << index << "]." << member.str();
  }
}

void emitRuntimeInvokeAdapter(std::ostringstream &os,
                              const HostKernelABI &kernel) {
  if (!isRuntimeInvokableGridKernel(kernel))
    return;

  os << "static inline int " << adapterNameFor(kernel) << "(\n"
     << "    const double *params, int64_t param_count,\n"
     << "    const tensorium_memref1d_f64 *buffers, int64_t buffer_count,\n"
     << "    const tensorium_host_grid_desc *grid) {\n"
     << "  if (param_count != "
     << static_cast<std::int64_t>(kernel.params.size())
     << " || buffer_count != "
     << static_cast<std::int64_t>(kernel.buffers.size()) << ")\n"
     << "    return -1;\n"
     << "  if (!grid)\n"
     << "    return -2;\n";
  if (!kernel.params.empty())
    os << "  if (!params)\n"
       << "    return -3;\n";
  if (!kernel.buffers.empty())
    os << "  if (!buffers)\n"
       << "    return -4;\n";
  if (kernel.kind == tensorium_mlir::abi::kKindRhsGridScf ||
      kernel.kind == tensorium_mlir::abi::kKindRhsGridAffine) {
    os << "  if (grid->nx <= 0 || grid->ny <= 0 || grid->nz <= 0 || "
          "grid->n_points <= 0)\n"
       << "    return -5;\n";
  }

  os << "  " << kernel.symbolName << "(";
  bool first = true;
  if (kernel.kind == tensorium_mlir::abi::kKindRhsGridScf ||
      kernel.kind == tensorium_mlir::abi::kKindRhsGridAffine) {
    for (llvm::StringRef expr : {"grid->nx", "grid->ny", "grid->nz",
                                 "grid->dx", "grid->dy", "grid->dz"}) {
      appendComma(os, first);
      os << expr.str();
    }
  }
  for (std::size_t i = 0; i < kernel.params.size(); ++i) {
    appendComma(os, first);
    os << "params[" << i << "]";
  }
  for (std::size_t i = 0; i < kernel.buffers.size(); ++i)
    emitMemrefRawArgs(os, "buffers", static_cast<std::int64_t>(i), first);
  os << ");\n"
     << "  return 0;\n"
     << "}\n\n";
}

void emitRuntimeInvokeAdapters(std::ostringstream &os,
                               const HostModuleABI &abi) {
  os << "#define TENSORIUM_HOST_KERNEL_ADAPTER_COUNT " << abi.kernels.size()
     << "\n\n";
  for (const auto &kernel : abi.kernels)
    emitRuntimeInvokeAdapter(os, kernel);

  os << "static const tensorium_host_kernel_adapter_desc "
        "tensorium_host_kernel_adapters["
     << (abi.kernels.empty() ? 1 : abi.kernels.size()) << "] = {\n";
  if (abi.kernels.empty()) {
    os << "  {0, 0}\n";
  } else {
    for (std::size_t i = 0; i < abi.kernels.size(); ++i) {
      const auto &kernel = abi.kernels[i];
      os << "  {" << cStringLiteral(kernel.symbolName) << ", ";
      if (isRuntimeInvokableGridKernel(kernel))
        os << "&" << adapterNameFor(kernel);
      else
        os << "0";
      os << "}" << (i + 1 == abi.kernels.size() ? "\n" : ",\n");
    }
  }
  os << "};\n\n";
}

void emitPrinterFlatHelper(std::ostringstream &os) {
  os << "static inline void tensorium_print_tensor_flat(\n"
     << "    const char *name, const double *data, int64_t point_index,\n"
     << "    int64_t n_points, int64_t rank, int64_t dim) {\n"
     << "  int64_t components = 1;\n"
     << "  for (int64_t axis = 0; axis < rank; ++axis)\n"
     << "    components *= dim;\n"
     << "  printf(\"%s = {\\n\", name);\n"
     << "  for (int64_t component = 0; component < components; ++component) {\n"
     << "    int64_t rem = component;\n"
     << "    int64_t divisor = components / dim;\n"
     << "    printf(\"  [\");\n"
     << "    for (int64_t axis = 0; axis < rank; ++axis) {\n"
     << "      const int64_t idx = rem / divisor;\n"
     << "      rem %= divisor;\n"
     << "      divisor = divisor > 1 ? divisor / dim : 1;\n"
     << "      if (axis != 0)\n"
     << "        printf(\",\");\n"
     << "      printf(\"%lld\", (long long)idx);\n"
     << "    }\n"
     << "    printf(\"] = %.17g\\n\", data[component * n_points + point_index]);\n"
     << "  }\n"
     << "  printf(\"}\\n\");\n"
     << "}\n\n";
}

void emitPrintRequest(
    std::ostringstream &os, const HostPrintABI &print) {
  const int rank = print.rank;
  const std::string label = cStringLiteral(print.label);
  const std::string field = makeHostCIdentifier(print.fieldName, "field");

  if (rank == 0) {
    os << "  printf(\"%s = %.17g\\n\", " << label << ", " << field
       << "[point_index]);\n";
    return;
  }

  if (rank == 1) {
    os << "  printf(\"%s = [\", " << label << ");\n"
       << "  for (int64_t tensorium_print_i = 0; tensorium_print_i < "
          "tensorium_dim; ++tensorium_print_i) {\n"
       << "    if (tensorium_print_i != 0)\n"
       << "      printf(\", \");\n"
       << "    printf(\"%.17g\", "
       << field << "[tensorium_print_i * n_points + point_index]);\n"
       << "  }\n"
       << "  printf(\"]\\n\");\n";
    return;
  }

  if (rank == 2) {
    os << "  printf(\"%s = [\\n\", " << label << ");\n"
       << "  for (int64_t tensorium_print_i = 0; tensorium_print_i < "
          "tensorium_dim; ++tensorium_print_i) {\n"
       << "    printf(\"  [\");\n"
       << "    for (int64_t tensorium_print_j = 0; tensorium_print_j < "
          "tensorium_dim; ++tensorium_print_j) {\n"
       << "      if (tensorium_print_j != 0)\n"
       << "        printf(\", \");\n"
       << "      const int64_t tensorium_component = "
          "tensorium_print_i * tensorium_dim + tensorium_print_j;\n"
       << "      printf(\"%.17g\", "
       << field << "[tensorium_component * n_points + point_index]);\n"
       << "    }\n"
       << "    printf(tensorium_print_i + 1 == tensorium_dim ? \"]\\n\" : "
          "\"],\\n\");\n"
       << "  }\n"
       << "  printf(\"]\\n\");\n";
    return;
  }

  os << "  tensorium_print_tensor_flat(" << label << ", " << field
     << ", point_index, n_points, " << rank << ", tensorium_dim);\n";
}

void emitPrintRequestHelper(std::ostringstream &os, const HostModuleABI &abi) {
  if (abi.prints.empty())
    return;

  emitPrinterFlatHelper(os);

  os << "static inline void tensorium_print_requested_fields_at(\n"
     << "    int64_t point_index, int64_t n_points";
  for (const auto &field : abi.printFields)
    os << ", const double *" << makeHostCIdentifier(field, "field");
  os << ") {\n"
     << "  const int64_t tensorium_dim = " << abi.dimension << ";\n"
     << "  if (point_index < 0 || point_index >= n_points) {\n"
     << "    fprintf(stderr, \"tensorium print point_index out of range\\n\");\n"
     << "    return;\n"
     << "  }\n";
  for (const auto &print : abi.prints)
    emitPrintRequest(os, print);
  os << "}\n";
}

std::string renderHostHeader(const HostModuleABI &abi) {
  std::ostringstream os;
  os << "#ifndef TENSORIUM_GENERATED_HOST_H\n"
     << "#define TENSORIUM_GENERATED_HOST_H\n\n"
     << "#include <stdint.h>\n"
     << "#include <stdio.h>\n\n"
     << "#ifndef TENSORIUM_GENERATED_HOST_MEMREF_TYPES_H\n"
     << "#define TENSORIUM_GENERATED_HOST_MEMREF_TYPES_H\n\n"
     << "typedef struct tensorium_memref1d_f64 {\n"
     << "  double *allocated;\n"
     << "  double *aligned;\n"
     << "  int64_t offset;\n"
     << "  int64_t size;\n"
     << "  int64_t stride;\n"
     << "} tensorium_memref1d_f64;\n\n"
     << "static inline tensorium_memref1d_f64 "
        "tensorium_make_memref1d_f64(double *data, int64_t size) {\n"
     << "  tensorium_memref1d_f64 ref = {data, data, 0, size, 1};\n"
     << "  return ref;\n"
     << "}\n\n"
     << "#endif /* TENSORIUM_GENERATED_HOST_MEMREF_TYPES_H */\n\n";

  emitRuntimeDescriptorTypes(os);
  emitRuntimeInvokeTypes(os);
  emitRuntimeDescriptors(os, abi);

  os
     << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";

  for (const auto &kernel : abi.kernels)
    emitRawPrototype(os, kernel);

  os << "\n#ifdef __cplusplus\n}\n#endif\n\n";

  for (const auto &kernel : abi.kernels) {
    emitConvenienceWrapper(os, kernel, abi.componentCounts);
    os << "\n";
  }

  emitRuntimeInvokeAdapters(os, abi);

  emitPrintRequestHelper(os, abi);
  if (!abi.prints.empty())
    os << "\n";

  os << "#endif /* TENSORIUM_GENERATED_HOST_H */\n";
  return os.str();
}

std::string renderHostHeader(const tensorium::backend::ModuleIR &module,
                             mlir::ModuleOp moduleOp) {
  return renderHostHeader(buildHostModuleABI(module, moduleOp));
}

bool emitLoweredLLVMIR(mlir::ModuleOp moduleOp, mlir::MLIRContext &ctx,
                       const MLIRGenOptions &opts,
                       std::string *llvmIRText) {
  if (!lowerModuleToLLVM(moduleOp, ctx, opts)) {
    llvm::errs() << "LLVM lowering pipeline failed\n";
    return false;
  }

  llvm::LLVMContext llvmCtx;
  auto llvmModule = mlir::translateModuleToLLVMIR(moduleOp, llvmCtx);
  if (!llvmModule) {
    llvm::errs() << "LLVM IR translation failed\n";
    return false;
  }

  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  llvmModule->print(stream, nullptr);
  stream.flush();

  if (llvmIRText) {
    *llvmIRText = std::move(buffer);
  } else {
    llvm::outs() << buffer;
  }
  return true;
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
buildMLIRModule(const tensorium::backend::ModuleIR &module,
                mlir::MLIRContext &ctx, const MLIRGenOptions &opts,
                bool *pipelineSuccess) {
  if (opts.mlirDisableThreading)
    ctx.disableMultithreading();
  ctx.printOpOnDiagnostic(opts.mlirPrintOpOnDiagnostic);

  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<tensorium::mlir::TensoriumDialect>();

  mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();
  auto moduleOp = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc));

  const auto fields = extractFields(module);
  llvm::SmallVector<mlir::Type, 8> allArgTypes;
  allArgTypes.reserve(fields.size());
  for (const auto &fd : fields) {
    allArgTypes.push_back(
        tensorium::mlir::FieldType::get(&ctx, b.getF64Type(), fd.up, fd.down));
  }

  std::vector<unsigned> initArgIndices = collectInitArgIndices(module, fields);
  std::vector<unsigned> rhsArgIndices = collectRhsArgIndices(module, fields);
  std::vector<unsigned> allArgIndices(fields.size());
  std::iota(allArgIndices.begin(), allArgIndices.end(), 0u);

  auto buildTypeFromIndices = [&](const std::vector<unsigned> &indices) {
    llvm::SmallVector<mlir::Type, 8> types;
    types.reserve(indices.size());
    for (unsigned idx : indices)
      types.push_back(allArgTypes[idx]);
    return b.getFunctionType(types, {});
  };

  auto initFunc = mlir::func::FuncOp::create(
      loc, tensorium_mlir::abi::kSymbolInit, buildTypeFromIndices(initArgIndices));
  auto rhsFunc = mlir::func::FuncOp::create(
      loc, tensorium_mlir::abi::kSymbolRhs, buildTypeFromIndices(rhsArgIndices));
  auto entryFunc = mlir::func::FuncOp::create(
      loc, tensorium_mlir::abi::kSymbolEntry, b.getFunctionType(allArgTypes, {}));

  auto mapFieldArgs = [&](mlir::Block *block, const std::vector<unsigned> &indices) {
    llvm::DenseMap<llvm::StringRef, mlir::Value> fieldArg;
    for (unsigned i = 0; i < indices.size(); ++i)
      fieldArg[fields[indices[i]].name] = block->getArgument(i);
    return fieldArg;
  };

  auto makeFieldNamesAttr = [&](const std::vector<unsigned> &indices) {
    llvm::SmallVector<mlir::Attribute> attrs;
    attrs.reserve(indices.size());
    for (unsigned idx : indices)
      attrs.push_back(b.getStringAttr(fields[idx].name));
    return b.getArrayAttr(attrs);
  };

  auto setCommonABIAttrs = [&](mlir::func::FuncOp fn, llvm::StringRef kind) {
    fn->setAttr(tensorium_mlir::abi::kAttrABIVersion,
                b.getI64IntegerAttr(
                    tensorium_mlir::abi::kGeneratedKernelABIVersion));
    fn->setAttr(tensorium_mlir::abi::kAttrABIKind, b.getStringAttr(kind));
    fn->setAttr(tensorium_mlir::abi::kAttrMemoryLayout,
                b.getStringAttr(
                    tensorium_mlir::abi::kMemLayoutSoAComponentMajor));
    fn->setAttr(tensorium_mlir::abi::kAttrMemrefABI,
                b.getStringAttr(tensorium_mlir::abi::kMemrefABI1DStridedF64));
  };

  setCommonABIAttrs(initFunc, tensorium_mlir::abi::kKindInitSource);
  setCommonABIAttrs(rhsFunc, tensorium_mlir::abi::kKindRhsSource);
  setCommonABIAttrs(entryFunc, tensorium_mlir::abi::kKindEntrySource);
  initFunc->setAttr(tensorium_mlir::abi::kAttrFieldNames,
                    makeFieldNamesAttr(initArgIndices));
  rhsFunc->setAttr(tensorium_mlir::abi::kAttrFieldNames,
                   makeFieldNamesAttr(rhsArgIndices));
  entryFunc->setAttr(tensorium_mlir::abi::kAttrFieldNames,
                     makeFieldNamesAttr(allArgIndices));

  auto *initBlock = initFunc.addEntryBlock();
  b.setInsertionPointToEnd(initBlock);
  auto initFieldArg = mapFieldArgs(initBlock, initArgIndices);
  emitInitialDataOps(b, loc, module, initFieldArg);
  b.create<mlir::func::ReturnOp>(loc);

  auto *rhsBlock = rhsFunc.addEntryBlock();
  b.setInsertionPointToEnd(rhsBlock);
  auto rhsFieldArg = mapFieldArgs(rhsBlock, rhsArgIndices);
  emitEvolutionOps(b, loc, module, rhsFieldArg);
  b.create<mlir::func::ReturnOp>(loc);

  auto *entryBlock = entryFunc.addEntryBlock();
  b.setInsertionPointToEnd(entryBlock);
  llvm::SmallVector<mlir::Value, 8> initCallArgs;
  initCallArgs.reserve(initArgIndices.size());
  for (unsigned idx : initArgIndices)
    initCallArgs.push_back(entryBlock->getArgument(idx));

  llvm::SmallVector<mlir::Value, 8> rhsCallArgs;
  rhsCallArgs.reserve(rhsArgIndices.size());
  for (unsigned idx : rhsArgIndices)
    rhsCallArgs.push_back(entryBlock->getArgument(idx));

  b.create<mlir::func::CallOp>(loc, tensorium_mlir::abi::kSymbolInit, mlir::TypeRange{},
                             initCallArgs);
  b.create<mlir::func::CallOp>(loc, tensorium_mlir::abi::kSymbolRhs, mlir::TypeRange{},
                             rhsCallArgs);
  b.create<mlir::func::ReturnOp>(loc);

  moduleOp->getOperation()->setAttr(
      tensorium_mlir::abi::kAttrABIVersion,
      b.getI64IntegerAttr(tensorium_mlir::abi::kGeneratedKernelABIVersion));
  moduleOp->getOperation()->setAttr(
      tensorium_mlir::abi::kAttrMemoryLayout,
      b.getStringAttr(tensorium_mlir::abi::kMemLayoutSoAComponentMajor));
  moduleOp->getOperation()->setAttr(
      tensorium_mlir::abi::kAttrMemrefABI,
      b.getStringAttr(tensorium_mlir::abi::kMemrefABI1DStridedF64));

  if (module.simulation) {
    moduleOp->getOperation()->setAttr(
        "tensorium.sim.dim", b.getI64IntegerAttr(module.simulation->dimension));
    moduleOp->getOperation()->setAttr(
        "tensorium.sim.coords",
        b.getStringAttr(coordSystemToAttr(module.simulation->coords)));
  }
  moduleOp->push_back(initFunc);
  moduleOp->push_back(rhsFunc);
  moduleOp->push_back(entryFunc);

  MLIRGenOptions pipelineOpts = opts;
  if (module.simulation) {
    pipelineOpts.order = module.simulation->spatial.order;
    if (!module.simulation->resolution.empty() &&
        module.simulation->resolution[0] > 0) {
      pipelineOpts.dx =
          1.0 / static_cast<double>(module.simulation->resolution[0]);
    }
  }

  mlir::PassManager pm(&ctx);
  if (pipelineOpts.mlirPrintIRAfterFailure) {
    pm.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/true);
  }
  if (pipelineOpts.mlirPassTiming) {
    llvm::errs() << "[Tensorium] pass timing: Tensorium MLIR pipeline\n";
    pm.enableTiming();
  }
  addEinsteinPipelineSafe(pm, pipelineOpts);
  addPostMLIRNormalizationPipeline(pm, pipelineOpts);

  const bool ok = mlir::succeeded(pm.run(*moduleOp));
  if (!ok)
    llvm::errs() << "Pipeline failed\n";
  if (pipelineSuccess)
    *pipelineSuccess = ok;
  return moduleOp;
}

bool emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts, std::string *mlirText) {
  mlir::MLIRContext ctx;
  bool pipelineOk = true;
  auto moduleOp = buildMLIRModule(module, ctx, opts, &pipelineOk);
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  moduleOp->print(stream);
  stream.flush();

  if (mlirText) {
    *mlirText = std::move(buffer);
  } else {
    llvm::outs() << buffer;
  }
  return pipelineOk;
}

bool emitLLVMIR(const tensorium::backend::ModuleIR &module,
                const MLIRGenOptions &opts, std::string *llvmIRText) {
  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  ctx.appendDialectRegistry(registry);
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  bool pipelineOk = true;
  auto moduleOp = buildMLIRModule(module, ctx, opts, &pipelineOk);
  if (!pipelineOk)
    return false;

  return emitLoweredLLVMIR(*moduleOp, ctx, opts, llvmIRText);
}

bool emitHostHeader(const tensorium::backend::ModuleIR &module,
                    const MLIRGenOptions &opts, std::string *headerText) {
  mlir::MLIRContext ctx;
  bool pipelineOk = true;
  auto moduleOp = buildMLIRModule(module, ctx, opts, &pipelineOk);
  if (!pipelineOk)
    return false;

  std::string buffer = renderHostHeader(module, *moduleOp);

  if (headerText) {
    *headerText = std::move(buffer);
  } else {
    llvm::outs() << buffer;
  }
  return true;
}

bool emitLLVMIRAndHostHeader(const tensorium::backend::ModuleIR &module,
                             const MLIRGenOptions &opts,
                             std::string *llvmIRText,
                             std::string *headerText) {
  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  ctx.appendDialectRegistry(registry);
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  bool pipelineOk = true;
  auto moduleOp = buildMLIRModule(module, ctx, opts, &pipelineOk);
  if (!pipelineOk)
    return false;

  std::string headerBuffer = renderHostHeader(module, *moduleOp);
  if (!emitLoweredLLVMIR(*moduleOp, ctx, opts, llvmIRText))
    return false;

  if (headerText) {
    *headerText = std::move(headerBuffer);
  } else {
    llvm::outs() << headerBuffer;
  }
  return true;
}

} // namespace tensorium_mlir
