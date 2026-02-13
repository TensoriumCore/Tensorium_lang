#pragma once

#include "tensorium/Backend/DomainIR.hpp"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include <string>
#include <vector>

namespace mlir {
class ArrayAttr;
class Location;
class OpBuilder;
} // namespace mlir

namespace tensorium_mlir {

struct FieldDesc {
  std::string name;
  unsigned up = 0;
  unsigned down = 0;
};

mlir::ArrayAttr makeIndexArrayAttr(mlir::OpBuilder &b,
                                   const std::vector<std::string> &idx);
mlir::ArrayAttr makeStringArrayAttr(mlir::OpBuilder &b,
                                    const std::vector<std::string> &v);

tensorium::mlir::FieldType asFieldType(mlir::OpBuilder &b,
                                       const tensorium::ir::TensorType &desc);
bool startsWith(const std::string &s, const char *prefix);

std::vector<FieldDesc> extractFields(const tensorium::backend::ModuleIR &module);

void collectExprFieldNames(const tensorium::backend::ExprIR *expr,
                           llvm::StringSet<> &out);
void collectInitExprFieldNames(const tensorium::backend::InitExprIR *expr,
                               const llvm::StringSet<> &knownFieldNames,
                               llvm::StringSet<> &out);
bool moduleUsesFieldName(const tensorium::backend::ModuleIR &module,
                         llvm::StringRef fieldName);

std::vector<unsigned>
collectInitArgIndices(const tensorium::backend::ModuleIR &module,
                      const std::vector<FieldDesc> &fields);
std::vector<unsigned>
collectRhsArgIndices(const tensorium::backend::ModuleIR &module,
                     const std::vector<FieldDesc> &fields);

[[noreturn]] void emitUnsupportedExprError(mlir::Location loc,
                                           const std::string &detail);
[[noreturn]] void emitExternLoweringError(mlir::Location loc,
                                          const std::string &callee);

} // namespace tensorium_mlir
