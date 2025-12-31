
#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace tensorium::semantic {

enum class IndexRoleKind : uint8_t {
  Free,
  Contracted,
  Summed,
  Dangling,
  Invalid
};

struct EinsteinAnalysisResult {
  llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 8>, 4> ins;
  llvm::SmallVector<llvm::StringRef, 8> out;
  llvm::SmallVector<llvm::StringRef, 16> all;

  llvm::MapVector<llvm::StringRef, int64_t> counts;

  llvm::DenseMap<llvm::StringRef, IndexRoleKind> roles;

  bool valid = true;
};

struct EinsteinAnalyzeOptions {
  bool allowSummed = false;
  bool allowDangling = false;
};

EinsteinAnalysisResult analyzeEinstein(llvm::ArrayRef<llvm::StringRef> outIdx,
                                       llvm::ArrayRef<llvm::StringRef> rhsIdx,
                                       const EinsteinAnalyzeOptions &opt = {});

llvm::StringRef roleToString(IndexRoleKind r);

} // namespace tensorium::semantic
