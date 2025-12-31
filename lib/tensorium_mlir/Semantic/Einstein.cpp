
#include "tensorium_mlir/Semantic/Einstein.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace tensorium::semantic {

llvm::StringRef roleToString(IndexRoleKind r) {
  switch (r) {
  case IndexRoleKind::Free:
    return "free";
  case IndexRoleKind::Contracted:
    return "contracted";
  case IndexRoleKind::Summed:
    return "summed";
  case IndexRoleKind::Dangling:
    return "dangling";
  case IndexRoleKind::Invalid:
    return "invalid";
  }
  return "invalid";
}

static llvm::SmallVector<llvm::StringRef, 16>
computeAllSorted(llvm::ArrayRef<llvm::SmallVector<llvm::StringRef, 8>> ins,
                 llvm::ArrayRef<llvm::StringRef> out) {
  llvm::SmallVector<llvm::StringRef, 16> all;
  llvm::DenseSet<llvm::StringRef> seen;

  for (auto x : out)
    if (seen.insert(x).second)
      all.push_back(x);

  for (auto &vec : ins)
    for (auto x : vec)
      if (seen.insert(x).second)
        all.push_back(x);

  return all;
}

EinsteinAnalysisResult analyzeEinstein(llvm::ArrayRef<llvm::StringRef> outIdx,
                                       llvm::ArrayRef<llvm::StringRef> rhsIdx,
                                       const EinsteinAnalyzeOptions &opt) {
  EinsteinAnalysisResult res;

  res.out.assign(outIdx.begin(), outIdx.end());
  res.ins.clear();
  res.ins.emplace_back(rhsIdx.begin(), rhsIdx.end());

  for (auto s : rhsIdx)
    res.counts[s] += 1;

  res.all = computeAllSorted(res.ins, res.out);
  llvm::SmallDenseSet<llvm::StringRef, 16> outSet;
  for (auto x : res.out)
    outSet.insert(x);

  res.valid = true;

  for (auto idx : res.all) {
    const int64_t c = res.counts.lookup(idx);
    const bool inOut = outSet.contains(idx);

    IndexRoleKind role = IndexRoleKind::Invalid;

    if (inOut) {
      role = IndexRoleKind::Free;
      if (c != 1) {
        res.valid = false;
        role = IndexRoleKind::Invalid;
      }
    } else {
      if (c == 2) {
        role = IndexRoleKind::Contracted;
      } else if (c > 2) {
        role = IndexRoleKind::Summed;
        if (!opt.allowSummed)
          res.valid = false;
      } else if (c == 1) {
        role = IndexRoleKind::Dangling;
        if (!opt.allowDangling)
          res.valid = false;
      } else {
        role = IndexRoleKind::Invalid;
        res.valid = false;
      }
    }

    res.roles[idx] = role;
  }

  return res;
}

} // namespace tensorium::semantic
