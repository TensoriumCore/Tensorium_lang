#include "tensorium_mlir/Target/MLIRGen/RhsEvaluator.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorium_mlir {
namespace {

struct RuntimeTensor {
  std::vector<double> data;
  std::vector<std::string> indices;
};

static std::size_t powU(std::size_t base, unsigned exp) {
  std::size_t out = 1;
  for (unsigned i = 0; i < exp; ++i)
    out *= base;
  return out;
}

static std::size_t flattenAxes(const std::vector<unsigned> &axes,
                               unsigned dim) {
  std::size_t out = 0;
  for (unsigned axis : axes)
    out = out * dim + axis;
  return out;
}

static std::vector<unsigned> unflattenAxes(std::size_t linear, unsigned rank,
                                           unsigned dim) {
  std::vector<unsigned> out(rank, 0);
  for (unsigned i = 0; i < rank; ++i) {
    const unsigned rev = rank - 1 - i;
    out[rev] = static_cast<unsigned>(linear % dim);
    linear /= dim;
  }
  return out;
}

static RhsEvalResult valueFromOperand(
    const llvm::DenseMap<::mlir::Value, RuntimeTensor> &values,
    ::mlir::Value operand, RuntimeTensor &out) {
  auto it = values.find(operand);
  if (it == values.end())
    return RhsEvalResult::failure("missing runtime value for operand");
  out = it->second;
  return RhsEvalResult::success();
}

static bool shiftPoint(const std::array<std::size_t, 3> &in,
                       const std::array<int64_t, 3> &delta,
                       const std::array<std::size_t, 3> &extents,
                       unsigned spatialDim, std::array<std::size_t, 3> &out) {
  out = in;
  for (unsigned axis = 0; axis < spatialDim; ++axis) {
    const int64_t shifted =
        static_cast<int64_t>(in[axis]) + static_cast<int64_t>(delta[axis]);
    if (shifted < 0 || shifted >= static_cast<int64_t>(extents[axis]))
      return false;
    out[axis] = static_cast<std::size_t>(shifted);
  }
  return true;
}

static std::size_t flattenPoint(const std::array<std::size_t, 3> &point,
                                const std::array<std::size_t, 3> &extents) {
  return (point[0] * extents[1] + point[1]) * extents[2] + point[2];
}

static std::array<int64_t, 3> getRefOffsets(tensorium::mlir::RefOp ref,
                                            unsigned spatialDim) {
  std::array<int64_t, 3> out{0, 0, 0};
  if (auto offsets = ref.getOffsetsAttr()) {
    const unsigned n = std::min<unsigned>(offsets.size(), spatialDim);
    for (unsigned i = 0; i < n; ++i) {
      auto intAttr = llvm::dyn_cast<::mlir::IntegerAttr>(offsets[i]);
      if (!intAttr)
        continue;
      out[i] = intAttr.getInt();
    }
  }
  return out;
}

static std::vector<std::string> getRefIndices(tensorium::mlir::RefOp ref) {
  std::vector<std::string> out;
  if (auto idx = ref.getIndicesAttr()) {
    out.reserve(idx.size());
    for (::mlir::Attribute attr : idx) {
      auto s = llvm::dyn_cast<::mlir::StringAttr>(attr);
      if (s)
        out.push_back(s.getValue().str());
    }
  }
  return out;
}

static RhsEvalResult loadRefComponentAtPoint(
    const RhsEvalDescriptor &desc, unsigned spatialDim, ::mlir::Value source,
    std::size_t componentIndex, const std::array<std::size_t, 3> &point,
    double &outValue) {
  auto arg = llvm::dyn_cast<::mlir::BlockArgument>(source);
  if (!arg)
    return RhsEvalResult::failure("ref source must be a rhs function argument");
  if (arg.getArgNumber() >= desc.args.size())
    return RhsEvalResult::failure("ref source argument index out of range");

  const auto &field = desc.args[arg.getArgNumber()];
  if (componentIndex >= field.components.size())
    return RhsEvalResult::failure("ref component index out of range");
  if (!field.components[componentIndex])
    return RhsEvalResult::failure("ref source component buffer is null");

  const std::size_t linear = flattenPoint(point, desc.grid.extents);
  (void)spatialDim;
  outValue = field.components[componentIndex][linear];
  return RhsEvalResult::success();
}

static RhsEvalResult evaluateRefAtPoint(const RhsEvalDescriptor &desc,
                                        unsigned spatialDim,
                                        tensorium::mlir::RefOp ref,
                                        const std::array<std::size_t, 3> &point,
                                        RuntimeTensor &out) {
  auto resTy =
      llvm::dyn_cast<tensorium::mlir::FieldType>(ref.getResult().getType());
  if (!resTy)
    return RhsEvalResult::failure("ref result is not tensorium.field");

  out.indices = getRefIndices(ref);
  if (out.indices.size() != resTy.getRank()) {
    return RhsEvalResult::failure(
        "ref indices count does not match result tensor rank");
  }

  const std::size_t componentCount = powU(spatialDim, resTy.getRank());
  out.data.assign(componentCount, 0.0);

  std::array<std::size_t, 3> shiftedPoint;
  const auto offsets = getRefOffsets(ref, spatialDim);
  if (!shiftPoint(point, offsets, desc.grid.extents, spatialDim, shiftedPoint)) {
    return RhsEvalResult::failure(
        "ref offsets move access outside grid (interior-only evaluator)");
  }

  for (std::size_t c = 0; c < componentCount; ++c) {
    double value = 0.0;
    auto loadRes =
        loadRefComponentAtPoint(desc, spatialDim, ref.getSource(), c,
                                shiftedPoint, value);
    if (!loadRes.ok)
      return loadRes;
    out.data[c] = value;
  }
  return RhsEvalResult::success();
}

static RhsEvalResult evalBinaryElementwise(const RuntimeTensor &lhs,
                                           const RuntimeTensor &rhs, char op,
                                           unsigned spatialDim,
                                           RuntimeTensor &out) {
  if (lhs.data.size() != rhs.data.size())
    return RhsEvalResult::failure("binary op expects matching tensor sizes");

  out.indices = lhs.indices;
  out.data.assign(lhs.data.size(), 0.0);

  if (lhs.indices == rhs.indices) {
    for (std::size_t i = 0; i < lhs.data.size(); ++i) {
      if (op == '+')
        out.data[i] = lhs.data[i] + rhs.data[i];
      else if (op == '-')
        out.data[i] = lhs.data[i] - rhs.data[i];
      else
        return RhsEvalResult::failure("unsupported elementwise binary op");
    }
    return RhsEvalResult::success();
  }

  if (lhs.indices.size() != rhs.indices.size()) {
    return RhsEvalResult::failure(
        "binary op expects matching tensor index ranks");
  }

  std::unordered_set<std::string> seen;
  for (const auto &name : lhs.indices) {
    if (!seen.insert(name).second)
      return RhsEvalResult::failure("binary op lhs has duplicate index names");
  }
  seen.clear();
  for (const auto &name : rhs.indices) {
    if (!seen.insert(name).second)
      return RhsEvalResult::failure("binary op rhs has duplicate index names");
  }

  std::unordered_map<std::string, unsigned> rhsPos;
  for (unsigned i = 0; i < rhs.indices.size(); ++i)
    rhsPos[rhs.indices[i]] = i;

  for (std::size_t lhsComp = 0; lhsComp < lhs.data.size(); ++lhsComp) {
    auto lhsAxes = unflattenAxes(lhsComp, lhs.indices.size(), spatialDim);
    std::unordered_map<std::string, unsigned> indexValues;
    for (unsigned i = 0; i < lhs.indices.size(); ++i)
      indexValues[lhs.indices[i]] = lhsAxes[i];

    std::vector<unsigned> rhsAxes(rhs.indices.size(), 0);
    for (unsigned i = 0; i < rhs.indices.size(); ++i) {
      auto it = indexValues.find(rhs.indices[i]);
      if (it == indexValues.end()) {
        return RhsEvalResult::failure(
            "binary op cannot align index '" + rhs.indices[i] + "'");
      }
      rhsAxes[i] = it->second;
    }
    const std::size_t rhsComp = flattenAxes(rhsAxes, spatialDim);

    if (op == '+')
      out.data[lhsComp] = lhs.data[lhsComp] + rhs.data[rhsComp];
    else if (op == '-')
      out.data[lhsComp] = lhs.data[lhsComp] - rhs.data[rhsComp];
    else
      return RhsEvalResult::failure("unsupported elementwise binary op");
  }
  return RhsEvalResult::success();
}

static RhsEvalResult evalMulTensor(const RuntimeTensor &lhs,
                                   const RuntimeTensor &rhs, RuntimeTensor &out) {
  out.indices = lhs.indices;
  out.indices.insert(out.indices.end(), rhs.indices.begin(), rhs.indices.end());
  out.data.assign(lhs.data.size() * rhs.data.size(), 0.0);

  for (std::size_t i = 0; i < lhs.data.size(); ++i) {
    for (std::size_t j = 0; j < rhs.data.size(); ++j)
      out.data[i * rhs.data.size() + j] = lhs.data[i] * rhs.data[j];
  }
  return RhsEvalResult::success();
}

static RhsEvalResult evalDivTensor(const RuntimeTensor &lhs,
                                   const RuntimeTensor &rhs, RuntimeTensor &out) {
  if (!rhs.indices.empty() || rhs.data.size() != 1) {
    return RhsEvalResult::failure("div rhs must be scalar");
  }
  out.indices = lhs.indices;
  out.data.assign(lhs.data.size(), 0.0);
  for (std::size_t i = 0; i < lhs.data.size(); ++i)
    out.data[i] = lhs.data[i] / rhs.data[0];
  return RhsEvalResult::success();
}

static std::vector<std::string> getContractedNames(tensorium::mlir::ContractOp op,
                                                   const RuntimeTensor &in) {
  std::vector<std::string> names;
  if (auto sumAttr = op->getAttrOfType<::mlir::ArrayAttr>("sum_indices")) {
    for (::mlir::Attribute attr : sumAttr) {
      auto s = llvm::dyn_cast<::mlir::StringAttr>(attr);
      if (s)
        names.push_back(s.getValue().str());
    }
  }
  if (!names.empty())
    return names;

  std::unordered_map<std::string, unsigned> count;
  for (const auto &name : in.indices)
    ++count[name];
  for (const auto &it : count) {
    if (it.second > 1)
      names.push_back(it.first);
  }
  return names;
}

static RhsEvalResult evalContractTensor(tensorium::mlir::ContractOp op,
                                        unsigned spatialDim,
                                        const RuntimeTensor &in,
                                        RuntimeTensor &out) {
  auto contracted = getContractedNames(op, in);
  std::unordered_set<std::string> contractedSet(contracted.begin(),
                                                contracted.end());

  out.indices.clear();
  for (const auto &name : in.indices) {
    if (!contractedSet.count(name))
      out.indices.push_back(name);
  }

  const std::size_t outCount = powU(spatialDim, out.indices.size());
  out.data.assign(outCount, 0.0);

  for (std::size_t outComp = 0; outComp < outCount; ++outComp) {
    const auto outAxes = unflattenAxes(outComp, out.indices.size(), spatialDim);
    std::unordered_map<std::string, unsigned> indexValues;
    for (std::size_t i = 0; i < out.indices.size(); ++i)
      indexValues[out.indices[i]] = outAxes[i];

    std::function<double(std::size_t)> evalLoop = [&](std::size_t depth) {
      if (depth == contracted.size()) {
        std::vector<unsigned> inAxes;
        inAxes.reserve(in.indices.size());
        for (const auto &name : in.indices) {
          auto it = indexValues.find(name);
          if (it == indexValues.end())
            return 0.0;
          inAxes.push_back(it->second);
        }
        return in.data[flattenAxes(inAxes, spatialDim)];
      }

      double acc = 0.0;
      const std::string &name = contracted[depth];
      for (unsigned axis = 0; axis < spatialDim; ++axis) {
        indexValues[name] = axis;
        acc += evalLoop(depth + 1);
      }
      return acc;
    };

    out.data[outComp] = evalLoop(0);
  }

  return RhsEvalResult::success();
}

static RhsEvalResult evalDerivTensor(
    const RhsEvalDescriptor &desc, unsigned spatialDim, tensorium::mlir::DerivOp deriv,
    const RuntimeTensor &in, RuntimeTensor &out) {
  auto derivIndex = deriv->getAttrOfType<::mlir::StringAttr>("index");
  if (!derivIndex)
    return RhsEvalResult::failure("deriv op missing index attribute");

  auto ref = deriv.getIn().getDefiningOp<tensorium::mlir::RefOp>();
  if (!ref) {
    return RhsEvalResult::failure(
        "rhs evaluator currently supports deriv only on tensorium.ref");
  }

  auto inTy = llvm::dyn_cast<tensorium::mlir::FieldType>(deriv.getIn().getType());
  auto outTy = llvm::dyn_cast<tensorium::mlir::FieldType>(deriv.getOut().getType());
  if (!inTy || !outTy)
    return RhsEvalResult::failure("deriv operand/result must be tensorium.field");

  out.indices = in.indices;
  out.indices.push_back(derivIndex.getValue().str());

  if (out.indices.size() != outTy.getRank()) {
    return RhsEvalResult::failure(
        "deriv output rank does not match tensor index layout");
  }

  const std::size_t inCount = in.data.size();
  out.data.assign(inCount * spatialDim, 0.0);

  for (std::size_t inComp = 0; inComp < inCount; ++inComp) {
    const auto inAxes = unflattenAxes(inComp, inTy.getRank(), spatialDim);
    for (unsigned axis = 0; axis < spatialDim; ++axis) {
      if (std::abs(desc.grid.spacing[axis]) < 1e-15) {
        return RhsEvalResult::failure("deriv requires non-zero grid spacing");
      }

      std::array<int64_t, 3> deltaPlus{0, 0, 0};
      std::array<int64_t, 3> deltaMinus{0, 0, 0};
      deltaPlus[axis] = 1;
      deltaMinus[axis] = -1;

      const auto refOffsets = getRefOffsets(ref, spatialDim);
      for (unsigned a = 0; a < spatialDim; ++a) {
        deltaPlus[a] += refOffsets[a];
        deltaMinus[a] += refOffsets[a];
      }

      std::array<std::size_t, 3> plusPoint;
      std::array<std::size_t, 3> minusPoint;
      if (!shiftPoint(desc.point, deltaPlus, desc.grid.extents, spatialDim,
                      plusPoint) ||
          !shiftPoint(desc.point, deltaMinus, desc.grid.extents, spatialDim,
                      minusPoint)) {
        return RhsEvalResult::failure(
            "deriv stencil reaches outside grid (interior-only evaluator)");
      }

      double plusVal = 0.0;
      auto plusRes = loadRefComponentAtPoint(
          desc, spatialDim, ref.getSource(), flattenAxes(inAxes, spatialDim),
          plusPoint, plusVal);
      if (!plusRes.ok)
        return plusRes;

      double minusVal = 0.0;
      auto minusRes = loadRefComponentAtPoint(
          desc, spatialDim, ref.getSource(), flattenAxes(inAxes, spatialDim),
          minusPoint, minusVal);
      if (!minusRes.ok)
        return minusRes;

      const double value =
          (plusVal - minusVal) / (2.0 * desc.grid.spacing[axis]);
      out.data[inComp * spatialDim + axis] = value;
    }
  }

  return RhsEvalResult::success();
}

static RhsEvalResult storeDtAssign(const RhsEvalDescriptor &desc,
                                   unsigned spatialDim,
                                   tensorium::mlir::DtAssignOp dt,
                                   const RuntimeTensor &rhs) {
  auto fieldArg = llvm::dyn_cast<::mlir::BlockArgument>(dt.getField());
  if (!fieldArg)
    return RhsEvalResult::failure("dt_assign field must be rhs argument");
  if (fieldArg.getArgNumber() >= desc.args.size())
    return RhsEvalResult::failure("dt_assign field argument index out of range");

  auto fieldTy =
      llvm::dyn_cast<tensorium::mlir::FieldType>(dt.getField().getType());
  if (!fieldTy)
    return RhsEvalResult::failure("dt_assign field is not tensorium.field");

  const std::size_t rank = fieldTy.getRank();
  if (dt.getIndices().size() != rank)
    return RhsEvalResult::failure("dt_assign indices/rank mismatch");
  if (rhs.indices.size() != rank)
    return RhsEvalResult::failure("dt_assign rhs rank/index mismatch");

  const auto &dest = desc.args[fieldArg.getArgNumber()];
  const std::size_t destComps = powU(spatialDim, rank);
  if (dest.components.size() != destComps)
    return RhsEvalResult::failure("dt_assign destination component count mismatch");

  std::vector<std::string> lhsIndices;
  lhsIndices.reserve(rank);
  for (::mlir::Attribute attr : dt.getIndices()) {
    auto s = llvm::dyn_cast<::mlir::StringAttr>(attr);
    if (!s)
      return RhsEvalResult::failure("dt_assign indices must be strings");
    lhsIndices.push_back(s.getValue().str());
  }

  const std::size_t pointLinear = flattenPoint(desc.point, desc.grid.extents);
  for (std::size_t destComp = 0; destComp < destComps; ++destComp) {
    const auto lhsAxes = unflattenAxes(destComp, rank, spatialDim);
    std::unordered_map<std::string, unsigned> indexValues;
    for (std::size_t i = 0; i < rank; ++i)
      indexValues[lhsIndices[i]] = lhsAxes[i];

    std::vector<unsigned> rhsAxes;
    rhsAxes.reserve(rank);
    for (const auto &name : rhs.indices) {
      auto it = indexValues.find(name);
      if (it == indexValues.end()) {
        return RhsEvalResult::failure(
            "dt_assign cannot map rhs index '" + name + "' to lhs indices");
      }
      rhsAxes.push_back(it->second);
    }

    const std::size_t rhsComp = flattenAxes(rhsAxes, spatialDim);
    if (rhsComp >= rhs.data.size())
      return RhsEvalResult::failure("dt_assign rhs component index out of range");
    if (!dest.components[destComp])
      return RhsEvalResult::failure("dt_assign destination component buffer is null");

    dest.components[destComp][pointLinear] = rhs.data[rhsComp];
  }
  return RhsEvalResult::success();
}

static RhsEvalResult validateDescriptor(const RhsEvalDescriptor &desc,
                                        ::mlir::func::FuncOp rhsFunc,
                                        unsigned spatialDim) {
  if (spatialDim == 0 || spatialDim > 3)
    return RhsEvalResult::failure("rhs evaluator expects spatialDim in [1,3]");

  if (desc.grid.spatialDim != spatialDim) {
    return RhsEvalResult::failure(
        "rhs evaluator descriptor spatialDim does not match module");
  }

  for (unsigned axis = 0; axis < spatialDim; ++axis) {
    if (desc.grid.extents[axis] == 0)
      return RhsEvalResult::failure("rhs evaluator grid extents must be > 0");
    if (!(desc.point[axis] < desc.grid.extents[axis])) {
      return RhsEvalResult::failure(
          "rhs evaluator point must be inside grid extents");
    }
  }

  if (desc.args.size() != rhsFunc.getNumArguments()) {
    return RhsEvalResult::failure(
        "rhs evaluator argument buffer count does not match @tensorium_rhs signature");
  }

  for (unsigned i = 0; i < rhsFunc.getNumArguments(); ++i) {
    auto argTy = llvm::dyn_cast<tensorium::mlir::FieldType>(
        rhsFunc.getArgument(i).getType());
    if (!argTy) {
      return RhsEvalResult::failure(
          "rhs evaluator expects tensorium.field argument types");
    }
    const std::size_t expectedComps = powU(spatialDim, argTy.getRank());
    if (desc.args[i].components.size() != expectedComps) {
      return RhsEvalResult::failure(
          "rhs evaluator component buffer count mismatch for argument " +
          std::to_string(i));
    }
    for (std::size_t c = 0; c < expectedComps; ++c) {
      if (!desc.args[i].components[c]) {
        return RhsEvalResult::failure(
            "rhs evaluator received null component buffer for argument " +
            std::to_string(i));
      }
    }
  }

  return RhsEvalResult::success();
}

} // namespace

RhsEvalResult evaluateTensoriumRHS(::mlir::ModuleOp module,
                                   const RhsEvalDescriptor &desc) {
  if (!module)
    return RhsEvalResult::failure("rhs evaluator got null module");

  auto rhsFunc = module.lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhsFunc)
    return RhsEvalResult::failure("rhs evaluator could not find @tensorium_rhs");
  if (rhsFunc.getBody().empty() || rhsFunc.getBody().front().empty()) {
    return RhsEvalResult::failure("rhs evaluator found empty @tensorium_rhs body");
  }

  unsigned spatialDim = desc.grid.spatialDim;
  if (auto dimAttr = module->getAttrOfType<::mlir::IntegerAttr>("tensorium.sim.dim")) {
    spatialDim = static_cast<unsigned>(dimAttr.getInt());
  }

  auto valid = validateDescriptor(desc, rhsFunc, spatialDim);
  if (!valid.ok)
    return valid;

  llvm::DenseMap<::mlir::Value, RuntimeTensor> values;
  for (::mlir::Operation &op : rhsFunc.getBody().front()) {
    if (llvm::isa<::mlir::func::ReturnOp>(&op))
      continue;

    if (auto c = llvm::dyn_cast<tensorium::mlir::ConstOp>(&op)) {
      RuntimeTensor out;
      out.data = {c.getValue().convertToDouble()};
      values[c.getResult()] = std::move(out);
      continue;
    }

    if (auto ref = llvm::dyn_cast<tensorium::mlir::RefOp>(&op)) {
      RuntimeTensor out;
      auto res = evaluateRefAtPoint(desc, spatialDim, ref, desc.point, out);
      if (!res.ok)
        return res;
      values[ref.getResult()] = std::move(out);
      continue;
    }

    if (auto add = llvm::dyn_cast<tensorium::mlir::AddOp>(&op)) {
      RuntimeTensor lhs, rhs, out;
      auto lhsRes = valueFromOperand(values, add.getLhs(), lhs);
      if (!lhsRes.ok)
        return lhsRes;
      auto rhsRes = valueFromOperand(values, add.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto outRes = evalBinaryElementwise(lhs, rhs, '+', spatialDim, out);
      if (!outRes.ok)
        return outRes;
      values[add.getRes()] = std::move(out);
      continue;
    }

    if (auto sub = llvm::dyn_cast<tensorium::mlir::SubOp>(&op)) {
      RuntimeTensor lhs, rhs, out;
      auto lhsRes = valueFromOperand(values, sub.getLhs(), lhs);
      if (!lhsRes.ok)
        return lhsRes;
      auto rhsRes = valueFromOperand(values, sub.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto outRes = evalBinaryElementwise(lhs, rhs, '-', spatialDim, out);
      if (!outRes.ok)
        return outRes;
      values[sub.getRes()] = std::move(out);
      continue;
    }

    if (auto mul = llvm::dyn_cast<tensorium::mlir::MulOp>(&op)) {
      RuntimeTensor lhs, rhs, out;
      auto lhsRes = valueFromOperand(values, mul.getLhs(), lhs);
      if (!lhsRes.ok)
        return lhsRes;
      auto rhsRes = valueFromOperand(values, mul.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto outRes = evalMulTensor(lhs, rhs, out);
      if (!outRes.ok)
        return outRes;
      values[mul.getRes()] = std::move(out);
      continue;
    }

    if (auto div = llvm::dyn_cast<tensorium::mlir::DivOp>(&op)) {
      RuntimeTensor lhs, rhs, out;
      auto lhsRes = valueFromOperand(values, div.getLhs(), lhs);
      if (!lhsRes.ok)
        return lhsRes;
      auto rhsRes = valueFromOperand(values, div.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto outRes = evalDivTensor(lhs, rhs, out);
      if (!outRes.ok)
        return outRes;
      values[div.getRes()] = std::move(out);
      continue;
    }

    if (auto promote = llvm::dyn_cast<tensorium::mlir::PromoteOp>(&op)) {
      RuntimeTensor in;
      auto inRes = valueFromOperand(values, promote.getIn(), in);
      if (!inRes.ok)
        return inRes;
      if (!in.indices.empty() || in.data.size() != 1) {
        return RhsEvalResult::failure(
            "promote expects scalar input in rhs evaluator");
      }

      auto outTy =
          llvm::dyn_cast<tensorium::mlir::FieldType>(promote.getOut().getType());
      if (!outTy)
        return RhsEvalResult::failure("promote result type is not tensorium.field");

      RuntimeTensor out;
      out.indices.assign(outTy.getRank(), "_");
      out.data.assign(powU(spatialDim, outTy.getRank()), in.data[0]);
      values[promote.getOut()] = std::move(out);
      continue;
    }

    if (auto deriv = llvm::dyn_cast<tensorium::mlir::DerivOp>(&op)) {
      RuntimeTensor in, out;
      auto inRes = valueFromOperand(values, deriv.getIn(), in);
      if (!inRes.ok)
        return inRes;
      auto outRes = evalDerivTensor(desc, spatialDim, deriv, in, out);
      if (!outRes.ok)
        return outRes;
      values[deriv.getOut()] = std::move(out);
      continue;
    }

    if (auto contract = llvm::dyn_cast<tensorium::mlir::ContractOp>(&op)) {
      RuntimeTensor in, out;
      auto inRes = valueFromOperand(values, contract.getIn(), in);
      if (!inRes.ok)
        return inRes;
      auto outRes = evalContractTensor(contract, spatialDim, in, out);
      if (!outRes.ok)
        return outRes;
      values[contract.getOut()] = std::move(out);
      continue;
    }

    if (auto dt = llvm::dyn_cast<tensorium::mlir::DtAssignOp>(&op)) {
      RuntimeTensor rhs;
      auto rhsRes = valueFromOperand(values, dt.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto storeRes = storeDtAssign(desc, spatialDim, dt, rhs);
      if (!storeRes.ok)
        return storeRes;
      continue;
    }

    return RhsEvalResult::failure("rhs evaluator unsupported op: " +
                                  op.getName().getStringRef().str());
  }

  return RhsEvalResult::success();
}

} // namespace tensorium_mlir
