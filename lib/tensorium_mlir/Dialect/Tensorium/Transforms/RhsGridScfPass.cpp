#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"

#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <llvm/ADT/APFloat.h>

using namespace mlir;

namespace tensorium::mlir {
namespace {

struct Shift3 {
  int x = 0;
  int y = 0;
  int z = 0;
};

struct TensorScalars {
  std::vector<std::string> indices;
  std::vector<Value> comps;
};

static std::size_t powU(std::size_t base, unsigned exp) {
  std::size_t out = 1;
  for (unsigned i = 0; i < exp; ++i)
    out *= base;
  return out;
}

static std::vector<unsigned> unflattenAxes(std::size_t linear, unsigned rank,
                                           unsigned dim) {
  std::vector<unsigned> out(rank, 0);
  for (unsigned i = 0; i < rank; ++i) {
    unsigned rev = rank - 1 - i;
    out[rev] = static_cast<unsigned>(linear % dim);
    linear /= dim;
  }
  return out;
}

static std::size_t flattenAxes(const std::vector<unsigned> &axes,
                               unsigned dim) {
  std::size_t out = 0;
  for (unsigned axis : axes)
    out = out * dim + axis;
  return out;
}

static std::vector<std::string> parseStringArrayAttr(ArrayAttr arr) {
  std::vector<std::string> out;
  if (!arr)
    return out;
  out.reserve(arr.size());
  for (Attribute attr : arr) {
    auto s = dyn_cast<StringAttr>(attr);
    if (!s)
      continue;
    out.push_back(s.getValue().str());
  }
  return out;
}

static std::vector<std::string>
parseStringArrayAttr(const std::optional<ArrayAttr> &arr) {
  if (!arr)
    return {};
  return parseStringArrayAttr(*arr);
}

static bool parseEinsumOutIndices(EinsumOp op, std::vector<std::string> &out) {
  out.clear();
  if (auto outAttr = op->getAttrOfType<ArrayAttr>("tin.idx.out")) {
    out = parseStringArrayAttr(outAttr);
    return !out.empty();
  }

  auto specAttr = op->getAttrOfType<StringAttr>("spec");
  if (!specAttr)
    return false;
  std::string spec = specAttr.getValue().str();
  auto pos = spec.find("->");
  if (pos == std::string::npos)
    return false;
  std::string rhs = spec.substr(pos + 2);
  for (char c : rhs) {
    if (c == ',' || c == ' ' || c == '\t')
      continue;
    out.push_back(std::string(1, c));
  }
  return !out.empty();
}

static std::vector<std::string> getContractedNames(ContractOp op,
                                                   const TensorScalars &in) {
  std::vector<std::string> names;
  if (auto sumAttr = op->getAttrOfType<ArrayAttr>("sum_indices")) {
    names = parseStringArrayAttr(sumAttr);
    if (!names.empty())
      return names;
  }

  std::unordered_map<std::string, unsigned> count;
  for (const auto &name : in.indices)
    ++count[name];
  for (const auto &it : count) {
    if (it.second > 1)
      names.push_back(it.first);
  }
  return names;
}

class RhsScalarizer {
public:
  RhsScalarizer(OpBuilder &b, Location loc, func::FuncOp srcRhs,
                Value nx, Value ny, Value nz, Value dx, Value dy, Value dz,
                Value ix, Value iy, Value iz,
                llvm::ArrayRef<Value> fieldMemrefs)
      : b(b), loc(loc), srcRhs(srcRhs), nx(nx), ny(ny), nz(nz), dx(dx), dy(dy),
        dz(dz), ix(ix), iy(iy), iz(iz), fieldMemrefs(fieldMemrefs.begin(),
                                                     fieldMemrefs.end()) {
    nPoints = arith::MulIOp::create(b, loc, nx, ny);
    nPoints = arith::MulIOp::create(b, loc, nPoints, nz);
  }

  LogicalResult lowerDtAssign(DtAssignOp dt) {
    auto fieldArg = dyn_cast<BlockArgument>(dt.getField());
    if (!fieldArg || fieldArg.getOwner() != &srcRhs.getBody().front()) {
      dt.emitError("rhs-grid-scf: dt_assign field must be rhs block argument");
      return failure();
    }

    if (fieldArg.getArgNumber() >= fieldMemrefs.size()) {
      dt.emitError("rhs-grid-scf: dt_assign field argument index out of range");
      return failure();
    }

    auto lhsTy = dyn_cast<FieldType>(dt.getField().getType());
    if (!lhsTy) {
      dt.emitError("rhs-grid-scf: dt_assign field must be tensorium.field");
      return failure();
    }

    auto rhsOr = evalValue(dt.getRhs(), Shift3{});
    if (failed(rhsOr))
      return failure();
    const TensorScalars &rhs = *rhsOr;

    std::vector<std::string> lhsIndices = parseStringArrayAttr(dt.getIndices());
    if (lhsIndices.size() != lhsTy.getRank()) {
      dt.emitError("rhs-grid-scf: dt_assign indices/rank mismatch");
      return failure();
    }

    const std::size_t lhsCount = powU(spatialDim, lhsTy.getRank());
    Value linear = pointLinear(Shift3{});

    for (std::size_t lhsComp = 0; lhsComp < lhsCount; ++lhsComp) {
      auto lhsAxes = unflattenAxes(lhsComp, lhsTy.getRank(), spatialDim);
      std::unordered_map<std::string, unsigned> values;
      for (std::size_t i = 0; i < lhsIndices.size(); ++i)
        values[lhsIndices[i]] = lhsAxes[i];

      auto rhsCompOr = componentFromIndexMap(rhs, values, dt.getOperation());
      if (!rhsCompOr)
        return failure();

      Value cComp = idxConst(static_cast<int64_t>(lhsComp));
      Value base = arith::MulIOp::create(b, loc, cComp, nPoints);
      Value flat = arith::AddIOp::create(b, loc, base, linear);
      Value outVal = rhs.comps[*rhsCompOr];
      memref::StoreOp::create(b, loc, outVal, fieldMemrefs[fieldArg.getArgNumber()],
                              ValueRange{flat});
    }

    return success();
  }

private:
  FailureOr<TensorScalars> evalValue(Value v, Shift3 shift) {
    if (auto c = dyn_cast_or_null<ConstOp>(v.getDefiningOp())) {
      TensorScalars out;
      out.indices.clear();
      out.comps.push_back(arith::ConstantFloatOp::create(
          b, loc, llvm::cast<FloatType>(b.getF64Type()),
          APFloat(c.getValue().convertToDouble())));
      return out;
    }

    if (auto p = dyn_cast_or_null<PromoteOp>(v.getDefiningOp())) {
      auto in = evalValue(p.getIn(), shift);
      if (failed(in))
        return failure();
      return *in;
    }

    if (auto ref = dyn_cast_or_null<RefOp>(v.getDefiningOp())) {
      auto srcArg = dyn_cast<BlockArgument>(ref.getSource());
      if (!srcArg || srcArg.getOwner() != &srcRhs.getBody().front()) {
        ref.emitError("rhs-grid-scf: ref source must be rhs block argument");
        return failure();
      }
      if (srcArg.getArgNumber() >= fieldMemrefs.size()) {
        ref.emitError("rhs-grid-scf: ref source arg index out of range");
        return failure();
      }

      auto resultTy = dyn_cast<FieldType>(ref.getResult().getType());
      if (!resultTy) {
        ref.emitError("rhs-grid-scf: ref result must be tensorium.field");
        return failure();
      }

      TensorScalars out;
      out.indices = parseStringArrayAttr(ref.getIndices());
      if (out.indices.size() != resultTy.getRank()) {
        ref.emitError("rhs-grid-scf: ref indices/rank mismatch");
        return failure();
      }

      std::array<int64_t, 3> refOffsets{0, 0, 0};
      if (auto off = ref.getOffsetsAttr()) {
        for (unsigned i = 0; i < std::min<std::size_t>(3, off.size()); ++i) {
          auto intAttr = dyn_cast<IntegerAttr>(off[i]);
          if (intAttr)
            refOffsets[i] = intAttr.getInt();
        }
      }

      Shift3 total{shift.x + static_cast<int>(refOffsets[0]),
                   shift.y + static_cast<int>(refOffsets[1]),
                   shift.z + static_cast<int>(refOffsets[2])};
      const std::size_t count = powU(spatialDim, resultTy.getRank());
      out.comps.reserve(count);

      Value lin = pointLinear(total);
      for (std::size_t comp = 0; comp < count; ++comp) {
        Value cComp = idxConst(static_cast<int64_t>(comp));
        Value base = arith::MulIOp::create(b, loc, cComp, nPoints);
        Value flat = arith::AddIOp::create(b, loc, base, lin);
        out.comps.push_back(memref::LoadOp::create(
            b, loc, fieldMemrefs[srcArg.getArgNumber()], ValueRange{flat}));
      }
      return out;
    }

    if (auto add = dyn_cast_or_null<AddOp>(v.getDefiningOp())) {
      auto lhs = evalValue(add.getLhs(), shift);
      auto rhs = evalValue(add.getRhs(), shift);
      if (failed(lhs) || failed(rhs))
        return failure();
      return evalAddSub(*lhs, *rhs, /*isSub=*/false, add.getOperation());
    }

    if (auto sub = dyn_cast_or_null<SubOp>(v.getDefiningOp())) {
      auto lhs = evalValue(sub.getLhs(), shift);
      auto rhs = evalValue(sub.getRhs(), shift);
      if (failed(lhs) || failed(rhs))
        return failure();
      return evalAddSub(*lhs, *rhs, /*isSub=*/true, sub.getOperation());
    }

    if (auto mul = dyn_cast_or_null<MulOp>(v.getDefiningOp())) {
      auto lhs = evalValue(mul.getLhs(), shift);
      auto rhs = evalValue(mul.getRhs(), shift);
      if (failed(lhs) || failed(rhs))
        return failure();

      TensorScalars out;
      out.indices = lhs->indices;
      out.indices.insert(out.indices.end(), rhs->indices.begin(), rhs->indices.end());
      out.comps.reserve(lhs->comps.size() * rhs->comps.size());
      for (Value l : lhs->comps) {
        for (Value r : rhs->comps)
          out.comps.push_back(arith::MulFOp::create(b, loc, l, r));
      }
      return out;
    }

    if (auto div = dyn_cast_or_null<DivOp>(v.getDefiningOp())) {
      auto lhs = evalValue(div.getLhs(), shift);
      auto rhs = evalValue(div.getRhs(), shift);
      if (failed(lhs) || failed(rhs))
        return failure();
      if (!rhs->indices.empty() || rhs->comps.size() != 1) {
        div.emitError("rhs-grid-scf: div rhs must be scalar");
        return failure();
      }
      TensorScalars out;
      out.indices = lhs->indices;
      out.comps.reserve(lhs->comps.size());
      for (Value l : lhs->comps)
        out.comps.push_back(arith::DivFOp::create(b, loc, l, rhs->comps[0]));
      return out;
    }

    if (auto deriv = dyn_cast_or_null<DerivOp>(v.getDefiningOp())) {
      auto derivIdxAttr = deriv->getAttrOfType<StringAttr>("index");
      if (!derivIdxAttr) {
        deriv.emitError("rhs-grid-scf: deriv missing index attribute");
        return failure();
      }

      auto in0 = evalValue(deriv.getIn(), shift);
      if (failed(in0))
        return failure();

      TensorScalars out;
      out.indices = in0->indices;
      out.indices.push_back(derivIdxAttr.getValue().str());
      out.comps.assign(in0->comps.size() * spatialDim, Value());

      Value two = arith::ConstantFloatOp::create(
          b, loc, llvm::cast<FloatType>(b.getF64Type()), APFloat(2.0));
      for (unsigned axis = 0; axis < spatialDim; ++axis) {
        Shift3 plus = shift;
        Shift3 minus = shift;
        if (axis == 0) {
          ++plus.x;
          --minus.x;
        } else if (axis == 1) {
          ++plus.y;
          --minus.y;
        } else {
          ++plus.z;
          --minus.z;
        }

        auto plusVal = evalValue(deriv.getIn(), plus);
        auto minusVal = evalValue(deriv.getIn(), minus);
        if (failed(plusVal) || failed(minusVal))
          return failure();
        if (plusVal->comps.size() != minusVal->comps.size() ||
            plusVal->comps.size() != in0->comps.size()) {
          deriv.emitError("rhs-grid-scf: inconsistent deriv operand component size");
          return failure();
        }

        Value spacing = axis == 0 ? dx : (axis == 1 ? dy : dz);
        Value denom = arith::MulFOp::create(b, loc, two, spacing);

        for (std::size_t c = 0; c < in0->comps.size(); ++c) {
          Value diff =
              arith::SubFOp::create(b, loc, plusVal->comps[c], minusVal->comps[c]);
          out.comps[c * spatialDim + axis] =
              arith::DivFOp::create(b, loc, diff, denom);
        }
      }
      return out;
    }

    if (auto contract = dyn_cast_or_null<ContractOp>(v.getDefiningOp())) {
      auto in = evalValue(contract.getIn(), shift);
      if (failed(in))
        return failure();

      const auto contracted = getContractedNames(contract, *in);
      const std::unordered_set<std::string> contractedSet(contracted.begin(),
                                                          contracted.end());

      TensorScalars out;
      for (const auto &name : in->indices) {
        if (!contractedSet.count(name))
          out.indices.push_back(name);
      }

      const std::size_t outCount = powU(spatialDim, out.indices.size());
      out.comps.reserve(outCount);

      for (std::size_t outComp = 0; outComp < outCount; ++outComp) {
        auto outAxes = unflattenAxes(outComp, out.indices.size(), spatialDim);
        std::unordered_map<std::string, unsigned> values;
        for (std::size_t i = 0; i < out.indices.size(); ++i)
          values[out.indices[i]] = outAxes[i];

        auto sum = sumOverContracted(*in, contracted, 0, values, contract.getOperation());
        if (!sum)
          return failure();
        out.comps.push_back(*sum);
      }
      return out;
    }

    if (auto einsum = dyn_cast_or_null<EinsumOp>(v.getDefiningOp())) {
      std::vector<TensorScalars> inputs;
      inputs.reserve(einsum.getInputs().size());
      for (Value inV : einsum.getInputs()) {
        auto in = evalValue(inV, shift);
        if (failed(in))
          return failure();
        inputs.push_back(*in);
      }

      std::vector<std::string> outIdx;
      if (!parseEinsumOutIndices(einsum, outIdx)) {
        einsum.emitError("rhs-grid-scf: cannot parse einsum output indices");
        return failure();
      }

      std::unordered_set<std::string> outSet(outIdx.begin(), outIdx.end());
      std::vector<std::string> contracted;
      std::unordered_set<std::string> seen;
      for (const auto &in : inputs) {
        for (const auto &name : in.indices) {
          if (!seen.insert(name).second)
            continue;
          if (!outSet.count(name))
            contracted.push_back(name);
        }
      }

      TensorScalars out;
      out.indices = outIdx;
      const std::size_t outCount = powU(spatialDim, out.indices.size());
      out.comps.reserve(outCount);

      for (std::size_t outComp = 0; outComp < outCount; ++outComp) {
        auto outAxes = unflattenAxes(outComp, out.indices.size(), spatialDim);
        std::unordered_map<std::string, unsigned> values;
        for (std::size_t i = 0; i < out.indices.size(); ++i)
          values[out.indices[i]] = outAxes[i];

        auto sum = sumEinsum(inputs, contracted, 0, values, einsum.getOperation());
        if (!sum)
          return failure();
        out.comps.push_back(*sum);
      }
      return out;
    }

    if (auto p = dyn_cast_or_null<ParamOp>(v.getDefiningOp())) {
      p.emitError("rhs-grid-scf: param in RHS is not supported in this pass");
      return failure();
    }

    if (auto c = dyn_cast_or_null<CoordOp>(v.getDefiningOp())) {
      c.emitError("rhs-grid-scf: coord in RHS is not supported in this pass");
      return failure();
    }

    if (auto *op = v.getDefiningOp()) {
      op->emitError("rhs-grid-scf: unsupported op in RHS scalarization: ")
          << op->getName().getStringRef();
    }
    return failure();
  }

  FailureOr<TensorScalars> evalAddSub(const TensorScalars &lhs,
                                      const TensorScalars &rhs, bool isSub,
                                      Operation *opForError) {
    TensorScalars out;
    out.indices = lhs.indices;

    const std::size_t outCount = powU(spatialDim, out.indices.size());
    if (outCount != lhs.comps.size()) {
      opForError->emitError(
          "rhs-grid-scf: lhs component count does not match lhs index rank");
      return failure();
    }

    out.comps.reserve(outCount);
    for (std::size_t outComp = 0; outComp < outCount; ++outComp) {
      auto outAxes = unflattenAxes(outComp, out.indices.size(), spatialDim);
      std::unordered_map<std::string, unsigned> values;
      for (std::size_t i = 0; i < out.indices.size(); ++i)
        values[out.indices[i]] = outAxes[i];

      auto lhsCompOr = componentFromIndexMap(lhs, values, opForError);
      auto rhsCompOr = componentFromIndexMap(rhs, values, opForError);
      if (!lhsCompOr || !rhsCompOr)
        return failure();

      Value val;
      if (isSub) {
        val = arith::SubFOp::create(b, loc, lhs.comps[*lhsCompOr],
                                    rhs.comps[*rhsCompOr]);
      } else {
        val = arith::AddFOp::create(b, loc, lhs.comps[*lhsCompOr],
                                    rhs.comps[*rhsCompOr]);
      }
      out.comps.push_back(val);
    }

    return out;
  }

  std::optional<std::size_t>
  componentFromIndexMap(const TensorScalars &tensor,
                        const std::unordered_map<std::string, unsigned> &values,
                        Operation *opForError) {
    if (tensor.indices.empty())
      return 0;

    std::vector<unsigned> axes;
    axes.reserve(tensor.indices.size());
    for (const auto &name : tensor.indices) {
      auto it = values.find(name);
      if (it == values.end()) {
        opForError->emitError("rhs-grid-scf: cannot map index '") << name
                                                                    << "'";
        return std::nullopt;
      }
      axes.push_back(it->second);
    }
    const std::size_t comp = flattenAxes(axes, spatialDim);
    if (comp >= tensor.comps.size()) {
      opForError->emitError("rhs-grid-scf: component index out of range");
      return std::nullopt;
    }
    return comp;
  }

  std::optional<Value>
  sumOverContracted(const TensorScalars &in,
                    const std::vector<std::string> &contracted,
                    std::size_t depth,
                    std::unordered_map<std::string, unsigned> &values,
                    Operation *opForError) {
    if (depth == contracted.size()) {
      auto compOr = componentFromIndexMap(in, values, opForError);
      if (!compOr)
        return std::nullopt;
      return in.comps[*compOr];
    }

    Value acc;
    bool first = true;
    const std::string &name = contracted[depth];
    for (unsigned axis = 0; axis < spatialDim; ++axis) {
      values[name] = axis;
      auto term = sumOverContracted(in, contracted, depth + 1, values, opForError);
      if (!term)
        return std::nullopt;
      if (first) {
        acc = *term;
        first = false;
      } else {
        acc = arith::AddFOp::create(b, loc, acc, *term);
      }
    }
    return acc;
  }

  std::optional<Value>
  sumEinsum(const std::vector<TensorScalars> &inputs,
            const std::vector<std::string> &contracted,
            std::size_t depth,
            std::unordered_map<std::string, unsigned> &values,
            Operation *opForError) {
    if (depth == contracted.size()) {
      Value prod;
      bool first = true;
      for (const TensorScalars &in : inputs) {
        auto compOr = componentFromIndexMap(in, values, opForError);
        if (!compOr)
          return std::nullopt;
        Value term = in.comps[*compOr];
        if (first) {
          prod = term;
          first = false;
        } else {
          prod = arith::MulFOp::create(b, loc, prod, term);
        }
      }
      return prod;
    }

    Value acc;
    bool first = true;
    const std::string &name = contracted[depth];
    for (unsigned axis = 0; axis < spatialDim; ++axis) {
      values[name] = axis;
      auto term = sumEinsum(inputs, contracted, depth + 1, values, opForError);
      if (!term)
        return std::nullopt;
      if (first) {
        acc = *term;
        first = false;
      } else {
        acc = arith::AddFOp::create(b, loc, acc, *term);
      }
    }
    return acc;
  }

  Value idxConst(int64_t v) {
    return arith::ConstantIndexOp::create(b, loc, v);
  }

  Value addIdx(Value base, int delta) {
    if (delta == 0)
      return base;
    Value c = idxConst(delta);
    return arith::AddIOp::create(b, loc, base, c);
  }

  Value pointLinear(Shift3 shift) {
    Value x = addIdx(ix, shift.x);
    Value y = addIdx(iy, shift.y);
    Value z = addIdx(iz, shift.z);
    Value xy = arith::MulIOp::create(b, loc, x, ny);
    Value xyy = arith::AddIOp::create(b, loc, xy, y);
    Value xyz = arith::MulIOp::create(b, loc, xyy, nz);
    return arith::AddIOp::create(b, loc, xyz, z);
  }

  OpBuilder &b;
  Location loc;
  func::FuncOp srcRhs;
  Value nx;
  Value ny;
  Value nz;
  Value dx;
  Value dy;
  Value dz;
  Value ix;
  Value iy;
  Value iz;
  SmallVector<Value> fieldMemrefs;
  Value nPoints;
  static constexpr unsigned spatialDim = 3;
};

struct RhsGridScfPass
    : public PassWrapper<RhsGridScfPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RhsGridScfPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TensoriumDialect, func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto rhs = module.lookupSymbol<func::FuncOp>("tensorium_rhs");
    if (!rhs)
      return;

    if (module.lookupSymbol<func::FuncOp>("tensorium_rhs_grid_scf"))
      return;

    OpBuilder b(&getContext());
    Location loc = rhs.getLoc();
    Type idxTy = b.getIndexType();
    Type f64 = b.getF64Type();
    Type dynMemF64 = MemRefType::get({ShapedType::kDynamic}, f64);

    SmallVector<Type> args;
    args.push_back(idxTy); // nx
    args.push_back(idxTy); // ny
    args.push_back(idxTy); // nz
    args.push_back(f64);   // dx
    args.push_back(f64);   // dy
    args.push_back(f64);   // dz
    for (Type argTy : rhs.getFunctionType().getInputs()) {
      if (!isa<FieldType>(argTy)) {
        rhs.emitError("rhs-grid-scf: expected tensorium.field arg in tensorium_rhs");
        signalPassFailure();
        return;
      }
      args.push_back(dynMemF64);
    }

    auto fnTy = b.getFunctionType(args, {});
    auto outFn = func::FuncOp::create(loc, "tensorium_rhs_grid_scf", fnTy);
    Block *entry = outFn.addEntryBlock();
    b.setInsertionPointToEnd(entry);

    Value nx = entry->getArgument(0);
    Value ny = entry->getArgument(1);
    Value nz = entry->getArgument(2);
    Value dx = entry->getArgument(3);
    Value dy = entry->getArgument(4);
    Value dz = entry->getArgument(5);

    SmallVector<Value> fieldMemrefs;
    fieldMemrefs.reserve(rhs.getNumArguments());
    for (unsigned i = 0; i < rhs.getNumArguments(); ++i)
      fieldMemrefs.push_back(entry->getArgument(6 + i));

    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value ubX = arith::SubIOp::create(b, loc, nx, c2);
    Value ubY = arith::SubIOp::create(b, loc, ny, c2);
    Value ubZ = arith::SubIOp::create(b, loc, nz, c2);

    auto loopX = scf::ForOp::create(b, loc, c2, ubX, c1);
    b.setInsertionPointToStart(loopX.getBody());
    auto loopY = scf::ForOp::create(b, loc, c2, ubY, c1);
    b.setInsertionPointToStart(loopY.getBody());
    auto loopZ = scf::ForOp::create(b, loc, c2, ubZ, c1);

    {
      OpBuilder ib = OpBuilder::atBlockBegin(loopZ.getBody());
      Value ix = loopX.getInductionVar();
      Value iy = loopY.getInductionVar();
      Value iz = loopZ.getInductionVar();

      RhsScalarizer scalarizer(ib, loc, rhs, nx, ny, nz, dx, dy, dz, ix, iy, iz,
                               fieldMemrefs);

      for (Operation &op : rhs.getBody().front().without_terminator()) {
        if (auto dt = dyn_cast<DtAssignOp>(&op)) {
          if (failed(scalarizer.lowerDtAssign(dt))) {
            signalPassFailure();
            return;
          }
        }
      }
    }

    b.setInsertionPointAfter(loopX);
    func::ReturnOp::create(b, loc);

    module.push_back(outFn);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTensoriumRhsGridScfPass() {
  return std::make_unique<RhsGridScfPass>();
}

} // namespace tensorium::mlir
