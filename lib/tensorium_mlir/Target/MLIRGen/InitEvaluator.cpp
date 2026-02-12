#include "tensorium_mlir/Target/MLIRGen/InitEvaluator.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace tensorium_mlir {
namespace {

enum class ValueKind { Scalar, Covector3, Tensor3x3, Metric4x4 };

struct RuntimeValue {
  ValueKind kind = ValueKind::Scalar;
  std::array<double, 16> data{};
  unsigned size = 1;
};

enum class ArgBindingKind { AlphaScalar, GammaCov2, GammaUCon2, Unknown };

struct ArgBinding {
  ArgBindingKind kind = ArgBindingKind::Unknown;
};

static RuntimeValue makeScalar(double value) {
  RuntimeValue out;
  out.kind = ValueKind::Scalar;
  out.size = 1;
  out.data[0] = value;
  return out;
}

static RuntimeValue makeCovector3() {
  RuntimeValue out;
  out.kind = ValueKind::Covector3;
  out.size = 3;
  return out;
}

static RuntimeValue makeTensor3x3() {
  RuntimeValue out;
  out.kind = ValueKind::Tensor3x3;
  out.size = 9;
  return out;
}

static RuntimeValue makeMetric4x4() {
  RuntimeValue out;
  out.kind = ValueKind::Metric4x4;
  out.size = 16;
  return out;
}

static bool isClose(double a, double b, double eps = 1e-12) {
  if (std::isnan(a) || std::isnan(b))
    return false;
  if (std::isinf(a) || std::isinf(b))
    return a == b;
  const double scale = 1.0 + std::max(std::abs(a), std::abs(b));
  return std::abs(a - b) <= eps * scale;
}

static bool isNearZero(double v, double eps = 1e-12) {
  if (std::isnan(v))
    return false;
  if (std::isinf(v))
    return false;
  return std::abs(v) <= eps;
}

static InitEvalResult
validateDescriptor(const InitEvalDescriptor &desc) {
  if (desc.nPoints == 0)
    return InitEvalResult::failure("init evaluator requires nPoints > 0");
  if (!desc.outputs.alpha) {
    return InitEvalResult::failure(
        "init evaluator missing output buffer: alpha");
  }
  for (unsigned c = 0; c < 9; ++c) {
    if (!desc.outputs.gamma[c]) {
      return InitEvalResult::failure("init evaluator missing gamma output "
                                     "component buffer " +
                                     std::to_string(c));
    }
    if (!desc.outputs.gammaU[c]) {
      return InitEvalResult::failure("init evaluator missing gammaU output "
                                     "component buffer " +
                                     std::to_string(c));
    }
  }
  return InitEvalResult::success();
}

static InitEvalResult
valueFromOperand(const llvm::DenseMap<::mlir::Value, RuntimeValue> &values,
                 ::mlir::Value operand, RuntimeValue &out) {
  auto it = values.find(operand);
  if (it == values.end())
    return InitEvalResult::failure("missing runtime value for operand");
  out = it->second;
  return InitEvalResult::success();
}

static InitEvalResult
elementwiseBinary(const RuntimeValue &lhs, const RuntimeValue &rhs,
                  char op, RuntimeValue &out) {
  if (lhs.kind == ValueKind::Scalar && rhs.kind == ValueKind::Scalar) {
    double v = 0.0;
    if (op == '+')
      v = lhs.data[0] + rhs.data[0];
    else if (op == '-')
      v = lhs.data[0] - rhs.data[0];
    else if (op == '*')
      v = lhs.data[0] * rhs.data[0];
    else if (op == '/')
      v = lhs.data[0] / rhs.data[0];
    else
      return InitEvalResult::failure("unsupported scalar binary op");
    out = makeScalar(v);
    return InitEvalResult::success();
  }

  if (op == '*' || op == '/') {
    if (lhs.kind != ValueKind::Scalar && rhs.kind == ValueKind::Scalar) {
      out = lhs;
      for (unsigned i = 0; i < lhs.size; ++i) {
        out.data[i] =
            (op == '*') ? lhs.data[i] * rhs.data[0] : lhs.data[i] / rhs.data[0];
      }
      return InitEvalResult::success();
    }
    if (lhs.kind == ValueKind::Scalar && rhs.kind != ValueKind::Scalar &&
        op == '*') {
      out = rhs;
      for (unsigned i = 0; i < rhs.size; ++i)
        out.data[i] = lhs.data[0] * rhs.data[i];
      return InitEvalResult::success();
    }
  }

  if (lhs.kind != rhs.kind || lhs.size != rhs.size) {
    return InitEvalResult::failure(
        "binary op type mismatch in init evaluator");
  }

  out = lhs;
  for (unsigned i = 0; i < lhs.size; ++i) {
    if (op == '+')
      out.data[i] = lhs.data[i] + rhs.data[i];
    else if (op == '-')
      out.data[i] = lhs.data[i] - rhs.data[i];
    else
      return InitEvalResult::failure(
          "unsupported elementwise binary op in init evaluator");
  }
  return InitEvalResult::success();
}

static InitEvalResult inverse3x3Symmetric(const RuntimeValue &gamma,
                                          RuntimeValue &gammaU) {
  if (gamma.kind != ValueKind::Tensor3x3 || gamma.size != 9) {
    return InitEvalResult::failure(
        "inverse3x3 expects tensor3x3 gamma value");
  }

  const double g00 = gamma.data[0];
  const double g01 = gamma.data[1];
  const double g02 = gamma.data[2];
  const double g10 = gamma.data[3];
  const double g11 = gamma.data[4];
  const double g12 = gamma.data[5];
  const double g20 = gamma.data[6];
  const double g21 = gamma.data[7];
  const double g22 = gamma.data[8];

  const bool diagonal = isNearZero(g01) && isNearZero(g02) && isNearZero(g10) &&
                        isNearZero(g12) && isNearZero(g20) && isNearZero(g21);

  gammaU = makeTensor3x3();
  if (diagonal) {
    gammaU.data[0] = 1.0 / g00;
    gammaU.data[1] = 0.0;
    gammaU.data[2] = 0.0;
    gammaU.data[3] = 0.0;
    gammaU.data[4] = 1.0 / g11;
    gammaU.data[5] = 0.0;
    gammaU.data[6] = 0.0;
    gammaU.data[7] = 0.0;
    gammaU.data[8] = 1.0 / g22;
    return InitEvalResult::success();
  }

  const double c00 = g11 * g22 - g12 * g21;
  const double c01 = -(g10 * g22 - g12 * g20);
  const double c02 = g10 * g21 - g11 * g20;
  const double c10 = -(g01 * g22 - g02 * g21);
  const double c11 = g00 * g22 - g02 * g20;
  const double c12 = -(g00 * g21 - g01 * g20);
  const double c20 = g01 * g12 - g02 * g11;
  const double c21 = -(g00 * g12 - g02 * g10);
  const double c22 = g00 * g11 - g01 * g10;

  const double det = g00 * c00 + g01 * c01 + g02 * c02;
  if (isNearZero(det)) {
    return InitEvalResult::failure(
        "decompose3p1_from_metric: singular spatial metric (det(gamma)=0)");
  }

  gammaU.data[0] = c00 / det;
  gammaU.data[1] = c10 / det;
  gammaU.data[2] = c20 / det;
  gammaU.data[3] = c01 / det;
  gammaU.data[4] = c11 / det;
  gammaU.data[5] = c21 / det;
  gammaU.data[6] = c02 / det;
  gammaU.data[7] = c12 / det;
  gammaU.data[8] = c22 / det;

  return InitEvalResult::success();
}

static InitEvalResult
loadArgValue(const InitEvalDescriptor &desc,
             const std::vector<ArgBinding> &argBindings, ::mlir::Value source,
             std::size_t point, RuntimeValue &out) {
  auto arg = llvm::dyn_cast<::mlir::BlockArgument>(source);
  if (!arg) {
    return InitEvalResult::failure(
        "init evaluator only supports ref from init function block arguments");
  }
  if (arg.getArgNumber() >= argBindings.size()) {
    return InitEvalResult::failure(
        "init evaluator block argument out of range");
  }

  switch (argBindings[arg.getArgNumber()].kind) {
  case ArgBindingKind::AlphaScalar:
    out = makeScalar(desc.outputs.alpha[point]);
    return InitEvalResult::success();
  case ArgBindingKind::GammaCov2:
    out = makeTensor3x3();
    for (unsigned c = 0; c < 9; ++c)
      out.data[c] = desc.outputs.gamma[c][point];
    return InitEvalResult::success();
  case ArgBindingKind::GammaUCon2:
    out = makeTensor3x3();
    for (unsigned c = 0; c < 9; ++c)
      out.data[c] = desc.outputs.gammaU[c][point];
    return InitEvalResult::success();
  case ArgBindingKind::Unknown:
    return InitEvalResult::failure(
        "init evaluator cannot read unknown field argument binding");
  }
  return InitEvalResult::failure("init evaluator failed to read argument");
}

static InitEvalResult
storeArgValue(const InitEvalDescriptor &desc,
              const std::vector<ArgBinding> &argBindings, ::mlir::Value target,
              std::size_t point, const RuntimeValue &rhs) {
  auto arg = llvm::dyn_cast<::mlir::BlockArgument>(target);
  if (!arg) {
    return InitEvalResult::failure(
        "init evaluator only supports assign to init function block arguments");
  }
  if (arg.getArgNumber() >= argBindings.size()) {
    return InitEvalResult::failure(
        "init evaluator assign target argument out of range");
  }

  switch (argBindings[arg.getArgNumber()].kind) {
  case ArgBindingKind::AlphaScalar:
    if (rhs.kind != ValueKind::Scalar)
      return InitEvalResult::failure("assign alpha expects scalar rhs");
    desc.outputs.alpha[point] = rhs.data[0];
    return InitEvalResult::success();
  case ArgBindingKind::GammaCov2:
    if (rhs.kind != ValueKind::Tensor3x3)
      return InitEvalResult::failure("assign gamma expects tensor3x3 rhs");
    for (unsigned c = 0; c < 9; ++c)
      desc.outputs.gamma[c][point] = rhs.data[c];
    return InitEvalResult::success();
  case ArgBindingKind::GammaUCon2:
    if (rhs.kind != ValueKind::Tensor3x3)
      return InitEvalResult::failure("assign gammaU expects tensor3x3 rhs");
    for (unsigned c = 0; c < 9; ++c)
      desc.outputs.gammaU[c][point] = rhs.data[c];
    return InitEvalResult::success();
  case ArgBindingKind::Unknown:
    return InitEvalResult::failure(
        "init evaluator cannot assign to unknown field argument binding");
  }
  return InitEvalResult::failure("init evaluator failed to assign argument");
}

static InitEvalResult
buildArgBindings(::mlir::func::FuncOp initFunc,
                 std::vector<ArgBinding> &argBindings) {
  argBindings.assign(initFunc.getNumArguments(), ArgBinding{});
  bool sawAlpha = false;
  bool sawGamma = false;
  bool sawGammaU = false;

  for (unsigned i = 0; i < initFunc.getNumArguments(); ++i) {
    auto argTy = llvm::dyn_cast<tensorium::mlir::FieldType>(
        initFunc.getArgument(i).getType());
    if (!argTy) {
      return InitEvalResult::failure(
          "init evaluator expected tensorium.field argument type");
    }

    if (argTy.getUp() == 0 && argTy.getDown() == 0 && !sawAlpha) {
      argBindings[i].kind = ArgBindingKind::AlphaScalar;
      sawAlpha = true;
      continue;
    }
    if (argTy.getUp() == 0 && argTy.getDown() == 2 && !sawGamma) {
      argBindings[i].kind = ArgBindingKind::GammaCov2;
      sawGamma = true;
      continue;
    }
    if (argTy.getUp() == 2 && argTy.getDown() == 0 && !sawGammaU) {
      argBindings[i].kind = ArgBindingKind::GammaUCon2;
      sawGammaU = true;
      continue;
    }
    argBindings[i].kind = ArgBindingKind::Unknown;
  }

  if (!sawAlpha || !sawGamma || !sawGammaU) {
    return InitEvalResult::failure(
        "init evaluator requires alpha/gamma/gammaU bindings in tensorium_init signature");
  }

  return InitEvalResult::success();
}

static InitEvalResult executeInitPoint(
    ::mlir::func::FuncOp initFunc, const InitEvalDescriptor &desc,
    const std::vector<ArgBinding> &argBindings, std::size_t point) {
  llvm::DenseMap<::mlir::Value, RuntimeValue> values;

  for (::mlir::Operation &op : initFunc.getBody().front()) {
    if (llvm::isa<::mlir::func::ReturnOp>(&op))
      continue;

    if (auto c = llvm::dyn_cast<tensorium::mlir::ConstOp>(&op)) {
      values[c.getResult()] = makeScalar(c.getValue().convertToDouble());
      continue;
    }

    if (auto param = llvm::dyn_cast<tensorium::mlir::ParamOp>(&op)) {
      auto it = desc.params.find(param.getName().str());
      if (it == desc.params.end()) {
        return InitEvalResult::failure("missing parameter '" +
                                       param.getName().str() + "'");
      }
      values[param.getResult()] = makeScalar(it->second);
      continue;
    }

    if (auto coord = llvm::dyn_cast<tensorium::mlir::CoordOp>(&op)) {
      double coordValue = 0.0;
      const std::string name = coord.getName().str();
      if (name == "r") {
        if (!desc.coords.r)
          return InitEvalResult::failure("missing coordinate array 'r'");
        coordValue = desc.coords.r[point];
      } else if (name == "theta") {
        if (!desc.coords.theta)
          return InitEvalResult::failure("missing coordinate array 'theta'");
        coordValue = desc.coords.theta[point];
      } else if (name == "phi") {
        if (!desc.coords.phi)
          return InitEvalResult::failure("missing coordinate array 'phi'");
        coordValue = desc.coords.phi[point];
      } else {
        return InitEvalResult::failure("unsupported coordinate '" + name + "'");
      }
      values[coord.getResult()] = makeScalar(coordValue);
      continue;
    }

    if (auto ref = llvm::dyn_cast<tensorium::mlir::RefOp>(&op)) {
      RuntimeValue loaded;
      auto loadedRes = loadArgValue(desc, argBindings, ref.getSource(), point, loaded);
      if (!loadedRes.ok)
        return loadedRes;
      values[ref.getResult()] = loaded;
      continue;
    }

    if (auto add = llvm::dyn_cast<tensorium::mlir::AddOp>(&op)) {
      RuntimeValue lhs, rhs, out;
      auto lhsRes = valueFromOperand(values, add.getLhs(), lhs);
      if (!lhsRes.ok)
        return lhsRes;
      auto rhsRes = valueFromOperand(values, add.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto outRes = elementwiseBinary(lhs, rhs, '+', out);
      if (!outRes.ok)
        return outRes;
      values[add.getRes()] = out;
      continue;
    }

    if (auto sub = llvm::dyn_cast<tensorium::mlir::SubOp>(&op)) {
      RuntimeValue lhs, rhs, out;
      auto lhsRes = valueFromOperand(values, sub.getLhs(), lhs);
      if (!lhsRes.ok)
        return lhsRes;
      auto rhsRes = valueFromOperand(values, sub.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto outRes = elementwiseBinary(lhs, rhs, '-', out);
      if (!outRes.ok)
        return outRes;
      values[sub.getRes()] = out;
      continue;
    }

    if (auto mul = llvm::dyn_cast<tensorium::mlir::MulOp>(&op)) {
      RuntimeValue lhs, rhs, out;
      auto lhsRes = valueFromOperand(values, mul.getLhs(), lhs);
      if (!lhsRes.ok)
        return lhsRes;
      auto rhsRes = valueFromOperand(values, mul.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto outRes = elementwiseBinary(lhs, rhs, '*', out);
      if (!outRes.ok)
        return outRes;
      values[mul.getRes()] = out;
      continue;
    }

    if (auto div = llvm::dyn_cast<tensorium::mlir::DivOp>(&op)) {
      RuntimeValue lhs, rhs, out;
      auto lhsRes = valueFromOperand(values, div.getLhs(), lhs);
      if (!lhsRes.ok)
        return lhsRes;
      auto rhsRes = valueFromOperand(values, div.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto outRes = elementwiseBinary(lhs, rhs, '/', out);
      if (!outRes.ok)
        return outRes;
      values[div.getRes()] = out;
      continue;
    }

    if (auto sin = llvm::dyn_cast<tensorium::mlir::SinOp>(&op)) {
      RuntimeValue inVal;
      auto inRes = valueFromOperand(values, sin.getIn(), inVal);
      if (!inRes.ok)
        return inRes;
      if (inVal.kind != ValueKind::Scalar)
        return InitEvalResult::failure("sin expects scalar input");
      values[sin.getOut()] = makeScalar(std::sin(inVal.data[0]));
      continue;
    }

    if (auto sqrt = llvm::dyn_cast<tensorium::mlir::SqrtOp>(&op)) {
      RuntimeValue inVal;
      auto inRes = valueFromOperand(values, sqrt.getIn(), inVal);
      if (!inRes.ok)
        return inRes;
      if (inVal.kind != ValueKind::Scalar)
        return InitEvalResult::failure("sqrt expects scalar input");
      values[sqrt.getOut()] = makeScalar(std::sqrt(inVal.data[0]));
      continue;
    }

    if (auto metric = llvm::dyn_cast<tensorium::mlir::Metric4Op>(&op)) {
      auto out = makeMetric4x4();
      auto comps = metric.getComponents();
      if (comps.size() != 16) {
        return InitEvalResult::failure(
            "metric4 expects 16 scalar components");
      }
      for (unsigned i = 0; i < 16; ++i) {
        RuntimeValue c;
        auto cRes = valueFromOperand(values, comps[i], c);
        if (!cRes.ok)
          return cRes;
        if (c.kind != ValueKind::Scalar)
          return InitEvalResult::failure("metric4 component must be scalar");
        out.data[i] = c.data[0];
      }
      values[metric.getMetric()] = out;
      continue;
    }

    if (auto decomp =
            llvm::dyn_cast<tensorium::mlir::Decompose3P1FromMetricOp>(&op)) {
      RuntimeValue metric;
      auto metricRes = valueFromOperand(values, decomp.getMetric4(), metric);
      if (!metricRes.ok)
        return metricRes;
      if (metric.kind != ValueKind::Metric4x4) {
        return InitEvalResult::failure(
            "decompose3p1_from_metric expects metric4 input");
      }

      for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = i + 1; j < 4; ++j) {
          if (!isClose(metric.data[i * 4 + j], metric.data[j * 4 + i])) {
            return InitEvalResult::failure(
                "decompose3p1_from_metric requires symmetric metric components");
          }
        }
      }

      RuntimeValue beta = makeCovector3();
      beta.data[0] = metric.data[1];
      beta.data[1] = metric.data[2];
      beta.data[2] = metric.data[3];

      RuntimeValue gamma = makeTensor3x3();
      gamma.data[0] = metric.data[5];
      gamma.data[1] = metric.data[6];
      gamma.data[2] = metric.data[7];
      gamma.data[3] = metric.data[9];
      gamma.data[4] = metric.data[10];
      gamma.data[5] = metric.data[11];
      gamma.data[6] = metric.data[13];
      gamma.data[7] = metric.data[14];
      gamma.data[8] = metric.data[15];

      RuntimeValue gammaU;
      auto invRes = inverse3x3Symmetric(gamma, gammaU);
      if (!invRes.ok)
        return invRes;

      double betaDot = 0.0;
      for (unsigned i = 0; i < 3; ++i) {
        double betaUpperI = 0.0;
        for (unsigned j = 0; j < 3; ++j)
          betaUpperI += gammaU.data[i * 3 + j] * beta.data[j];
        betaDot += beta.data[i] * betaUpperI;
      }

      const double alphaSq = betaDot - metric.data[0];
      if (alphaSq < 0.0 && !isNearZero(alphaSq)) {
        return InitEvalResult::failure(
            "decompose3p1_from_metric produced negative alpha^2");
      }
      RuntimeValue alpha = makeScalar(std::sqrt(std::max(0.0, alphaSq)));

      values[decomp.getAlpha()] = alpha;
      values[decomp.getBeta()] = beta;
      values[decomp.getGamma()] = gamma;
      values[decomp.getGammaU()] = gammaU;
      continue;
    }

    if (auto init3p1 = llvm::dyn_cast<tensorium::mlir::Init3P1Op>(&op)) {
      RuntimeValue alphaIn, betaIn, gammaIn, gammaUIn;
      auto aRes = valueFromOperand(values, init3p1.getAlphaIn(), alphaIn);
      if (!aRes.ok)
        return aRes;
      auto bRes = valueFromOperand(values, init3p1.getBetaIn(), betaIn);
      if (!bRes.ok)
        return bRes;
      auto gRes = valueFromOperand(values, init3p1.getGammaIn(), gammaIn);
      if (!gRes.ok)
        return gRes;
      auto guRes = valueFromOperand(values, init3p1.getGammaUIn(), gammaUIn);
      if (!guRes.ok)
        return guRes;
      values[init3p1.getAlpha()] = alphaIn;
      values[init3p1.getBeta()] = betaIn;
      values[init3p1.getGamma()] = gammaIn;
      values[init3p1.getGammaU()] = gammaUIn;
      continue;
    }

    if (auto assign = llvm::dyn_cast<tensorium::mlir::AssignOp>(&op)) {
      if (assign.getIndices().size() != 0) {
        return InitEvalResult::failure(
            "init evaluator only supports whole-field assign (no indices)");
      }
      RuntimeValue rhs;
      auto rhsRes = valueFromOperand(values, assign.getRhs(), rhs);
      if (!rhsRes.ok)
        return rhsRes;
      auto storeRes =
          storeArgValue(desc, argBindings, assign.getField(), point, rhs);
      if (!storeRes.ok)
        return storeRes;
      continue;
    }

    return InitEvalResult::failure("unsupported init op in evaluator: " +
                                   op.getName().getStringRef().str());
  }

  return InitEvalResult::success();
}

} // namespace

InitEvalResult evaluateTensoriumInit(::mlir::ModuleOp module,
                                     const InitEvalDescriptor &desc) {
  auto validRes = validateDescriptor(desc);
  if (!validRes.ok)
    return validRes;

  auto initFunc = module.lookupSymbol<::mlir::func::FuncOp>("tensorium_init");
  if (!initFunc) {
    return InitEvalResult::failure(
        "init evaluator could not find @tensorium_init");
  }

  if (initFunc.getBody().empty() || initFunc.getBody().front().empty()) {
    return InitEvalResult::failure(
        "init evaluator found empty @tensorium_init body");
  }

  std::vector<ArgBinding> argBindings;
  auto bindRes = buildArgBindings(initFunc, argBindings);
  if (!bindRes.ok)
    return bindRes;

  for (std::size_t p = 0; p < desc.nPoints; ++p) {
    auto pointRes = executeInitPoint(initFunc, desc, argBindings, p);
    if (!pointRes.ok) {
      return InitEvalResult::failure("point " + std::to_string(p) + ": " +
                                     pointRes.message);
    }
  }

  return InitEvalResult::success();
}

} // namespace tensorium_mlir
