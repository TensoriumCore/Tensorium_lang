#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/API/Compiler.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Validation/IRCanonicalize.hpp"
#include "tensorium/Validation/IRVerifier.hpp"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "tensorium_mlir/Target/MLIRGen/GeneratedKernelABI.h"
#include "tensorium_mlir/Target/MLIRGen/InitEvaluator.h"
#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "tensorium_mlir/Target/MLIRGen/RhsEvaluator.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace tensorium;

static std::string readFile(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open fixture: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static backend::ModuleIR buildModuleFromSource(const std::string &source,
                                               CompilationMode mode) {
  Lexer lex(source.c_str());
  Parser parser(lex);
  Program prog = parser.parseProgram();
  SemanticAnalyzer sem(prog, mode);
  return backend::BackendBuilder::build(prog, sem);
}

static backend::ModuleIR buildModuleFromFile(const std::string &path,
                                             CompilationMode mode) {
  return buildModuleFromSource(readFile(path), mode);
}

static bool verifyCanonicalIR(const backend::ModuleIR &mod,
                              const std::string &label) {
  auto verify = validation::verifyIR(mod);
  if (verify.ok())
    return true;

  for (const auto &diag : verify.diags)
    std::cerr << "FAIL(" << label << "): " << diag.message << "\n";
  return false;
}

static tensorium_mlir::MLIRGenOptions makeExecutablePipelineOpts() {
  return tensorium_mlir::makeMLIRGenOptions(
      tensorium_mlir::OptimizationLevel::O2);
}

static bool testMLIRGenOptimizationPassOptions() {
  tensorium_mlir::MLIRPassOptions passOptions;
  passOptions.enableDissipationPass = true;
  passOptions.dx = 0.25;
  passOptions.order = 4;
  passOptions.dissipationStrength = 0.05;
  passOptions.enableMLIRCanonicalizePass = false;
  passOptions.enableMLIRCSEPass = false;
  passOptions.enableMLIRInlinePass = true;
  passOptions.mlirDisableThreading = true;

  auto opts = tensorium_mlir::makeMLIRGenOptions(
      tensorium_mlir::OptimizationLevel::O2, passOptions);

  if (!opts.enableEinsteinLoweringPass || !opts.enableStencilLoweringPass ||
      !opts.enableDissipationPass || !opts.enableMLIRInlinePass) {
    std::cerr << "FAIL: optimization preset/pass options were not merged\n";
    return false;
  }
  if (opts.enableMLIRCanonicalizePass || opts.enableMLIRCSEPass) {
    std::cerr << "FAIL: post-MLIR pass options were not applied\n";
    return false;
  }
  if (opts.dx != 0.25 || opts.order != 4 ||
      opts.dissipationStrength != 0.05 || !opts.mlirDisableThreading) {
    std::cerr << "FAIL: numeric/diagnostic pass options were not applied\n";
    return false;
  }
  return true;
}

static ::mlir::OwningOpRef<::mlir::ModuleOp>
buildMLIRModuleFromSourceWithOpts(const std::string &source,
                                  CompilationMode mode,
                                  ::mlir::MLIRContext &ctx,
                                  const tensorium_mlir::MLIRGenOptions &opts) {
  backend::ModuleIR mod = buildModuleFromSource(source, mode);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);
  auto verify = validation::verifyIR(mod);
  if (!verify.ok()) {
    std::ostringstream oss;
    oss << "IR verification failed for inline source";
    for (const auto &diag : verify.diags)
      oss << "\n  - " << diag.message;
    throw std::runtime_error(oss.str());
  }
  return tensorium_mlir::buildMLIRModule(mod, ctx, opts);
}

static ::mlir::OwningOpRef<::mlir::ModuleOp>
buildMLIRModuleFromFileWithOpts(const std::string &path, CompilationMode mode,
                                ::mlir::MLIRContext &ctx,
                                const tensorium_mlir::MLIRGenOptions &opts) {
  return buildMLIRModuleFromSourceWithOpts(readFile(path), mode, ctx, opts);
}

static ::mlir::OwningOpRef<::mlir::ModuleOp>
buildMLIRModuleFromFile(const std::string &path, CompilationMode mode,
                        ::mlir::MLIRContext &ctx) {
  return buildMLIRModuleFromFileWithOpts(path, mode, ctx,
                                         makeExecutablePipelineOpts());
}

static bool isConstValue(::mlir::Value v, double expected, double eps = 1e-12) {
  auto c = v.getDefiningOp<tensorium::mlir::ConstOp>();
  if (!c)
    return false;
  return std::abs(c.getValue().convertToDouble() - expected) <= eps;
}

static bool isParamNamedValue(::mlir::Value v, llvm::StringRef name) {
  auto p = v.getDefiningOp<tensorium::mlir::ParamOp>();
  return p && p.getName() == name;
}

static bool isCoordNamedValue(::mlir::Value v, llvm::StringRef name) {
  auto c = v.getDefiningOp<tensorium::mlir::CoordOp>();
  return c && c.getName() == name;
}

static bool almostEqual(double got, double expected, double relTol = 1e-12,
                        double absTol = 1e-12) {
  if (std::isnan(got) || std::isnan(expected))
    return false;
  const double scale = std::max(std::abs(expected), 1.0);
  return std::abs(got - expected) <= std::max(absTol, relTol * scale);
}

struct InitEvalBuffers {
  double alpha[1] = {0.0};
  std::array<std::array<double, 1>, 3> beta{};
  std::array<std::array<double, 1>, 16> metric4{};
  std::array<std::array<double, 1>, 9> gamma{};
  std::array<std::array<double, 1>, 9> gammaU{};
  std::array<double *, 3> betaPtrs{};
  std::array<double *, 16> metric4Ptrs{};
  std::array<double *, 9> gammaPtrs{};
  std::array<double *, 9> gammaUPtrs{};
};

struct InitEvalContext {
  InitEvalBuffers buffers;
  std::array<double, 1> r{};
  std::array<double, 1> theta{};
  std::array<double, 1> phi{};
  tensorium_mlir::InitEvalDescriptor desc;
};

static void setupSinglePointInitContext(InitEvalContext &ctx, double M, double r,
                                        double theta, double phi) {
  ctx.r[0] = r;
  ctx.theta[0] = theta;
  ctx.phi[0] = phi;

  for (unsigned c = 0; c < 3; ++c) {
    ctx.buffers.betaPtrs[c] = ctx.buffers.beta[c].data();
    ctx.buffers.beta[c][0] = std::numeric_limits<double>::quiet_NaN();
  }
  for (unsigned c = 0; c < 9; ++c) {
    ctx.buffers.gammaPtrs[c] = ctx.buffers.gamma[c].data();
    ctx.buffers.gammaUPtrs[c] = ctx.buffers.gammaU[c].data();
    ctx.buffers.gamma[c][0] = std::numeric_limits<double>::quiet_NaN();
    ctx.buffers.gammaU[c][0] = std::numeric_limits<double>::quiet_NaN();
  }
  for (unsigned c = 0; c < 16; ++c) {
    ctx.buffers.metric4Ptrs[c] = ctx.buffers.metric4[c].data();
    ctx.buffers.metric4[c][0] = std::numeric_limits<double>::quiet_NaN();
  }
  ctx.buffers.alpha[0] = std::numeric_limits<double>::quiet_NaN();

  ctx.desc.nPoints = 1;
  ctx.desc.params["M"] = M;
  ctx.desc.coords.r = ctx.r.data();
  ctx.desc.coords.theta = ctx.theta.data();
  ctx.desc.coords.phi = ctx.phi.data();
  ctx.desc.outputs.alpha = ctx.buffers.alpha;
  ctx.desc.outputs.beta = ctx.buffers.betaPtrs;
  ctx.desc.outputs.metric4 = ctx.buffers.metric4Ptrs;
  ctx.desc.outputs.gamma = ctx.buffers.gammaPtrs;
  ctx.desc.outputs.gammaU = ctx.buffers.gammaUPtrs;
}

static std::string formatMatrix3x3(const std::array<std::array<double, 1>, 9> &m) {
  std::ostringstream os;
  os << "[[" << m[0][0] << ", " << m[1][0] << ", " << m[2][0] << "], "
     << "[" << m[3][0] << ", " << m[4][0] << ", " << m[5][0] << "], "
     << "[" << m[6][0] << ", " << m[7][0] << ", " << m[8][0] << "]]";
  return os.str();
}

static std::string
formatMatrix4x4(const std::array<std::array<double, 1>, 16> &m) {
  std::ostringstream os;
  os << "[[" << m[0][0] << ", " << m[1][0] << ", " << m[2][0] << ", "
     << m[3][0] << "], "
     << "[" << m[4][0] << ", " << m[5][0] << ", " << m[6][0] << ", "
     << m[7][0] << "], "
     << "[" << m[8][0] << ", " << m[9][0] << ", " << m[10][0] << ", "
     << m[11][0] << "], "
     << "[" << m[12][0] << ", " << m[13][0] << ", " << m[14][0] << ", "
     << m[15][0] << "]]";
  return os.str();
}

static std::string formatVector3(const std::array<std::array<double, 1>, 3> &v) {
  std::ostringstream os;
  os << "[" << v[0][0] << ", " << v[1][0] << ", " << v[2][0] << "]";
  return os.str();
}

struct EvalPoint3 {
  double r = 0.0;
  double theta = 0.0;
  double phi = 0.0;
};

static bool valueDependsOnImpl(
    ::mlir::Value root,
    const std::function<bool(::mlir::Operation *)> &predicate,
    std::unordered_set<::mlir::Operation *> &visited) {
  ::mlir::Operation *def = root.getDefiningOp();
  if (!def)
    return false;
  if (predicate(def))
    return true;
  if (!visited.insert(def).second)
    return false;
  for (::mlir::Value operand : def->getOperands()) {
    if (valueDependsOnImpl(operand, predicate, visited))
      return true;
  }
  return false;
}

static bool valueDependsOn(
    ::mlir::Value root,
    const std::function<bool(::mlir::Operation *)> &predicate) {
  std::unordered_set<::mlir::Operation *> visited;
  return valueDependsOnImpl(root, predicate, visited);
}

static std::string joinStringArrayAttr(::mlir::ArrayAttr arr) {
  if (!arr)
    return "-";
  std::string out;
  bool first = true;
  for (::mlir::Attribute attr : arr) {
    auto s = llvm::dyn_cast<::mlir::StringAttr>(attr);
    if (!s)
      continue;
    if (!first)
      out += ",";
    out += s.getValue().str();
    first = false;
  }
  return out;
}

static std::vector<std::string>
parseStringArrayAttr(::mlir::ArrayAttr arr) {
  std::vector<std::string> out;
  if (!arr)
    return out;
  out.reserve(arr.size());
  for (::mlir::Attribute attr : arr) {
    auto s = llvm::dyn_cast<::mlir::StringAttr>(attr);
    if (s)
      out.push_back(s.getValue().str());
  }
  return out;
}

static std::vector<int64_t> parseI64ArrayAttr(::mlir::ArrayAttr arr) {
  std::vector<int64_t> out;
  if (!arr)
    return out;
  out.reserve(arr.size());
  for (::mlir::Attribute attr : arr) {
    auto i = llvm::dyn_cast<::mlir::IntegerAttr>(attr);
    if (i)
      out.push_back(i.getInt());
  }
  return out;
}

static bool isStaticF64Memref(::mlir::Type type, int64_t size) {
  auto memTy = llvm::dyn_cast<::mlir::MemRefType>(type);
  return memTy && memTy.getRank() == 1 && memTy.getShape()[0] == size &&
         memTy.getElementType().isF64();
}

static bool isDynamicF64Memref(::mlir::Type type) {
  auto memTy = llvm::dyn_cast<::mlir::MemRefType>(type);
  return memTy && memTy.getRank() == 1 &&
         memTy.getShape()[0] == ::mlir::ShapedType::kDynamic &&
         memTy.getElementType().isF64();
}

static std::optional<int64_t> getConstantIndexValueForTest(::mlir::Value value) {
  if (auto indexOp = value.getDefiningOp<::mlir::arith::ConstantIndexOp>())
    return indexOp.value();
  if (auto constOp = value.getDefiningOp<::mlir::arith::ConstantOp>()) {
    if (!value.getType().isIndex())
      return std::nullopt;
    if (auto intAttr = llvm::dyn_cast<::mlir::IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

static std::optional<double> getConstantF64ValueForTest(::mlir::Value value) {
  if (auto floatOp = value.getDefiningOp<::mlir::arith::ConstantFloatOp>())
    return floatOp.value().convertToDouble();
  if (auto constOp = value.getDefiningOp<::mlir::arith::ConstantOp>()) {
    if (auto floatAttr = llvm::dyn_cast<::mlir::FloatAttr>(constOp.getValue()))
      return floatAttr.getValue().convertToDouble();
  }
  return std::nullopt;
}

static bool collectInitPointConstantStores(
    ::mlir::func::FuncOp initPoint, unsigned firstOutputArg,
    std::array<double, 1> &alpha, std::array<double, 9> &gamma,
    std::array<double, 9> &gammaU) {
  std::array<bool, 1> alphaSeen{};
  std::array<bool, 9> gammaSeen{};
  std::array<bool, 9> gammaUSeen{};
  bool ok = true;

  initPoint.walk([&](::mlir::memref::StoreOp store) {
    if (!ok)
      return;
    auto outputArg = llvm::dyn_cast<::mlir::BlockArgument>(store.getMemref());
    if (!outputArg || outputArg.getOwner() != &initPoint.getBody().front()) {
      ok = false;
      return;
    }
    const unsigned argNumber = outputArg.getArgNumber();
    if (argNumber < firstOutputArg || argNumber > firstOutputArg + 2) {
      ok = false;
      return;
    }
    if (store.getIndices().size() != 1) {
      ok = false;
      return;
    }

    auto component = getConstantIndexValueForTest(store.getIndices().front());
    auto value = getConstantF64ValueForTest(store.getValue());
    if (!component || !value) {
      ok = false;
      return;
    }

    if (argNumber == firstOutputArg) {
      if (*component != 0) {
        ok = false;
        return;
      }
      alpha[0] = *value;
      alphaSeen[0] = true;
      return;
    }

    if (*component < 0 || *component >= 9) {
      ok = false;
      return;
    }
    const auto idx = static_cast<std::size_t>(*component);
    if (argNumber == firstOutputArg + 1) {
      gamma[idx] = *value;
      gammaSeen[idx] = true;
    } else {
      gammaU[idx] = *value;
      gammaUSeen[idx] = true;
    }
  });

  if (!ok)
    return false;
  for (bool seen : alphaSeen)
    if (!seen)
      return false;
  for (bool seen : gammaSeen)
    if (!seen)
      return false;
  for (bool seen : gammaUSeen)
    if (!seen)
      return false;
  return true;
}

static bool verifyCommonGeneratedABIAttrs(::mlir::func::FuncOp fn,
                                          const char *expectedKind,
                                          std::string &error) {
  if (!fn) {
    error = "missing function for ABI check";
    return false;
  }

  auto version = fn->getAttrOfType<::mlir::IntegerAttr>(
      tensorium_mlir::abi::kAttrABIVersion);
  if (!version ||
      version.getInt() != tensorium_mlir::abi::kGeneratedKernelABIVersion) {
    error = std::string("invalid ABI version on ") +
            fn.getSymName().str();
    return false;
  }

  auto kind = fn->getAttrOfType<::mlir::StringAttr>(
      tensorium_mlir::abi::kAttrABIKind);
  if (!kind || kind.getValue() != expectedKind) {
    error = std::string("invalid ABI kind on ") + fn.getSymName().str();
    return false;
  }

  auto memLayout = fn->getAttrOfType<::mlir::StringAttr>(
      tensorium_mlir::abi::kAttrMemoryLayout);
  if (!memLayout ||
      memLayout.getValue() != tensorium_mlir::abi::kMemLayoutSoAComponentMajor) {
    error = std::string("invalid memory layout attr on ") +
            fn.getSymName().str();
    return false;
  }

  auto memrefABI = fn->getAttrOfType<::mlir::StringAttr>(
      tensorium_mlir::abi::kAttrMemrefABI);
  if (!memrefABI ||
      memrefABI.getValue() != tensorium_mlir::abi::kMemrefABI1DStridedF64) {
    error = std::string("invalid memref ABI attr on ") +
            fn.getSymName().str();
    return false;
  }

  return true;
}

static std::string trimCopy(const std::string &s) {
  std::size_t begin = 0;
  while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])))
    ++begin;
  std::size_t end = s.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(s[end - 1])))
    --end;
  return s.substr(begin, end - begin);
}

static bool llvmFunctionArgTypeTokens(const std::string &llvmIR,
                                      const std::string &name,
                                      std::vector<std::string> &typesOut) {
  typesOut.clear();
  const std::string needle = "define void @" + name + "(";
  const std::size_t pos = llvmIR.find(needle);
  if (pos == std::string::npos)
    return false;

  const std::size_t open = llvmIR.find('(', pos);
  const std::size_t close = llvmIR.find(')', open);
  if (open == std::string::npos || close == std::string::npos || close <= open)
    return false;

  const std::string args = llvmIR.substr(open + 1, close - open - 1);
  if (trimCopy(args).empty())
    return true;

  std::size_t start = 0;
  while (start < args.size()) {
    std::size_t comma = args.find(',', start);
    std::string piece =
        trimCopy(args.substr(start, comma == std::string::npos
                                       ? std::string::npos
                                       : comma - start));
    if (!piece.empty()) {
      std::size_t space = piece.find(' ');
      typesOut.push_back(piece.substr(0, space));
    }

    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }

  return true;
}

static std::string typeShapeKey(::mlir::Type ty) {
  auto fieldTy = llvm::dyn_cast<tensorium::mlir::FieldType>(ty);
  if (!fieldTy)
    return "?";
  return std::to_string(fieldTy.getUp()) + "/" +
         std::to_string(fieldTy.getDown());
}

static std::vector<std::string>
collectRhsTensoriumSignature(::mlir::ModuleOp module) {
  std::vector<std::string> sig;
  auto rhs = module.lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhs)
    return sig;

  for (::mlir::Operation &op : rhs.getBody().front()) {
    std::string key = op.getName().getStringRef().str();

    if (auto ref = llvm::dyn_cast<tensorium::mlir::RefOp>(&op)) {
      key += "|idx=" +
             (ref.getIndices() ? joinStringArrayAttr(*ref.getIndices()) : "-");
      key += "|src=" + typeShapeKey(ref.getSource().getType());
    } else if (auto ctr = llvm::dyn_cast<tensorium::mlir::ContractOp>(&op)) {
      key += "|sum=" +
             joinStringArrayAttr(ctr->getAttrOfType<::mlir::ArrayAttr>(
                 "sum_indices"));
    } else if (auto deriv = llvm::dyn_cast<tensorium::mlir::DerivOp>(&op)) {
      key += "|idx=" + deriv->getAttrOfType<::mlir::StringAttr>("index")
                          .getValue()
                          .str();
    } else if (auto dt = llvm::dyn_cast<tensorium::mlir::DtAssignOp>(&op)) {
      key += "|lhs=" + joinStringArrayAttr(dt.getIndices());
      key += "|rhs=" + typeShapeKey(dt.getRhs().getType());
    }

    sig.push_back(std::move(key));
  }

  return sig;
}

static std::string exprCanonicalKey(const backend::ExprIR *expr) {
  using backend::ExprIR;
  if (!expr)
    return "null";

  switch (expr->kind) {
  case ExprIR::Kind::Number:
    return "num";
  case ExprIR::Kind::Var: {
    auto *var = static_cast<const backend::VarIR *>(expr);
    std::string out = "var(" + var->name + ";";
    for (size_t i = 0; i < var->tensorIndexNames.size(); ++i) {
      if (i)
        out += ",";
      out += var->tensorIndexNames[i];
    }
    out += ")";
    return out;
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const backend::BinaryIR *>(expr);
    return "bin(" + bin->op + "," + exprCanonicalKey(bin->lhs.get()) + "," +
           exprCanonicalKey(bin->rhs.get()) + ")";
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<const backend::CallIR *>(expr);
    std::string out = "call(" + call->callee;
    for (const auto &arg : call->args)
      out += "," + exprCanonicalKey(arg.get());
    out += ")";
    return out;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<const backend::TensorProductIR *>(expr);
    return "prod(" + exprCanonicalKey(prod->lhs.get()) + "," +
           exprCanonicalKey(prod->rhs.get()) + ")";
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<const backend::ContractionIR *>(expr);
    std::string out = "ctr(";
    for (size_t i = 0; i < ctr->summedIndices.size(); ++i) {
      if (i)
        out += ",";
      out += ctr->summedIndices[i];
    }
    out += ";" + exprCanonicalKey(ctr->in.get()) + ")";
    return out;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<const backend::IndexRenameIR *>(expr);
    return "rename(" + rename->from + "->" + rename->to + ";" +
           exprCanonicalKey(rename->in.get()) + ")";
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<const backend::IndexPermuteIR *>(expr);
    std::string out = "perm(";
    for (size_t i = 0; i < perm->order.size(); ++i) {
      if (i)
        out += ",";
      out += perm->order[i];
    }
    out += ";" + exprCanonicalKey(perm->in.get()) + ")";
    return out;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<const backend::TraceIR *>(expr);
    std::string out = "trace(";
    for (size_t i = 0; i < trace->tracedIndices.size(); ++i) {
      if (i)
        out += ",";
      out += trace->tracedIndices[i];
    }
    out += ";" + exprCanonicalKey(trace->in.get()) + ")";
    return out;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<const backend::PartialDerivativeIR *>(expr);
    return "pd(" + diff->coordIndex + ";" + exprCanonicalKey(diff->in.get()) +
           ")";
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<const backend::GradientIR *>(expr);
    return "grad(" + exprCanonicalKey(grad->in.get()) + ")";
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<const backend::CovariantDerivativeIR *>(expr);
    return "cd(" + diff->derivIndex + ";" + exprCanonicalKey(diff->in.get()) +
           ")";
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<const backend::DivergenceIR *>(expr);
    return "div(" + div->contractedIndex + ";" + exprCanonicalKey(div->in.get()) +
           ")";
  }
  }
  return "?";
}

struct IRStats {
  int contractions = 0;
  int partials = 0;
  int gradients = 0;
  int divergences = 0;
  int covariant = 0;
  int renames = 0;
};

static void collectExprKinds(const backend::ExprIR *expr,
                             std::vector<backend::ExprIR::Kind> &kinds) {
  using backend::ExprIR;
  if (!expr)
    return;
  kinds.push_back(expr->kind);

  switch (expr->kind) {
  case ExprIR::Kind::Number:
  case ExprIR::Kind::Var:
    return;
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const backend::BinaryIR *>(expr);
    collectExprKinds(bin->lhs.get(), kinds);
    collectExprKinds(bin->rhs.get(), kinds);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<const backend::CallIR *>(expr);
    for (const auto &arg : call->args)
      collectExprKinds(arg.get(), kinds);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<const backend::TensorProductIR *>(expr);
    collectExprKinds(prod->lhs.get(), kinds);
    collectExprKinds(prod->rhs.get(), kinds);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<const backend::ContractionIR *>(expr);
    collectExprKinds(ctr->in.get(), kinds);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<const backend::IndexRenameIR *>(expr);
    collectExprKinds(rename->in.get(), kinds);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<const backend::IndexPermuteIR *>(expr);
    collectExprKinds(perm->in.get(), kinds);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<const backend::TraceIR *>(expr);
    collectExprKinds(trace->in.get(), kinds);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<const backend::PartialDerivativeIR *>(expr);
    collectExprKinds(diff->in.get(), kinds);
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<const backend::GradientIR *>(expr);
    collectExprKinds(grad->in.get(), kinds);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<const backend::CovariantDerivativeIR *>(expr);
    collectExprKinds(diff->in.get(), kinds);
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<const backend::DivergenceIR *>(expr);
    collectExprKinds(div->in.get(), kinds);
    return;
  }
  }
}

static void collectExprStats(const backend::ExprIR *expr, IRStats &stats) {
  using backend::ExprIR;
  if (!expr)
    return;

  switch (expr->kind) {
  case ExprIR::Kind::Number:
  case ExprIR::Kind::Var:
    return;
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const backend::BinaryIR *>(expr);
    collectExprStats(bin->lhs.get(), stats);
    collectExprStats(bin->rhs.get(), stats);
    return;
  }
  case ExprIR::Kind::Call: {
    auto *call = static_cast<const backend::CallIR *>(expr);
    for (const auto &arg : call->args)
      collectExprStats(arg.get(), stats);
    return;
  }
  case ExprIR::Kind::TensorProduct: {
    auto *prod = static_cast<const backend::TensorProductIR *>(expr);
    collectExprStats(prod->lhs.get(), stats);
    collectExprStats(prod->rhs.get(), stats);
    return;
  }
  case ExprIR::Kind::Contraction: {
    auto *ctr = static_cast<const backend::ContractionIR *>(expr);
    stats.contractions += 1;
    collectExprStats(ctr->in.get(), stats);
    return;
  }
  case ExprIR::Kind::IndexRename: {
    auto *rename = static_cast<const backend::IndexRenameIR *>(expr);
    stats.renames += 1;
    collectExprStats(rename->in.get(), stats);
    return;
  }
  case ExprIR::Kind::IndexPermute: {
    auto *perm = static_cast<const backend::IndexPermuteIR *>(expr);
    collectExprStats(perm->in.get(), stats);
    return;
  }
  case ExprIR::Kind::Trace: {
    auto *trace = static_cast<const backend::TraceIR *>(expr);
    collectExprStats(trace->in.get(), stats);
    return;
  }
  case ExprIR::Kind::PartialDerivative: {
    auto *diff = static_cast<const backend::PartialDerivativeIR *>(expr);
    stats.partials += 1;
    collectExprStats(diff->in.get(), stats);
    return;
  }
  case ExprIR::Kind::Gradient: {
    auto *grad = static_cast<const backend::GradientIR *>(expr);
    stats.gradients += 1;
    collectExprStats(grad->in.get(), stats);
    return;
  }
  case ExprIR::Kind::CovariantDerivative: {
    auto *diff = static_cast<const backend::CovariantDerivativeIR *>(expr);
    stats.covariant += 1;
    collectExprStats(diff->in.get(), stats);
    return;
  }
  case ExprIR::Kind::Divergence: {
    auto *div = static_cast<const backend::DivergenceIR *>(expr);
    stats.divergences += 1;
    collectExprStats(div->in.get(), stats);
    return;
  }
  }
}

static bool testConTensor3Lowering() {
  static const char *kSource = R"(
    field con_tensor3 A[i,j,k]

    simulation {
      dimension = 1
      resolution = [8]
      time { dt = 0.1 integrator = euler }
      spatial { scheme = fd derivative = centered order = 2 }
    }

    evolution E {
      dt A[i,j,k] = A[i,j,k]
    }
  )";

  backend::ModuleIR mod =
      buildModuleFromSource(kSource, CompilationMode::Executable);

  for (const auto &field : mod.fields) {
    if (field.name != "A")
      continue;

    if (field.kind != backend::FieldKind::ConTensor3) {
      std::cerr << "FAIL: expected backend kind ConTensor3 for field A\n";
      return false;
    }
    if (field.tensorType.up != 3 || field.tensorType.down != 0) {
      std::cerr << "FAIL: expected field A variance up=3 down=0\n";
      return false;
    }
    return true;
  }

  std::cerr << "FAIL: field A not found in lowered backend module\n";
  return false;
}

static bool testIndexSetPolicy() {
  for (char idx : core::kTensorIndices) {
    if (!core::isTensorIndexChar(idx)) {
      std::cerr << "FAIL: expected accepted tensor index char '" << idx
                << "'\n";
      return false;
    }
    if (!core::isTensorIndexName(std::string(1, idx))) {
      std::cerr << "FAIL: expected accepted tensor index name '" << idx
                << "'\n";
      return false;
    }
  }

  const char rejectedChars[] = {'a', 'x', 'z', '0', 'I', '_'};
  for (char idx : rejectedChars) {
    if (core::isTensorIndexChar(idx)) {
      std::cerr << "FAIL: expected rejected tensor index char '" << idx
                << "'\n";
      return false;
    }
  }

  const std::string rejectedNames[] = {"", "ij", "p", "i0", "_", "theta"};
  for (const auto &name : rejectedNames) {
    if (core::isTensorIndexName(name)) {
      std::cerr << "FAIL: expected rejected tensor index name '" << name
                << "'\n";
      return false;
    }
  }

  return true;
}

static bool testIRTensorTypeMappingForExternCall() {
  static const char *kSource = R"(
    extern cov_tensor3 foo_cov(cov_tensor3)
    extern con_tensor3 foo_con(con_tensor3)
    field cov_tensor3 C[i,j,k]
    field con_tensor3 U[i,j,k]

    evolution E {
      dt C[i,j,k] = foo_cov(C[i,j,k])
      dt U[i,j,k] = foo_con(U[i,j,k])
    }
  )";

  backend::ModuleIR mod =
      buildModuleFromSource(kSource, CompilationMode::Symbolic);

  if (mod.evolutions.empty() || mod.evolutions[0].equations.size() != 2) {
    std::cerr << "FAIL: expected two equations in IR extern mapping test\n";
    return false;
  }

  auto checkCall = [](const backend::EquationIR &eq, int retUp, int retDown,
                      int argUp, int argDown) {
    auto *call = dynamic_cast<const backend::CallIR *>(eq.rhs.get());
    if (!call)
      return false;
    if (call->returnType.up != retUp || call->returnType.down != retDown)
      return false;
    if (call->paramTypes.size() != 1)
      return false;
    if (call->paramTypes[0].up != argUp || call->paramTypes[0].down != argDown)
      return false;
    return true;
  };

  const auto &covEq = mod.evolutions[0].equations[0];
  const auto &conEq = mod.evolutions[0].equations[1];
  if (!checkCall(covEq, 0, 3, 0, 3)) {
    std::cerr << "FAIL: covariant extern tensor signature mapping mismatch\n";
    return false;
  }
  if (!checkCall(conEq, 3, 0, 3, 0)) {
    std::cerr << "FAIL: contravariant extern tensor signature mapping mismatch\n";
    return false;
  }

  return true;
}

static bool testIRCanonicalGradientFromFixture() {
  backend::ModuleIR mod = buildModuleFromFile(
      "tests/ir/canonical/01_gradient_sugar.tn", CompilationMode::Symbolic);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);

  if (!verifyCanonicalIR(mod, "gradient"))
    return false;

  const auto *rhs = mod.evolutions[0].equations[0].rhs.get();
  if (!rhs || rhs->kind != backend::ExprIR::Kind::PartialDerivative) {
    std::cerr << "FAIL: expected gradient sugar to canonicalize to partial "
                 "derivative\n";
    return false;
  }

  IRStats stats;
  collectExprStats(rhs, stats);
  if (stats.gradients != 0 || stats.partials == 0) {
    std::cerr << "FAIL: expected no gradient nodes and at least one partial "
                 "derivative node\n";
    return false;
  }
  return true;
}

static bool testIRCanonicalDivergenceFromFixture() {
  backend::ModuleIR mod = buildModuleFromFile(
      "tests/ir/canonical/02_divergence_sugar.tn", CompilationMode::Symbolic);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);

  if (!verifyCanonicalIR(mod, "divergence"))
    return false;

  const auto *rhs = mod.evolutions[0].equations[0].rhs.get();
  if (!rhs || rhs->kind != backend::ExprIR::Kind::Contraction) {
    std::cerr << "FAIL: expected divergence sugar to canonicalize to "
                 "contraction(covariant_derivative(.))\n";
    return false;
  }

  auto *ctr = static_cast<const backend::ContractionIR *>(rhs);
  if (!ctr->in || ctr->in->kind != backend::ExprIR::Kind::CovariantDerivative) {
    std::cerr << "FAIL: divergence canonical form must contain covariant "
                 "derivative\n";
    return false;
  }
  if (ctr->summedIndices.empty()) {
    std::cerr << "FAIL: divergence canonical contraction must carry summed "
                 "index\n";
    return false;
  }

  IRStats stats;
  collectExprStats(rhs, stats);
  if (stats.divergences != 0 || stats.contractions == 0 || stats.covariant == 0) {
    std::cerr << "FAIL: expected divergence eliminated into contraction + "
                 "covariant derivative\n";
    return false;
  }
  return true;
}

static bool testIRCanonicalTraceFromFixture() {
  backend::ModuleIR mod = buildModuleFromFile(
      "tests/ir/canonical/03_trace_from_contract.tn", CompilationMode::Symbolic);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);

  if (!verifyCanonicalIR(mod, "trace"))
    return false;

  const auto *rhs = mod.evolutions[0].equations[0].rhs.get();
  if (!rhs || rhs->kind != backend::ExprIR::Kind::Contraction) {
    std::cerr
        << "FAIL: expected trace/contract forms to canonicalize to contraction\n";
    return false;
  }

  auto *ctr = static_cast<const backend::ContractionIR *>(rhs);
  if (ctr->summedIndices.empty()) {
    std::cerr << "FAIL: canonical contraction must contain summed indices\n";
    return false;
  }
  return true;
}

static bool testIRCanonicalEinsteinRenameInsert() {
  backend::ModuleIR mod;
  backend::EvolutionIR evo;
  evo.name = "E";
  backend::EquationIR eq;
  eq.fieldName = "S";

  auto makeVar = [](const std::string &name,
                    const std::vector<std::string> &indices) {
    auto var = std::make_unique<backend::VarIR>(name, backend::VarKind::Field);
    var->tensorIndexNames = indices;
    return var;
  };

  auto lhs = std::make_unique<backend::TensorProductIR>(makeVar("A", {"j"}),
                                                         makeVar("B", {"j"}));
  auto product = std::make_unique<backend::TensorProductIR>(std::move(lhs),
                                                             makeVar("C", {"j"}));
  auto contraction = std::make_unique<backend::ContractionIR>(std::move(product));
  contraction->summedIndices = {"j"};
  eq.rhs = std::move(contraction);
  evo.equations.push_back(std::move(eq));
  mod.evolutions.push_back(std::move(evo));

  validation::canonicalizeEinsteinIR(mod);

  const auto *rhs = mod.evolutions[0].equations[0].rhs.get();
  if (!rhs || rhs->kind != backend::ExprIR::Kind::Contraction) {
    std::cerr << "FAIL: expected contraction root after canonicalization\n";
    return false;
  }

  auto *ctr = static_cast<const backend::ContractionIR *>(rhs);
  if (ctr->summedIndices.size() != 1 || ctr->summedIndices[0] != "i") {
    std::cerr << "FAIL: expected canonical dummy index alpha-renamed to 'i'\n";
    return false;
  }

  std::vector<backend::ExprIR::Kind> kinds;
  collectExprKinds(rhs, kinds);
  if (llvm::find(kinds, backend::ExprIR::Kind::IndexRename) != kinds.end()) {
    std::cerr
        << "FAIL: canonical Einstein form should eliminate residual index_rename nodes\n";
    return false;
  }
  return true;
}

static bool testIRVerifierRejectsUncanonicalizedGradient() {
  backend::ModuleIR mod;
  backend::EvolutionIR evo;
  evo.name = "E";
  backend::EquationIR eq;
  eq.fieldName = "S";

  auto var = std::make_unique<backend::VarIR>("phi", backend::VarKind::Field);
  eq.rhs = std::make_unique<backend::GradientIR>(std::move(var));
  evo.equations.push_back(std::move(eq));
  mod.evolutions.push_back(std::move(evo));

  auto verify = validation::verifyIR(mod);
  if (verify.ok()) {
    std::cerr << "FAIL: verifier should reject uncanonicalized gradient nodes\n";
    return false;
  }

  bool found = false;
  for (const auto &diag : verify.diags) {
    if (diag.message.find("uncanonicalized gradient") != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    std::cerr << "FAIL: expected uncanonicalized gradient diagnostic\n";
    return false;
  }

  return true;
}

static bool testSchwarzschildCanonicalPatterns() {
  const std::vector<std::string> fixtures = {
      "tests/fixtures/gr/schwarzschild_2d.tn",
      "tests/fixtures/gr/schwarzschild_3d.tn",
  };

  for (const auto &fixture : fixtures) {
    backend::ModuleIR mod =
        buildModuleFromFile(fixture, CompilationMode::Symbolic);
    validation::canonicalizeDifferentialIR(mod);
    validation::canonicalizeEinsteinIR(mod);

    if (!verifyCanonicalIR(mod, fixture))
      return false;

    IRStats stats;
    for (const auto &evo : mod.evolutions) {
      for (const auto &eq : evo.equations)
        collectExprStats(eq.rhs.get(), stats);
      for (const auto &tmp : evo.temporaries)
        collectExprStats(tmp.rhs.get(), stats);
    }

    if (stats.contractions == 0 || stats.partials == 0) {
      std::cerr << "FAIL(" << fixture
                << "): expected canonical IR to contain contraction + partial "
                   "derivative nodes\n";
      return false;
    }
    if (stats.gradients != 0 || stats.divergences != 0) {
      std::cerr << "FAIL(" << fixture
                << "): expected canonical IR to eliminate gradient/divergence "
                   "sugar\n";
      return false;
    }
  }

  return true;
}

static bool testEinsteinCanonicalEquivalence() {
  const std::vector<std::pair<std::string, std::string>> pairs = {
      {"tests/semantic/einstein/canon/01_contract_ij.tn",
       "tests/semantic/einstein/canon/02_contract_mn.tn"},
  };

  tensorium_mlir::MLIRGenOptions opts;
  opts.enableMLIRCanonicalizePass = true;
  opts.enableMLIRCSEPass = true;

  for (const auto &pair : pairs) {
    backend::ModuleIR modA =
        buildModuleFromFile(pair.first, CompilationMode::Symbolic);
    validation::canonicalizeDifferentialIR(modA);
    validation::canonicalizeEinsteinIR(modA);
    if (!verifyCanonicalIR(modA, pair.first))
      return false;

    std::string keyBefore =
        exprCanonicalKey(modA.evolutions[0].equations[0].rhs.get());
    validation::canonicalizeEinsteinIR(modA);
    std::string keyAfter =
        exprCanonicalKey(modA.evolutions[0].equations[0].rhs.get());
    if (keyBefore != keyAfter) {
      std::cerr << "FAIL(" << pair.first
                << "): Einstein canonicalization is not idempotent\n";
      return false;
    }

    ::mlir::MLIRContext ctxA;
    auto mlirA = buildMLIRModuleFromFileWithOpts(pair.first,
                                                 CompilationMode::Symbolic,
                                                 ctxA, opts);
    ::mlir::MLIRContext ctxB;
    auto mlirB = buildMLIRModuleFromFileWithOpts(pair.second,
                                                 CompilationMode::Symbolic,
                                                 ctxB, opts);

    auto sigA = collectRhsTensoriumSignature(*mlirA);
    auto sigB = collectRhsTensoriumSignature(*mlirB);
    if (sigA.empty() || sigB.empty()) {
      std::cerr << "FAIL(" << pair.first << "," << pair.second
                << "): missing tensorium_rhs signature for canonical comparison\n";
      return false;
    }
    if (sigA != sigB) {
      std::cerr << "FAIL(" << pair.first << "," << pair.second
                << "): equivalent Einstein forms produced different normalized MLIR\n";
      return false;
    }
  }

  return true;
}

struct InitRhsLayout {
  ::mlir::func::FuncOp initFunc;
  ::mlir::func::FuncOp rhsFunc;
  ::mlir::func::FuncOp entryFunc;
  ::mlir::func::CallOp initCall;
  ::mlir::func::CallOp rhsCall;
};

static bool verifyInitRhsLayout(::mlir::ModuleOp module, InitRhsLayout &layout,
                                std::string &error) {
  layout.initFunc = module.lookupSymbol<::mlir::func::FuncOp>("tensorium_init");
  layout.rhsFunc = module.lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  layout.entryFunc = module.lookupSymbol<::mlir::func::FuncOp>("tensorium_entry");
  if (!layout.initFunc || !layout.rhsFunc || !layout.entryFunc) {
    error = "missing tensorium_init/tensorium_rhs/tensorium_entry";
    return false;
  }

  llvm::SmallVector<::mlir::func::CallOp, 2> calls;
  for (::mlir::Operation &op : layout.entryFunc.getBody().front()) {
    if (auto call = llvm::dyn_cast<::mlir::func::CallOp>(&op)) {
      calls.push_back(call);
      continue;
    }
    if (!llvm::isa<::mlir::func::ReturnOp>(&op)) {
      error = "tensorium_entry must only contain func.call + return";
      return false;
    }
  }
  if (calls.size() != 2) {
    error = "tensorium_entry must contain exactly 2 calls";
    return false;
  }
  if (calls[0].getCallee() != "tensorium_init" ||
      calls[1].getCallee() != "tensorium_rhs") {
    error = "tensorium_entry call order must be init then rhs";
    return false;
  }
  layout.initCall = calls[0];
  layout.rhsCall = calls[1];

  bool initHasMetric = false;
  bool initHasDecompose = false;
  bool initHasAssign = false;
  bool initHasDtAssign = false;
  for (::mlir::Operation &op : layout.initFunc.getBody().front()) {
    initHasMetric |= llvm::isa<tensorium::mlir::Metric4Op>(&op);
    initHasDecompose |=
        llvm::isa<tensorium::mlir::Decompose3P1FromMetricOp>(&op);
    initHasAssign |= llvm::isa<tensorium::mlir::AssignOp>(&op);
    initHasDtAssign |= llvm::isa<tensorium::mlir::DtAssignOp>(&op);
  }
  if (!initHasMetric || !initHasDecompose || !initHasAssign || initHasDtAssign) {
    error = "tensorium_init placement invariant failed";
    return false;
  }

  bool rhsHasDtAssign = false;
  bool rhsHasForbiddenOps = false;
  for (::mlir::Operation &op : layout.rhsFunc.getBody().front()) {
    rhsHasDtAssign |= llvm::isa<tensorium::mlir::DtAssignOp>(&op);
    rhsHasForbiddenOps |= llvm::isa<tensorium::mlir::Metric4Op>(&op);
    rhsHasForbiddenOps |=
        llvm::isa<tensorium::mlir::Decompose3P1FromMetricOp>(&op);
    rhsHasForbiddenOps |= llvm::isa<tensorium::mlir::Init3P1Op>(&op);
    rhsHasForbiddenOps |= llvm::isa<tensorium::mlir::AssignOp>(&op);
  }
  if (!rhsHasDtAssign || rhsHasForbiddenOps) {
    error = "tensorium_rhs placement invariant failed";
    return false;
  }

  return true;
}

static bool testSchwarzschildMLIRVerification() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/schwarzschild_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitRhsLayout layout;
  std::string layoutError;
  if (!verifyInitRhsLayout(*module, layout, layoutError)) {
    std::cerr << "FAIL: " << layoutError << "\n";
    return false;
  }

  if (layout.initFunc.getNumArguments() != 3) {
    std::cerr << "FAIL: expected tensorium_init signature to have 3 arguments for Schwarzschild fixture\n";
    return false;
  }
  if (layout.rhsFunc.getNumArguments() != 6) {
    std::cerr << "FAIL: expected tensorium_rhs signature to have 6 arguments for Schwarzschild fixture\n";
    return false;
  }
  if (layout.initCall.getNumOperands() != layout.initFunc.getNumArguments()) {
    std::cerr << "FAIL: entry->init call arity mismatch\n";
    return false;
  }
  if (layout.rhsCall.getNumOperands() != layout.rhsFunc.getNumArguments()) {
    std::cerr << "FAIL: entry->rhs call arity mismatch\n";
    return false;
  }

  auto collectEntryArgOrder = [](::mlir::func::CallOp call,
                                 std::vector<unsigned> &out,
                                 const char *label) {
    out.clear();
    out.reserve(call.getNumOperands());
    for (::mlir::Value operand : call.getArgOperands()) {
      auto arg = llvm::dyn_cast<::mlir::BlockArgument>(operand);
      if (!arg) {
        std::cerr << "FAIL: " << label
                  << " call operand is not an entry block argument\n";
        return false;
      }
      out.push_back(arg.getArgNumber());
    }
    return true;
  };

  std::vector<unsigned> initArgOrder;
  std::vector<unsigned> rhsArgOrder;
  if (!collectEntryArgOrder(layout.initCall, initArgOrder, "tensorium_init") ||
      !collectEntryArgOrder(layout.rhsCall, rhsArgOrder, "tensorium_rhs")) {
    return false;
  }

  const std::vector<unsigned> expectedInitArgs = {2, 5, 6};
  const std::vector<unsigned> expectedRhsArgs = {2, 3, 4, 5, 6, 7};
  if (initArgOrder != expectedInitArgs) {
    std::cerr << "FAIL: unexpected tensorium_init entry forwarding order\n";
    return false;
  }
  if (rhsArgOrder != expectedRhsArgs) {
    std::cerr << "FAIL: unexpected tensorium_rhs entry forwarding order\n";
    return false;
  }

  tensorium::mlir::Metric4Op metricOp;
  tensorium::mlir::Decompose3P1FromMetricOp decomposeOp;
  tensorium::mlir::Init3P1Op init3p1Op;
  bool hasLegacySplitOp = false;
  bool hasParamM = false;
  bool hasCoordR = false;
  bool hasCoordTheta = false;
  bool hasSin = false;

  int initAssignCount = 0;
  std::vector<::mlir::Value> initAssignedEntryValues;
  auto pushUniqueValue = [](std::vector<::mlir::Value> &vals, ::mlir::Value v) {
    if (llvm::find(vals, v) == vals.end())
      vals.push_back(v);
  };

  for (::mlir::Operation &op : layout.initFunc.getBody().front()) {
    if (auto metric = llvm::dyn_cast<tensorium::mlir::Metric4Op>(&op))
      metricOp = metric;
    if (auto decomp = llvm::dyn_cast<tensorium::mlir::Decompose3P1FromMetricOp>(&op))
      decomposeOp = decomp;
    if (auto init = llvm::dyn_cast<tensorium::mlir::Init3P1Op>(&op))
      init3p1Op = init;
    if (auto assign = llvm::dyn_cast<tensorium::mlir::AssignOp>(&op)) {
      ++initAssignCount;
      if (assign.getRhs() == init3p1Op.getAlpha() ||
          assign.getRhs() == init3p1Op.getGamma() ||
          assign.getRhs() == init3p1Op.getGammaU()) {
        auto arg = llvm::dyn_cast<::mlir::BlockArgument>(assign.getField());
        if (!arg) {
          std::cerr << "FAIL: init assign target must be a block argument\n";
          return false;
        }
        if (arg.getArgNumber() >= layout.initCall.getNumOperands()) {
          std::cerr << "FAIL: init assign target arg out of init call bounds\n";
          return false;
        }
        pushUniqueValue(initAssignedEntryValues,
                        layout.initCall.getOperand(arg.getArgNumber()));
      }
    }
    if (auto param = llvm::dyn_cast<tensorium::mlir::ParamOp>(&op)) {
      if (param.getName() == "M")
        hasParamM = true;
    }
    if (auto coord = llvm::dyn_cast<tensorium::mlir::CoordOp>(&op)) {
      if (coord.getName() == "r")
        hasCoordR = true;
      if (coord.getName() == "theta")
        hasCoordTheta = true;
    }
    if (llvm::isa<tensorium::mlir::SinOp>(&op))
      hasSin = true;
    if (op.getName().getStringRef() == "tensorium.split3p1")
      hasLegacySplitOp = true;
    if (op.hasAttr("alpha_expr") || op.hasAttr("gamma_diag") ||
        op.hasAttr("components")) {
      std::cerr << "FAIL: forbidden legacy string attr found on op '"
                << op.getName().getStringRef().str() << "'\n";
      return false;
    }
  }

  bool rhsDtTargetsValid = true;
  int rhsDtAssignCount = 0;
  int rhsScalarDtAssignCount = 0;
  int rhsTensorDtAssignCount = 0;
  std::vector<::mlir::Value> rhsReadEntryValues;

  for (::mlir::Operation &op : layout.rhsFunc.getBody().front()) {
    if (auto dt = llvm::dyn_cast<tensorium::mlir::DtAssignOp>(&op)) {
      ++rhsDtAssignCount;
      if (!llvm::isa<::mlir::BlockArgument>(dt.getField()))
        rhsDtTargetsValid = false;
      auto idx = dt.getIndices();
      if (idx.size() == 0) {
        ++rhsScalarDtAssignCount;
      } else if (idx.size() == 2) {
        auto i0 = llvm::dyn_cast<::mlir::StringAttr>(idx[0]);
        auto i1 = llvm::dyn_cast<::mlir::StringAttr>(idx[1]);
        if (!i0 || !i1 || i0.getValue() != "i" || i1.getValue() != "j")
          rhsDtTargetsValid = false;
        ++rhsTensorDtAssignCount;
      } else {
        rhsDtTargetsValid = false;
      }
    }
    if (auto ref = llvm::dyn_cast<tensorium::mlir::RefOp>(&op)) {
      if (auto arg = llvm::dyn_cast<::mlir::BlockArgument>(ref.getSource())) {
        if (arg.getArgNumber() < layout.rhsCall.getNumOperands()) {
          pushUniqueValue(rhsReadEntryValues,
                          layout.rhsCall.getOperand(arg.getArgNumber()));
        }
      }
    }
  }

  if (!metricOp || !decomposeOp || !init3p1Op) {
    std::cerr << "FAIL: expected metric4 + decompose3p1_from_metric + init3p1 in tensorium_init\n";
    return false;
  }
  if (initAssignCount != 3 || initAssignedEntryValues.size() != 3) {
    std::cerr << "FAIL: tensorium_init must bind alpha/gamma/gammaU via tensorium.assign\n";
    return false;
  }
  for (::mlir::Value v : initAssignedEntryValues) {
    if (llvm::find(rhsReadEntryValues, v) == rhsReadEntryValues.end()) {
      std::cerr << "FAIL: RHS does not read one of the fields assigned in init\n";
      return false;
    }
  }
  if (!rhsDtTargetsValid || rhsDtAssignCount != 2 || rhsScalarDtAssignCount != 1 ||
      rhsTensorDtAssignCount != 1) {
    std::cerr << "FAIL: tensorium_rhs dt_assign must target only H and K\n";
    return false;
  }
  if (hasLegacySplitOp) {
    std::cerr << "FAIL: legacy tensorium.split3p1 op should not be emitted\n";
    return false;
  }
  if (decomposeOp->getNumOperands() != 1) {
    std::cerr << "FAIL: decompose3p1_from_metric must take exactly one metric operand\n";
    return false;
  }
  if (decomposeOp.getMetric4() != metricOp.getMetric()) {
    std::cerr << "FAIL: decompose3p1_from_metric must consume metric4 result\n";
    return false;
  }
  if (init3p1Op->getNumOperands() != 4) {
    std::cerr << "FAIL: init3p1 must take exactly 4 operands\n";
    return false;
  }
  if (init3p1Op.getAlphaIn() != decomposeOp.getAlpha() ||
      init3p1Op.getBetaIn() != decomposeOp.getBeta() ||
      init3p1Op.getGammaIn() != decomposeOp.getGamma() ||
      init3p1Op.getGammaUIn() != decomposeOp.getGammaU()) {
    std::cerr << "FAIL: init3p1 inputs must come from decompose3p1_from_metric outputs\n";
    return false;
  }
  if (!hasParamM || !hasCoordR || !hasCoordTheta || !hasSin) {
    std::cerr << "FAIL: expected param/coord/sin ops for Schwarzschild metric\n";
    return false;
  }

  auto metricComps = metricOp.getComponents();
  if (metricComps.size() != 16) {
    std::cerr << "FAIL: metric4 must carry 16 SSA components\n";
    return false;
  }
  ::mlir::Value g00 = metricComps[0];
  ::mlir::Value g33 = metricComps[15];

  auto g00Neg = g00.getDefiningOp<tensorium::mlir::SubOp>();
  if (!g00Neg || !isConstValue(g00Neg.getLhs(), 0.0)) {
    std::cerr << "FAIL: expected g_tt = -(...) lowering form\n";
    return false;
  }

  auto fSub = g00Neg.getRhs().getDefiningOp<tensorium::mlir::SubOp>();
  if (!fSub || !isConstValue(fSub.getLhs(), 1.0)) {
    std::cerr << "FAIL: expected f = 1 - 2*M/r in metric lowering\n";
    return false;
  }

  auto twoMrDiv = fSub.getRhs().getDefiningOp<tensorium::mlir::DivOp>();
  if (!twoMrDiv || !isCoordNamedValue(twoMrDiv.getRhs(), "r")) {
    std::cerr << "FAIL: expected 2*M/r denominator to be coordinate r\n";
    return false;
  }

  auto twoMMul = twoMrDiv.getLhs().getDefiningOp<tensorium::mlir::MulOp>();
  if (!twoMMul) {
    std::cerr << "FAIL: expected 2*M multiplication in Schwarzschild factor\n";
    return false;
  }
  const bool hasConst2ParamM =
      (isConstValue(twoMMul.getLhs(), 2.0) &&
       isParamNamedValue(twoMMul.getRhs(), "M")) ||
      (isConstValue(twoMMul.getRhs(), 2.0) &&
       isParamNamedValue(twoMMul.getLhs(), "M"));
  if (!hasConst2ParamM) {
    std::cerr << "FAIL: expected factor 2*M in Schwarzschild factor\n";
    return false;
  }
  int twoMrCount = 0;
  for (::mlir::Operation &op : layout.initFunc.getBody().front()) {
    auto div = llvm::dyn_cast<tensorium::mlir::DivOp>(&op);
    if (!div || !isCoordNamedValue(div.getRhs(), "r"))
      continue;
    auto mul = div.getLhs().getDefiningOp<tensorium::mlir::MulOp>();
    if (!mul)
      continue;
    const bool match =
        (isConstValue(mul.getLhs(), 2.0) && isParamNamedValue(mul.getRhs(), "M")) ||
        (isConstValue(mul.getRhs(), 2.0) && isParamNamedValue(mul.getLhs(), "M"));
    if (match)
      ++twoMrCount;
  }
  if (twoMrCount != 1) {
    std::cerr << "FAIL: expected CSE to keep a single 2*M/r computation, got "
              << twoMrCount << "\n";
    return false;
  }

  if (!valueDependsOn(g33, [](::mlir::Operation *op) {
        return llvm::isa<tensorium::mlir::SinOp>(op);
      }) ||
      !valueDependsOn(g33, [](::mlir::Operation *op) {
        auto coord = llvm::dyn_cast<tensorium::mlir::CoordOp>(op);
        return coord && coord.getName() == "theta";
      })) {
    std::cerr << "FAIL: g_phph must depend on sin(theta)\n";
    return false;
  }

  auto valueFeedsContract = [](::mlir::Value v) {
    for (::mlir::Operation *user : v.getUsers()) {
      auto mul = llvm::dyn_cast<tensorium::mlir::MulOp>(user);
      if (!mul)
        continue;
      for (::mlir::Operation *mulUser : mul.getRes().getUsers()) {
        if (llvm::isa<tensorium::mlir::ContractOp>(mulUser))
          return true;
      }
    }
    return false;
  };

  bool gammaUFromInitAssignedFeedsContract = false;
  bool gammaUContractUsesNonInitSource = false;
  bool rhsBuildsLocalGammaU = false;
  bool sawGammaURef = false;
  ::mlir::Value alphaRef;
  ::mlir::Value gammaRef;
  auto isScalarField = [](::mlir::Value v) {
    auto ty = llvm::dyn_cast<tensorium::mlir::FieldType>(v.getType());
    return ty && ty.getRank() == 0;
  };
  for (::mlir::Operation &op : layout.rhsFunc.getBody().front()) {
    if (llvm::isa<tensorium::mlir::BuildConTensor2Op>(&op))
      rhsBuildsLocalGammaU = true;

    auto ref = llvm::dyn_cast<tensorium::mlir::RefOp>(&op);
    if (!ref)
      continue;
    auto srcTy = llvm::dyn_cast<tensorium::mlir::FieldType>(ref.getSource().getType());
    if (!srcTy)
      continue;
    auto idx = ref.getIndices();
    if (srcTy.getUp() == 2 && srcTy.getDown() == 0 && idx && idx->size() == 2) {
      sawGammaURef = true;
      if (valueFeedsContract(ref.getResult())) {
        bool fromInitAssignedField = false;
        if (auto arg = llvm::dyn_cast<::mlir::BlockArgument>(ref.getSource())) {
          if (arg.getArgNumber() < layout.rhsCall.getNumOperands()) {
            ::mlir::Value entryOperand =
                layout.rhsCall.getOperand(arg.getArgNumber());
            fromInitAssignedField =
                llvm::find(initAssignedEntryValues, entryOperand) !=
                initAssignedEntryValues.end();
          }
        }

        if (fromInitAssignedField) {
          gammaUFromInitAssignedFeedsContract = true;
        } else {
          gammaUContractUsesNonInitSource = true;
        }
      }
    }
    if (srcTy.getUp() == 0 && srcTy.getDown() == 2 && idx && idx->size() == 2)
      gammaRef = ref.getResult();
  }
  if (!sawGammaURef || !gammaRef) {
    std::cerr << "FAIL: expected gamma/gammaU refs in tensorium_rhs\n";
    return false;
  }
  if (rhsBuildsLocalGammaU) {
    std::cerr << "FAIL: tensorium_rhs must not construct local gammaU values\n";
    return false;
  }
  if (!gammaUFromInitAssignedFeedsContract) {
    std::cerr << "FAIL: contract must consume gammaU loaded from init-assigned field\n";
    return false;
  }
  if (gammaUContractUsesNonInitSource) {
    std::cerr << "FAIL: contract must not consume gammaU from non-init source\n";
    return false;
  }

  for (::mlir::Operation &op : layout.rhsFunc.getBody().front()) {
    auto mul = llvm::dyn_cast<tensorium::mlir::MulOp>(&op);
    if (!mul)
      continue;
    if (mul.getLhs() == gammaRef && isScalarField(mul.getRhs()))
      alphaRef = mul.getRhs();
    if (mul.getRhs() == gammaRef && isScalarField(mul.getLhs()))
      alphaRef = mul.getLhs();
  }
  if (!alphaRef) {
    std::cerr << "FAIL: expected alpha scalar ref multiplied with gamma in tensorium_rhs\n";
    return false;
  }

  bool alphaGammaMulFound = false;
  for (::mlir::Operation &op : layout.rhsFunc.getBody().front()) {
    auto mul = llvm::dyn_cast<tensorium::mlir::MulOp>(&op);
    if (!mul)
      continue;
    if ((mul.getLhs() == alphaRef && mul.getRhs() == gammaRef) ||
        (mul.getRhs() == alphaRef && mul.getLhs() == gammaRef)) {
      alphaGammaMulFound = true;
      break;
    }
  }
  if (!alphaGammaMulFound) {
    std::cerr << "FAIL: expected dt K to use alpha*gamma field values\n";
    return false;
  }

  return true;
}

struct InitNormCounts {
  int twoMrDiv = 0;
  int sinTheta = 0;
};

static InitNormCounts countInitNormalizationPatterns(::mlir::func::FuncOp initFunc) {
  InitNormCounts counts;
  for (::mlir::Operation &op : initFunc.getBody().front()) {
    if (auto div = llvm::dyn_cast<tensorium::mlir::DivOp>(&op)) {
      if (!isCoordNamedValue(div.getRhs(), "r"))
        continue;
      auto mul = div.getLhs().getDefiningOp<tensorium::mlir::MulOp>();
      if (!mul)
        continue;
      const bool twoMr =
          (isConstValue(mul.getLhs(), 2.0) &&
           isParamNamedValue(mul.getRhs(), "M")) ||
          (isConstValue(mul.getRhs(), 2.0) &&
           isParamNamedValue(mul.getLhs(), "M"));
      if (twoMr)
        ++counts.twoMrDiv;
    }

    if (auto sin = llvm::dyn_cast<tensorium::mlir::SinOp>(&op)) {
      if (isCoordNamedValue(sin.getIn(), "theta"))
        ++counts.sinTheta;
    }
  }
  return counts;
}

static bool testMLIRNormalizationPasses() {
  tensorium_mlir::MLIRGenOptions optsNoNorm = makeExecutablePipelineOpts();
  optsNoNorm.enableMLIRCanonicalizePass = false;
  optsNoNorm.enableMLIRCSEPass = false;
  optsNoNorm.enableMLIRInlinePass = false;

  tensorium_mlir::MLIRGenOptions optsNorm = makeExecutablePipelineOpts();
  optsNorm.enableMLIRCanonicalizePass = true;
  optsNorm.enableMLIRCSEPass = true;
  optsNorm.enableMLIRInlinePass = false;

  ::mlir::MLIRContext ctxNoNorm;
  auto rawModule = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable,
      ctxNoNorm, optsNoNorm);

  ::mlir::MLIRContext ctxNorm;
  auto normModule = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable,
      ctxNorm, optsNorm);

  InitRhsLayout rawLayout;
  std::string rawErr;
  if (!verifyInitRhsLayout(*rawModule, rawLayout, rawErr)) {
    std::cerr << "FAIL: raw MLIR layout invalid before normalization: " << rawErr
              << "\n";
    return false;
  }

  InitRhsLayout normLayout;
  std::string normErr;
  if (!verifyInitRhsLayout(*normModule, normLayout, normErr)) {
    std::cerr << "FAIL: normalized MLIR layout invalid: " << normErr << "\n";
    return false;
  }

  InitNormCounts rawCounts = countInitNormalizationPatterns(rawLayout.initFunc);
  InitNormCounts normCounts = countInitNormalizationPatterns(normLayout.initFunc);

  if (normCounts.twoMrDiv != 1) {
    std::cerr << "FAIL: expected normalized init to keep one 2*M/r, got "
              << normCounts.twoMrDiv << "\n";
    return false;
  }
  if (normCounts.sinTheta != 1) {
    std::cerr << "FAIL: expected normalized init to keep one sin(theta), got "
              << normCounts.sinTheta << "\n";
    return false;
  }

  if (rawCounts.twoMrDiv <= normCounts.twoMrDiv) {
    std::cerr << "FAIL: normalization should reduce duplicated 2*M/r, raw="
              << rawCounts.twoMrDiv << " normalized=" << normCounts.twoMrDiv
              << "\n";
    return false;
  }
  if (rawCounts.sinTheta <= normCounts.sinTheta) {
    std::cerr << "FAIL: normalization should reduce duplicated sin(theta), raw="
              << rawCounts.sinTheta << " normalized=" << normCounts.sinTheta
              << "\n";
    return false;
  }

  return true;
}

static bool testSchwarzschildInitNumericPoint() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/schwarzschild_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitEvalContext evalCtx;
  const double M = 1.0;
  const double r = 10.0;
  const double theta = std::acos(-1.0) * 0.5;
  const double phi = 0.0;
  setupSinglePointInitContext(evalCtx, M, r, theta, phi);

  auto result = tensorium_mlir::evaluateTensoriumInit(*module, evalCtx.desc);
  if (!result.ok) {
    std::cerr << "FAIL: init evaluator failed at reference point: "
              << result.message << "\n";
    return false;
  }

  const double f = 1.0 - 2.0 * M / r;
  const double alphaExpected = std::sqrt(f);
  std::cout << std::setprecision(17)
            << "[numeric] Schwarzschild reference point"
            << " M=" << M << " r=" << r << " theta=" << theta << "\n"
            << "  g_uv        got=" << formatMatrix4x4(evalCtx.buffers.metric4)
            << " expected=[[" << (-f) << ", 0, 0, 0], [0, " << (1.0 / f)
            << ", 0, 0], [0, 0, " << (r * r) << ", 0], [0, 0, 0, "
            << (r * r) << "]]\n"
            << "  alpha       got=" << evalCtx.buffers.alpha[0]
            << " expected=" << alphaExpected << "\n"
            << "  Gamma_ij    got=" << formatMatrix3x3(evalCtx.buffers.gamma)
            << " expected=[[" << (1.0 / f) << ", 0, 0], [0, " << (r * r)
            << ", 0], [0, 0, " << (r * r) << "]]\n"
            << "  GammaU^ij   got=" << formatMatrix3x3(evalCtx.buffers.gammaU)
            << " expected=[[" << f << ", 0, 0], [0, " << (1.0 / (r * r))
            << ", 0], [0, 0, " << (1.0 / (r * r)) << "]]\n";

  if (!almostEqual(evalCtx.buffers.alpha[0], alphaExpected)) {
    std::cerr << "FAIL: alpha mismatch at reference point, got "
              << evalCtx.buffers.alpha[0] << " expected " << alphaExpected
              << "\n";
    return false;
  }

  if (!almostEqual(evalCtx.buffers.gamma[0][0], 1.0 / f) ||
      !almostEqual(evalCtx.buffers.gamma[4][0], r * r) ||
      !almostEqual(evalCtx.buffers.gamma[8][0], r * r)) {
    std::cerr << "FAIL: gamma diagonal mismatch at reference point\n";
    return false;
  }

  if (!almostEqual(evalCtx.buffers.gammaU[0][0], f) ||
      !almostEqual(evalCtx.buffers.gammaU[4][0], 1.0 / (r * r)) ||
      !almostEqual(evalCtx.buffers.gammaU[8][0], 1.0 / (r * r))) {
    std::cerr << "FAIL: gammaU diagonal mismatch at reference point\n";
    return false;
  }

  return true;
}

static bool testSchwarzschildInitMetricLoweringPass() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableStencilLoweringPass = false;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable, ctx,
      opts);

  auto initFunc = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_init");
  if (!initFunc) {
    std::cerr << "FAIL: missing tensorium_init after metric lowering\n";
    return false;
  }

  int metricOps = 0;
  int decomposeOps = 0;
  int init3p1Ops = 0;
  int buildGammaOps = 0;
  int buildGammaUOps = 0;
  for (::mlir::Operation &op : initFunc.getBody().front()) {
    metricOps += llvm::isa<tensorium::mlir::Metric4Op>(&op) ? 1 : 0;
    decomposeOps +=
        llvm::isa<tensorium::mlir::Decompose3P1FromMetricOp>(&op) ? 1 : 0;
    init3p1Ops += llvm::isa<tensorium::mlir::Init3P1Op>(&op) ? 1 : 0;
    buildGammaOps += llvm::isa<tensorium::mlir::BuildCovTensor2Op>(&op) ? 1 : 0;
    buildGammaUOps += llvm::isa<tensorium::mlir::BuildConTensor2Op>(&op) ? 1 : 0;
  }

  if (metricOps != 0 || decomposeOps != 0 || init3p1Ops != 0) {
    std::cerr << "FAIL: metric lowering pass must remove metric4/decompose/init3p1 "
                 "from tensorium_init\n";
    return false;
  }
  if (buildGammaOps == 0 || buildGammaUOps == 0) {
    std::cerr << "FAIL: metric lowering pass must materialize gamma/gammaU builders\n";
    return false;
  }

  InitEvalContext evalCtx;
  const double M = 1.0;
  const double r = 10.0;
  const double theta = std::acos(-1.0) * 0.5;
  setupSinglePointInitContext(evalCtx, M, r, theta, 0.0);

  auto result = tensorium_mlir::evaluateTensoriumInit(*module, evalCtx.desc);
  if (!result.ok) {
    std::cerr << "FAIL: init evaluator failed after metric lowering: "
              << result.message << "\n";
    return false;
  }

  const double f = 1.0 - 2.0 * M / r;
  if (!almostEqual(evalCtx.buffers.alpha[0], std::sqrt(f)) ||
      !almostEqual(evalCtx.buffers.gamma[0][0], 1.0 / f) ||
      !almostEqual(evalCtx.buffers.gamma[4][0], r * r) ||
      !almostEqual(evalCtx.buffers.gamma[8][0], r * r) ||
      !almostEqual(evalCtx.buffers.gammaU[0][0], f) ||
      !almostEqual(evalCtx.buffers.gammaU[4][0], 1.0 / (r * r)) ||
      !almostEqual(evalCtx.buffers.gammaU[8][0], 1.0 / (r * r))) {
    std::cerr << "FAIL: metric lowering changed Schwarzschild init numerics\n";
    return false;
  }

  return true;
}

static bool testSchwarzschildInitPointStdLowering() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableStencilLoweringPass = false;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable, ctx,
      opts);

  auto initPoint = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_init_point");
  if (!initPoint) {
    std::cerr << "FAIL: missing tensorium_init_point after init-to-std lowering\n";
    return false;
  }

  if (initPoint.getNumArguments() != 7) {
    std::cerr << "FAIL: tensorium_init_point must have 7 arguments, got "
              << initPoint.getNumArguments() << "\n";
    return false;
  }

  for (unsigned i = 0; i < 4; ++i) {
    if (!initPoint.getArgument(i).getType().isF64()) {
      std::cerr << "FAIL: tensorium_init_point arg " << i
                << " must be f64\n";
      return false;
    }
  }

  auto checkMemRefArg = [&](unsigned argIndex, int64_t expectedSize) {
    auto memTy = llvm::dyn_cast<::mlir::MemRefType>(
        initPoint.getArgument(argIndex).getType());
    if (!memTy || memTy.getRank() != 1 || memTy.getShape()[0] != expectedSize ||
        !memTy.getElementType().isF64()) {
      std::cerr << "FAIL: tensorium_init_point arg " << argIndex
                << " must be memref<" << expectedSize << "xf64>\n";
      return false;
    }
    return true;
  };
  if (!checkMemRefArg(4, 1) || !checkMemRefArg(5, 9) || !checkMemRefArg(6, 9))
    return false;

  bool hasMemrefStore = false;
  for (::mlir::Operation &op : initPoint.getBody().front()) {
    if (op.getName().getDialectNamespace() == "tensorium") {
      std::cerr << "FAIL: tensorium_init_point must not keep tensorium ops, found "
                << op.getName().getStringRef().str() << "\n";
      return false;
    }
    if (op.getName().getStringRef() == "memref.store")
      hasMemrefStore = true;
  }
  if (!hasMemrefStore) {
    std::cerr << "FAIL: tensorium_init_point must contain memref.store writes\n";
    return false;
  }

  return true;
}

static bool testSchwarzschildInitGridScfLowering() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridScfPass = true;
  opts.enableStencilLoweringPass = false;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable, ctx,
      opts);

  auto initGrid =
      module->lookupSymbol<::mlir::func::FuncOp>("tensorium_init_grid_scf");
  if (!initGrid) {
    std::cerr << "FAIL: missing tensorium_init_grid_scf after SCF init lowering\n";
    return false;
  }
  if (initGrid.getNumArguments() != 7) {
    std::cerr << "FAIL: tensorium_init_grid_scf must have 7 arguments, got "
              << initGrid.getNumArguments() << "\n";
    return false;
  }

  if (!initGrid.getArgument(0).getType().isF64()) {
    std::cerr << "FAIL: tensorium_init_grid_scf arg 0 must be f64 (M)\n";
    return false;
  }

  auto checkDynMemRef = [&](unsigned argIndex) {
    auto memTy = llvm::dyn_cast<::mlir::MemRefType>(
        initGrid.getArgument(argIndex).getType());
    if (!memTy || memTy.getRank() != 1 || memTy.getShape()[0] != ::mlir::ShapedType::kDynamic ||
        !memTy.getElementType().isF64()) {
      std::cerr << "FAIL: tensorium_init_grid_scf arg " << argIndex
                << " must be memref<?xf64>\n";
      return false;
    }
    return true;
  };
  for (unsigned arg = 1; arg < 7; ++arg) {
    if (!checkDynMemRef(arg))
      return false;
  }

  bool hasScfFor = false;
  bool callsInitPoint = false;
  bool hasTensoriumOp = false;
  std::string tensoriumOpName;
  initGrid.walk([&](::mlir::Operation *op) {
    if (llvm::isa<::mlir::scf::ForOp>(op))
      hasScfFor = true;
    if (auto call = llvm::dyn_cast<::mlir::func::CallOp>(op)) {
      if (call.getCallee() == "tensorium_init_point")
        callsInitPoint = true;
    }
    if (op != initGrid.getOperation() &&
        op->getName().getDialectNamespace() == "tensorium") {
      hasTensoriumOp = true;
      tensoriumOpName = op->getName().getStringRef().str();
    }
  });
  if (hasTensoriumOp) {
    std::cerr << "FAIL: tensorium_init_grid_scf must not keep tensorium ops, found "
              << tensoriumOpName << "\n";
    return false;
  }

  if (!hasScfFor) {
    std::cerr << "FAIL: tensorium_init_grid_scf must contain scf.for\n";
    return false;
  }
  if (!callsInitPoint) {
    std::cerr << "FAIL: tensorium_init_grid_scf must call tensorium_init_point\n";
    return false;
  }
  return true;
}

static bool testSchwarzschildInitGridAffineLowering() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridAffinePass = true;
  opts.enableStencilLoweringPass = false;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable, ctx,
      opts);

  auto initGrid =
      module->lookupSymbol<::mlir::func::FuncOp>("tensorium_init_grid_affine");
  if (!initGrid) {
    std::cerr << "FAIL: missing tensorium_init_grid_affine after affine init lowering\n";
    return false;
  }
  if (initGrid.getNumArguments() != 7) {
    std::cerr << "FAIL: tensorium_init_grid_affine must have 7 arguments, got "
              << initGrid.getNumArguments() << "\n";
    return false;
  }

  if (!initGrid.getArgument(0).getType().isF64()) {
    std::cerr << "FAIL: tensorium_init_grid_affine arg 0 must be f64 (M)\n";
    return false;
  }

  auto checkDynMemRef = [&](unsigned argIndex) {
    auto memTy = llvm::dyn_cast<::mlir::MemRefType>(
        initGrid.getArgument(argIndex).getType());
    if (!memTy || memTy.getRank() != 1 ||
        memTy.getShape()[0] != ::mlir::ShapedType::kDynamic ||
        !memTy.getElementType().isF64()) {
      std::cerr << "FAIL: tensorium_init_grid_affine arg " << argIndex
                << " must be memref<?xf64>\n";
      return false;
    }
    return true;
  };
  for (unsigned arg = 1; arg < 7; ++arg) {
    if (!checkDynMemRef(arg))
      return false;
  }

  bool hasAffineFor = false;
  bool hasMemrefStore = false;
  bool hasTensoriumOp = false;
  std::string tensoriumOpName;
  initGrid.walk([&](::mlir::Operation *op) {
    if (llvm::isa<::mlir::affine::AffineForOp>(op))
      hasAffineFor = true;
    hasMemrefStore |= (op->getName().getStringRef() == "memref.store");
    if (op != initGrid.getOperation() &&
        op->getName().getDialectNamespace() == "tensorium") {
      hasTensoriumOp = true;
      tensoriumOpName = op->getName().getStringRef().str();
    }
  });
  if (hasTensoriumOp) {
    std::cerr << "FAIL: tensorium_init_grid_affine must not keep tensorium ops, found "
              << tensoriumOpName << "\n";
    return false;
  }
  if (!hasAffineFor) {
    std::cerr << "FAIL: tensorium_init_grid_affine must contain affine.for\n";
    return false;
  }
  if (!hasMemrefStore) {
    std::cerr << "FAIL: tensorium_init_grid_affine must write grid outputs\n";
    return false;
  }
  return true;
}

static bool testSchwarzschildRhsGridScfLowering() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  opts.enableRhsGridScfPass = true;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable, ctx,
      opts);

  auto rhsGrid =
      module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs_grid_scf");
  if (!rhsGrid) {
    std::cerr << "FAIL: missing tensorium_rhs_grid_scf after rhs SCF lowering\n";
    return false;
  }

  auto rhs = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhs) {
    std::cerr << "FAIL: missing source tensorium_rhs for rhs SCF lowering test\n";
    return false;
  }

  const unsigned expectedArgs = 6 + rhs.getNumArguments();
  if (rhsGrid.getNumArguments() != expectedArgs) {
    std::cerr << "FAIL: tensorium_rhs_grid_scf must have " << expectedArgs
              << " args, got " << rhsGrid.getNumArguments() << "\n";
    return false;
  }

  for (unsigned i = 0; i < 3; ++i) {
    if (!rhsGrid.getArgument(i).getType().isIndex()) {
      std::cerr << "FAIL: tensorium_rhs_grid_scf arg " << i
                << " must be index\n";
      return false;
    }
  }
  for (unsigned i = 3; i < 6; ++i) {
    if (!rhsGrid.getArgument(i).getType().isF64()) {
      std::cerr << "FAIL: tensorium_rhs_grid_scf arg " << i
                << " must be f64\n";
      return false;
    }
  }
  for (unsigned i = 6; i < expectedArgs; ++i) {
    auto memTy = llvm::dyn_cast<::mlir::MemRefType>(rhsGrid.getArgument(i).getType());
    if (!memTy || memTy.getRank() != 1 ||
        memTy.getShape()[0] != ::mlir::ShapedType::kDynamic ||
        !memTy.getElementType().isF64()) {
      std::cerr << "FAIL: tensorium_rhs_grid_scf arg " << i
                << " must be memref<?xf64>\n";
      return false;
    }
  }

  bool hasFor = false;
  bool hasStore = false;
  bool hasTensoriumOp = false;
  std::string tensoriumOpName;
  rhsGrid.walk([&](::mlir::Operation *op) {
    hasFor |= llvm::isa<::mlir::scf::ForOp>(op);
    hasStore |= (op->getName().getStringRef() == "memref.store");
    if (op != rhsGrid.getOperation() &&
        op->getName().getDialectNamespace() == "tensorium") {
      hasTensoriumOp = true;
      tensoriumOpName = op->getName().getStringRef().str();
    }
  });

  if (!hasFor) {
    std::cerr << "FAIL: tensorium_rhs_grid_scf must contain scf.for\n";
    return false;
  }
  if (!hasStore) {
    std::cerr << "FAIL: tensorium_rhs_grid_scf must contain memref.store\n";
    return false;
  }
  if (hasTensoriumOp) {
    std::cerr << "FAIL: tensorium_rhs_grid_scf must not contain tensorium ops, found "
              << tensoriumOpName << "\n";
    return false;
  }

  return true;
}

static bool testRhsGridScfRejectsImplicitParams() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  opts.enableRhsGridScfPass = true;

  const std::string source = R"(
field scalar phi

simulation {
  dimension = 1
  resolution = [16]
  time { dt = 0.05 integrator = euler }
  spatial { scheme = fd derivative = centered order = 2 }
}

evolution ParamImplicit {
  dt phi = M * phi
}
)";

  try {
    (void)buildMLIRModuleFromSourceWithOpts(source, CompilationMode::Executable,
                                            ctx, opts);
    std::cerr << "FAIL: implicit parameter 'M' should be rejected in strict "
                 "semantic mode\n";
    return false;
  } catch (const std::exception &ex) {
    const std::string msg = ex.what();
    if (msg.find("Unknown identifier: M") == std::string::npos) {
      std::cerr << "FAIL: expected unknown-identifier error for implicit "
                   "parameter, got: "
                << msg << "\n";
      return false;
    }
  }

  return true;
}

static bool testRhsExplicitParamDeclarationAccepted() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();

  const std::string source = R"(
field scalar phi

params { M }

simulation {
  dimension = 1
  resolution = [16]
  time { dt = 0.05 integrator = euler }
  spatial { scheme = fd derivative = centered order = 2 }
}

evolution ParamDeclared {
  dt phi = M * phi
}
)";

  auto module = buildMLIRModuleFromSourceWithOpts(
      source, CompilationMode::Executable, ctx, opts);
  auto rhs = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhs) {
    std::cerr << "FAIL: missing tensorium_rhs for explicit parameter test\n";
    return false;
  }

  bool hasParamOp = false;
  rhs.walk([&](::mlir::Operation *op) {
    if (llvm::isa<tensorium::mlir::ParamOp>(op))
      hasParamOp = true;
  });

  if (!hasParamOp) {
    std::cerr << "FAIL: expected tensorium.param op for declared parameter\n";
    return false;
  }

  return true;
}

static bool testRhsGridScfLoweringSupportsCoords() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  opts.enableRhsGridScfPass = true;

  const std::string source = R"(
field vector beta[i]

simulation {
  dimension = 3
  resolution = [8, 8, 8]
  time { dt = 0.01 integrator = euler }
  spatial { scheme = fd derivative = centered order = 2 }
}

evolution CoordInRhs {
  dt beta[i] = i * beta[i]
}
)";

  auto module = buildMLIRModuleFromSourceWithOpts(
      source, CompilationMode::Executable, ctx, opts);

  auto rhsGrid =
      module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs_grid_scf");
  if (!rhsGrid) {
    std::cerr << "FAIL: missing tensorium_rhs_grid_scf for coord lowering test\n";
    return false;
  }

  if (rhsGrid.getNumArguments() != 7) {
    std::cerr << "FAIL: expected tensorium_rhs_grid_scf to have 7 args "
                 "(nx,ny,nz,dx,dy,dz,beta), got "
              << rhsGrid.getNumArguments() << "\n";
    return false;
  }

  auto memTy = llvm::dyn_cast<::mlir::MemRefType>(rhsGrid.getArgument(6).getType());
  if (!memTy || memTy.getRank() != 1 ||
      memTy.getShape()[0] != ::mlir::ShapedType::kDynamic ||
      !memTy.getElementType().isF64()) {
    std::cerr << "FAIL: rhs-grid-scf arg 6 must be memref<?xf64>\n";
    return false;
  }

  bool hasTensoriumOp = false;
  bool hasIndexToFloat = false;
  bool spacingArgUsed = false;
  rhsGrid.walk([&](::mlir::Operation *op) {
    if (op != rhsGrid.getOperation() &&
        op->getName().getDialectNamespace() == "tensorium") {
      hasTensoriumOp = true;
    }
    if (op->getName().getStringRef() == "arith.sitofp")
      hasIndexToFloat = true;
    if (op->getName().getStringRef() != "arith.mulf")
      return;
    for (::mlir::Value operand : op->getOperands()) {
      if (operand == rhsGrid.getArgument(3))
        spacingArgUsed = true;
    }
  });

  if (hasTensoriumOp) {
    std::cerr << "FAIL: rhs-grid-scf with coords must not keep tensorium ops\n";
    return false;
  }
  if (!hasIndexToFloat) {
    std::cerr << "FAIL: rhs-grid-scf must cast loop index to float for coord op\n";
    return false;
  }
  if (!spacingArgUsed) {
    std::cerr << "FAIL: rhs-grid-scf must use spacing argument when lowering coord op\n";
    return false;
  }

  return true;
}

static bool testSchwarzschildRhsGridAffineLowering() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  opts.enableRhsGridAffinePass = true;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable, ctx,
      opts);

  auto rhsGrid =
      module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs_grid_affine");
  if (!rhsGrid) {
    std::cerr
        << "FAIL: missing tensorium_rhs_grid_affine after rhs affine lowering\n";
    return false;
  }

  auto rhs = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhs) {
    std::cerr
        << "FAIL: missing source tensorium_rhs for rhs affine lowering test\n";
    return false;
  }

  const unsigned expectedArgs = 6 + rhs.getNumArguments();
  if (rhsGrid.getNumArguments() != expectedArgs) {
    std::cerr << "FAIL: tensorium_rhs_grid_affine must have " << expectedArgs
              << " args, got " << rhsGrid.getNumArguments() << "\n";
    return false;
  }

  for (unsigned i = 0; i < 3; ++i) {
    if (!rhsGrid.getArgument(i).getType().isIndex()) {
      std::cerr << "FAIL: tensorium_rhs_grid_affine arg " << i
                << " must be index\n";
      return false;
    }
  }
  for (unsigned i = 3; i < 6; ++i) {
    if (!rhsGrid.getArgument(i).getType().isF64()) {
      std::cerr << "FAIL: tensorium_rhs_grid_affine arg " << i
                << " must be f64\n";
      return false;
    }
  }
  for (unsigned i = 6; i < expectedArgs; ++i) {
    auto memTy =
        llvm::dyn_cast<::mlir::MemRefType>(rhsGrid.getArgument(i).getType());
    if (!memTy || memTy.getRank() != 1 ||
        memTy.getShape()[0] != ::mlir::ShapedType::kDynamic ||
        !memTy.getElementType().isF64()) {
      std::cerr << "FAIL: tensorium_rhs_grid_affine arg " << i
                << " must be memref<?xf64>\n";
      return false;
    }
  }

  bool hasFor = false;
  bool hasStore = false;
  bool hasTensoriumOp = false;
  std::string tensoriumOpName;
  rhsGrid.walk([&](::mlir::Operation *op) {
    hasFor |= llvm::isa<::mlir::affine::AffineForOp>(op);
    hasStore |= (op->getName().getStringRef() == "memref.store");
    if (op != rhsGrid.getOperation() &&
        op->getName().getDialectNamespace() == "tensorium") {
      hasTensoriumOp = true;
      tensoriumOpName = op->getName().getStringRef().str();
    }
  });

  if (!hasFor) {
    std::cerr << "FAIL: tensorium_rhs_grid_affine must contain affine.for\n";
    return false;
  }
  if (!hasStore) {
    std::cerr << "FAIL: tensorium_rhs_grid_affine must contain memref.store\n";
    return false;
  }
  if (hasTensoriumOp) {
    std::cerr
        << "FAIL: tensorium_rhs_grid_affine must not contain tensorium ops, found "
        << tensoriumOpName << "\n";
    return false;
  }

  return true;
}

static bool testGeneratedKernelABIMetadata() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridAffinePass = true;
  opts.enableRhsGridAffinePass = true;
  opts.enableStencilLoweringPass = false;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable, ctx,
      opts);

  auto modVersion = module->getOperation()->getAttrOfType<::mlir::IntegerAttr>(
      tensorium_mlir::abi::kAttrABIVersion);
  if (!modVersion ||
      modVersion.getInt() != tensorium_mlir::abi::kGeneratedKernelABIVersion) {
    std::cerr << "FAIL: module missing generated ABI version attr\n";
    return false;
  }
  auto modLayout = module->getOperation()->getAttrOfType<::mlir::StringAttr>(
      tensorium_mlir::abi::kAttrMemoryLayout);
  if (!modLayout ||
      modLayout.getValue() != tensorium_mlir::abi::kMemLayoutSoAComponentMajor) {
    std::cerr << "FAIL: module missing generated memory layout attr\n";
    return false;
  }
  auto modMemref = module->getOperation()->getAttrOfType<::mlir::StringAttr>(
      tensorium_mlir::abi::kAttrMemrefABI);
  if (!modMemref ||
      modMemref.getValue() != tensorium_mlir::abi::kMemrefABI1DStridedF64) {
    std::cerr << "FAIL: module missing generated memref ABI attr\n";
    return false;
  }

  auto init = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolInit);
  auto rhs = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolRhs);
  auto entry = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolEntry);
  auto initPoint = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolInitPoint);
  auto initGrid = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolInitGridAffine);
  auto rhsGrid = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolRhsGridAffine);

  std::string abiErr;
  if (!verifyCommonGeneratedABIAttrs(init, tensorium_mlir::abi::kKindInitSource,
                                     abiErr) ||
      !verifyCommonGeneratedABIAttrs(rhs, tensorium_mlir::abi::kKindRhsSource,
                                     abiErr) ||
      !verifyCommonGeneratedABIAttrs(entry,
                                     tensorium_mlir::abi::kKindEntrySource,
                                     abiErr) ||
      !verifyCommonGeneratedABIAttrs(initPoint,
                                     tensorium_mlir::abi::kKindInitPoint,
                                     abiErr) ||
      !verifyCommonGeneratedABIAttrs(initGrid,
                                     tensorium_mlir::abi::kKindInitGridAffine,
                                     abiErr) ||
      !verifyCommonGeneratedABIAttrs(rhsGrid,
                                     tensorium_mlir::abi::kKindRhsGridAffine,
                                     abiErr)) {
    std::cerr << "FAIL: " << abiErr << "\n";
    return false;
  }

  auto expectEqVec = [](const std::vector<std::string> &got,
                        const std::vector<std::string> &expected) {
    return got == expected;
  };

  auto initPointParams = parseStringArrayAttr(
      initPoint->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrParamNames));
  auto initPointCoords = parseStringArrayAttr(
      initPoint->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrCoordNames));
  auto initPointOutputs = parseStringArrayAttr(
      initPoint->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrOutputNames));
  auto initPointWrites = parseI64ArrayAttr(
      initPoint->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrWriteArgIndices));
  if (!expectEqVec(initPointParams, {"M"}) ||
      !expectEqVec(initPointCoords, {"r", "theta", "phi"}) ||
      !expectEqVec(initPointOutputs, {"alpha", "gamma", "gammaU"}) ||
      initPointWrites != std::vector<int64_t>({4, 5, 6})) {
    std::cerr << "FAIL: init_point ABI metadata does not match expected ABI v1\n";
    return false;
  }

  auto initGridOutputs = parseStringArrayAttr(
      initGrid->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrOutputNames));
  auto initGridWrites = parseI64ArrayAttr(
      initGrid->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrWriteArgIndices));
  if (!expectEqVec(initGridOutputs, {"alpha", "gamma", "gammaU"}) ||
      initGridWrites != std::vector<int64_t>({4, 5, 6})) {
    std::cerr << "FAIL: init_grid_affine ABI metadata does not match expected ABI v1\n";
    return false;
  }

  auto rhsFieldNames = parseStringArrayAttr(
      rhs->getAttrOfType<::mlir::ArrayAttr>(tensorium_mlir::abi::kAttrFieldNames));
  auto rhsGridFieldNames = parseStringArrayAttr(
      rhsGrid->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrFieldNames));
  auto rhsGridOutputs = parseStringArrayAttr(
      rhsGrid->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrOutputNames));
  auto rhsGridWrites = parseI64ArrayAttr(
      rhsGrid->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrWriteArgIndices));
  if (!expectEqVec(rhsFieldNames,
                   {"alpha", "phi", "H", "gamma", "gammaU", "K"})) {
    std::cerr << "FAIL: unexpected tensorium_rhs field order in ABI metadata\n";
    return false;
  }
  if (rhsGridFieldNames != rhsFieldNames ||
      !expectEqVec(rhsGridOutputs, {"H", "K"}) ||
      rhsGridWrites != std::vector<int64_t>({8, 11})) {
    std::cerr << "FAIL: rhs_grid_affine ABI metadata mismatch\n";
    return false;
  }

  return true;
}

static bool testSpatialOffdiagNoParamABI() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridAffinePass = true;
  opts.enableRhsGridAffinePass = true;
  opts.enableStencilLoweringPass = false;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/spatial_offdiag_3d.tn", CompilationMode::Executable,
      ctx, opts);

  auto initPoint = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolInitPoint);
  auto initGrid = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolInitGridAffine);
  auto rhsGrid = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolRhsGridAffine);
  if (!initPoint || !initGrid || !rhsGrid) {
    std::cerr << "FAIL: missing generated spatial offdiag ABI functions\n";
    return false;
  }

  std::string abiErr;
  if (!verifyCommonGeneratedABIAttrs(initPoint,
                                     tensorium_mlir::abi::kKindInitPoint,
                                     abiErr) ||
      !verifyCommonGeneratedABIAttrs(initGrid,
                                     tensorium_mlir::abi::kKindInitGridAffine,
                                     abiErr) ||
      !verifyCommonGeneratedABIAttrs(rhsGrid,
                                     tensorium_mlir::abi::kKindRhsGridAffine,
                                     abiErr)) {
    std::cerr << "FAIL: " << abiErr << "\n";
    return false;
  }

  auto expectNoParams = [](const ::mlir::func::FuncOp fn,
                           const char *label) {
    auto params = parseStringArrayAttr(fn->getAttrOfType<::mlir::ArrayAttr>(
        tensorium_mlir::abi::kAttrParamNames));
    if (!params.empty()) {
      std::cerr << "FAIL: " << label
                << " must not expose implicit ABI params\n";
      return false;
    }
    return true;
  };
  if (!expectNoParams(initPoint, "tensorium_init_point") ||
      !expectNoParams(initGrid, "tensorium_init_grid_affine") ||
      !expectNoParams(rhsGrid, "tensorium_rhs_grid_affine")) {
    return false;
  }

  auto initInternalParams = parseStringArrayAttr(
      initPoint->getAttrOfType<::mlir::ArrayAttr>(
          "tensorium.init.param_names"));
  if (!initInternalParams.empty()) {
    std::cerr << "FAIL: tensorium.init.param_names must stay empty\n";
    return false;
  }

  if (initPoint.getNumArguments() != 6) {
    std::cerr << "FAIL: spatial init_point must have 6 args, got "
              << initPoint.getNumArguments() << "\n";
    return false;
  }
  for (unsigned arg = 0; arg < 3; ++arg) {
    if (!initPoint.getArgument(arg).getType().isF64()) {
      std::cerr << "FAIL: spatial init_point coord arg " << arg
                << " must be f64\n";
      return false;
    }
  }
  if (!isStaticF64Memref(initPoint.getArgument(3).getType(), 1) ||
      !isStaticF64Memref(initPoint.getArgument(4).getType(), 9) ||
      !isStaticF64Memref(initPoint.getArgument(5).getType(), 9)) {
    std::cerr << "FAIL: spatial init_point output arg layout mismatch\n";
    return false;
  }

  if (initGrid.getNumArguments() != 6) {
    std::cerr << "FAIL: spatial init_grid_affine must have 6 args, got "
              << initGrid.getNumArguments() << "\n";
    return false;
  }
  for (unsigned arg = 0; arg < initGrid.getNumArguments(); ++arg) {
    if (!isDynamicF64Memref(initGrid.getArgument(arg).getType())) {
      std::cerr << "FAIL: spatial init_grid_affine arg " << arg
                << " must be memref<?xf64>\n";
      return false;
    }
  }

  auto initPointWrites = parseI64ArrayAttr(
      initPoint->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrWriteArgIndices));
  auto initGridWrites = parseI64ArrayAttr(
      initGrid->getAttrOfType<::mlir::ArrayAttr>(
          tensorium_mlir::abi::kAttrWriteArgIndices));
  if (initPointWrites != std::vector<int64_t>({3, 4, 5}) ||
      initGridWrites != std::vector<int64_t>({3, 4, 5})) {
    std::cerr << "FAIL: spatial no-param write_arg_indices mismatch\n";
    return false;
  }

  return true;
}

static bool testSpatialOffdiagGeneratedSplit3p1Constants() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableStencilLoweringPass = false;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/spatial_offdiag_3d.tn", CompilationMode::Executable,
      ctx, opts);
  auto initPoint = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolInitPoint);
  if (!initPoint) {
    std::cerr << "FAIL: missing spatial init_point for constants test\n";
    return false;
  }

  std::array<double, 1> alpha{};
  std::array<double, 9> gamma{};
  std::array<double, 9> gammaU{};
  if (!collectInitPointConstantStores(initPoint, 3, alpha, gamma, gammaU)) {
    std::cerr << "FAIL: spatial init_point did not lower to constant stores\n";
    return false;
  }

  const std::array<double, 9> gammaExpected = {
      2.0, 1.0, 0.0,
      1.0, 3.0, 0.0,
      0.0, 0.0, 4.0};
  const std::array<double, 9> gammaUExpected = {
      0.6, -0.2, 0.0,
      -0.2, 0.4, 0.0,
      0.0, 0.0, 0.25};

  if (!almostEqual(alpha[0], 1.0)) {
    std::cerr << "FAIL: generated spatial alpha constant mismatch\n";
    return false;
  }
  for (std::size_t i = 0; i < gammaExpected.size(); ++i) {
    if (!almostEqual(gamma[i], gammaExpected[i]) ||
        !almostEqual(gammaU[i], gammaUExpected[i])) {
      std::cerr << "FAIL: generated spatial 3+1 constant mismatch at component "
                << i << "\n";
      return false;
    }
  }
  if (!almostEqual(gamma[1], gamma[3]) ||
      !almostEqual(gammaU[1], gammaU[3])) {
    std::cerr << "FAIL: enforce_symmetry lost off-diagonal components\n";
    return false;
  }

  return true;
}

static bool testSpatialOffdiagInitGridAffineNoLoopAlloc() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridAffinePass = true;
  opts.enableStencilLoweringPass = false;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/spatial_offdiag_3d.tn", CompilationMode::Executable,
      ctx, opts);
  auto initGrid = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolInitGridAffine);
  if (!initGrid) {
    std::cerr << "FAIL: missing spatial init_grid_affine\n";
    return false;
  }

  bool hasAffineFor = false;
  bool hasAlloc = false;
  bool hasDealloc = false;
  bool callsInitPoint = false;
  initGrid.walk([&](::mlir::Operation *op) {
    hasAffineFor |= llvm::isa<::mlir::affine::AffineForOp>(op);
    hasAlloc |= llvm::isa<::mlir::memref::AllocOp>(op);
    hasDealloc |= llvm::isa<::mlir::memref::DeallocOp>(op);
    if (auto call = llvm::dyn_cast<::mlir::func::CallOp>(op))
      callsInitPoint |= (call.getCallee() == tensorium_mlir::abi::kSymbolInitPoint);
  });

  if (!hasAffineFor) {
    std::cerr << "FAIL: spatial init_grid_affine must contain affine.for\n";
    return false;
  }
  if (hasAlloc || hasDealloc || callsInitPoint) {
    std::cerr << "FAIL: spatial constant init_grid_affine must not allocate "
                 "or call init_point per grid point\n";
    return false;
  }
  return true;
}

static bool testSpatialOffdiagRhsCompactHessianAffineLowering() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  opts.enableRhsGridAffinePass = true;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/spatial_offdiag_3d.tn", CompilationMode::Executable,
      ctx, opts);
  auto rhsGrid = module->lookupSymbol<::mlir::func::FuncOp>(
      tensorium_mlir::abi::kSymbolRhsGridAffine);
  if (!rhsGrid) {
    std::cerr << "FAIL: missing spatial rhs_grid_affine\n";
    return false;
  }

  int affineLoopCount = 0;
  int radiusOneLowerBounds = 0;
  bool hasRadiusTwoLowerBound = false;
  bool hasDxDxDenom = false;
  bool hasDyDyDenom = false;
  bool hasDzDzDenom = false;
  bool hasMixedFourFactor = false;
  bool hasCopy = false;
  bool hasAlloc = false;

  auto isArgSquare = [&](::mlir::arith::MulFOp mul, unsigned arg) {
    return mul.getLhs() == rhsGrid.getArgument(arg) &&
           mul.getRhs() == rhsGrid.getArgument(arg);
  };

  rhsGrid.walk([&](::mlir::Operation *op) {
    if (auto loop = llvm::dyn_cast<::mlir::affine::AffineForOp>(op)) {
      ++affineLoopCount;
      if (loop.hasConstantLowerBound() && loop.getConstantLowerBound() == 1)
        ++radiusOneLowerBounds;
      if (loop.hasConstantLowerBound() && loop.getConstantLowerBound() == 2)
        hasRadiusTwoLowerBound = true;
    }
    hasCopy |= llvm::isa<::mlir::memref::CopyOp>(op);
    hasAlloc |= llvm::isa<::mlir::memref::AllocOp>(op);
    if (auto mul = llvm::dyn_cast<::mlir::arith::MulFOp>(op)) {
      hasDxDxDenom |= isArgSquare(mul, 3);
      hasDyDyDenom |= isArgSquare(mul, 4);
      hasDzDzDenom |= isArgSquare(mul, 5);
      auto lhsConst = getConstantF64ValueForTest(mul.getLhs());
      auto rhsConst = getConstantF64ValueForTest(mul.getRhs());
      hasMixedFourFactor |=
          (lhsConst && almostEqual(*lhsConst, 4.0)) ||
          (rhsConst && almostEqual(*rhsConst, 4.0));
    }
  });

  if (affineLoopCount < 3 || radiusOneLowerBounds < 3 ||
      hasRadiusTwoLowerBound) {
    std::cerr << "FAIL: spatial Hessian affine loops must use radius 1 bounds\n";
    return false;
  }
  if (!hasDxDxDenom || !hasDyDyDenom || !hasDzDzDenom ||
      !hasMixedFourFactor) {
    std::cerr << "FAIL: spatial Hessian lowering did not expose compact "
                 "diagonal and centered mixed denominators\n";
    return false;
  }
  if (hasCopy || hasAlloc) {
    std::cerr << "FAIL: spatial rhs_grid_affine must not copy or allocate "
                 "dead snapshots\n";
    return false;
  }

  return true;
}

static bool testLoweredGridLLVMABISignature() {
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridAffinePass = true;
  opts.enableRhsGridAffinePass = true;
  opts.enableStripSourceFuncsPass = true;
  opts.enableStencilLoweringPass = false;

  backend::ModuleIR mod =
      buildModuleFromFile("tests/fixtures/gr/schwarzschild_3d.tn",
                          CompilationMode::Executable);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);
  auto verify = validation::verifyIR(mod);
  if (!verify.ok()) {
    std::cerr << "FAIL: IR verification failed before ABI LLVM signature test\n";
    return false;
  }

  std::string llvmIR;
  if (!tensorium_mlir::emitLLVMIR(mod, opts, &llvmIR)) {
    std::cerr << "FAIL: emitLLVMIR failed for ABI LLVM signature test\n";
    return false;
  }

  auto checkMemrefGroups = [&](const std::vector<std::string> &types,
                               std::size_t base, std::size_t groups) {
    static const std::array<const char *, 5> kMemRefGroup = {
        "ptr", "ptr", "i64", "i64", "i64"};
    if (types.size() < base + groups * kMemRefGroup.size())
      return false;
    for (std::size_t g = 0; g < groups; ++g) {
      for (std::size_t i = 0; i < kMemRefGroup.size(); ++i) {
        if (types[base + g * kMemRefGroup.size() + i] != kMemRefGroup[i])
          return false;
      }
    }
    return true;
  };

  std::vector<std::string> initPointTypes;
  if (!llvmFunctionArgTypeTokens(
          llvmIR, tensorium_mlir::abi::kSymbolInitPoint, initPointTypes)) {
    std::cerr << "FAIL: missing LLVM signature for tensorium_init_point\n";
    return false;
  }
  if (initPointTypes.size() != 19 || initPointTypes[0] != "double" ||
      initPointTypes[1] != "double" || initPointTypes[2] != "double" ||
      initPointTypes[3] != "double" ||
      !checkMemrefGroups(initPointTypes, 4, 3)) {
    std::cerr << "FAIL: tensorium_init_point LLVM ABI signature mismatch\n";
    return false;
  }

  std::vector<std::string> initGridTypes;
  if (!llvmFunctionArgTypeTokens(
          llvmIR, tensorium_mlir::abi::kSymbolInitGridAffine, initGridTypes)) {
    std::cerr << "FAIL: missing LLVM signature for tensorium_init_grid_affine\n";
    return false;
  }
  if (initGridTypes.size() != 31 || initGridTypes[0] != "double" ||
      !checkMemrefGroups(initGridTypes, 1, 6)) {
    std::cerr << "FAIL: tensorium_init_grid_affine LLVM ABI signature mismatch\n";
    return false;
  }

  std::vector<std::string> rhsGridTypes;
  if (!llvmFunctionArgTypeTokens(
          llvmIR, tensorium_mlir::abi::kSymbolRhsGridAffine, rhsGridTypes)) {
    std::cerr << "FAIL: missing LLVM signature for tensorium_rhs_grid_affine\n";
    return false;
  }
  if (rhsGridTypes.size() != 36 || rhsGridTypes[0] != "i64" ||
      rhsGridTypes[1] != "i64" || rhsGridTypes[2] != "i64" ||
      rhsGridTypes[3] != "double" || rhsGridTypes[4] != "double" ||
      rhsGridTypes[5] != "double" || !checkMemrefGroups(rhsGridTypes, 6, 6)) {
    std::cerr << "FAIL: tensorium_rhs_grid_affine LLVM ABI signature mismatch\n";
    return false;
  }

  return true;
}

static bool testStripSourceFuncsAfterGridLowering() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridAffinePass = true;
  opts.enableRhsGridAffinePass = true;
  opts.enableStripSourceFuncsPass = true;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_3d.tn", CompilationMode::Executable, ctx,
      opts);

  if (module->lookupSymbol<::mlir::func::FuncOp>("tensorium_init") ||
      module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs") ||
      module->lookupSymbol<::mlir::func::FuncOp>("tensorium_entry")) {
    std::cerr << "FAIL: strip-source-funcs must remove tensorium_init/rhs/entry\n";
    return false;
  }

  if (!module->lookupSymbol<::mlir::func::FuncOp>("tensorium_init_grid_affine")) {
    std::cerr << "FAIL: missing tensorium_init_grid_affine after strip-source-funcs\n";
    return false;
  }
  if (!module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs_grid_affine")) {
    std::cerr << "FAIL: missing tensorium_rhs_grid_affine after strip-source-funcs\n";
    return false;
  }

  bool hasTensoriumOp = false;
  std::string tensoriumOpName;
  module->walk([&](::mlir::Operation *op) {
    if (op->getName().getDialectNamespace() == "tensorium") {
      hasTensoriumOp = true;
      tensoriumOpName = op->getName().getStringRef().str();
    }
  });
  if (hasTensoriumOp) {
    std::cerr << "FAIL: strip-source-funcs module must not contain tensorium ops, found "
              << tensoriumOpName << "\n";
    return false;
  }

  return true;
}

static bool testStripSourceFuncsRhsOnly() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableRhsGridAffinePass = true;
  opts.enableStripSourceFuncsPass = true;

  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/covariant_rank1_3d.tn", CompilationMode::Executable,
      ctx, opts);

  if (module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs")) {
    std::cerr << "FAIL: rhs-only strip-source-funcs must remove tensorium_rhs\n";
    return false;
  }
  if (module->lookupSymbol<::mlir::func::FuncOp>("tensorium_entry")) {
    std::cerr
        << "FAIL: rhs-only strip-source-funcs must remove tensorium_entry\n";
    return false;
  }
  if (!module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs_grid_affine")) {
    std::cerr
        << "FAIL: rhs-only strip-source-funcs missing tensorium_rhs_grid_affine\n";
    return false;
  }
  return true;
}

static bool testLoweredGridModuleLLVMIREmission() {
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableMetricLoweringPass = true;
  opts.enableInitStdLoweringPass = true;
  opts.enableInitGridAffinePass = true;
  opts.enableRhsGridAffinePass = true;
  opts.enableStripSourceFuncsPass = true;
  opts.enableStencilLoweringPass = false;

  backend::ModuleIR mod =
      buildModuleFromFile("tests/fixtures/gr/schwarzschild_3d.tn",
                          CompilationMode::Executable);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);
  auto verify = validation::verifyIR(mod);
  if (!verify.ok()) {
    std::cerr << "FAIL: IR verification failed before LLVM emission test\n";
    return false;
  }

  std::string llvmIR;
  if (!tensorium_mlir::emitLLVMIR(mod, opts, &llvmIR)) {
    std::cerr << "FAIL: emitLLVMIR failed for lowered grid module\n";
    return false;
  }

  if (llvmIR.find("tensorium_init_grid_affine") == std::string::npos) {
    std::cerr << "FAIL: LLVM IR missing tensorium_init_grid_affine symbol\n";
    return false;
  }
  if (llvmIR.find("tensorium_rhs_grid_affine") == std::string::npos) {
    std::cerr << "FAIL: LLVM IR missing tensorium_rhs_grid_affine symbol\n";
    return false;
  }
  if (llvmIR.find("@tensorium_entry") != std::string::npos ||
      llvmIR.find("@tensorium_init(") != std::string::npos ||
      llvmIR.find("@tensorium_rhs(") != std::string::npos) {
    std::cerr << "FAIL: LLVM IR should not expose source tensorium_init/rhs/entry symbols\n";
    return false;
  }

  return true;
}

static bool testLoweredGridHostHeaderEmission() {
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
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

  backend::ModuleIR mod =
      buildModuleFromFile("tests/fixtures/gr/schwarzschild_ricci_3d.tn",
                          CompilationMode::Executable);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);
  auto verify = validation::verifyIR(mod);
  if (!verify.ok()) {
    std::cerr << "FAIL: IR verification failed before host header test\n";
    return false;
  }

  std::string header;
  if (!tensorium_mlir::emitHostHeader(mod, opts, &header)) {
    std::cerr << "FAIL: emitHostHeader failed for lowered Ricci grid module\n";
    return false;
  }

  if (header.find("tensorium_call_init_grid_affine") == std::string::npos ||
      header.find("tensorium_call_rhs_grid_affine") == std::string::npos ||
      header.find("double *Christoffel") == std::string::npos ||
      header.find("27 * n_points") == std::string::npos ||
      header.find("9 * n_points") == std::string::npos) {
    std::cerr << "FAIL: generated host header missing expected grid wrappers "
                 "or component sizes\n";
    return false;
  }

  return true;
}

static bool testCompilerApiCompileFileToLLVMIR() {
  tensorium::api::CompileOptions compileOpts;
  compileOpts.mode = CompilationMode::Executable;

  tensorium_mlir::MLIRGenOptions mlirOpts = makeExecutablePipelineOpts();
  mlirOpts.enableMetricLoweringPass = true;
  mlirOpts.enableInitStdLoweringPass = true;
  mlirOpts.enableInitGridAffinePass = true;
  mlirOpts.enableRhsGridAffinePass = true;
  mlirOpts.enableStripSourceFuncsPass = true;
  mlirOpts.enableStencilLoweringPass = false;

  std::string llvmIR;
  try {
    llvmIR = tensorium::api::compileFileToLLVMIR("tests/01_scalar_minimal.tn",
                                                  compileOpts, mlirOpts);
  } catch (const std::exception &ex) {
    std::cerr << "FAIL: compiler API compileFileToLLVMIR threw: " << ex.what()
              << "\n";
    return false;
  }

  if (llvmIR.find("tensorium_rhs_grid_affine") == std::string::npos) {
    std::cerr
        << "FAIL: compiler API LLVM output missing tensorium_rhs_grid_affine\n";
    return false;
  }

  return true;
}

static bool testCompilerApiSymbolicWarningPropagation() {
  const std::string source = R"(
field scalar phi

evolution NoSimulation {
  dt phi = phi
}
)";

  tensorium::api::CompileOptions compileOpts;
  compileOpts.mode = CompilationMode::Symbolic;

  tensorium::api::CompileResult result;
  try {
    result = tensorium::api::parseAndValidateSource(source, compileOpts);
  } catch (const std::exception &ex) {
    std::cerr << "FAIL: compiler API parseAndValidateSource threw: "
              << ex.what() << "\n";
    return false;
  }

  bool sawMissingSimulationWarning = false;
  for (const auto &warn : result.warnings) {
    if (warn.find("W1001: missing simulation block in symbolic mode") !=
        std::string::npos) {
      sawMissingSimulationWarning = true;
      break;
    }
  }

  if (!sawMissingSimulationWarning) {
    std::cerr << "FAIL: compiler API did not return missing-simulation warning "
                 "in symbolic mode\n";
    return false;
  }

  return true;
}

static bool testSchwarzschildInitThetaZeroNoNaN() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/schwarzschild_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitEvalContext evalCtx;
  setupSinglePointInitContext(evalCtx, 1.0, 10.0, 0.0, 0.0);
  auto result = tensorium_mlir::evaluateTensoriumInit(*module, evalCtx.desc);
  if (!result.ok) {
    std::cerr << "FAIL: init evaluator failed at theta=0: "
              << result.message << "\n";
    return false;
  }

  if (std::isnan(evalCtx.buffers.gamma[8][0])) {
    std::cerr << "FAIL: gamma_phiphi must not be NaN at theta=0\n";
    return false;
  }
  if (!almostEqual(evalCtx.buffers.gamma[8][0], 0.0)) {
    std::cerr << "FAIL: gamma_phiphi mismatch at theta=0, got "
              << evalCtx.buffers.gamma[8][0] << "\n";
    return false;
  }
  std::cout << std::setprecision(17)
            << "[numeric] Schwarzschild theta=0 edge case"
            << " gamma_phiphi=" << evalCtx.buffers.gamma[8][0] << "\n"
            << "  g_uv        got=" << formatMatrix4x4(evalCtx.buffers.metric4)
            << "\n"
            << "  Gamma_ij    got=" << formatMatrix3x3(evalCtx.buffers.gamma)
            << "\n"
            << "  GammaU^ij   got=" << formatMatrix3x3(evalCtx.buffers.gammaU)
            << "\n";
  return true;
}

static bool testSchwarzschildInitHorizonIEEE() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/schwarzschild_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitEvalContext evalCtx;
  setupSinglePointInitContext(evalCtx, 1.0, 2.0, std::acos(-1.0) * 0.5, 0.0);
  auto result = tensorium_mlir::evaluateTensoriumInit(*module, evalCtx.desc);
  if (!result.ok) {
    std::cerr << "FAIL: init evaluator unexpectedly rejected r=2M: "
              << result.message << "\n";
    return false;
  }

  if (!almostEqual(evalCtx.buffers.alpha[0], 0.0)) {
    std::cerr << "FAIL: alpha must be zero at r=2M in current front contract\n";
    return false;
  }
  if (!std::isinf(evalCtx.buffers.gamma[0][0])) {
    std::cerr << "FAIL: gamma_rr expected to be inf at r=2M\n";
    return false;
  }
  if (std::isnan(evalCtx.buffers.gammaU[0][0]) ||
      !almostEqual(evalCtx.buffers.gammaU[0][0], 0.0)) {
    std::cerr << "FAIL: gammaU_rr expected to be finite zero at r=2M\n";
    return false;
  }
  std::cout << std::setprecision(17)
            << "[numeric] Schwarzschild horizon edge case (r=2M)"
            << " alpha=" << evalCtx.buffers.alpha[0]
            << " gamma_rr=" << evalCtx.buffers.gamma[0][0]
            << " gammaU_rr=" << evalCtx.buffers.gammaU[0][0] << "\n"
            << "  g_uv        got=" << formatMatrix4x4(evalCtx.buffers.metric4)
            << "\n"
            << "  Gamma_ij    got=" << formatMatrix3x3(evalCtx.buffers.gamma)
            << "\n"
            << "  GammaU^ij   got=" << formatMatrix3x3(evalCtx.buffers.gammaU)
            << "\n";
  return true;
}

static bool testReissnerNordstromInitNumericPoint() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/reissner_nordstrom_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitEvalContext evalCtx;
  const double M = 1.0;
  const double Q = 0.5;
  const double r = 10.0;
  const double theta = std::acos(-1.0) * 0.5;
  setupSinglePointInitContext(evalCtx, M, r, theta, 0.0);
  evalCtx.desc.params["Q"] = Q;

  auto result = tensorium_mlir::evaluateTensoriumInit(*module, evalCtx.desc);
  if (!result.ok) {
    std::cerr << "FAIL: RN init evaluator failed at reference point: "
              << result.message << "\n";
    return false;
  }

  const double f = 1.0 - 2.0 * M / r + (Q * Q) / (r * r);
  std::cout << std::setprecision(17)
            << "[numeric] Reissner-Nordstrom reference point"
            << " M=" << M << " Q=" << Q << " r=" << r
            << " theta=" << theta << "\n"
            << "  g_uv        got=" << formatMatrix4x4(evalCtx.buffers.metric4)
            << " expected=[[" << (-f) << ", 0, 0, 0], [0, " << (1.0 / f)
            << ", 0, 0], [0, 0, " << (r * r) << ", 0], [0, 0, 0, "
            << (r * r) << "]]\n"
            << "  alpha       got=" << evalCtx.buffers.alpha[0]
            << " expected=" << std::sqrt(f) << "\n"
            << "  Gamma_ij    got=" << formatMatrix3x3(evalCtx.buffers.gamma)
            << " expected=[[" << (1.0 / f) << ", 0, 0], [0, " << (r * r)
            << ", 0], [0, 0, " << (r * r) << "]]\n"
            << "  GammaU^ij   got=" << formatMatrix3x3(evalCtx.buffers.gammaU)
            << " expected=[[" << f << ", 0, 0], [0, " << (1.0 / (r * r))
            << ", 0], [0, 0, " << (1.0 / (r * r)) << "]]\n";

  if (!almostEqual(evalCtx.buffers.alpha[0], std::sqrt(f)) ||
      !almostEqual(evalCtx.buffers.gamma[0][0], 1.0 / f) ||
      !almostEqual(evalCtx.buffers.gamma[4][0], r * r) ||
      !almostEqual(evalCtx.buffers.gamma[8][0], r * r) ||
      !almostEqual(evalCtx.buffers.gammaU[0][0], f) ||
      !almostEqual(evalCtx.buffers.gammaU[4][0], 1.0 / (r * r)) ||
      !almostEqual(evalCtx.buffers.gammaU[8][0], 1.0 / (r * r))) {
    std::cerr << "FAIL: RN numeric init mismatch\n";
    return false;
  }
  return true;
}

static bool testSpatialOffdiagInitNumericPoint() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/spatial_offdiag_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitEvalContext evalCtx;
  setupSinglePointInitContext(evalCtx, 1.0, 0.0, 0.0, 0.0);

  auto result = tensorium_mlir::evaluateTensoriumInit(*module, evalCtx.desc);
  if (!result.ok) {
    std::cerr << "FAIL: spatial offdiag init evaluator failed: "
              << result.message << "\n";
    return false;
  }

  const double gammaExpected[9] = {
      2.0, 1.0, 0.0,
      1.0, 3.0, 0.0,
      0.0, 0.0, 4.0};
  const double gammaUExpected[9] = {
      0.6, -0.2, 0.0,
      -0.2, 0.4, 0.0,
      0.0, 0.0, 0.25};

  std::cout << std::setprecision(17)
            << "[numeric] Spatial offdiag reference point\n"
            << "  g_uv        got=" << formatMatrix4x4(evalCtx.buffers.metric4)
            << " expected=[[-1, 0, 0, 0], [0, 2, 1, 0], [0, 1, 3, 0], [0, 0, 0, 4]]\n"
            << "  alpha       got=" << evalCtx.buffers.alpha[0]
            << " expected=1\n"
            << "  Gamma_ij    got=" << formatMatrix3x3(evalCtx.buffers.gamma)
            << " expected=[[2, 1, 0], [1, 3, 0], [0, 0, 4]]\n"
            << "  GammaU^ij   got=" << formatMatrix3x3(evalCtx.buffers.gammaU)
            << " expected=[[0.6, -0.2, 0], [-0.2, 0.4, 0], [0, 0, 0.25]]\n";

  if (!almostEqual(evalCtx.buffers.alpha[0], 1.0)) {
    std::cerr << "FAIL: spatial offdiag alpha mismatch\n";
    return false;
  }
  for (unsigned i = 0; i < 9; ++i) {
    if (!almostEqual(evalCtx.buffers.gamma[i][0], gammaExpected[i])) {
      std::cerr << "FAIL: spatial offdiag gamma mismatch at component " << i
                << "\n";
      return false;
    }
    if (!almostEqual(evalCtx.buffers.gammaU[i][0], gammaUExpected[i])) {
      std::cerr << "FAIL: spatial offdiag gammaU mismatch at component " << i
                << "\n";
      return false;
    }
  }
  return true;
}

static bool testKerrLikeInitNumericPoint() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/kerr_like_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitEvalContext evalCtx;
  const double M = 1.0;
  const double a = 0.3;
  const double r = 10.0;
  const double theta = std::acos(-1.0) * 0.5;
  setupSinglePointInitContext(evalCtx, M, r, theta, 0.0);
  evalCtx.desc.params["a"] = a;

  auto result = tensorium_mlir::evaluateTensoriumInit(*module, evalCtx.desc);
  if (!result.ok) {
    std::cerr << "FAIL: Kerr-like init evaluator failed: " << result.message
              << "\n";
    return false;
  }

  const double sin2 = std::sin(theta) * std::sin(theta);
  const double f = 1.0 - 2.0 * M / r;
  const double betaPhi = -(2.0 * a * M / r) * sin2;
  const double gammaUPhPhi = 1.0 / (r * r * sin2);
  const double betaDot = betaPhi * (gammaUPhPhi * betaPhi);
  const double alphaExpected = std::sqrt(f + betaDot);

  std::cout << std::setprecision(17)
            << "[numeric] Kerr-like reference point"
            << " M=" << M << " a=" << a << " r=" << r
            << " theta=" << theta << "\n"
            << "  g_uv        got=" << formatMatrix4x4(evalCtx.buffers.metric4)
            << " expected=[[" << (-f) << ", 0, 0, " << betaPhi
            << "], [0, " << (1.0 / f) << ", 0, 0], [0, 0, " << (r * r)
            << ", 0], [" << betaPhi << ", 0, 0, " << (r * r * sin2)
            << "]]\n"
            << "  alpha       got=" << evalCtx.buffers.alpha[0]
            << " expected=" << alphaExpected << "\n"
            << "  Gamma_ij    got=" << formatMatrix3x3(evalCtx.buffers.gamma)
            << " expected=[[" << (1.0 / f) << ", 0, 0], [0, " << (r * r)
            << ", 0], [0, 0, " << (r * r * sin2) << "]]\n"
            << "  GammaU^ij   got=" << formatMatrix3x3(evalCtx.buffers.gammaU)
            << " expected=[[" << f << ", 0, 0], [0, " << (1.0 / (r * r))
            << ", 0], [0, 0, " << gammaUPhPhi << "]]\n";

  if (!almostEqual(evalCtx.buffers.alpha[0], alphaExpected) ||
      !almostEqual(evalCtx.buffers.gamma[0][0], 1.0 / f) ||
      !almostEqual(evalCtx.buffers.gamma[4][0], r * r) ||
      !almostEqual(evalCtx.buffers.gamma[8][0], r * r * sin2) ||
      !almostEqual(evalCtx.buffers.gammaU[0][0], f) ||
      !almostEqual(evalCtx.buffers.gammaU[4][0], 1.0 / (r * r)) ||
      !almostEqual(evalCtx.buffers.gammaU[8][0], gammaUPhPhi)) {
    std::cerr << "FAIL: Kerr-like numeric init mismatch\n";
    return false;
  }

  return true;
}

static bool testKerrLikeReconstructMetricPoint() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/kerr_like_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitEvalContext evalCtx;
  const double M = 1.0;
  const double a = 0.3;
  const double r = 10.0;
  const double theta = std::acos(-1.0) * 0.5;
  setupSinglePointInitContext(evalCtx, M, r, theta, 0.0);
  evalCtx.desc.params["a"] = a;

  auto result = tensorium_mlir::evaluateTensoriumInit(*module, evalCtx.desc);
  if (!result.ok) {
    std::cerr << "FAIL: Kerr-like reconstruction evaluator failed: "
              << result.message << "\n";
    return false;
  }

  const double alpha = evalCtx.buffers.alpha[0];
  const double beta[3] = {evalCtx.buffers.beta[0][0], evalCtx.buffers.beta[1][0],
                          evalCtx.buffers.beta[2][0]};

  double betaUpper[3] = {0.0, 0.0, 0.0};
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j)
      betaUpper[i] += evalCtx.buffers.gammaU[i * 3 + j][0] * beta[j];
  }

  const double betaDot =
      beta[0] * betaUpper[0] + beta[1] * betaUpper[1] + beta[2] * betaUpper[2];
  const double g00Recon = -alpha * alpha + betaDot;
  const double g0Recon[3] = {beta[0], beta[1], beta[2]};

  const double g00In = evalCtx.buffers.metric4[0][0];
  const double g0In[3] = {evalCtx.buffers.metric4[1][0], evalCtx.buffers.metric4[2][0],
                          evalCtx.buffers.metric4[3][0]};
  const unsigned spatialMap[9] = {5, 6, 7, 9, 10, 11, 13, 14, 15};

  std::cout << std::setprecision(17)
            << "[numeric] Kerr-like reconstruction check\n"
            << "  beta_i      got=" << formatVector3(evalCtx.buffers.beta)
            << "\n"
            << "  g00 recon   got=" << g00Recon << " in=" << g00In << "\n"
            << "  g0i recon   got=[" << g0Recon[0] << ", " << g0Recon[1] << ", "
            << g0Recon[2] << "]"
            << " in=[" << g0In[0] << ", " << g0In[1] << ", " << g0In[2]
            << "]\n";

  if (!almostEqual(g00Recon, g00In)) {
    std::cerr << "FAIL: reconstructed g00 mismatch\n";
    return false;
  }

  for (unsigned i = 0; i < 3; ++i) {
    if (!almostEqual(g0Recon[i], g0In[i])) {
      std::cerr << "FAIL: reconstructed g0i mismatch at i=" << i << "\n";
      return false;
    }
  }

  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      const double gijRecon = evalCtx.buffers.gamma[i * 3 + j][0];
      const double gijIn = evalCtx.buffers.metric4[spatialMap[i * 3 + j]][0];
      if (!almostEqual(gijRecon, gijIn)) {
        std::cerr << "FAIL: reconstructed gij mismatch at (" << i << "," << j
                  << ")\n";
        return false;
      }
    }
  }

  // Keep symmetry in check on g_ti/g_it for the same evaluated metric point.
  if (!almostEqual(evalCtx.buffers.metric4[1][0], evalCtx.buffers.metric4[4][0]) ||
      !almostEqual(evalCtx.buffers.metric4[2][0], evalCtx.buffers.metric4[8][0]) ||
      !almostEqual(evalCtx.buffers.metric4[3][0], evalCtx.buffers.metric4[12][0])) {
    std::cerr << "FAIL: metric4 symmetry mismatch on time-space components\n";
    return false;
  }

  return true;
}

static bool testKerrLikeHasNonZeroBetaPhi() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/kerr_like_3d.tn",
                                        CompilationMode::Executable, ctx);

  const double M = 1.0;
  const double r = 10.0;
  const double theta = std::acos(-1.0) * 0.5;

  InitEvalContext evalShift;
  setupSinglePointInitContext(evalShift, M, r, theta, 0.0);
  evalShift.desc.params["a"] = 0.3;
  auto shiftedRes = tensorium_mlir::evaluateTensoriumInit(*module, evalShift.desc);
  if (!shiftedRes.ok) {
    std::cerr << "FAIL: Kerr-like beta sanity (a=0.3) failed: "
              << shiftedRes.message << "\n";
    return false;
  }

  InitEvalContext evalNoShift;
  setupSinglePointInitContext(evalNoShift, M, r, theta, 0.0);
  evalNoShift.desc.params["a"] = 0.0;
  auto noShiftRes =
      tensorium_mlir::evaluateTensoriumInit(*module, evalNoShift.desc);
  if (!noShiftRes.ok) {
    std::cerr << "FAIL: Kerr-like beta sanity (a=0) failed: "
              << noShiftRes.message << "\n";
    return false;
  }

  const double betaPhiShift = evalShift.buffers.beta[2][0];
  const double betaPhiNoShift = evalNoShift.buffers.beta[2][0];

  std::cout << std::setprecision(17)
            << "[numeric] Kerr-like beta sanity\n"
            << "  beta_i (a=0.3) got=" << formatVector3(evalShift.buffers.beta)
            << "\n"
            << "  beta_i (a=0.0) got=" << formatVector3(evalNoShift.buffers.beta)
            << "\n";

  if (!(std::abs(betaPhiShift) > 1e-12)) {
    std::cerr << "FAIL: expected non-zero beta_phi for a=0.3\n";
    return false;
  }
  if (!almostEqual(betaPhiNoShift, 0.0)) {
    std::cerr << "FAIL: expected beta_phi == 0 for a=0\n";
    return false;
  }

  return true;
}

static bool testSchwarzschildChristoffelNumericPoint() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/schwarzschild_christoffel_3d.tn",
      CompilationMode::Executable, ctx, opts);

  auto rhsFunc = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhsFunc) {
    std::cerr << "FAIL: missing @tensorium_rhs for Christoffel numeric test\n";
    return false;
  }
  if (rhsFunc.getNumArguments() != 3) {
    std::cerr << "FAIL: expected @tensorium_rhs(gamma,gammaU,dtGamma) signature\n";
    return false;
  }

  constexpr std::size_t nr = 9;
  constexpr std::size_t nt = 9;
  constexpr std::size_t np = 9;
  constexpr std::size_t nPoints = nr * nt * np;
  constexpr std::size_t center = 4;

  const auto linearIndex = [](std::size_t ir, std::size_t it, std::size_t ip) {
    return (ir * nt + it) * np + ip;
  };

  const double M = 1.0;
  const double r0 = 10.0;
  const double theta0 = 1.0;
  const double dr = 1.0e-4;
  const double dtheta = 1.0e-4;
  const double dphi = 1.0e-4;

  std::array<std::vector<double>, 9> gamma;
  std::array<std::vector<double>, 9> gammaU;
  std::array<std::vector<double>, 27> dtGamma;
  std::array<double *, 9> gammaPtrs{};
  std::array<double *, 9> gammaUPtrs{};
  std::array<double *, 27> dtGammaPtrs{};
  for (unsigned c = 0; c < 9; ++c) {
    gamma[c].assign(nPoints, 0.0);
    gammaU[c].assign(nPoints, 0.0);
    gammaPtrs[c] = gamma[c].data();
    gammaUPtrs[c] = gammaU[c].data();
  }
  for (unsigned c = 0; c < 27; ++c) {
    dtGamma[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    dtGammaPtrs[c] = dtGamma[c].data();
  }

  for (std::size_t ir = 0; ir < nr; ++ir) {
    const double r = r0 + (static_cast<double>(ir) - static_cast<double>(center)) * dr;
    for (std::size_t it = 0; it < nt; ++it) {
      const double theta =
          theta0 + (static_cast<double>(it) - static_cast<double>(center)) * dtheta;
      const double sinTheta = std::sin(theta);
      const double sin2 = sinTheta * sinTheta;
      const double f = 1.0 - 2.0 * M / r;
      for (std::size_t ip = 0; ip < np; ++ip) {
        (void)ip;
        const std::size_t p = linearIndex(ir, it, ip);
        gamma[0][p] = 1.0 / f;
        gamma[4][p] = r * r;
        gamma[8][p] = r * r * sin2;
        gammaU[0][p] = f;
        gammaU[4][p] = 1.0 / (r * r);
        gammaU[8][p] = 1.0 / (r * r * sin2);
      }
    }
  }

  tensorium_mlir::RhsEvalDescriptor desc;
  desc.grid.spatialDim = 3;
  desc.grid.extents = {nr, nt, np};
  desc.grid.spacing = {dr, dtheta, dphi};
  desc.point = {center, center, center};
  desc.args.resize(3);
  desc.args[0].components.assign(gammaPtrs.begin(), gammaPtrs.end());
  desc.args[1].components.assign(gammaUPtrs.begin(), gammaUPtrs.end());
  desc.args[2].components.assign(dtGammaPtrs.begin(), dtGammaPtrs.end());

  auto evalRes = tensorium_mlir::evaluateTensoriumRHS(*module, desc);
  if (!evalRes.ok) {
    std::cerr << "FAIL: rhs evaluator failed for Christoffel test: "
              << evalRes.message << "\n";
    return false;
  }

  const std::size_t p0 = linearIndex(center, center, center);
  const auto comp3 = [](unsigned i, unsigned j, unsigned k) {
    return i * 9 + j * 3 + k;
  };

  const double r = r0;
  const double theta = theta0;
  const double f = 1.0 - 2.0 * M / r;
  const double sinTheta = std::sin(theta);
  const double sin2 = sinTheta * sinTheta;

  const double gamma_r_rr = dtGamma[comp3(0, 0, 0)][p0];
  const double gamma_r_thth = dtGamma[comp3(0, 1, 1)][p0];
  const double gamma_r_phph = dtGamma[comp3(0, 2, 2)][p0];
  const double gamma_th_rth = dtGamma[comp3(1, 0, 1)][p0];
  const double gamma_ph_rph = dtGamma[comp3(2, 0, 2)][p0];
  const double gamma_ph_thph = dtGamma[comp3(2, 1, 2)][p0];

  const double expected_r_rr = -M / (r * (r - 2.0 * M));
  const double expected_r_thth = -r * f;
  const double expected_r_phph = -r * f * sin2;
  const double expected_th_rth = 1.0 / r;
  const double expected_ph_rph = 1.0 / r;
  const double expected_ph_thph = std::cos(theta) / sinTheta;

  const auto finite = [](double v) { return std::isfinite(v); };
  const auto closeFD = [](double got, double expected) {
    return almostEqual(got, expected, 1e-8, 1e-8);
  };

  std::cout << std::setprecision(17)
            << "[numeric] Schwarzschild Christoffel point M=1 r=10 theta=1\n"
            << "  Gamma^r_rr      got=" << gamma_r_rr
            << " expected=" << expected_r_rr << "\n"
            << "  Gamma^r_thetatheta got=" << gamma_r_thth
            << " expected=" << expected_r_thth << "\n"
            << "  Gamma^r_phiphi  got=" << gamma_r_phph
            << " expected=" << expected_r_phph << "\n"
            << "  Gamma^theta_rtheta got=" << gamma_th_rth
            << " expected=" << expected_th_rth << "\n"
            << "  Gamma^phi_rphi  got=" << gamma_ph_rph
            << " expected=" << expected_ph_rph << "\n"
            << "  Gamma^phi_thetaphi got=" << gamma_ph_thph
            << " expected=" << expected_ph_thph << "\n";

  if (!finite(gamma_r_rr) || !finite(gamma_r_thth) || !finite(gamma_r_phph) ||
      !finite(gamma_th_rth) || !finite(gamma_ph_rph) ||
      !finite(gamma_ph_thph) || !closeFD(gamma_r_rr, expected_r_rr) ||
      !closeFD(gamma_r_thth, expected_r_thth) ||
      !closeFD(gamma_r_phph, expected_r_phph) ||
      !closeFD(gamma_th_rth, expected_th_rth) ||
      !closeFD(gamma_ph_rph, expected_ph_rph) ||
      !closeFD(gamma_ph_thph, expected_ph_thph)) {
    std::cerr << "FAIL: Schwarzschild Christoffel numeric mismatch\n";
    return false;
  }

  return true;
}

static bool testCovariantDerivativeRankOneNumericPoint() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/covariant_rank1_3d.tn", CompilationMode::Executable,
      ctx, opts);

  auto rhsFunc = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhsFunc) {
    std::cerr << "FAIL: missing @tensorium_rhs for covariant derivative test\n";
    return false;
  }

  if (rhsFunc.getNumArguments() != 5) {
    std::cerr << "FAIL: expected @tensorium_rhs(Christoffel,V,W,nablaV,nablaW) "
                 "signature\n";
    return false;
  }

  bool hasCovariantDerivAttr = false;
  rhsFunc.walk([&](tensorium::mlir::DerivOp deriv) {
    if (auto cov = deriv->getAttrOfType<::mlir::BoolAttr>("covariant")) {
      if (cov.getValue())
        hasCovariantDerivAttr = true;
    }
  });
  if (hasCovariantDerivAttr) {
    std::cerr << "FAIL: covariant derivatives must be lowered to explicit "
                 "Christoffel terms (no covariant deriv attrs)\n";
    return false;
  }

  constexpr std::size_t nr = 9;
  constexpr std::size_t nt = 9;
  constexpr std::size_t np = 9;
  constexpr std::size_t nPoints = nr * nt * np;
  constexpr std::size_t center = 4;
  const auto linearIndex = [](std::size_t ir, std::size_t it, std::size_t ip) {
    return (ir * nt + it) * np + ip;
  };
  const std::size_t p0 = linearIndex(center, center, center);
  const auto comp3 = [](unsigned a, unsigned b, unsigned c) {
    return a * 9 + b * 3 + c;
  };
  const auto comp2 = [](unsigned a, unsigned b) { return a * 3 + b; };

  std::array<std::vector<double>, 27> christoffel;
  std::array<std::vector<double>, 3> covectorV;
  std::array<std::vector<double>, 3> vectorW;
  std::array<std::vector<double>, 9> outNablaV;
  std::array<std::vector<double>, 9> outNablaW;
  std::array<double *, 27> christoffelPtrs{};
  std::array<double *, 3> covectorVPtrs{};
  std::array<double *, 3> vectorWPtrs{};
  std::array<double *, 9> outNablaVPtrs{};
  std::array<double *, 9> outNablaWPtrs{};

  for (unsigned c = 0; c < 27; ++c) {
    christoffel[c].assign(nPoints, 0.0);
    christoffelPtrs[c] = christoffel[c].data();
  }
  for (unsigned c = 0; c < 3; ++c) {
    covectorV[c].assign(nPoints, 0.0);
    vectorW[c].assign(nPoints, 0.0);
    covectorVPtrs[c] = covectorV[c].data();
    vectorWPtrs[c] = vectorW[c].data();
  }
  for (unsigned c = 0; c < 9; ++c) {
    outNablaV[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    outNablaW[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    outNablaVPtrs[c] = outNablaV[c].data();
    outNablaWPtrs[c] = outNablaW[c].data();
  }

  // Non-zero components:
  // Gamma^0_{0 1} = 2.0  -> contributes to covector correction term.
  // Gamma^0_{1 2} = 3.0  -> contributes to vector correction term.
  for (std::size_t p = 0; p < nPoints; ++p) {
    christoffel[comp3(0, 0, 1)][p] = 2.0;
    christoffel[comp3(0, 1, 2)][p] = 3.0;

    covectorV[0][p] = 1.0;
    covectorV[1][p] = 2.0;
    covectorV[2][p] = 3.0;

    vectorW[0][p] = 4.0;
    vectorW[1][p] = 5.0;
    vectorW[2][p] = 6.0;
  }

  tensorium_mlir::RhsEvalDescriptor desc;
  desc.grid.spatialDim = 3;
  desc.grid.extents = {nr, nt, np};
  desc.grid.spacing = {1.0, 1.0, 1.0};
  desc.point = {center, center, center};
  desc.args.resize(5);
  desc.args[0].components.assign(christoffelPtrs.begin(), christoffelPtrs.end());
  desc.args[1].components.assign(covectorVPtrs.begin(), covectorVPtrs.end());
  desc.args[2].components.assign(vectorWPtrs.begin(), vectorWPtrs.end());
  desc.args[3].components.assign(outNablaVPtrs.begin(), outNablaVPtrs.end());
  desc.args[4].components.assign(outNablaWPtrs.begin(), outNablaWPtrs.end());

  auto evalRes = tensorium_mlir::evaluateTensoriumRHS(*module, desc);
  if (!evalRes.ok) {
    std::cerr << "FAIL: rhs evaluator failed for covariant derivative test: "
              << evalRes.message << "\n";
    return false;
  }

  const double nablaV_01 = outNablaV[comp2(0, 1)][p0];
  const double nablaW_01 = outNablaW[comp2(0, 1)][p0];
  const double expectedNablaV_01 = -2.0; // -Gamma^m_{0 1} V_m = -2*1
  const double expectedNablaW_01 = 18.0; // +Gamma^0_{1 m} W^m = +3*6

  std::cout << std::setprecision(17)
            << "[numeric] Covariant rank-1 point\n"
            << "  nabla_j(V_i) component [0,1] got=" << nablaV_01
            << " expected=" << expectedNablaV_01 << "\n"
            << "  nabla_j(W^i) component [0,1] got=" << nablaW_01
            << " expected=" << expectedNablaW_01 << "\n";

  if (!almostEqual(nablaV_01, expectedNablaV_01, 1e-10, 1e-10) ||
      !almostEqual(nablaW_01, expectedNablaW_01, 1e-10, 1e-10)) {
    std::cerr << "FAIL: covariant derivative rank-1 numeric mismatch\n";
    return false;
  }
  return true;
}

static bool testNablaMetricPathVarianceMatrix() {
  const std::vector<std::string> fixtures = {
      "tests/64_valid_nabla_contravariant_scalar.tn",
      "tests/68_valid_nabla_covector.tn",
      "tests/69_valid_nabla_mixed_tensor.tn",
      "tests/70_valid_nabla_contravariant_vector.tn",
      "tests/73_valid_nabla_contravariant_covector.tn",
      "tests/74_valid_nabla_contravariant_mixed_tensor.tn",
  };

  for (const auto &fixture : fixtures) {
    backend::ModuleIR mod =
        buildModuleFromFile(fixture, CompilationMode::Executable);
    validation::canonicalizeDifferentialIR(mod);
    validation::canonicalizeEinsteinIR(mod);
    if (!verifyCanonicalIR(mod, fixture))
      return false;

    IRStats stats;
    for (const auto &evo : mod.evolutions) {
      for (const auto &eq : evo.equations)
        collectExprStats(eq.rhs.get(), stats);
      for (const auto &tmp : evo.temporaries)
        collectExprStats(tmp.rhs.get(), stats);
    }

    if (stats.covariant != 0) {
      std::cerr << "FAIL(" << fixture
                << "): nabla sugar must be expanded (no covariant nodes remain)\n";
      return false;
    }
  }
  return true;
}

static bool testNablaConnectionFallbackCovariantOnly() {
  static const char *kSource = R"(
    field mixed_tensor(up=1,down=2) Christoffel[i,j,k]
    field covector V[i]
    field vector W[i]
    field mixed_tensor(up=1,down=1) A[i,j]
    field cov_tensor2 nablaV[i,j]
    field mixed_tensor(up=1,down=1) nablaW[i,j]
    field mixed_tensor(up=1,down=2) nablaA[i,j,k]

    simulation {
      dimension = 3
      resolution = [9,9,9]
      time { dt = 0.01 integrator = euler }
      spatial { scheme = fd derivative = centered order = 2 }
    }

    evolution FallbackCovariantOnly {
      dt nablaV[i,j] = nabla_j(V[i])
      dt nablaW[i,j] = nabla_j(W[i])
      dt nablaA[i,j,k] = nabla_k(A[i,j])
    }
  )";

  backend::ModuleIR mod =
      buildModuleFromSource(kSource, CompilationMode::Executable);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);
  if (!verifyCanonicalIR(mod, "fallback_covariant_only"))
    return false;

  IRStats stats;
  for (const auto &evo : mod.evolutions) {
    for (const auto &eq : evo.equations)
      collectExprStats(eq.rhs.get(), stats);
    for (const auto &tmp : evo.temporaries)
      collectExprStats(tmp.rhs.get(), stats);
  }
  if (stats.covariant != 0) {
    std::cerr << "FAIL: fallback covariant-only program still contains "
                 "covariant derivative IR nodes\n";
    return false;
  }
  return true;
}

static bool testNablaConnectionFallbackContravariantRequiresInverseMetric() {
  static const char *kSource = R"(
    field mixed_tensor(up=1,down=2) Christoffel[i,j,k]
    field vector W[i]
    field con_tensor2 nablaUpW[i,j]

    simulation {
      dimension = 3
      resolution = [9,9,9]
      time { dt = 0.01 integrator = euler }
      spatial { scheme = fd derivative = centered order = 2 }
    }

    evolution FallbackContravariantMissingInverse {
      dt nablaUpW[i,j] = nabla^j(W[i])
    }
  )";

  try {
    (void)buildModuleFromSource(kSource, CompilationMode::Executable);
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    if (msg.find("inverse_metric") == std::string::npos) {
      std::cerr << "FAIL: expected missing inverse_metric error, got: " << msg
                << "\n";
      return false;
    }
    return true;
  }

  std::cerr << "FAIL: expected contravariant nabla fallback to require "
               "inverse_metric\n";
  return false;
}

static bool testNablaConnectionFallbackContravariantNonScalarCompiles() {
  static const char *kSource = R"(
    field mixed_tensor(up=1,down=2) Christoffel[i,j,k]
    field inverse_metric gammaU[i,j]
    field covector V[i]
    field vector W[i]
    field mixed_tensor(up=1,down=1) A[i,j]
    field mixed_tensor(up=1,down=1) nablaUpV[i,j]
    field con_tensor2 nablaUpW[i,j]
    field mixed_tensor(up=2,down=1) nablaUpA[i,k,j]

    simulation {
      dimension = 3
      resolution = [9,9,9]
      time { dt = 0.01 integrator = euler }
      spatial { scheme = fd derivative = centered order = 2 }
    }

    evolution FallbackContravariantNonScalar {
      dt nablaUpV[i,j] = nabla^i(V[j])
      dt nablaUpW[i,j] = nabla^j(W[i])
      dt nablaUpA[i,k,j] = nabla^k(A[i,j])
    }
  )";

  backend::ModuleIR mod =
      buildModuleFromSource(kSource, CompilationMode::Executable);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);
  if (!verifyCanonicalIR(mod, "fallback_contravariant_nonscalar"))
    return false;

  IRStats stats;
  for (const auto &evo : mod.evolutions) {
    for (const auto &eq : evo.equations)
      collectExprStats(eq.rhs.get(), stats);
    for (const auto &tmp : evo.temporaries)
      collectExprStats(tmp.rhs.get(), stats);
  }
  if (stats.covariant != 0) {
    std::cerr << "FAIL: contravariant fallback program still contains "
                 "covariant derivative IR nodes\n";
    return false;
  }
  return true;
}

static bool testCovariantDerivativeAllCasesNumericPoint() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/covariant_all_cases_3d.tn", CompilationMode::Executable,
      ctx, opts);

  auto rhsFunc = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhsFunc) {
    std::cerr << "FAIL: missing @tensorium_rhs for all-cases covariant test\n";
    return false;
  }
  const unsigned rhsArgCount = rhsFunc.getNumArguments();
  if (rhsArgCount != 7 && rhsArgCount != 8) {
    std::cerr << "FAIL: expected @tensorium_rhs with 7 or 8 field arguments for "
                 "all-cases covariant test, got "
              << rhsArgCount << "\n";
    return false;
  }

  constexpr std::size_t nr = 9;
  constexpr std::size_t nt = 9;
  constexpr std::size_t np = 9;
  constexpr std::size_t nPoints = nr * nt * np;
  constexpr std::size_t center = 4;
  const auto linearIndex = [](std::size_t ir, std::size_t it, std::size_t ip) {
    return (ir * nt + it) * np + ip;
  };
  const std::size_t p0 = linearIndex(center, center, center);
  const auto comp3 = [](unsigned a, unsigned b, unsigned c) {
    return a * 9 + b * 3 + c;
  };
  const auto comp2 = [](unsigned a, unsigned b) { return a * 3 + b; };

  std::array<std::vector<double>, 27> christoffel;
  std::array<std::vector<double>, 9> inverseMetric;
  std::array<std::vector<double>, 3> covectorV;
  std::array<std::vector<double>, 3> vectorW;
  std::array<std::vector<double>, 9> mixedA;
  std::array<std::vector<double>, 9> outNablaV;
  std::array<std::vector<double>, 9> outNablaW;
  std::array<std::vector<double>, 27> outNablaA;

  std::array<double *, 27> christoffelPtrs{};
  std::array<double *, 9> inverseMetricPtrs{};
  std::array<double *, 3> covectorVPtrs{};
  std::array<double *, 3> vectorWPtrs{};
  std::array<double *, 9> mixedAPtrs{};
  std::array<double *, 9> outNablaVPtrs{};
  std::array<double *, 9> outNablaWPtrs{};
  std::array<double *, 27> outNablaAPtrs{};

  for (unsigned c = 0; c < 27; ++c) {
    christoffel[c].assign(nPoints, 0.0);
    outNablaA[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    christoffelPtrs[c] = christoffel[c].data();
    outNablaAPtrs[c] = outNablaA[c].data();
  }
  for (unsigned c = 0; c < 9; ++c) {
    inverseMetric[c].assign(nPoints, 0.0);
    mixedA[c].assign(nPoints, 0.0);
    outNablaV[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    outNablaW[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    inverseMetricPtrs[c] = inverseMetric[c].data();
    mixedAPtrs[c] = mixedA[c].data();
    outNablaVPtrs[c] = outNablaV[c].data();
    outNablaWPtrs[c] = outNablaW[c].data();
  }
  for (unsigned c = 0; c < 3; ++c) {
    covectorV[c].assign(nPoints, 0.0);
    vectorW[c].assign(nPoints, 0.0);
    covectorVPtrs[c] = covectorV[c].data();
    vectorWPtrs[c] = vectorW[c].data();
  }

  // Connection and field values chosen so selected components have closed-form
  // expected values.
  for (std::size_t p = 0; p < nPoints; ++p) {
    // Gamma^0_{0 1} = 2 => contributes to covector correction.
    christoffel[comp3(0, 0, 1)][p] = 2.0;
    // Gamma^0_{1 2} = 3 => contributes to vector and mixed (+) corrections.
    christoffel[comp3(0, 1, 2)][p] = 3.0;
    // Gamma^2_{2 1} = 5 => contributes to mixed (-) correction.
    christoffel[comp3(2, 2, 1)][p] = 5.0;

    // Identity inverse metric => nabla^1 == nabla_1 for selected probes.
    inverseMetric[comp2(0, 0)][p] = 1.0;
    inverseMetric[comp2(1, 1)][p] = 1.0;
    inverseMetric[comp2(2, 2)][p] = 1.0;

    covectorV[0][p] = 1.0;
    vectorW[2][p] = 6.0;

    mixedA[comp2(2, 2)][p] = 7.0;
    mixedA[comp2(0, 2)][p] = 11.0;
  }

  tensorium_mlir::RhsEvalDescriptor desc;
  desc.grid.spatialDim = 3;
  desc.grid.extents = {nr, nt, np};
  desc.grid.spacing = {1.0, 1.0, 1.0};
  desc.point = {center, center, center};
  desc.args.resize(rhsArgCount);
  unsigned arg = 0;
  desc.args[arg++].components.assign(christoffelPtrs.begin(), christoffelPtrs.end());
  if (rhsArgCount == 8) {
    desc.args[arg++].components.assign(inverseMetricPtrs.begin(),
                                       inverseMetricPtrs.end());
  }
  desc.args[arg++].components.assign(covectorVPtrs.begin(), covectorVPtrs.end());
  desc.args[arg++].components.assign(vectorWPtrs.begin(), vectorWPtrs.end());
  desc.args[arg++].components.assign(mixedAPtrs.begin(), mixedAPtrs.end());
  desc.args[arg++].components.assign(outNablaVPtrs.begin(), outNablaVPtrs.end());
  desc.args[arg++].components.assign(outNablaWPtrs.begin(), outNablaWPtrs.end());
  desc.args[arg++].components.assign(outNablaAPtrs.begin(), outNablaAPtrs.end());

  auto evalRes = tensorium_mlir::evaluateTensoriumRHS(*module, desc);
  if (!evalRes.ok) {
    std::cerr << "FAIL: rhs evaluator failed for all-cases covariant test: "
              << evalRes.message << "\n";
    return false;
  }

  const double nablaV_01 = outNablaV[comp2(0, 1)][p0];
  const double nablaW_01 = outNablaW[comp2(0, 1)][p0];
  const double nablaA_021 = outNablaA[comp3(0, 2, 1)][p0];

  const double expectedNablaV_01 = -2.0;
  const double expectedNablaW_01 = 18.0;
  const double expectedNablaA_021 = -34.0;

  std::cout << std::setprecision(17)
            << "[numeric] Covariant all-cases fallback point\n"
            << "  nabla_j(V_i) [0,1] got=" << nablaV_01
            << " expected=" << expectedNablaV_01 << "\n"
            << "  nabla_j(W^i) [0,1] got=" << nablaW_01
            << " expected=" << expectedNablaW_01 << "\n"
            << "  nabla_k(A^i_j) [0,2,1] got=" << nablaA_021
            << " expected=" << expectedNablaA_021 << "\n";

  if (!almostEqual(nablaV_01, expectedNablaV_01, 1e-10, 1e-10) ||
      !almostEqual(nablaW_01, expectedNablaW_01, 1e-10, 1e-10) ||
      !almostEqual(nablaA_021, expectedNablaA_021, 1e-10, 1e-10)) {
    std::cerr << "FAIL: covariant all-cases numeric mismatch\n";
    return false;
  }
  return true;
}

static bool testContravariantDerivativeAllCasesNumericPoint() {
  ::mlir::MLIRContext ctx;
  tensorium_mlir::MLIRGenOptions opts = makeExecutablePipelineOpts();
  opts.enableStencilLoweringPass = false;
  auto module = buildMLIRModuleFromFileWithOpts(
      "tests/fixtures/gr/contravariant_all_cases_3d.tn",
      CompilationMode::Executable, ctx, opts);

  auto rhsFunc = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_rhs");
  if (!rhsFunc) {
    std::cerr << "FAIL: missing @tensorium_rhs for all-cases contravariant test\n";
    return false;
  }
  if (rhsFunc.getNumArguments() != 8) {
    std::cerr << "FAIL: expected @tensorium_rhs with 8 field arguments for "
                 "all-cases contravariant test, got "
              << rhsFunc.getNumArguments() << "\n";
    return false;
  }

  constexpr std::size_t nr = 9;
  constexpr std::size_t nt = 9;
  constexpr std::size_t np = 9;
  constexpr std::size_t nPoints = nr * nt * np;
  constexpr std::size_t center = 4;
  const auto linearIndex = [](std::size_t ir, std::size_t it, std::size_t ip) {
    return (ir * nt + it) * np + ip;
  };
  const std::size_t p0 = linearIndex(center, center, center);
  const auto comp3 = [](unsigned a, unsigned b, unsigned c) {
    return a * 9 + b * 3 + c;
  };
  const auto comp2 = [](unsigned a, unsigned b) { return a * 3 + b; };

  std::array<std::vector<double>, 27> christoffel;
  std::array<std::vector<double>, 9> inverseMetric;
  std::array<std::vector<double>, 3> covectorV;
  std::array<std::vector<double>, 3> vectorW;
  std::array<std::vector<double>, 9> mixedA;
  std::array<std::vector<double>, 9> outNablaUpV;
  std::array<std::vector<double>, 9> outNablaUpW;
  std::array<std::vector<double>, 27> outNablaUpA;

  std::array<double *, 27> christoffelPtrs{};
  std::array<double *, 9> inverseMetricPtrs{};
  std::array<double *, 3> covectorVPtrs{};
  std::array<double *, 3> vectorWPtrs{};
  std::array<double *, 9> mixedAPtrs{};
  std::array<double *, 9> outNablaUpVPtrs{};
  std::array<double *, 9> outNablaUpWPtrs{};
  std::array<double *, 27> outNablaUpAPtrs{};

  for (unsigned c = 0; c < 27; ++c) {
    christoffel[c].assign(nPoints, 0.0);
    outNablaUpA[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    christoffelPtrs[c] = christoffel[c].data();
    outNablaUpAPtrs[c] = outNablaUpA[c].data();
  }
  for (unsigned c = 0; c < 9; ++c) {
    inverseMetric[c].assign(nPoints, 0.0);
    mixedA[c].assign(nPoints, 0.0);
    outNablaUpV[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    outNablaUpW[c].assign(nPoints, std::numeric_limits<double>::quiet_NaN());
    inverseMetricPtrs[c] = inverseMetric[c].data();
    mixedAPtrs[c] = mixedA[c].data();
    outNablaUpVPtrs[c] = outNablaUpV[c].data();
    outNablaUpWPtrs[c] = outNablaUpW[c].data();
  }
  for (unsigned c = 0; c < 3; ++c) {
    covectorV[c].assign(nPoints, 0.0);
    vectorW[c].assign(nPoints, 0.0);
    covectorVPtrs[c] = covectorV[c].data();
    vectorWPtrs[c] = vectorW[c].data();
  }

  for (std::size_t p = 0; p < nPoints; ++p) {
    // Same setup as covariant numeric tests, then raise with identity gammaU.
    christoffel[comp3(0, 0, 1)][p] = 2.0;
    christoffel[comp3(0, 1, 2)][p] = 3.0;
    christoffel[comp3(2, 2, 1)][p] = 5.0;

    inverseMetric[comp2(0, 0)][p] = 1.0;
    inverseMetric[comp2(1, 1)][p] = 1.0;
    inverseMetric[comp2(2, 2)][p] = 1.0;

    covectorV[0][p] = 1.0;
    vectorW[2][p] = 6.0;

    mixedA[comp2(2, 2)][p] = 7.0;
    mixedA[comp2(0, 2)][p] = 11.0;
  }

  tensorium_mlir::RhsEvalDescriptor desc;
  desc.grid.spatialDim = 3;
  desc.grid.extents = {nr, nt, np};
  desc.grid.spacing = {1.0, 1.0, 1.0};
  desc.point = {center, center, center};
  desc.args.resize(8);
  desc.args[0].components.assign(christoffelPtrs.begin(), christoffelPtrs.end());
  desc.args[1].components.assign(inverseMetricPtrs.begin(),
                                 inverseMetricPtrs.end());
  desc.args[2].components.assign(covectorVPtrs.begin(), covectorVPtrs.end());
  desc.args[3].components.assign(vectorWPtrs.begin(), vectorWPtrs.end());
  desc.args[4].components.assign(mixedAPtrs.begin(), mixedAPtrs.end());
  desc.args[5].components.assign(outNablaUpVPtrs.begin(), outNablaUpVPtrs.end());
  desc.args[6].components.assign(outNablaUpWPtrs.begin(), outNablaUpWPtrs.end());
  desc.args[7].components.assign(outNablaUpAPtrs.begin(), outNablaUpAPtrs.end());

  auto evalRes = tensorium_mlir::evaluateTensoriumRHS(*module, desc);
  if (!evalRes.ok) {
    std::cerr << "FAIL: rhs evaluator failed for all-cases contravariant test: "
              << evalRes.message << "\n";
    return false;
  }

  const double nablaUpV_10 = outNablaUpV[comp2(1, 0)][p0];
  const double nablaUpW_01 = outNablaUpW[comp2(0, 1)][p0];
  const double nablaUpA_012 = outNablaUpA[comp3(0, 1, 2)][p0];

  const double expectedNablaUpV_10 = -2.0;
  const double expectedNablaUpW_01 = 18.0;
  const double expectedNablaUpA_012 = -34.0;

  std::cout << std::setprecision(17)
            << "[numeric] Contravariant all-cases fallback point\n"
            << "  nabla^i(V_j) [1,0] got=" << nablaUpV_10
            << " expected=" << expectedNablaUpV_10 << "\n"
            << "  nabla^j(W^i) [0,1] got=" << nablaUpW_01
            << " expected=" << expectedNablaUpW_01 << "\n"
            << "  nabla^k(A^i_j) [0,1,2] got=" << nablaUpA_012
            << " expected=" << expectedNablaUpA_012 << "\n";

  if (!almostEqual(nablaUpV_10, expectedNablaUpV_10, 1e-10, 1e-10) ||
      !almostEqual(nablaUpW_01, expectedNablaUpW_01, 1e-10, 1e-10) ||
      !almostEqual(nablaUpA_012, expectedNablaUpA_012, 1e-10, 1e-10)) {
    std::cerr << "FAIL: contravariant all-cases numeric mismatch\n";
    return false;
  }
  return true;
}

static bool testSchwarzschildChristoffelMLIRStructure() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile(
      "tests/fixtures/gr/schwarzschild_christoffel_3d.tn",
      CompilationMode::Executable, ctx);

  InitRhsLayout layout;
  std::string layoutError;
  if (!verifyInitRhsLayout(*module, layout, layoutError)) {
    std::cerr << "FAIL: " << layoutError << "\n";
    return false;
  }

  tensorium::mlir::Init3P1Op init3p1Op;
  std::vector<::mlir::Value> initAssignedEntryValues;
  auto pushUniqueValue = [](std::vector<::mlir::Value> &vals, ::mlir::Value v) {
    if (llvm::find(vals, v) == vals.end())
      vals.push_back(v);
  };

  for (::mlir::Operation &op : layout.initFunc.getBody().front()) {
    if (auto init = llvm::dyn_cast<tensorium::mlir::Init3P1Op>(&op))
      init3p1Op = init;
    if (auto assign = llvm::dyn_cast<tensorium::mlir::AssignOp>(&op)) {
      if (!init3p1Op)
        continue;
      if (assign.getRhs() == init3p1Op.getAlpha() ||
          assign.getRhs() == init3p1Op.getGamma() ||
          assign.getRhs() == init3p1Op.getGammaU()) {
        auto arg = llvm::dyn_cast<::mlir::BlockArgument>(assign.getField());
        if (!arg || arg.getArgNumber() >= layout.initCall.getNumOperands()) {
          std::cerr << "FAIL: init assign target mapping is invalid\n";
          return false;
        }
        pushUniqueValue(initAssignedEntryValues,
                        layout.initCall.getOperand(arg.getArgNumber()));
      }
    }
  }

  auto valueFeedsContraction = [](::mlir::Value v) {
    for (::mlir::Operation *user : v.getUsers()) {
      if (llvm::isa<tensorium::mlir::ContractOp>(user) ||
          llvm::isa<tensorium::mlir::EinsumOp>(user)) {
        return true;
      }
    }
    for (::mlir::Operation *user : v.getUsers()) {
      auto mul = llvm::dyn_cast<tensorium::mlir::MulOp>(user);
      if (!mul)
        continue;
      for (::mlir::Operation *mulUser : mul.getRes().getUsers()) {
        if (llvm::isa<tensorium::mlir::ContractOp>(mulUser) ||
            llvm::isa<tensorium::mlir::EinsumOp>(mulUser)) {
          return true;
        }
      }
    }
    return false;
  };

  int addCount = 0;
  int subCount = 0;
  int mulCount = 0;
  int contractCount = 0;
  int einsumCount = 0;
  int dtAssignCount = 0;
  bool hasChristoffelMagicOp = false;
  bool rhsBuildsLocalGammaU = false;
  bool gammaUFromInitAssignedFeedsContract = false;
  bool gammaUContractUsesNonInitSource = false;

  for (::mlir::Operation &op : layout.rhsFunc.getBody().front()) {
    addCount += llvm::isa<tensorium::mlir::AddOp>(&op) ? 1 : 0;
    subCount += llvm::isa<tensorium::mlir::SubOp>(&op) ? 1 : 0;
    mulCount += llvm::isa<tensorium::mlir::MulOp>(&op) ? 1 : 0;
    contractCount += llvm::isa<tensorium::mlir::ContractOp>(&op) ? 1 : 0;
    einsumCount += llvm::isa<tensorium::mlir::EinsumOp>(&op) ? 1 : 0;
    rhsBuildsLocalGammaU |= llvm::isa<tensorium::mlir::BuildConTensor2Op>(&op);
    hasChristoffelMagicOp |= (op.getName().getStringRef() == "tensorium.christoffel");

    if (auto dt = llvm::dyn_cast<tensorium::mlir::DtAssignOp>(&op)) {
      ++dtAssignCount;
      auto idx = dt.getIndices();
      if (idx.size() != 3) {
        std::cerr << "FAIL: Christoffel dt_assign must carry 3 indices\n";
        return false;
      }
    }

    auto ref = llvm::dyn_cast<tensorium::mlir::RefOp>(&op);
    if (!ref)
      continue;
    auto srcTy =
        llvm::dyn_cast<tensorium::mlir::FieldType>(ref.getSource().getType());
    if (!srcTy || srcTy.getUp() != 2 || srcTy.getDown() != 0)
      continue;
    auto idx = ref.getIndices();
    if (!idx || idx->size() != 2)
      continue;
    if (!valueFeedsContraction(ref.getResult()))
      continue;

    bool fromInitAssignedField = false;
    if (auto arg = llvm::dyn_cast<::mlir::BlockArgument>(ref.getSource())) {
      if (arg.getArgNumber() < layout.rhsCall.getNumOperands()) {
        ::mlir::Value entryOperand =
            layout.rhsCall.getOperand(arg.getArgNumber());
        fromInitAssignedField =
            llvm::find(initAssignedEntryValues, entryOperand) !=
            initAssignedEntryValues.end();
      }
    }

    if (fromInitAssignedField)
      gammaUFromInitAssignedFeedsContract = true;
    else
      gammaUContractUsesNonInitSource = true;
  }

  if (hasChristoffelMagicOp) {
    std::cerr << "FAIL: Christoffel lowering emitted forbidden magic op\n";
    return false;
  }
  if (rhsBuildsLocalGammaU) {
    std::cerr << "FAIL: tensorium_rhs must not construct local gammaU values\n";
    return false;
  }
  if (!gammaUFromInitAssignedFeedsContract || gammaUContractUsesNonInitSource) {
    std::cerr << "FAIL: Christoffel contraction must consume init-assigned gammaU field\n";
    return false;
  }
  if (addCount < 1 || subCount < 1 || mulCount < 2 || dtAssignCount != 1) {
    std::cerr << "FAIL: Christoffel MLIR structure is incomplete\n";
    return false;
  }
  if (contractCount == 0 && einsumCount == 0) {
    std::cerr << "FAIL: Christoffel MLIR must include contract or einsum contraction ops\n";
    return false;
  }

  return true;
}

static bool testInitRhsInvariantRejectsMetricInRhs() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/schwarzschild_3d.tn",
                                        CompilationMode::Executable, ctx);

  InitRhsLayout layout;
  std::string err;
  if (!verifyInitRhsLayout(*module, layout, err)) {
    std::cerr << "FAIL: baseline layout must be valid before negative mutation: "
              << err << "\n";
    return false;
  }

  tensorium::mlir::Metric4Op metricInInit;
  for (::mlir::Operation &op : layout.initFunc.getBody().front()) {
    if (auto metric = llvm::dyn_cast<tensorium::mlir::Metric4Op>(&op)) {
      metricInInit = metric;
      break;
    }
  }
  if (!metricInInit) {
    std::cerr << "FAIL: could not locate metric4 op for negative invariant test\n";
    return false;
  }

  ::mlir::Operation *clonedMetric = metricInInit->clone();
  layout.rhsFunc.getBody().front().push_front(clonedMetric);

  InitRhsLayout mutatedLayout;
  std::string mutatedErr;
  if (verifyInitRhsLayout(*module, mutatedLayout, mutatedErr)) {
    std::cerr << "FAIL: expected invariant checker to reject metric4 inside tensorium_rhs\n";
    return false;
  }
  return true;
}

int main() {
  struct NamedTest {
    const char *name;
    bool (*fn)();
  };

  const NamedTest tests[] = {
      {"testMLIRGenOptimizationPassOptions",
       &testMLIRGenOptimizationPassOptions},
      {"testConTensor3Lowering", &testConTensor3Lowering},
      {"testIndexSetPolicy", &testIndexSetPolicy},
      {"testIRTensorTypeMappingForExternCall", &testIRTensorTypeMappingForExternCall},
      {"testIRCanonicalGradientFromFixture", &testIRCanonicalGradientFromFixture},
      {"testIRCanonicalDivergenceFromFixture", &testIRCanonicalDivergenceFromFixture},
      {"testIRCanonicalTraceFromFixture", &testIRCanonicalTraceFromFixture},
      {"testIRCanonicalEinsteinRenameInsert", &testIRCanonicalEinsteinRenameInsert},
      {"testIRVerifierRejectsUncanonicalizedGradient", &testIRVerifierRejectsUncanonicalizedGradient},
      {"testSchwarzschildCanonicalPatterns", &testSchwarzschildCanonicalPatterns},
      {"testEinsteinCanonicalEquivalence", &testEinsteinCanonicalEquivalence},
      {"testSchwarzschildMLIRVerification", &testSchwarzschildMLIRVerification},
      {"testMLIRNormalizationPasses", &testMLIRNormalizationPasses},
      {"testSchwarzschildInitNumericPoint", &testSchwarzschildInitNumericPoint},
      {"testSchwarzschildInitMetricLoweringPass",
       &testSchwarzschildInitMetricLoweringPass},
      {"testSchwarzschildInitPointStdLowering",
       &testSchwarzschildInitPointStdLowering},
      {"testSchwarzschildInitGridScfLowering",
       &testSchwarzschildInitGridScfLowering},
      {"testSchwarzschildInitGridAffineLowering",
       &testSchwarzschildInitGridAffineLowering},
      {"testSchwarzschildRhsGridScfLowering",
       &testSchwarzschildRhsGridScfLowering},
      {"testRhsGridScfRejectsImplicitParams",
       &testRhsGridScfRejectsImplicitParams},
      {"testRhsExplicitParamDeclarationAccepted",
       &testRhsExplicitParamDeclarationAccepted},
      {"testRhsGridScfLoweringSupportsCoords",
       &testRhsGridScfLoweringSupportsCoords},
      {"testSchwarzschildRhsGridAffineLowering",
       &testSchwarzschildRhsGridAffineLowering},
      {"testGeneratedKernelABIMetadata", &testGeneratedKernelABIMetadata},
      {"testSpatialOffdiagNoParamABI", &testSpatialOffdiagNoParamABI},
      {"testSpatialOffdiagGeneratedSplit3p1Constants",
       &testSpatialOffdiagGeneratedSplit3p1Constants},
      {"testSpatialOffdiagInitGridAffineNoLoopAlloc",
       &testSpatialOffdiagInitGridAffineNoLoopAlloc},
      {"testSpatialOffdiagRhsCompactHessianAffineLowering",
       &testSpatialOffdiagRhsCompactHessianAffineLowering},
      {"testLoweredGridLLVMABISignature", &testLoweredGridLLVMABISignature},
      {"testStripSourceFuncsAfterGridLowering",
       &testStripSourceFuncsAfterGridLowering},
      {"testStripSourceFuncsRhsOnly", &testStripSourceFuncsRhsOnly},
      {"testLoweredGridModuleLLVMIREmission",
       &testLoweredGridModuleLLVMIREmission},
      {"testLoweredGridHostHeaderEmission",
       &testLoweredGridHostHeaderEmission},
      {"testCompilerApiCompileFileToLLVMIR",
       &testCompilerApiCompileFileToLLVMIR},
      {"testCompilerApiSymbolicWarningPropagation",
       &testCompilerApiSymbolicWarningPropagation},
      {"testSchwarzschildInitThetaZeroNoNaN",
       &testSchwarzschildInitThetaZeroNoNaN},
      {"testSchwarzschildInitHorizonIEEE", &testSchwarzschildInitHorizonIEEE},
      {"testReissnerNordstromInitNumericPoint",
       &testReissnerNordstromInitNumericPoint},
      {"testSpatialOffdiagInitNumericPoint", &testSpatialOffdiagInitNumericPoint},
      {"testKerrLikeInitNumericPoint", &testKerrLikeInitNumericPoint},
      {"testKerrLikeReconstructMetricPoint",
       &testKerrLikeReconstructMetricPoint},
      {"testKerrLikeHasNonZeroBetaPhi", &testKerrLikeHasNonZeroBetaPhi},
      {"testNablaMetricPathVarianceMatrix", &testNablaMetricPathVarianceMatrix},
      {"testNablaConnectionFallbackCovariantOnly",
       &testNablaConnectionFallbackCovariantOnly},
      {"testNablaConnectionFallbackContravariantRequiresInverseMetric",
       &testNablaConnectionFallbackContravariantRequiresInverseMetric},
      {"testNablaConnectionFallbackContravariantNonScalarCompiles",
       &testNablaConnectionFallbackContravariantNonScalarCompiles},
      {"testCovariantDerivativeRankOneNumericPoint",
       &testCovariantDerivativeRankOneNumericPoint},
      {"testCovariantDerivativeAllCasesNumericPoint",
       &testCovariantDerivativeAllCasesNumericPoint},
      {"testContravariantDerivativeAllCasesNumericPoint",
       &testContravariantDerivativeAllCasesNumericPoint},
      {"testSchwarzschildChristoffelNumericPoint",
       &testSchwarzschildChristoffelNumericPoint},
      {"testSchwarzschildChristoffelMLIRStructure",
       &testSchwarzschildChristoffelMLIRStructure},
      {"testInitRhsInvariantRejectsMetricInRhs", &testInitRhsInvariantRejectsMetricInRhs},
  };

  bool ok = true;
  for (const auto &test : tests) {
    try {
      if (!test.fn()) {
        std::cerr << "FAIL: " << test.name << "\n";
        ok = false;
      }
    } catch (const std::exception &e) {
      std::cerr << "FAIL: " << test.name << " threw: " << e.what() << "\n";
      ok = false;
    }
  }

  if (!ok)
    return 1;

  std::cout << "All unit tests passed\n";
  return 0;
}
