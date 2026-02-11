#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Validation/IRCanonicalize.hpp"
#include "tensorium/Validation/IRVerifier.hpp"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
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
  tensorium_mlir::MLIRGenOptions opts;
  opts.enableStencilLoweringPass = true;
  opts.enableEinsteinLoweringPass = true;
  opts.enableIndexAnalyzePass = true;
  opts.enableEinsteinAnalyzeEinsumPass = true;
  opts.enableEinsteinCanonicalizePass = true;
  opts.enableEinsteinValidityPass = true;
  return opts;
}

static ::mlir::OwningOpRef<::mlir::ModuleOp>
buildMLIRModuleFromFile(const std::string &path, CompilationMode mode,
                        ::mlir::MLIRContext &ctx) {
  backend::ModuleIR mod = buildModuleFromFile(path, mode);
  validation::canonicalizeDifferentialIR(mod);
  validation::canonicalizeEinsteinIR(mod);
  auto verify = validation::verifyIR(mod);
  if (!verify.ok()) {
    std::ostringstream oss;
    oss << "IR verification failed for " << path;
    for (const auto &diag : verify.diags)
      oss << "\n  - " << diag.message;
    throw std::runtime_error(oss.str());
  }
  return tensorium_mlir::buildMLIRModule(mod, ctx, makeExecutablePipelineOpts());
}

static bool isConstValue(::mlir::Value v, double expected, double eps = 1e-12) {
  auto c = v.getDefiningOp<tensorium::mlir::ConstOp>();
  if (!c)
    return false;
  return std::abs(c.getValue().convertToDouble() - expected) <= eps;
}

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

struct IRStats {
  int contractions = 0;
  int partials = 0;
  int gradients = 0;
  int divergences = 0;
  int covariant = 0;
  int renames = 0;
};

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
  if (!rhs || rhs->kind != backend::ExprIR::Kind::Trace) {
    std::cerr << "FAIL: expected contract(A[i,i]) to canonicalize to trace(A)\n";
    return false;
  }

  auto *trace = static_cast<const backend::TraceIR *>(rhs);
  if (trace->tracedIndices.empty()) {
    std::cerr << "FAIL: canonical trace must contain traced indices\n";
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
  if (!ctr->in || ctr->in->kind != backend::ExprIR::Kind::IndexRename) {
    std::cerr << "FAIL: expected alpha-rename insertion for risky index capture\n";
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

static bool testSchwarzschildMLIRVerification() {
  ::mlir::MLIRContext ctx;
  auto module = buildMLIRModuleFromFile("tests/fixtures/gr/schwarzschild_3d.tn",
                                        CompilationMode::Executable, ctx);

  auto func = module->lookupSymbol<::mlir::func::FuncOp>("tensorium_entry");
  if (!func) {
    std::cerr << "FAIL: missing tensorium_entry function in MLIR module\n";
    return false;
  }

  tensorium::mlir::Metric4Op metricOp;
  tensorium::mlir::Init3P1Op init3p1Op;
  bool hasLegacySplitOp = false;
  bool hasParamM = false;
  bool hasCoordR = false;
  bool hasCoordTheta = false;
  bool hasSin = false;

  for (::mlir::Operation &op : func.getBody().front()) {
    if (auto metric = llvm::dyn_cast<tensorium::mlir::Metric4Op>(&op))
      metricOp = metric;
    if (auto init = llvm::dyn_cast<tensorium::mlir::Init3P1Op>(&op))
      init3p1Op = init;
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
    if (op.hasAttr("alpha_expr") || op.hasAttr("gamma_diag")) {
      std::cerr << "FAIL: forbidden legacy string attr found on op '"
                << op.getName().getStringRef().str() << "'\n";
      return false;
    }
  }

  if (!metricOp || !init3p1Op) {
    std::cerr << "FAIL: expected both metric4 and init3p1 ops\n";
    return false;
  }
  if (hasLegacySplitOp) {
    std::cerr << "FAIL: legacy tensorium.split3p1 op should not be emitted\n";
    return false;
  }
  if (init3p1Op->getNumOperands() != 4) {
    std::cerr << "FAIL: init3p1 must take exactly 4 operands\n";
    return false;
  }
  if (metricOp->hasAttr("components")) {
    std::cerr << "FAIL: metric4 must not encode components as string attrs\n";
    return false;
  }
  if (!hasParamM || !hasCoordR || !hasCoordTheta || !hasSin) {
    std::cerr << "FAIL: expected param/coord/sin ops for Schwarzschild metric\n";
    return false;
  }

  auto isParamMValue = [](::mlir::Value v) {
    auto p = v.getDefiningOp<tensorium::mlir::ParamOp>();
    return p && p.getName() == "M";
  };
  auto isCoordValue = [](::mlir::Value v, llvm::StringRef name) {
    auto c = v.getDefiningOp<tensorium::mlir::CoordOp>();
    return c && c.getName() == name;
  };

  auto metricComps = metricOp.getComponents();
  if (metricComps.size() != 16) {
    std::cerr << "FAIL: metric4 must carry 16 SSA components\n";
    return false;
  }
  ::mlir::Value g00 = metricComps[0];
  ::mlir::Value g11 = metricComps[5];
  ::mlir::Value g22 = metricComps[10];
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
  if (!twoMrDiv || !isCoordValue(twoMrDiv.getRhs(), "r")) {
    std::cerr << "FAIL: expected 2*M/r denominator to be coordinate r\n";
    return false;
  }

  auto twoMMul = twoMrDiv.getLhs().getDefiningOp<tensorium::mlir::MulOp>();
  if (!twoMMul) {
    std::cerr << "FAIL: expected 2*M multiplication in Schwarzschild factor\n";
    return false;
  }
  const bool hasConst2ParamM =
      (isConstValue(twoMMul.getLhs(), 2.0) && isParamMValue(twoMMul.getRhs())) ||
      (isConstValue(twoMMul.getRhs(), 2.0) && isParamMValue(twoMMul.getLhs()));
  if (!hasConst2ParamM) {
    std::cerr << "FAIL: expected factor 2*M in Schwarzschild factor\n";
    return false;
  }

  auto alphaSqrt = init3p1Op.getAlphaIn().getDefiningOp<tensorium::mlir::SqrtOp>();
  if (!alphaSqrt) {
    std::cerr << "FAIL: expected alpha input to init3p1 to be sqrt(...)\n";
    return false;
  }
  if (!valueDependsOn(alphaSqrt.getIn(), [&](::mlir::Operation *op) {
        return op == fSub.getOperation();
      })) {
    std::cerr << "FAIL: alpha sqrt input does not depend on f = 1-2*M/r\n";
    return false;
  }

  auto gammaUBuild =
      init3p1Op.getGammaUIn().getDefiningOp<tensorium::mlir::BuildConTensor2Op>();
  if (!gammaUBuild || gammaUBuild.getComponents().size() != 9) {
    std::cerr << "FAIL: expected gammaU diagonal tensor builder\n";
    return false;
  }
  auto gammaUComps = gammaUBuild.getComponents();
  for (int idx : {1, 2, 3, 5, 6, 7}) {
    if (!isConstValue(gammaUComps[idx], 0.0)) {
      std::cerr << "FAIL: gammaU off-diagonal terms must be zero\n";
      return false;
    }
  }
  auto inv11 = gammaUComps[0].getDefiningOp<tensorium::mlir::DivOp>();
  auto inv22 = gammaUComps[4].getDefiningOp<tensorium::mlir::DivOp>();
  auto inv33 = gammaUComps[8].getDefiningOp<tensorium::mlir::DivOp>();
  if (!inv11 || !inv22 || !inv33 || !isConstValue(inv11.getLhs(), 1.0) ||
      !isConstValue(inv22.getLhs(), 1.0) || !isConstValue(inv33.getLhs(), 1.0) ||
      inv11.getRhs() != g11 || inv22.getRhs() != g22 || inv33.getRhs() != g33) {
    std::cerr << "FAIL: gammaU diag must be {1/g11, 1/g22, 1/g33}\n";
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

  ::mlir::Value gammaURef;
  ::mlir::Value alphaRef;
  ::mlir::Value gammaRef;
  for (::mlir::Operation &op : func.getBody().front()) {
    auto ref = llvm::dyn_cast<tensorium::mlir::RefOp>(&op);
    if (!ref)
      continue;
    if (ref.getSource() == init3p1Op.getGammaU())
      gammaURef = ref.getResult();
    if (ref.getSource() == init3p1Op.getAlpha())
      alphaRef = ref.getResult();
    if (ref.getSource() == init3p1Op.getGamma())
      gammaRef = ref.getResult();
  }
  if (!gammaURef || !alphaRef || !gammaRef) {
    std::cerr << "FAIL: expected refs sourced from init3p1 alpha/gamma/gammaU\n";
    return false;
  }

  bool gammaUFeedsContract = false;
  for (::mlir::Operation *user : gammaURef.getUsers()) {
    auto mul = llvm::dyn_cast<tensorium::mlir::MulOp>(user);
    if (!mul)
      continue;
    for (::mlir::Operation *mulUser : mul.getRes().getUsers()) {
      if (llvm::isa<tensorium::mlir::ContractOp>(mulUser)) {
        gammaUFeedsContract = true;
        break;
      }
    }
    if (gammaUFeedsContract)
      break;
  }
  if (!gammaUFeedsContract) {
    std::cerr << "FAIL: gammaU from init3p1 is not used in contract use-def chain\n";
    return false;
  }

  bool alphaGammaMulFound = false;
  for (::mlir::Operation *user : alphaRef.getUsers()) {
    auto mul = llvm::dyn_cast<tensorium::mlir::MulOp>(user);
    if (!mul)
      continue;
    if (mul.getLhs() == gammaRef || mul.getRhs() == gammaRef) {
      alphaGammaMulFound = true;
      break;
    }
  }
  if (!alphaGammaMulFound) {
    std::cerr << "FAIL: expected dt K to use alpha*gamma values from init3p1\n";
    return false;
  }

  return true;
}

int main() {
  bool ok = true;
  ok &= testConTensor3Lowering();
  ok &= testIndexSetPolicy();
  ok &= testIRTensorTypeMappingForExternCall();
  ok &= testIRCanonicalGradientFromFixture();
  ok &= testIRCanonicalDivergenceFromFixture();
  ok &= testIRCanonicalTraceFromFixture();
  ok &= testIRCanonicalEinsteinRenameInsert();
  ok &= testIRVerifierRejectsUncanonicalizedGradient();
  ok &= testSchwarzschildCanonicalPatterns();
  ok &= testSchwarzschildMLIRVerification();

  if (!ok)
    return 1;

  std::cout << "All unit tests passed\n";
  return 0;
}
