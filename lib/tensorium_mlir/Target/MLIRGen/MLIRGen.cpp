#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.h"
#include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h"
#include "tensorium_mlir/Dialect/Tensorium/Transform/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace tensorium_mlir {

namespace {

struct FieldDesc {
  std::string name;
  unsigned rank = 0;
  tensorium::mlir::Variance variance = tensorium::mlir::Variance::Scalar;
};

static mlir::ArrayAttr makeIndexArrayAttr(mlir::OpBuilder &b,
                                          const std::vector<std::string> &idx) {
  llvm::SmallVector<mlir::Attribute, 4> names;
  for (const auto &s : idx)
    names.push_back(b.getStringAttr(s));
  return b.getArrayAttr(names);
}

static bool startsWith(const std::string &s, const char *prefix) {
  size_t n = std::char_traits<char>::length(prefix);
  return s.size() >= n && s.compare(0, n, prefix) == 0;
}

static std::vector<FieldDesc>
extractFields(const tensorium::backend::ModuleIR &module) {
  std::vector<FieldDesc> out;
  for (const auto &f : module.fields) {
    FieldDesc d;
    d.name = f.name;
    d.rank = f.up + f.down;
    if (f.up == 0 && f.down == 0)
      d.variance = tensorium::mlir::Variance::Scalar;
    else if (f.up > 0 && f.down == 0)
      d.variance = tensorium::mlir::Variance::Contravariant;
    else if (f.up == 0 && f.down > 0)
      d.variance = tensorium::mlir::Variance::Covariant;
    else
      d.variance = tensorium::mlir::Variance::Mixed;
    out.push_back(std::move(d));
  }
  return out;
}

static mlir::Value
emitExpr(mlir::OpBuilder &b, mlir::Location loc, mlir::Type f64,
         const tensorium::backend::ExprIR *e,
         const llvm::DenseMap<llvm::StringRef, mlir::Value> &fieldArg) {
  using namespace tensorium::backend;
  if (!e)
    return {};

  switch (e->kind) {
  case ExprIR::Kind::Number: {
    auto *n = static_cast<const NumberIR *>(e);
    return b.create<tensorium::mlir::ConstOp>(loc, f64,
                                              b.getF64FloatAttr(n->value));
  }
  case ExprIR::Kind::Var: {
    auto *v = static_cast<const VarIR *>(e);
    auto it = fieldArg.find(v->name);
    if (it == fieldArg.end())
      return {};

    mlir::ArrayAttr indicesAttr;
    if (!v->tensorIndexNames.empty()) {
      llvm::SmallVector<mlir::Attribute, 4> idxList;
      for (const auto &s : v->tensorIndexNames)
        idxList.push_back(b.getStringAttr(s));
      indicesAttr = b.getArrayAttr(idxList);
    }

    auto r = b.create<tensorium::mlir::RefOp>(loc, f64, it->second,
                                              b.getStringAttr("field"),
                                              indicesAttr, mlir::ArrayAttr());

    return r.getResult();
  }
  case ExprIR::Kind::Binary: {
    auto *bin = static_cast<const BinaryIR *>(e);
    auto L = emitExpr(b, loc, f64, bin->lhs.get(), fieldArg);
    auto R = emitExpr(b, loc, f64, bin->rhs.get(), fieldArg);
    if (!L || !R)
      return {};

    if (bin->op == "+")
      return b.create<tensorium::mlir::AddOp>(loc, f64, L, R);
    if (bin->op == "*")
      return b.create<tensorium::mlir::MulOp>(loc, f64, L, R);
    if (bin->op == "-")
      return b.create<tensorium::mlir::SubOp>(loc, f64, L, R);
    return {};
  }
  case ExprIR::Kind::Call: {
    auto *c = static_cast<const CallIR *>(e);
    if (startsWith(c->callee, "d_") && c->callee.size() == 3) {
      if (c->args.empty())
        return {};
      auto arg0 = emitExpr(b, loc, f64, c->args[0].get(), fieldArg);
      auto deriv =
          b.create<tensorium::mlir::DerivOp>(loc, arg0.getType(), arg0);
      deriv->setAttr("index", b.getStringAttr(std::string(1, c->callee[2])));
      return deriv.getResult();
    }
    if (c->callee == "contract") {
      if (c->args.empty())
        return {};
      auto arg0 = emitExpr(b, loc, f64, c->args[0].get(), fieldArg);
      return b.create<tensorium::mlir::ContractOp>(loc, f64, arg0);
    }
    return {};
  }
  default:
    return {};
  }
}
} // namespace

static void addEinsteinPipelineSafe(::mlir::PassManager &pm,
                                    const MLIRGenOptions &opts) {

  if (opts.enableEinsteinLoweringPass) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinLoweringPass());
  }

  const bool needValidity = opts.enableEinsteinValidityPass;
  const bool needCanon = opts.enableEinsteinCanonicalizePass;
  const bool needAnalyze = opts.enableEinsteinAnalyzeEinsumPass || needValidity;
  const bool needIndex = opts.enableIndexAnalyzePass || needValidity;

  if (needIndex) {
    pm.addPass(tensorium::mlir::createTensoriumIndexAnalyzePass());
  }

  if (needAnalyze) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinAnalyzeEinsumPass());
  }

  if (needCanon) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinCanonicalizePass());
  }

  if (needValidity) {
    pm.addPass(tensorium::mlir::createTensoriumEinsteinValidityPass());
  }

  if (opts.enableStencilLoweringPass) {
    pm.addPass(tensorium::mlir::createTensoriumStencilLoweringPass(opts.dx,
                                                                   opts.order));
  }
  if (opts.enableDissipationPass) {
    pm.addPass(tensorium::mlir::createTensoriumDissipationPass(
        opts.dissipationStrength, opts.dx));
  }
}

void emitMLIR(const tensorium::backend::ModuleIR &module,
              const MLIRGenOptions &opts) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<tensorium::mlir::TensoriumDialect>();

  mlir::OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();
  auto moduleOp = mlir::ModuleOp::create(loc);

  const auto fields = extractFields(module);
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (const auto &fd : fields) {
    argTypes.push_back(tensorium::mlir::FieldType::get(&ctx, b.getF64Type(),
                                                       fd.rank, fd.variance));
  }

  auto funcTy = b.getFunctionType(argTypes, {});
  auto f = b.create<mlir::func::FuncOp>(loc, "tensorium_entry", funcTy);
  auto *entry = f.addEntryBlock();
  b.setInsertionPointToEnd(entry);

  llvm::DenseMap<llvm::StringRef, mlir::Value> fieldArg;
  for (unsigned i = 0; i < fields.size(); ++i) {
    fieldArg[fields[i].name] = entry->getArgument(i);
  }

  auto f64 = b.getF64Type();
  for (const auto &evo : module.evolutions) {
    for (const auto &eq : evo.equations) {
      auto it = fieldArg.find(eq.fieldName);
      if (it == fieldArg.end())
        continue;
      auto rhsV = emitExpr(b, loc, f64, eq.rhs.get(), fieldArg);
      if (!rhsV)
        continue;
      b.create<tensorium::mlir::DtAssignOp>(loc, it->second, rhsV,
                                            makeIndexArrayAttr(b, eq.indices));
    }
  }
  b.create<mlir::func::ReturnOp>(loc);
  moduleOp.push_back(f);

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
  addEinsteinPipelineSafe(pm, pipelineOpts);

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(moduleOp))) {
    llvm::errs() << "Pipeline failed\n";
  }
  moduleOp.print(llvm::outs());
}

} // namespace tensorium_mlir
