#include "tensorium/AST/ASTPrinter.hpp"
#include "tensorium/Basic/Diagnostics.hpp"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"

#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Backend/DomainIR.hpp"
#include "tensorium/Backend/IRPrinter.hpp"
#include "tensorium/Runtime/CpuRuntime.hpp"
#include "tensorium/Runtime/Eval.hpp"
#include "tensorium/Solver/ConstraintSolver.hpp"
#include "tensorium/Validation/IRCanonicalize.hpp"
#include "tensorium/Validation/IRVerifier.hpp"
#include "tensorium/Validation/ProgramValidator.hpp"
#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace tensorium;

static std::string readFile(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("cannot open file: " + path);

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

static void writeFile(const std::string &path, const std::string &content) {
  std::ofstream file(path);
  if (!file)
    throw std::runtime_error("cannot open output file: " + path);
  file << content;
  if (!file.good())
    throw std::runtime_error("failed to write output file: " + path);
}

static void printUsage(std::ostream &os) {
  os << "usage: Tensorium_cc [options] file1.tn [file2.tn ...]\n"
     << "  -O0|-O1|-O2|-O3              select clang-style Tensorium pipeline preset\n"
     << "  --dump-mlir|--emit-mlir PATH  emit MLIR after selected Tensorium passes\n"
     << "  --dump-llvm-ir|--emit-llvm PATH emit final LLVM IR\n"
     << "  --emit-host-header PATH       emit C host-call wrappers for generated kernels\n"
     << "  --tensorium-rhs-grid-parallel-lower\n"
     << "                                emit scf.parallel/OpenMP-ready RHS grid kernels\n"
     << "  --tensorium-dx VALUE          finite-difference grid spacing for stencil passes\n"
     << "  --tensorium-stencil-order N   stencil order, supported values depend on pass\n"
     << "  --tensorium-dissipation-strength VALUE\n"
     << "                                artificial dissipation strength\n"
     << "  --mlir-inline                 run MLIR inliner in post-generation pipeline\n"
     << "  --mlir-no-canonicalize        disable post-generation canonicalizer\n"
     << "  --mlir-no-cse                 disable post-generation CSE\n"
     << "  --mlir-pass-timing            report per-pass MLIR/LLVM lowering timings\n"
     << "  --export-constraint-csv PATH export reconstructed radial CTT data\n"
     << "  --mlir-best-effort            do not fail on Tensorium MLIR pipeline errors\n";
}

static bool parseOptimizationLevelArg(
    const std::string &arg, tensorium_mlir::OptimizationLevel &out) {
  if (arg == "-O0" || arg == "--O0" || arg == "--tensorium-O0") {
    out = tensorium_mlir::OptimizationLevel::O0;
    return true;
  }
  if (arg == "-O1" || arg == "--O1" || arg == "--tensorium-O1") {
    out = tensorium_mlir::OptimizationLevel::O1;
    return true;
  }
  if (arg == "-O2" || arg == "--O2" || arg == "--tensorium-O2") {
    out = tensorium_mlir::OptimizationLevel::O2;
    return true;
  }
  if (arg == "-O3" || arg == "--O3" || arg == "--tensorium-O3") {
    out = tensorium_mlir::OptimizationLevel::O3;
    return true;
  }
  return false;
}

static bool takeOptionValue(const std::string &arg, const std::string &option,
                            int &i, int argc, char **argv,
                            std::string &out) {
  const std::string prefix = option + "=";
  if (arg.rfind(prefix, 0) == 0) {
    out = arg.substr(prefix.size());
    if (out.empty())
      throw std::runtime_error(option + " requires a non-empty value");
    return true;
  }
  if (arg == option) {
    if (i + 1 >= argc)
      throw std::runtime_error(option + " expects a value");
    out = argv[++i];
    if (out.empty())
      throw std::runtime_error(option + " requires a non-empty value");
    return true;
  }
  return false;
}

static double parseDoubleOption(const std::string &option,
                                const std::string &value) {
  try {
    size_t consumed = 0;
    double parsed = std::stod(value, &consumed);
    if (consumed != value.size())
      throw std::runtime_error("");
    return parsed;
  } catch (...) {
    throw std::runtime_error(option + " expects a floating point value");
  }
}

static int parseIntOption(const std::string &option,
                          const std::string &value) {
  try {
    size_t consumed = 0;
    int parsed = std::stoi(value, &consumed);
    if (consumed != value.size())
      throw std::runtime_error("");
    return parsed;
  } catch (...) {
    throw std::runtime_error(option + " expects an integer value");
  }
}

static void writeConstraintCsv(
    const std::string &path,
    const tensorium::solver::ConstraintSolution &solution) {
  if (!solution.physicalCtt)
    throw std::runtime_error(
        "constraint CSV export requires a reconstruct ctt block");
  const auto &physical = *solution.physicalCtt;
  const auto &conformalFactor =
      solution.unknowns.at(physical.conformalFactorUnknown);
  const auto &radialVector =
      solution.unknowns.at(physical.radialVectorPotentialUnknown);

  std::ofstream file(path);
  if (!file)
    throw std::runtime_error("cannot open constraint CSV output: " + path);
  file << std::setprecision(17);
  file << "domain,r,conformal_factor,radial_vector,mean_curvature,"
          "gamma_radial,gamma_tangential,k_radial,k_tangential\n";
  for (const auto &domain : solution.domains) {
    for (std::size_t local = 0; local < domain.pointCount; ++local) {
      const std::size_t i = domain.offset + local;
      file << domain.name << ',' << solution.coordinates[i] << ','
           << conformalFactor[i] << ',' << radialVector[i] << ','
           << physical.meanCurvature[i] << ','
           << physical.spatialMetricRadial[i] << ','
           << physical.spatialMetricTangential[i] << ','
           << physical.extrinsicCurvatureRadial[i] << ','
           << physical.extrinsicCurvatureTangential[i] << '\n';
    }
  }
  if (!file.good())
    throw std::runtime_error("failed to write constraint CSV output: " + path);
}

static void printIndexedType(const IndexedExpr *e) {
  if (!e)
    return;
  std::cout << "[u=" << e->inferredType.up
            << ",d=" << e->inferredType.down << "]";
}

static void printIndexedExpr(const IndexedExpr *e) {
  if (auto n = dynamic_cast<const IndexedNumber *>(e)) {
    std::cout << n->value;
    printIndexedType(e);
    return;
  }

  if (auto v = dynamic_cast<const IndexedVar *>(e)) {
    std::cout << v->name << "[";

    switch (v->kind) {
    case IndexedVarKind::Field:
      std::cout << "field";
      break;
    case IndexedVarKind::Parameter:
      std::cout << "param";
      break;
    case IndexedVarKind::Local:
      std::cout << "local";
      break;
    case IndexedVarKind::Coordinate:
      std::cout << "coord:" << v->coordIndex;
      break;
    case IndexedVarKind::Unknown:
      std::cout << "unknown";
      break;
    }

    if (!v->tensorIndexNames.empty()) {
      std::cout << ";";
      for (size_t i = 0; i < v->tensorIndexNames.size(); ++i) {
        std::cout << v->tensorIndexNames[i];
        if (i + 1 < v->tensorIndexNames.size())
          std::cout << ",";
      }
    }

    std::cout << "]";
    printIndexedType(e);
    return;
  }

  if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
    std::cout << "(";
    printIndexedExpr(b->lhs.get());
    std::cout << " " << b->op << " ";
    printIndexedExpr(b->rhs.get());
    std::cout << ")";
    printIndexedType(e);
    return;
  }

  if (auto c = dynamic_cast<const IndexedCall *>(e)) {
    std::cout << c->callee << "(";
    for (size_t i = 0; i < c->args.size(); ++i) {
      printIndexedExpr(c->args[i].get());
      if (i + 1 < c->args.size())
        std::cout << ", ";
    }
    std::cout << ")";
    printIndexedType(e);
    return;
  }
}

int main(int argc, char **argv) {
  bool dumpAST = false;
  bool dumpIndexed = false;
  bool dumpBackend = false;
  bool dumpBackendExpr = false;
  bool runCpu = false;
  bool solveConstraints = false;
  std::string constraintCsvPath;
  std::unordered_map<std::string, double> constraintParameters;
  size_t steps = 10;
  double initScalar = 1.0;
  double initAlpha = 2.0;
  bool dumpMLIR = false;
  bool dumpLLVMIR = false;
  std::string emitMLIRPath;
  std::string emitLLVMIRPath;
  std::string emitHostHeaderPath;
  tensorium_mlir::MLIRPassOptions passOptions;
  bool validateOnly = false;
  bool failOnMLIRPipelineFailure = true;
  bool hasOptimizationLevel = false;
  tensorium_mlir::OptimizationLevel optimizationLevel =
      tensorium_mlir::OptimizationLevel::O0;
  ColorMode colorMode = ColorMode::Auto;
  CompilationMode compilationMode = CompilationMode::Executable;

  if (argc < 2) {
    printUsage(std::cerr);
    return 1;
  }

  std::vector<std::string> files;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string optionValue;

    if (arg == "--help" || arg == "-h") {
      printUsage(std::cout);
      return 0;
    } else if (parseOptimizationLevelArg(arg, optimizationLevel)) {
      hasOptimizationLevel = true;
    } else if (arg == "--dump-ast") {
      dumpAST = true;
    } else if (arg == "--dump-indexed") {
      dumpIndexed = true;
    } else if (arg == "--dump-backend") {
      dumpBackend = true;
    } else if (arg == "--dump-backend-expr") {
      dumpBackendExpr = true;
    } else if (arg == "--tensorium-noop") {
      passOptions.enableNoOpPass = true;
    } else if (arg == "--tensorium-analyze") {
      passOptions.enableAnalysisPass = true;
    } else if (arg == "--run-cpu") {
      runCpu = true;
    } else if (arg == "--solve-constraints") {
      solveConstraints = true;
    } else if (arg == "--export-constraint-csv" ||
               arg.rfind("--export-constraint-csv=", 0) == 0) {
      if (arg == "--export-constraint-csv") {
        if (i + 1 >= argc)
          throw std::runtime_error("--export-constraint-csv expects a path");
        constraintCsvPath = argv[++i];
      } else {
        constraintCsvPath =
            arg.substr(std::string("--export-constraint-csv=").size());
      }
      if (constraintCsvPath.empty())
        throw std::runtime_error("--export-constraint-csv expects a path");
      solveConstraints = true;
    } else if (arg == "--param" || arg.rfind("--param=", 0) == 0) {
      std::string assignment;
      if (arg == "--param") {
        if (i + 1 >= argc)
          throw std::runtime_error("--param expects name=value");
        assignment = argv[++i];
      } else {
        assignment = arg.substr(std::string("--param=").size());
      }
      const size_t equals = assignment.find('=');
      if (equals == std::string::npos || equals == 0 ||
          equals + 1 == assignment.size())
        throw std::runtime_error("--param expects name=value");
      const std::string name = assignment.substr(0, equals);
      const std::string valueText = assignment.substr(equals + 1);
      size_t consumed = 0;
      const double value = std::stod(valueText, &consumed);
      if (consumed != valueText.size())
        throw std::runtime_error("--param value must be a number");
      constraintParameters[name] = value;
    } else if (arg == "--tensorium-einstein-lower") {
      passOptions.enableEinsteinLoweringPass = true;
    } else if (arg == "--tensorium-index-analyze") {
      passOptions.enableIndexAnalyzePass = true;
    } else if (arg == "--tensorium-index-role-analyze") {
      passOptions.enableIndexRoleAnalysisPass = true;
    } else if (arg == "--tensorium-einstein-validate") {
      passOptions.enableEinsteinValidityPass = true;
    } else if (arg == "--tensorium-einstein-canonicalize") {
      passOptions.enableEinsteinCanonicalizePass = true;
    } else if (arg == "--tensorium-einstein-analyze-einsum") {
      passOptions.enableEinsteinAnalyzeEinsumPass = true;
    } else if (arg == "--tensorium-metric-lower") {
      passOptions.enableMetricLoweringPass = true;
    } else if (arg == "--tensorium-init-std-lower") {
      passOptions.enableInitStdLoweringPass = true;
    } else if (arg == "--tensorium-init-grid-scf-lower") {
      passOptions.enableInitGridScfPass = true;
    } else if (arg == "--tensorium-init-grid-affine-lower") {
      passOptions.enableInitGridAffinePass = true;
    } else if (arg == "--tensorium-rhs-grid-scf-lower") {
      passOptions.enableRhsGridScfPass = true;
    } else if (arg == "--tensorium-rhs-grid-affine-lower") {
      passOptions.enableRhsGridAffinePass = true;
    } else if (arg == "--tensorium-rhs-grid-parallel-lower") {
      passOptions.enableRhsGridParallelPass = true;
    } else if (arg == "--tensorium-strip-source-funcs") {
      passOptions.enableStripSourceFuncsPass = true;
    } else if (arg == "--tensorium-stencil-lower") {
      passOptions.enableStencilLoweringPass = true;
    } else if (arg == "--tensorium-dissipation") {
      passOptions.enableDissipationPass = true;
    } else if (takeOptionValue(arg, "--tensorium-dx", i, argc, argv,
                               optionValue)) {
      passOptions.dx = parseDoubleOption("--tensorium-dx", optionValue);
    } else if (takeOptionValue(arg, "--tensorium-stencil-order", i, argc, argv,
                               optionValue)) {
      passOptions.order =
          parseIntOption("--tensorium-stencil-order", optionValue);
    } else if (takeOptionValue(arg, "--tensorium-dissipation-strength", i, argc,
                               argv, optionValue)) {
      passOptions.dissipationStrength =
          parseDoubleOption("--tensorium-dissipation-strength", optionValue);
    } else if (arg == "--dump-mlir") {
      dumpMLIR = true;
    } else if (arg.rfind("--emit-mlir=", 0) == 0) {
      emitMLIRPath = arg.substr(std::string("--emit-mlir=").size());
      if (emitMLIRPath.empty())
        throw std::runtime_error("--emit-mlir requires a non-empty path");
    } else if (arg == "--emit-mlir") {
      if (i + 1 >= argc)
        throw std::runtime_error("--emit-mlir expects a file path");
      emitMLIRPath = argv[++i];
      if (emitMLIRPath.empty())
        throw std::runtime_error("--emit-mlir requires a non-empty path");
    } else if (arg == "--dump-llvm-ir") {
      dumpLLVMIR = true;
    } else if (arg.rfind("--emit-llvm=", 0) == 0) {
      emitLLVMIRPath = arg.substr(std::string("--emit-llvm=").size());
      if (emitLLVMIRPath.empty())
        throw std::runtime_error("--emit-llvm requires a non-empty path");
    } else if (arg == "--emit-llvm") {
      if (i + 1 >= argc)
        throw std::runtime_error("--emit-llvm expects a file path");
      emitLLVMIRPath = argv[++i];
      if (emitLLVMIRPath.empty())
        throw std::runtime_error("--emit-llvm requires a non-empty path");
    } else if (arg.rfind("--emit-host-header=", 0) == 0) {
      emitHostHeaderPath =
          arg.substr(std::string("--emit-host-header=").size());
      if (emitHostHeaderPath.empty())
        throw std::runtime_error(
            "--emit-host-header requires a non-empty path");
    } else if (arg == "--emit-host-header") {
      if (i + 1 >= argc)
        throw std::runtime_error("--emit-host-header expects a file path");
      emitHostHeaderPath = argv[++i];
      if (emitHostHeaderPath.empty())
        throw std::runtime_error(
            "--emit-host-header requires a non-empty path");
    } else if (arg == "--mlir-disable-threading") {
      passOptions.mlirDisableThreading = true;
    } else if (arg == "--mlir-print-op-on-diagnostic") {
      passOptions.mlirPrintOpOnDiagnostic = true;
    } else if (arg == "--mlir-print-ir-after-failure") {
      passOptions.mlirPrintIRAfterFailure = true;
    } else if (arg == "--mlir-pass-timing") {
      passOptions.mlirPassTiming = true;
    } else if (arg == "--mlir-inline") {
      passOptions.enableMLIRInlinePass = true;
    } else if (arg == "--mlir-no-canonicalize") {
      passOptions.enableMLIRCanonicalizePass = false;
    } else if (arg == "--mlir-no-cse") {
      passOptions.enableMLIRCSEPass = false;
    } else if (arg == "--mlir-strict-pipeline") {
      failOnMLIRPipelineFailure = true;
    } else if (arg == "--mlir-best-effort") {
      failOnMLIRPipelineFailure = false;
    } else if (arg == "--color=always") {
      colorMode = ColorMode::Always;
    } else if (arg == "--color=never") {
      colorMode = ColorMode::Never;
    } else if (arg == "--color=auto") {
      colorMode = ColorMode::Auto;
    } else if (arg == "--color") {
      if (i + 1 >= argc)
        throw std::runtime_error("--color expects one of: auto, always, never");
      std::string modeArg = argv[++i];
      if (modeArg == "always")
        colorMode = ColorMode::Always;
      else if (modeArg == "never")
        colorMode = ColorMode::Never;
      else if (modeArg == "auto")
        colorMode = ColorMode::Auto;
      else
        throw std::runtime_error("--color expects one of: auto, always, never");
    } else if (arg == "--validate") {
      validateOnly = true;
    } else if (arg == "--steps") {
      if (i + 1 >= argc)
        throw std::runtime_error("--steps expects an integer");
      steps = std::stoul(argv[++i]);
    } else if (arg == "--init") {
      if (i + 1 >= argc)
        throw std::runtime_error("--init expects a float");
      initScalar = std::stod(argv[++i]);
    } else if (arg == "--init-alpha") {
      if (i + 1 >= argc)
        throw std::runtime_error("--init-alpha expects a float");
      initAlpha = std::stod(argv[++i]);
    } else if (arg == "--symbolic") {
      compilationMode = CompilationMode::Symbolic;
    } else {
      files.push_back(arg);
    }
  }

  if (files.empty()) {
    PrintDiagnosticOptions opts;
    opts.colorMode = colorMode;
    printDiagnostic(std::cerr, "<command line>", {},
                    DiagnosticLevel::Error, "no input files", {},
                    "E9001", opts);
    return 1;
  }

  if ((!emitMLIRPath.empty() || !emitLLVMIRPath.empty() ||
       !emitHostHeaderPath.empty()) &&
      files.size() != 1) {
    PrintDiagnosticOptions opts;
    opts.colorMode = colorMode;
    printDiagnostic(std::cerr, "<command line>", {}, DiagnosticLevel::Error,
                    "--emit-mlir/--emit-llvm/--emit-host-header require exactly one input file",
                    {}, "E9002", opts);
    return 1;
  }
  if (!constraintCsvPath.empty() && files.size() != 1) {
    PrintDiagnosticOptions opts;
    opts.colorMode = colorMode;
    printDiagnostic(std::cerr, "<command line>", {}, DiagnosticLevel::Error,
                    "--export-constraint-csv requires exactly one input file",
                    {}, "E9003", opts);
    return 1;
  }

  const bool hasExplicitTensoriumPipelineSelection =
      passOptions.enableNoOpPass || passOptions.enableAnalysisPass ||
      passOptions.enableEinsteinLoweringPass ||
      passOptions.enableIndexRoleAnalysisPass ||
      passOptions.enableEinsteinValidityPass ||
      passOptions.enableIndexAnalyzePass ||
      passOptions.enableEinsteinCanonicalizePass ||
      passOptions.enableEinsteinAnalyzeEinsumPass ||
      passOptions.enableMetricLoweringPass ||
      passOptions.enableInitStdLoweringPass ||
      passOptions.enableInitGridScfPass ||
      passOptions.enableInitGridAffinePass ||
      passOptions.enableRhsGridScfPass ||
      passOptions.enableRhsGridAffinePass ||
      passOptions.enableStripSourceFuncsPass ||
      passOptions.enableStencilLoweringPass ||
      passOptions.enableDissipationPass;

  // Make LLVM IR emission usable out-of-the-box for executable kernels.
  if ((dumpLLVMIR || !emitLLVMIRPath.empty() || !emitHostHeaderPath.empty()) &&
      compilationMode == CompilationMode::Executable &&
      !hasExplicitTensoriumPipelineSelection) {
    passOptions.enableMetricLoweringPass = true;
    passOptions.enableInitStdLoweringPass = true;
    passOptions.enableInitGridAffinePass = true;
    passOptions.enableRhsGridAffinePass = true;
    passOptions.enableStripSourceFuncsPass = true;
  }

  PrintDiagnosticOptions diagPrintOpts;
  diagPrintOpts.colorMode = colorMode;
  std::string currentPath;
  std::string currentSource;

  try {
    for (const auto &path : files) {
      bool fileOK = true;
      currentPath = path;
      currentSource.clear();
      std::cerr << "[Tensorium] parsing " << path << "\n";

      currentSource = readFile(path);

      Lexer lex(currentSource.c_str());
      Parser parser(lex);
      Program prog = parser.parseProgram();

      SemanticAnalyzer sem(prog, compilationMode);
      for (const auto &warn : sem.getWarnings()) {
        printDiagnostic(std::cerr, path, currentSource,
                        DiagnosticLevel::Warning, warn, {}, "", diagPrintOpts);
      }
      std::vector<IndexedEvolution> indexedEvos;

      if (dumpIndexed) {
        for (const auto &evo : prog.evolutions) {
          indexedEvos.push_back(sem.analyzeEvolution(evo));
        }
      }
      if (dumpAST) {
        std::cout << "\n=== AST DUMP (" << path << ") ===\n";
        printProgram(prog);
        std::cout << "==============================\n";
      }

      if (dumpIndexed) {
        std::cout << "\n=== INDEXED AST (" << path << ") ===\n";

        for (const auto &evo : indexedEvos) {
          std::cout << "Evolution " << evo.name << " {\n";

          for (const auto &eq : evo.equations) {
            std::cout << "  dt " << eq.fieldName;

            if (!eq.indices.empty()) {
              std::cout << "[";
              for (size_t i = 0; i < eq.indices.size(); ++i) {
                std::cout << eq.indices[i];
                if (i + 1 < eq.indices.size())
                  std::cout << ",";
              }
              std::cout << "]";
            }

            std::cout << " = ";
            printIndexedExpr(eq.rhs.get());
            std::cout << "\n";
          }

          std::cout << "}\n";
        }

        std::cout << "==============================\n";
      }
      auto mod = tensorium::backend::BackendBuilder::build(prog, sem);
      tensorium::validation::canonicalizeDifferentialIR(mod);
      tensorium::validation::canonicalizeEinsteinIR(mod);
      auto irResult = tensorium::validation::verifyIR(mod);
      for (const auto &d : irResult.diags) {
        const auto level = d.kind == tensorium::validation::Diagnostic::Kind::Error
                               ? DiagnosticLevel::Error
                               : DiagnosticLevel::Warning;
        printDiagnostic(std::cerr, path, currentSource, level, d.message, {}, "",
                        diagPrintOpts);
      }
      if (!irResult.ok())
        return 1;
      if (validateOnly) {
        auto result = tensorium::validation::validateProgram(mod);

        for (const auto &d : result.diags) {
          const auto level = d.kind == tensorium::validation::Diagnostic::Kind::Error
                                 ? DiagnosticLevel::Error
                                 : DiagnosticLevel::Warning;
          printDiagnostic(std::cerr, path, currentSource, level, d.message, {},
                          "", diagPrintOpts);
        }

        if (!result.ok())
          return 1;

        std::cout << "[Tensorium] validation OK: " << path << "\n";
        continue;
      }
      if (dumpBackend) {
        std::cout << "\n=== BACKEND IR (" << path << ") ===\n";
        if (mod.simulation) {
          std::cout << "Simulation:\n";
          std::cout << "  dim = " << mod.simulation->dimension << "\n";
          std::cout << "  dt  = " << mod.simulation->time.dt << "\n";
        }

        if (mod.constraintProblem) {
          const auto &problem = *mod.constraintProblem;
          std::cout << "ConstraintProblem:\n";
          std::cout << "  " << problem.name << " (" << problem.domains.size()
                    << " domains, " << problem.unknowns.size() << " unknowns, "
                    << problem.equations.size() << " residuals)\n";
        }

        std::cout << "Fields:\n";
        for (const auto &f : mod.fields) {
          std::cout << "  " << f.name << " (up=" << f.tensorType.up << ",down=" << f.tensorType.down
                    << ")\n";
        }

        if (!mod.prints.empty()) {
          std::cout << "Prints:\n";
          for (const auto &p : mod.prints) {
            std::cout << "  " << p.label << " (up=" << p.tensorType.up
                      << ",down=" << p.tensorType.down << ")\n";
          }
        }

        std::cout << "Evolutions:\n";
        for (const auto &evo : mod.evolutions) {
          std::cout << "  Evolution " << evo.name << " ("
                    << evo.equations.size() << " eqs)\n";
        }
        std::cout << "==============================\n";
      }
      if (dumpBackendExpr) {
        std::cout << "\n=== BACKEND IR FULL (" << path << ") ===\n";
        tensorium::backend::printModuleIR(mod);
        std::cout << "==============================\n";
      }

      if (solveConstraints) {
        tensorium::solver::ConstraintSolveRequest request;
        request.parameters = constraintParameters;
        auto solution = tensorium::solver::solveRadialConstraintProblem(
            mod, request);
        if (!solution.converged) {
          throw std::runtime_error(
              "constraint Newton solve did not converge after " +
              std::to_string(solution.iterations) +
              " iterations; residual_inf=" +
              std::to_string(solution.residualNorm));
        }
        std::cout << std::setprecision(17);
        std::cout << "[ConstraintSolve] converged=true iterations="
                  << solution.iterations
                  << " residual_inf=" << solution.residualNorm
                  << " domains=" << solution.domains.size() << "\n";
        for (const auto &domain : solution.domains) {
          std::cout << "[ConstraintSolve] domain=" << domain.name
                    << " offset=" << domain.offset
                    << " points=" << domain.pointCount
                    << " compactified="
                    << (domain.compactified ? "true" : "false") << "\n";
        }
        for (const auto &unknownLayout : solution.unknownLayouts) {
          const auto &values = solution.unknowns.at(unknownLayout.name);
          std::cout << "[ConstraintSolve] unknown=" << unknownLayout.name
                    << " points=" << unknownLayout.pointsPerComponent
                    << " components=" << unknownLayout.componentCount
                    << " values=" << values.size();
          if (unknownLayout.componentCount == 1 && !values.empty())
            std::cout << " inner=" << values.front()
                      << " outer=" << values.back();
          std::cout << "\n";
          if (unknownLayout.componentCount > 1) {
            for (std::size_t component = 0;
                 component < unknownLayout.componentCount; ++component) {
              const std::size_t offset =
                  component * unknownLayout.pointsPerComponent;
              std::cout << "[ConstraintSolve] unknown=" << unknownLayout.name
                        << " component=" << component
                        << " inner=" << values[offset]
                        << " outer="
                        << values[offset + unknownLayout.pointsPerComponent - 1]
                        << "\n";
            }
          }
        }
        if (solution.physicalCtt) {
          const auto &physical = *solution.physicalCtt;
          std::cout << "[ConstraintSolve] physical_ctt basis="
                    << physical.basis
                    << " points=" << physical.meanCurvature.size()
                    << " gamma_radial_inner="
                    << physical.spatialMetricRadial.front()
                    << " gamma_radial_outer="
                    << physical.spatialMetricRadial.back()
                    << " k_radial_inner="
                    << physical.extrinsicCurvatureRadial.front()
                    << " k_radial_outer="
                    << physical.extrinsicCurvatureRadial.back() << "\n";
        }
        if (!constraintCsvPath.empty()) {
          writeConstraintCsv(constraintCsvPath, solution);
          std::cout << "[ConstraintSolve] exported_csv=" << constraintCsvPath
                    << "\n";
        }
      }

      auto makeMLIRGenOptions = [&]() {
        const auto level =
            hasOptimizationLevel ? optimizationLevel
                                 : tensorium_mlir::OptimizationLevel::O0;
        return tensorium_mlir::makeMLIRGenOptions(level, passOptions);
      };
      const bool emitLLVMOutput = dumpLLVMIR || !emitLLVMIRPath.empty();
      const bool emitHostHeaderOutput = !emitHostHeaderPath.empty();
      const bool emitCombinedLLVMAndHostHeader =
          emitLLVMOutput && emitHostHeaderOutput;

      if (dumpMLIR || !emitMLIRPath.empty()) {
        if (dumpMLIR)
          std::cerr << "\n=== MLIR DUMP (" << path << ") ===\n";
        auto opts = makeMLIRGenOptions();
        std::string mlirText;
        const bool pipelineOK = tensorium_mlir::emitMLIR(mod, opts, &mlirText);
        if (dumpMLIR)
          std::cout << mlirText;
        if (!emitMLIRPath.empty())
          writeFile(emitMLIRPath, mlirText);
        if (!pipelineOK) {
          fileOK = false;
          printDiagnostic(std::cerr, path, currentSource, DiagnosticLevel::Error,
                          "MLIR pipeline failed", {}, "E3101",
                          diagPrintOpts);
          if (failOnMLIRPipelineFailure)
            return 1;
        }
        if (dumpMLIR)
          std::cerr << "==============================\n";
      }
      if (emitCombinedLLVMAndHostHeader) {
        if (dumpLLVMIR)
          std::cerr << "\n=== LLVM IR DUMP (" << path << ") ===\n";
        auto opts = makeMLIRGenOptions();
        std::string llvmIRText;
        std::string headerText;
        const bool pipelineOK = tensorium_mlir::emitLLVMIRAndHostHeader(
            mod, opts, &llvmIRText, &headerText);
        if (dumpLLVMIR)
          std::cout << llvmIRText;
        if (pipelineOK && !emitLLVMIRPath.empty())
          writeFile(emitLLVMIRPath, llvmIRText);
        if (pipelineOK)
          writeFile(emitHostHeaderPath, headerText);
        if (!pipelineOK) {
          fileOK = false;
          printDiagnostic(std::cerr, path, currentSource, DiagnosticLevel::Error,
                          "LLVM IR and host header generation pipeline failed",
                          {}, "E3102", diagPrintOpts);
          if (failOnMLIRPipelineFailure)
            return 1;
        }
        if (dumpLLVMIR)
          std::cerr << "==============================\n";
      }
      if (!emitCombinedLLVMAndHostHeader && emitLLVMOutput) {
        if (dumpLLVMIR)
          std::cerr << "\n=== LLVM IR DUMP (" << path << ") ===\n";
        auto opts = makeMLIRGenOptions();
        std::string llvmIRText;
        const bool pipelineOK = tensorium_mlir::emitLLVMIR(mod, opts, &llvmIRText);
        if (dumpLLVMIR)
          std::cout << llvmIRText;
        if (pipelineOK && !emitLLVMIRPath.empty())
          writeFile(emitLLVMIRPath, llvmIRText);
        if (!pipelineOK) {
          fileOK = false;
          printDiagnostic(std::cerr, path, currentSource, DiagnosticLevel::Error,
                          "LLVM IR pipeline failed", {}, "E3102",
                          diagPrintOpts);
          if (failOnMLIRPipelineFailure)
            return 1;
        }
        if (dumpLLVMIR)
          std::cerr << "==============================\n";
      }
      if (!emitCombinedLLVMAndHostHeader && emitHostHeaderOutput) {
        auto opts = makeMLIRGenOptions();
        std::string headerText;
        const bool pipelineOK =
            tensorium_mlir::emitHostHeader(mod, opts, &headerText);
        if (pipelineOK)
          writeFile(emitHostHeaderPath, headerText);
        if (!pipelineOK) {
          fileOK = false;
          printDiagnostic(std::cerr, path, currentSource, DiagnosticLevel::Error,
                          "host header generation pipeline failed", {},
                          "E3103", diagPrintOpts);
          if (failOnMLIRPipelineFailure)
            return 1;
        }
      }
      if (runCpu) {
        tensorium::runtime::RunOptions opt;
        opt.steps = steps;

        auto st = tensorium::runtime::initState1D(mod, initScalar, initAlpha);
        tensorium::runtime::runEuler1D(mod, st, opt);

        for (const auto &kv : st.fields) {
          std::cout << "\n[CPU] Field " << kv.first << " first values: ";
          for (size_t i = 0; i < kv.second.size() && i < 8; ++i) {
            std::cout << kv.second[i] << " ";
          }
          std::cout << "\n";
        }
      }
      if (fileOK)
        std::cout << "[Tensorium] OK: " << path << "\n";
      else
        std::cout << "[Tensorium] FAILED: " << path << "\n";
    }

  } catch (const DiagnosticError &d) {
    const std::string contextPath =
        currentPath.empty() ? "<command line>" : currentPath;
    printDiagnostic(std::cerr, contextPath, currentSource, d.level(),
                    d.message(), d.location(), d.code(), diagPrintOpts);
    return 1;
  } catch (const std::exception &e) {
    const std::string contextPath =
        currentPath.empty() ? "<command line>" : currentPath;
    printDiagnostic(std::cerr, contextPath, currentSource,
                    DiagnosticLevel::Error, e.what(), {}, "", diagPrintOpts);
    return 1;
  }

  return 0;
}
