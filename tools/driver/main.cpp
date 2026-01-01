#include "tensorium/AST/ASTPrinter.hpp"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"

#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Backend/DomainIR.hpp"
#include "tensorium/Backend/IRPrinter.hpp"
#include "tensorium/Runtime/CpuRuntime.hpp"
#include "tensorium/Runtime/Eval.hpp"
#include "tensorium/Sema/ProgramValidator.hpp"
#include "tensorium_mlir/Target/MLIRGen/MLIRGen.h"

#include <fstream>
#include <iostream>
#include <sstream>
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

static void printIndexedExpr(const IndexedExpr *e) {
  if (auto n = dynamic_cast<const IndexedNumber *>(e)) {
    std::cout << n->value;
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
    return;
  }

  if (auto b = dynamic_cast<const IndexedBinary *>(e)) {
    std::cout << "(";
    printIndexedExpr(b->lhs.get());
    std::cout << " " << b->op << " ";
    printIndexedExpr(b->rhs.get());
    std::cout << ")";
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
    return;
  }
}

int main(int argc, char **argv) {
  bool dumpAST = false;
  bool dumpIndexed = false;
  bool dumpBackend, dumpBackendExpr = false;
  bool runCpu = false;
  size_t steps = 10;
  double initScalar = 1.0;
  double initAlpha = 2.0;
  bool dumpMLIR = false;
  bool enableNoOpPass = false;
  bool enableAnalysisPass = false;
  bool validateOnly = false;
  bool enableEinsteinLoweringPass = false;
  bool enableEinsteinValidityPass = false;
  bool enableIndexAnalyzePass = false;
  bool enableEinsteinCanonicalizePass = false;
  bool enableEinsteinAnalyzeEinsumPass = false;
  bool enableStencilLoweringPass = false;
  bool enableDissipationPass = false;
  CompilationMode compilationMode = CompilationMode::Executable;

  if (argc < 2) {
    std::cerr << "usage: Tensorium_cc [--dump-ast] file1.tn [file2.tn ...]\n";
    return 1;
  }

  std::vector<std::string> files;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--dump-ast") {
      dumpAST = true;
    } else if (arg == "--dump-indexed") {
      dumpIndexed = true;
    } else if (arg == "--dump-backend") {
      dumpBackend = true;
    } else if (arg == "--dump-backend-expr") {
      dumpBackendExpr = true;
    } else if (arg == "--tensorium-noop") {
      enableNoOpPass = true;
    } else if (arg == "--tensorium-analyze") {
      enableAnalysisPass = true;
    } else if (arg == "--run-cpu") {
      runCpu = true;
    } else if (arg == "--tensorium-einstein-lower") {
      enableEinsteinLoweringPass = true;
    } else if (arg == "--tensorium-index-analyze") {
      enableIndexAnalyzePass = true;
    } else if (arg == "--tensorium-einstein-validate") {
      enableEinsteinValidityPass = true;
    } else if (arg == "--tensorium-einstein-canonicalize") {
      enableEinsteinCanonicalizePass = true;
    } else if (arg == "--tensorium-einstein-analyze-einsum") {
      enableEinsteinAnalyzeEinsumPass = true;
    } else if (arg == "--tensorium-stencil-lower") {
      enableStencilLoweringPass = true;
    } else if (arg == "--tensorium-dissipation") {
      enableDissipationPass = true;
    } else if (arg == "--dump-mlir") {
      dumpMLIR = true;
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
    std::cerr << "error: no input files\n";
    return 1;
  }

  try {
    for (const auto &path : files) {
      std::cout << "[Tensorium] parsing " << path << "\n";

      std::string src = readFile(path);

      Lexer lex(src.c_str());
      Parser parser(lex);
      Program prog = parser.parseProgram();

      SemanticAnalyzer sem(prog, compilationMode);
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
      const bool isSymbolicMode = compilationMode == CompilationMode::Symbolic;
      const bool executableRequested = validateOnly || dumpMLIR || runCpu;
      if (isSymbolicMode && executableRequested) {
        std::cerr << "Symbolically valid but not executable: missing simulation metadata\n";
        continue;
      }
      if (validateOnly) {
        auto result = tensorium::sema::validateProgram(mod);

        for (const auto &d : result.diags) {
          std::cerr << (d.kind == tensorium::sema::Diagnostic::Kind::Error
                            ? "error: "
                            : "warning: ")
                    << d.message << "\n";
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

        std::cout << "Fields:\n";
        for (const auto &f : mod.fields) {
          std::cout << "  " << f.name << " (up=" << f.up << ",down=" << f.down
                    << ")\n";
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
      if (dumpMLIR) {
        std::cout << "\n=== MLIR DUMP (" << path << ") ===\n";

        tensorium_mlir::MLIRGenOptions opts;
        opts.enableNoOpPass = enableNoOpPass;
        opts.enableAnalysisPass = enableAnalysisPass;

        opts.enableEinsteinLoweringPass = enableEinsteinLoweringPass;
        opts.enableEinsteinValidityPass = enableEinsteinValidityPass;
        opts.enableIndexAnalyzePass = enableIndexAnalyzePass;
        opts.enableEinsteinCanonicalizePass = enableEinsteinCanonicalizePass;
        opts.enableEinsteinAnalyzeEinsumPass = enableEinsteinAnalyzeEinsumPass;
        opts.enableStencilLoweringPass = enableStencilLoweringPass;
		opts.enableDissipationPass = enableDissipationPass;
        tensorium_mlir::emitMLIR(mod, opts);
        std::cout << "==============================\n";
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
      std::cout << "[Tensorium] OK: " << path << "\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Tensorium error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
