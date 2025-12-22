#include "tensorium/AST/ASTPrinter.hpp"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"

#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Backend/IR.hpp"
#include "tensorium/Backend/IRPrinter.hpp"
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
  bool dumpAST, dumpIndexed = false;
  bool dumpBackend, dumpBackendExpr = false;
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
    } else if (arg == "--dump-backend")
      dumpBackend = true;
    else if (arg == "--dump-backend-expr")
      dumpBackendExpr = true;
    else {
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

      SemanticAnalyzer sem(prog);

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
      if (dumpBackend) {
        auto mod = tensorium::backend::BackendBuilder::build(prog, sem);

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
        auto mod = tensorium::backend::BackendBuilder::build(prog, sem);

        std::cout << "\n=== BACKEND IR FULL (" << path << ") ===\n";
        tensorium::backend::printModuleIR(mod);
        std::cout << "==============================\n";
      }
      std::cout << "[Tensorium] OK: " << path << "\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Tensorium error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
